import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

DEFAULT_VITE = os.getenv('VITE_API_BASE_URL', "http://127.0.0.1:8000")
DEFAULT_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))
DEFAULT_MAX_BODY = int(os.getenv("MAX_BODY_CHARS", "1200"))

ValidationResult = Tuple[List[str], List[str], List[str]]


@dataclass
class TestContext:
    VITE: str
    timeout: float
    max_body_chars: int
    schedule_game: Optional[Dict[str, Any]] = None
    prediction: Optional[Dict[str, Any]] = None


@dataclass
class EndpointTest:
    name: str
    method: str
    path: str
    build_payload: Optional[Callable[[TestContext], Optional[Dict[str, Any]]]] = None
    validate: Optional[Callable[[TestContext, Any, Optional[Dict[str, Any]]], ValidationResult]] = None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... (truncated, {len(text)} chars total)"


def _format_json(data: Any, limit: int) -> str:
    try:
        rendered = json.dumps(data, indent=2, default=str, sort_keys=True)
    except TypeError:
        rendered = str(data)
    return _truncate(rendered, limit)


def _require_type(
    errors: List[str],
    value: Any,
    expected: Tuple[type, ...],
    path: str,
) -> None:
    if not isinstance(value, expected):
        exp = " or ".join(t.__name__ for t in expected)
        errors.append(f"{path}: expected {exp}, got {type(value).__name__}")


def _parse_json(response: requests.Response) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return response.json(), None
    except ValueError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _format_detail_errors(detail: Any) -> List[str]:
    if isinstance(detail, list):
        lines = []
        for item in detail:
            if isinstance(item, dict):
                loc = ".".join(str(p) for p in item.get("loc", []))
                msg = item.get("msg", "")
                typ = item.get("type", "")
                label = f"{loc}: {msg}".strip(": ")
                if typ:
                    label = f"{label} ({typ})"
                lines.append(label)
            else:
                lines.append(str(item))
        return lines
    if detail is None:
        return []
    return [str(detail)]


def _response_hints(detail_lines: List[str]) -> List[str]:
    hints = []
    joined = " ".join(detail_lines).lower()
    if "models" in joined or "model" in joined:
        hints.append("Check MODELS_DIR and ensure model artifacts load at startup.")
    if "dataset" in joined:
        hints.append("Check DATA_DIR/DATASET_PATH and confirm the dataset exists and loads.")
    if "messages cannot be empty" in joined:
        hints.append("Ensure /llm/chat sends at least one message.")
    if "model engine not initialized" in joined:
        hints.append("Verify backend startup logs for model initialization errors.")
    return hints


def _print_fail(
    test: EndpointTest,
    url: str,
    elapsed_ms: Optional[int],
    message: str,
    payload: Optional[Dict[str, Any]] = None,
    response: Optional[requests.Response] = None,
    response_json: Optional[Any] = None,
    json_error: Optional[str] = None,
    max_body_chars: int = DEFAULT_MAX_BODY,
) -> None:
    timing = f"{elapsed_ms}ms" if elapsed_ms is not None else "n/a"
    print(f"[FAIL] {test.name}: {test.method} {url} ({timing})")
    print(f"  error: {message}")
    if payload is not None:
        print(f"  request: {_format_json(payload, max_body_chars)}")
    if response is not None:
        print(f"  status: {response.status_code}")
        content_type = response.headers.get("content-type", "")
        if content_type:
            print(f"  content-type: {content_type}")
    if json_error:
        print(f"  json_error: {json_error}")
    if response_json is not None:
        detail = response_json.get("detail") if isinstance(response_json, dict) else None
        detail_lines = _format_detail_errors(detail)
        if detail_lines:
            print("  detail:")
            for line in detail_lines:
                print(f"    - {line}")
            for hint in _response_hints(detail_lines):
                print(f"  hint: {hint}")
        print(f"  body: {_format_json(response_json, max_body_chars)}")
    elif response is not None and response.text:
        print(f"  body: {_truncate(response.text, max_body_chars)}")


def _print_pass(
    test: EndpointTest,
    url: str,
    elapsed_ms: int,
    notes: List[str],
    warnings: List[str],
) -> None:
    note_str = f" - {notes[0]}" if notes else ""
    print(f"[PASS] {test.name}: {test.method} {url} ({elapsed_ms}ms){note_str}")
    for line in notes[1:]:
        print(f"  note: {line}")
    for warn in warnings:
        print(f"  warn: {warn}")


def _send_request(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]],
    timeout: float,
) -> Tuple[Optional[requests.Response], int, Optional[str]]:
    start = time.perf_counter()
    try:
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=payload, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return response, elapsed_ms, None
    except requests.RequestException as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return None, elapsed_ms, f"{type(exc).__name__}: {exc}"


def _validate_health(ctx: TestContext, data: Any, payload: Optional[Dict[str, Any]]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []
    if not isinstance(data, dict):
        return ["root: expected object"], warnings, notes
    for key in ("status", "mode", "reason"):
        if key not in data:
            errors.append(f"missing field: {key}")
    status = data.get("status")
    if isinstance(status, str):
        notes.append(f"status={status}")
        if status == "unhealthy":
            warnings.append("service reports unhealthy")
    mode = data.get("mode")
    if isinstance(mode, str):
        notes.append(f"mode={mode}")
    return errors, warnings, notes


def _validate_debug(ctx: TestContext, data: Any, payload: Optional[Dict[str, Any]]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []
    if not isinstance(data, dict):
        return ["root: expected object"], warnings, notes
    for key in ("status", "timestamp", "config", "dataset_info"):
        if key not in data:
            errors.append(f"missing field: {key}")
    if "dataset_info" in data and isinstance(data["dataset_info"], dict):
        shape = data["dataset_info"].get("shape")
        if shape is not None:
            notes.append(f"dataset_shape={shape}")
    return errors, warnings, notes


def _validate_schedule(ctx: TestContext, data: Any, payload: Optional[Dict[str, Any]]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []
    if not isinstance(data, dict):
        return ["root: expected object"], warnings, notes
    games = data.get("games")
    if games is None:
        errors.append("missing field: games")
        return errors, warnings, notes
    _require_type(errors, games, (list,), "games")
    if isinstance(games, list):
        notes.append(f"games={len(games)}")
        if not games:
            warnings.append("schedule returned zero games")
        else:
            ctx.schedule_game = games[0]
            season = games[0].get("season")
            week = games[0].get("week")
            if season is not None and week is not None:
                notes.append(f"first_game={season}W{week}")
    return errors, warnings, notes


def _build_predict_payload(ctx: TestContext) -> Dict[str, Any]:
    fallback = {
        "home_team": "LAC",
        "away_team": "KC",
        "season": 2025,
        "week": 1,
    }
    game = ctx.schedule_game or {}
    home = game.get("home_team")
    away = game.get("away_team")
    season = game.get("season")
    week = game.get("week")
    if home and away and season is not None and week is not None:
        try:
            return {
                "home_team": str(home),
                "away_team": str(away),
                "season": int(season),
                "week": int(week),
            }
        except (TypeError, ValueError):
            return fallback
    return fallback


def _validate_predict(ctx: TestContext, data: Any, payload: Optional[Dict[str, Any]]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []
    if not isinstance(data, dict):
        return ["root: expected object"], warnings, notes
    required = [
        "home_score",
        "away_score",
        "home_win_probability",
        "away_win_probability",
        "point_diff",
        "prediction_source",
        "game_id",
    ]
    for key in required:
        if key not in data:
            errors.append(f"missing field: {key}")
    p_home = data.get("home_win_probability")
    if isinstance(p_home, (int, float)):
        if not (0.0 <= float(p_home) <= 1.0):
            errors.append(f"home_win_probability out of range: {p_home}")
    else:
        errors.append("home_win_probability: expected number")
    if "home_score" in data and "away_score" in data:
        notes.append(f"score={data.get('home_score')}:{data.get('away_score')}")
    if "prediction_source" in data:
        notes.append(f"source={data.get('prediction_source')}")
    ctx.prediction = data
    return errors, warnings, notes


def _build_explain_payload(ctx: TestContext) -> Dict[str, Any]:
    payload = _build_predict_payload(ctx)
    if ctx.prediction:
        payload["prediction"] = ctx.prediction
    return payload


def _validate_explain(ctx: TestContext, data: Any, payload: Optional[Dict[str, Any]]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []
    if not isinstance(data, dict):
        return ["root: expected object"], warnings, notes
    for key in ("game_id", "explanation", "bullets", "caveats", "used_llm"):
        if key not in data:
            errors.append(f"missing field: {key}")
    if "explanation" in data and isinstance(data["explanation"], str):
        notes.append(f"explanation_len={len(data['explanation'])}")
    return errors, warnings, notes


def _build_chat_payload(ctx: TestContext) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "Summarize the prediction in one sentence."}
        ]
    }
    if ctx.prediction:
        payload["prediction"] = ctx.prediction
    return payload


def _validate_chat(ctx: TestContext, data: Any, payload: Optional[Dict[str, Any]]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []
    if not isinstance(data, dict):
        return ["root: expected object"], warnings, notes
    for key in ("reply", "used_llm"):
        if key not in data:
            errors.append(f"missing field: {key}")
    if "reply" in data and isinstance(data["reply"], str):
        notes.append(f"reply_len={len(data['reply'])}")
    if "used_llm" in data:
        notes.append(f"used_llm={data.get('used_llm')}")
    return errors, warnings, notes


def _validate_history(ctx: TestContext, data: Any, payload: Optional[Dict[str, Any]]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []
    if not isinstance(data, dict):
        return ["root: expected object"], warnings, notes
    entries = data.get("entries")
    total = data.get("total")
    if entries is None:
        errors.append("missing field: entries")
    else:
        _require_type(errors, entries, (list,), "entries")
        if isinstance(entries, list):
            notes.append(f"entries={len(entries)}")
    if total is None:
        errors.append("missing field: total")
    else:
        _require_type(errors, total, (int,), "total")
    return errors, warnings, notes


def _validate_status(ctx: TestContext, data: Any, payload: Optional[Dict[str, Any]]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    notes: List[str] = []
    if not isinstance(data, dict):
        return ["root: expected object"], warnings, notes
    for key in ("health", "dataset", "history"):
        if key not in data:
            errors.append(f"missing field: {key}")
    if "health" in data and isinstance(data["health"], dict):
        status = data["health"].get("status")
        if status:
            notes.append(f"health={status}")
    return errors, warnings, notes


def _run_tests(ctx: TestContext) -> int:
    tests: List[EndpointTest] = [
        EndpointTest("Health", "GET", "/health", validate=_validate_health),
        EndpointTest("Debug", "GET", "/debug", validate=_validate_debug),
        EndpointTest("Schedule", "GET", "/schedule/next-week", validate=_validate_schedule),
        EndpointTest("Predict", "POST", "/predict", build_payload=_build_predict_payload, validate=_validate_predict),
        EndpointTest("Predict Explain", "POST", "/predict/explain", build_payload=_build_explain_payload, validate=_validate_explain),
        EndpointTest("LLM Chat", "POST", "/llm/chat", build_payload=_build_chat_payload, validate=_validate_chat),
        EndpointTest("History", "GET", "/history?limit=5", validate=_validate_history),
        EndpointTest("Status Overview", "GET", "/status/overview", validate=_validate_status),
    ]

    failures = 0
    for test in tests:
        payload = test.build_payload(ctx) if test.build_payload else None
        url = ctx.VITE.rstrip("/") + test.path
        response, elapsed_ms, request_error = _send_request(
            test.method, url, payload, ctx.timeout
        )
        if request_error:
            failures += 1
            _print_fail(
                test,
                url,
                elapsed_ms,
                request_error,
                payload=payload,
                max_body_chars=ctx.max_body_chars,
            )
            continue
        if response is None:
            failures += 1
            _print_fail(
                test,
                url,
                elapsed_ms,
                "No response received",
                payload=payload,
                max_body_chars=ctx.max_body_chars,
            )
            continue
        response_json, json_error = _parse_json(response)
        if response.status_code != 200:
            failures += 1
            _print_fail(
                test,
                url,
                elapsed_ms,
                f"Unexpected status {response.status_code}",
                payload=payload,
                response=response,
                response_json=response_json,
                json_error=json_error,
                max_body_chars=ctx.max_body_chars,
            )
            continue
        if json_error:
            failures += 1
            _print_fail(
                test,
                url,
                elapsed_ms,
                "Expected JSON response",
                payload=payload,
                response=response,
                response_json=None,
                json_error=json_error,
                max_body_chars=ctx.max_body_chars,
            )
            continue
        if test.validate:
            errors, warnings, notes = test.validate(ctx, response_json, payload)
            if errors:
                failures += 1
                _print_fail(
                    test,
                    url,
                    elapsed_ms,
                    "; ".join(errors),
                    payload=payload,
                    response=response,
                    response_json=response_json,
                    max_body_chars=ctx.max_body_chars,
                )
                continue
            _print_pass(test, url, elapsed_ms, notes, warnings)
        else:
            _print_pass(test, url, elapsed_ms, [], [])

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test NFL ML API endpoints.")
    parser.add_argument("--base-url", default=DEFAULT_VITE, help="API base URL")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout (seconds)")
    parser.add_argument("--max-body-chars", type=int, default=DEFAULT_MAX_BODY, help="Max response chars to print")
    args = parser.parse_args()

    ctx = TestContext(
        VITE=args.VITE,
        timeout=args.timeout,
        max_body_chars=args.max_body_chars,
    )

    failures = _run_tests(ctx)
    if failures:
        print(f"\n{failures} endpoint test(s) failed.")
        sys.exit(1)
    print("\nAll endpoints passed.")


if __name__ == "__main__":
    main()
