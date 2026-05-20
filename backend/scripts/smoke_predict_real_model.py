r"""
Smoke check for the real-model /predict path.

Run from the repository root:
    python backend\scripts\smoke_predict_real_model.py

The script prefers FastAPI's in-process TestClient. If importing the app or
creating the client fails, it falls back to an already-running backend at
SMOKE_PREDICT_BASE_URL or http://127.0.0.1:8000.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREDICT_PAYLOAD = {
    "season": 2025,
    "week": 1,
    "home_team": "BUF",
    "away_team": "BAL",
}

EXPECTED = {
    "fallback_used": False,
    "probability_source": "win_classifier",
    "row_source": "dataset_exact",
    "win_classifier_used": True,
}


class SmokeFailure(RuntimeError):
    """Raised when the smoke check receives an unhealthy prediction payload."""


def _dig(payload: dict[str, Any], *path: str) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _first_present(payload: dict[str, Any], paths: list[tuple[str, ...]]) -> tuple[Any, str | None]:
    for path in paths:
        value = _dig(payload, *path)
        if value is not None:
            return value, ".".join(path)
    return None, None


def _post_with_testclient() -> tuple[dict[str, Any], str]:
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        response = client.post("/predict", json=PREDICT_PAYLOAD)
    if response.status_code != 200:
        raise SmokeFailure(
            "In-process /predict returned "
            f"HTTP {response.status_code}: {response.text}"
        )
    return response.json(), "FastAPI TestClient"


def _post_with_http() -> tuple[dict[str, Any], str]:
    base_url = os.getenv("SMOKE_PREDICT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    url = f"{base_url}/predict"
    body = json.dumps(PREDICT_PAYLOAD).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
            if response.status != 200:
                raise SmokeFailure(
                    f"HTTP /predict returned {response.status}: {response_body}"
                )
            return json.loads(response_body), url
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SmokeFailure(f"HTTP /predict returned {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SmokeFailure(
            f"Could not reach {url}. Start the backend or set SMOKE_PREDICT_BASE_URL. "
            f"Original error: {exc.reason}"
        ) from exc


def _post_predict() -> tuple[dict[str, Any], str]:
    try:
        return _post_with_testclient()
    except SmokeFailure:
        raise
    except Exception as exc:
        print(
            "In-process TestClient path was not usable; falling back to HTTP. "
            f"Reason: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        if os.getenv("SMOKE_VERBOSE_IMPORT_ERRORS") == "1":
            traceback.print_exc()
        return _post_with_http()


def _diagnostics(payload: dict[str, Any]) -> dict[str, tuple[Any, str | None]]:
    return {
        "fallback_used": _first_present(
            payload,
            [
                ("fallback_used",),
                ("diagnostics", "fallback_used"),
                ("model_diagnostics", "fallback_used"),
                ("explanation_fields", "fallback_used"),
            ],
        ),
        "probability_source": _first_present(
            payload,
            [
                ("probability_source",),
                ("diagnostics", "probability_source"),
                ("model_diagnostics", "probability_source"),
                ("explanation_fields", "probability_source"),
            ],
        ),
        "row_source": _first_present(
            payload,
            [
                ("row_source",),
                ("selected_row_source",),
                ("diagnostics", "row_source"),
                ("model_diagnostics", "row_source"),
                ("explanation_fields", "row_source"),
                ("explanation_fields", "selected_row_source"),
            ],
        ),
        "win_classifier_used": _first_present(
            payload,
            [
                ("win_classifier_used",),
                ("diagnostics", "win_classifier_used"),
                ("model_diagnostics", "win_classifier_used"),
                ("explanation_fields", "win_classifier_used"),
            ],
        ),
    }


def _assert_real_model_diagnostics(payload: dict[str, Any]) -> None:
    diagnostics = _diagnostics(payload)
    failures: list[str] = []

    for name, expected_value in EXPECTED.items():
        actual_value, source_path = diagnostics[name]
        if source_path is None:
            failures.append(
                f"Missing diagnostic '{name}'. Checked top-level, diagnostics, "
                "model_diagnostics, and explanation_fields aliases."
            )
            continue
        if actual_value != expected_value:
            failures.append(
                f"Expected {name}={expected_value!r} from {source_path}, "
                f"got {actual_value!r}."
            )

    if failures:
        debug_subset = {
            key: diagnostics[key][0]
            for key in ("fallback_used", "probability_source", "row_source", "win_classifier_used")
        }
        raise SmokeFailure(
            "Real-model diagnostics check failed for "
            f"{PREDICT_PAYLOAD['season']} week {PREDICT_PAYLOAD['week']} "
            f"{PREDICT_PAYLOAD['home_team']} vs {PREDICT_PAYLOAD['away_team']}:\n"
            + "\n".join(f"- {failure}" for failure in failures)
            + "\nObserved diagnostics: "
            + json.dumps(debug_subset, sort_keys=True)
        )


def main() -> int:
    try:
        payload, transport = _post_predict()
        _assert_real_model_diagnostics(payload)
    except SmokeFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        "PASS: /predict returned real-model diagnostics for "
        f"{PREDICT_PAYLOAD['season']} week {PREDICT_PAYLOAD['week']} "
        f"{PREDICT_PAYLOAD['home_team']} vs {PREDICT_PAYLOAD['away_team']} "
        f"via {transport}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
