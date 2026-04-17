#!/usr/bin/env python3
"""
Verify that the deployed or local FastAPI backend exposes the current public API
surface and sends the expected CORS headers for the active frontend origins.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BACKEND_URL = "https://nfl-predict-ecf5a5bd34fe.herokuapp.com"
DEFAULT_TIMEOUT_SEC = 30
DEFAULT_ORIGINS = (
    "https://new-nfl-predict.vercel.app",
)


def _request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[int, str, dict[str, str]]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = Request(
        url,
        data=data,
        headers=headers or {},
        method=method,
    )
    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SEC) as response:
            body = response.read().decode("utf-8")
            return response.status, body, dict(response.headers.items())
    except HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        return exc.code, body, dict(exc.headers.items())
    except (TimeoutError, socket.timeout) as exc:  # pragma: no cover - exercised in live usage
        raise SystemExit(f"Timed out while contacting backend: {exc}") from exc
    except URLError as exc:  # pragma: no cover - exercised in live usage
        raise SystemExit(f"Unable to reach backend: {exc.reason}") from exc


def _parse_json(body: str) -> dict[str, Any] | list[Any] | None:
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _check_health(base_url: str) -> list[str]:
    issues: list[str] = []
    status, body, _headers = _request(f"{base_url}/health")
    if status != 200:
        issues.append(f"/health returned {status}")
        return issues

    payload = _parse_json(body)
    if not isinstance(payload, dict):
        issues.append("/health did not return JSON")
        return issues

    if "status" not in payload or "production_ready" not in payload:
        issues.append("/health response is missing expected readiness fields")

    return issues


def _check_status_overview(base_url: str) -> list[str]:
    issues: list[str] = []
    status, body, _headers = _request(
        f"{base_url}/status/overview",
        headers={"X-User-Id": "ci-verifier@example.com"},
    )
    if status != 200:
        issues.append(f"/status/overview returned {status}")
        return issues

    payload = _parse_json(body)
    if not isinstance(payload, dict):
        issues.append("/status/overview did not return JSON")
        return issues

    if "health" not in payload or "dataset" not in payload:
        issues.append("/status/overview response is missing expected keys")

    return issues


def _check_predict_contract(base_url: str) -> list[str]:
    issues: list[str] = []
    status, body, _headers = _request(
        f"{base_url}/predict",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-User-Id": "ci-verifier@example.com",
        },
        payload={
            "home_team": "KC",
            "away_team": "BUF",
            "season": 2025,
            "week": 10,
        },
    )

    if status not in {200, 503}:
        issues.append(f"/predict returned unexpected status {status}")
        return issues

    payload = _parse_json(body)
    if not isinstance(payload, dict):
        issues.append("/predict did not return a JSON object")
        return issues

    if status == 200:
        required = {"home_score", "away_score", "home_win_probability", "away_win_probability"}
        missing = sorted(required.difference(payload.keys()))
        if missing:
            issues.append(f"/predict 200 response is missing keys: {', '.join(missing)}")
    else:
        if "detail" not in payload:
            issues.append("/predict 503 response is missing structured error detail")

    return issues


def _check_cors(base_url: str, origins: tuple[str, ...]) -> list[str]:
    issues: list[str] = []
    for origin in origins:
        status, _body, headers = _request(
            f"{base_url}/health",
            method="OPTIONS",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type,X-User-Id",
            },
        )
        if status != 200:
            issues.append(f"CORS preflight for {origin} returned {status}")
            continue

        allow_origin = headers.get("Access-Control-Allow-Origin") or headers.get("access-control-allow-origin")
        if allow_origin != origin and allow_origin != "*":
            issues.append(f"CORS allow-origin mismatch for {origin}: {allow_origin!r}")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the current API + CORS contract.")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL, help="Backend origin to verify.")
    parser.add_argument(
        "--origin",
        action="append",
        dest="origins",
        default=None,
        help="Origin to use for CORS preflight checks. Repeat to test multiple origins.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print passing checks as well as failures.")
    args = parser.parse_args()

    base_url = args.backend_url.rstrip("/")
    origins = tuple(args.origins or DEFAULT_ORIGINS)
    checks = {
        "health": _check_health(base_url),
        "status_overview": _check_status_overview(base_url),
        "predict": _check_predict_contract(base_url),
        "cors": _check_cors(base_url, origins),
    }

    failures = {name: issues for name, issues in checks.items() if issues}

    for name, issues in checks.items():
        if not issues:
            if args.verbose:
                print(f"[pass] {name}")
            continue
        print(f"[fail] {name}")
        for issue in issues:
            print(f"  - {issue}")

    if failures:
        sys.exit(1)

    print("API + CORS verification passed.")


if __name__ == "__main__":
    main()
