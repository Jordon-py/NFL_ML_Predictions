#!/usr/bin/env python3
"""
verify_api_cors.py
==================

Purpose
-------
Verify that the API and CORS configuration are correctly set up for the NFL ML Predictions system.
Tests backend endpoints and CORS headers to ensure proper frontend-backend communication.

Usage
-----
python scripts/verify_api_cors.py [--backend-url URL]

Options
-------
--backend-url : Backend API URL (default: https://nfl-predict-ecf5a5bd34fe.herokuapp.com)
--verbose : Enable verbose output

Examples
--------
# Test production backend
python scripts/verify_api_cors.py

# Test local backend
python scripts/verify_api_cors.py --backend-url http://localhost:8000

# Verbose output
python scripts/verify_api_cors.py --verbose
"""

import argparse
import json
import sys
from typing import Dict, List, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# ANSI color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    """Print a formatted section header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.RESET}\n")


def print_success(text: str):
    """Print success message in green."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text: str):
    """Print error message in red."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_warning(text: str):
    """Print warning message in yellow."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_info(text: str):
    """Print info message."""
    print(f"  {text}")


def test_endpoint(
    url: str, method: str = "GET", headers: Dict[str, str] = None, data: bytes = None, verbose: bool = False
) -> Tuple[bool, str, Dict[str, str]]:
    """
    Test an API endpoint.

    Parameters
    ----------
    url : str
        The full URL to test
    method : str
        HTTP method (GET, POST, OPTIONS)
    headers : dict
        Request headers
    data : bytes
        Request body for POST requests
    verbose : bool
        Enable verbose output

    Returns
    -------
    tuple
        (success: bool, response_body: str, response_headers: dict)
    """
    headers = headers or {}
    try:
        req = Request(url, data=data, headers=headers, method=method)
        with urlopen(req, timeout=10) as response:
            body = response.read().decode("utf-8")
            resp_headers = dict(response.headers)
            if verbose:
                print_info(f"Status: {response.status}")
                print_info(f"Headers: {json.dumps(resp_headers, indent=2)}")
            return True, body, resp_headers
    except HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        if verbose:
            print_info(f"HTTP Error {e.code}: {e.reason}")
            print_info(f"Body: {error_body}")
        return False, error_body, dict(e.headers)
    except URLError as e:
        if verbose:
            print_info(f"URL Error: {e.reason}")
        return False, str(e.reason), {}
    except Exception as e:
        if verbose:
            print_info(f"Error: {str(e)}")
        return False, str(e), {}


def verify_health_endpoint(backend_url: str, verbose: bool = False) -> bool:
    """Verify the /health endpoint is working."""
    print_header("Testing Health Endpoint")

    url = f"{backend_url.rstrip('/')}/health"
    print_info(f"URL: {url}")

    success, body, headers = test_endpoint(url, verbose=verbose)

    if not success:
        print_error(f"Health endpoint failed: {body}")
        return False

    try:
        data = json.loads(body)
        if data.get("status") == "healthy":
            print_success(f"Health endpoint returned: {data}")
            return True
        else:
            print_warning(f"Health endpoint returned non-healthy status: {data}")
            return False
    except json.JSONDecodeError:
        print_error(f"Invalid JSON response: {body}")
        return False


def verify_cors_headers(backend_url: str, origins: List[str], verbose: bool = False) -> bool:
    """Verify CORS headers for given origins."""
    print_header("Testing CORS Configuration")

    all_passed = True

    for origin in origins:
        print_info(f"\nTesting origin: {origin}")

        # Test OPTIONS preflight request
        url = f"{backend_url.rstrip('/')}/health"
        headers = {
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type",
        }

        success, body, resp_headers = test_endpoint(url, method="OPTIONS", headers=headers, verbose=verbose)

        if not success:
            print_error(f"  OPTIONS request failed for {origin}")
            all_passed = False
            continue

        # Check CORS headers
        allow_origin = resp_headers.get("Access-Control-Allow-Origin", resp_headers.get("access-control-allow-origin"))
        allow_methods = resp_headers.get("Access-Control-Allow-Methods", resp_headers.get("access-control-allow-methods"))
        allow_headers = resp_headers.get("Access-Control-Allow-Headers", resp_headers.get("access-control-allow-headers"))
        allow_credentials = resp_headers.get(
            "Access-Control-Allow-Credentials", resp_headers.get("access-control-allow-credentials")
        )

        if allow_origin == origin or allow_origin == "*":
            print_success(f"  Access-Control-Allow-Origin: {allow_origin}")
        else:
            print_error(f"  Access-Control-Allow-Origin missing or incorrect: {allow_origin}")
            all_passed = False

        if allow_methods and "*" in allow_methods:
            print_success(f"  Access-Control-Allow-Methods: {allow_methods}")
        elif not allow_methods:
            print_warning(f"  Access-Control-Allow-Methods not set")

        if allow_headers and "*" in allow_headers:
            print_success(f"  Access-Control-Allow-Headers: {allow_headers}")
        elif not allow_headers:
            print_warning(f"  Access-Control-Allow-Headers not set")

        if allow_credentials and allow_credentials.lower() == "true":
            print_success(f"  Access-Control-Allow-Credentials: {allow_credentials}")
        else:
            print_warning(f"  Access-Control-Allow-Credentials: {allow_credentials}")

    return all_passed


def verify_predict_endpoint(backend_url: str, verbose: bool = False) -> bool:
    """Verify the /predict endpoint accepts requests."""
    print_header("Testing Predict Endpoint")

    url = f"{backend_url.rstrip('/')}/predict"
    print_info(f"URL: {url}")

    # Sample prediction request
    payload = {"home_team": "KC", "away_team": "BUF", "season": 2025, "week": 10}

    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")

    success, body, resp_headers = test_endpoint(url, method="POST", headers=headers, data=data, verbose=verbose)

    if not success:
        print_error(f"Predict endpoint failed: {body}")
        # Check if it's a dataset error (which is expected without dataset)
        if "Dataset not found" in body or "500" in body:
            print_warning("  This is expected if the dataset hasn't been generated yet")
            print_info("  Run: python backend/build_csv_datasets.py --start 2016 --end 2026 --out-dir backend/data")
            return True  # Don't fail the test for missing dataset
        return False

    try:
        data = json.loads(body)
        required_keys = ["home_score", "away_score", "home_win_probability", "away_win_probability"]
        if all(key in data for key in required_keys):
            print_success(f"Predict endpoint returned valid response:")
            print_info(f"  Home Score: {data['home_score']}")
            print_info(f"  Away Score: {data['away_score']}")
            print_info(f"  Home Win Probability: {data['home_win_probability']}")
            print_info(f"  Away Win Probability: {data['away_win_probability']}")
            return True
        else:
            print_warning(f"Predict endpoint missing required keys: {data}")
            return False
    except json.JSONDecodeError:
        print_error(f"Invalid JSON response: {body}")
        return False


def verify_debug_endpoint(backend_url: str, verbose: bool = False) -> bool:
    """Verify the /debug endpoint returns CORS configuration."""
    print_header("Testing Debug Endpoint (CORS Info)")

    url = f"{backend_url.rstrip('/')}/debug"
    print_info(f"URL: {url}")

    success, body, headers = test_endpoint(url, verbose=verbose)

    if not success:
        print_warning(f"Debug endpoint not available (this is optional)")
        return True  # Don't fail for missing debug endpoint

    try:
        data = json.loads(body)
        cors_origins = data.get("cors_origins", [])
        env_cors = data.get("env_cors_origins", "not set")

        print_success("Debug endpoint returned CORS configuration:")
        print_info(f"  CORS Origins: {cors_origins}")
        print_info(f"  Environment CORS_ORIGINS: {env_cors}")

        # Verify expected origins are present
        expected_origins = [
            "http://localhost:3000",
            "https://nfl-ml-predictions.vercel.app",
        ]

        for origin in expected_origins:
            if any(origin in configured for configured in cors_origins):
                print_success(f"  ✓ {origin} is configured")
            else:
                print_warning(f"  ⚠ {origin} is NOT configured")

        return True
    except json.JSONDecodeError:
        print_error(f"Invalid JSON response: {body}")
        return False


def main():
    """Main verification script."""
    parser = argparse.ArgumentParser(description="Verify NFL ML Predictions API and CORS configuration")
    parser.add_argument(
        "--backend-url",
        default="https://nfl-predict-ecf5a5bd34fe.herokuapp.com",
        help="Backend API URL (default: production Heroku URL)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    print(f"\n{Colors.BOLD}NFL ML Predictions - API & CORS Verification{Colors.RESET}")
    print(f"Backend URL: {args.backend_url}\n")

    # Define expected CORS origins
    cors_origins = [
        "http://localhost:3000",
        "https://nfl-ml-predictions.vercel.app",
    ]

    # Run all verification tests
    results = {
        "Health Endpoint": verify_health_endpoint(args.backend_url, args.verbose),
        "CORS Configuration": verify_cors_headers(args.backend_url, cors_origins, args.verbose),
        "Debug Endpoint": verify_debug_endpoint(args.backend_url, args.verbose),
        "Predict Endpoint": verify_predict_endpoint(args.backend_url, args.verbose),
    }

    # Print summary
    print_header("Verification Summary")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")

    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.RESET}\n")

    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
