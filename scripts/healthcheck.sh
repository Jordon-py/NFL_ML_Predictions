#!/bin/bash
# ===========================================
# healthcheck.sh — HTTP health check with retry
# ===========================================
# Usage: ./healthcheck.sh <base_url> [max_retries] [delay_seconds]
#
# Checks:
#   - /api/health        (fast check)
#   - /api/health/deep   (deep check: models, deps)
#   - /api/metrics       (Prometheus endpoint)
#
# Exit codes:
#   0 = all checks passed
#   1 = one or more checks failed after retries
# ===========================================

set -e

BASE_URL="${1:-http://localhost:8000}"
MAX_RETRIES="${2:-5}"
DELAY="${3:-10}"

# Remove trailing slash if present
BASE_URL="${BASE_URL%/}"

echo "=== Health Check ==="
echo "Base URL: $BASE_URL"
echo "Max retries: $MAX_RETRIES"
echo "Delay between retries: ${DELAY}s"
echo ""

check_endpoint() {
    local endpoint="$1"
    local description="$2"
    local url="${BASE_URL}${endpoint}"
    
    echo "Checking $description ($url)..."
    
    for i in $(seq 1 $MAX_RETRIES); do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 "$url" 2>/dev/null || echo "000")
        
        if [ "$HTTP_CODE" = "200" ]; then
            echo "  ✓ $description: OK (HTTP $HTTP_CODE)"
            return 0
        else
            echo "  ⏳ Attempt $i/$MAX_RETRIES: HTTP $HTTP_CODE"
            if [ "$i" -lt "$MAX_RETRIES" ]; then
                sleep "$DELAY"
            fi
        fi
    done
    
    echo "  ✗ $description: FAILED after $MAX_RETRIES attempts"
    return 1
}

FAILED=0

# Check /api/health (required)
if ! check_endpoint "/api/health" "Health endpoint"; then
    FAILED=1
fi

# Check /api/health/deep (required if exists)
if ! check_endpoint "/api/health/deep" "Deep health check"; then
    echo "  (Deep health check not critical, continuing...)"
fi

# Check /api/metrics (optional)
if ! check_endpoint "/api/metrics" "Metrics endpoint"; then
    echo "  (Metrics endpoint not critical, continuing...)"
fi

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "=== All critical health checks passed ==="
    exit 0
else
    echo "=== Health checks FAILED ==="
    exit 1
fi
