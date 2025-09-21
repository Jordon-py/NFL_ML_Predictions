#!/usr/bin/env python3
"""
Production API Testing Suite
============================

Comprehensive testing of the NFL Prediction API endpoints to validate
production readiness and ensure all functionality works correctly.

Usage:
    # Test local development server
    python test_api.py --host localhost --port 8000
    
    # Test production deployment
    python test_api.py --host your-app.herokuapp.com --port 443 --https
"""

import argparse
import json
import logging
import sys
import time
from typing import Dict, Any
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    print("❌ requests library required: pip install requests")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APITester:
    """Comprehensive API testing suite."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30
        self.passed = 0
        self.failed = 0
    
    def test_endpoint(self, method: str, endpoint: str, expected_status: int = 200, 
                     data: Dict = None, description: str = None) -> Dict[str, Any]:
        """Test a single API endpoint."""
        url = urljoin(self.base_url + '/', endpoint.lstrip('/'))
        test_name = description or f"{method} {endpoint}"
        
        logger.info(f"Testing: {test_name}")
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if response.status_code == expected_status:
                logger.info(f"✓ {test_name} - Status: {response.status_code}")
                self.passed += 1
                
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return {"raw_response": response.text}
            else:
                logger.error(f"✗ {test_name} - Expected: {expected_status}, Got: {response.status_code}")
                logger.error(f"  Response: {response.text[:200]}...")
                self.failed += 1
                return {"error": f"Status {response.status_code}", "response": response.text}
                
        except requests.RequestException as e:
            logger.error(f"✗ {test_name} - Request failed: {e}")
            self.failed += 1
            return {"error": str(e)}
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        logger.info("\n--- Health Check Tests ---")
        
        result = self.test_endpoint('GET', '/health', description="Health check")
        
        if 'error' not in result:
            if result.get('status') == 'healthy':
                logger.info("✓ API reports healthy status")
                self.passed += 1
            else:
                logger.error(f"✗ API reports unhealthy: {result.get('reason', 'Unknown')}")
                self.failed += 1
    
    def test_info_endpoint(self):
        """Test the root info endpoint."""
        logger.info("\n--- Info Endpoint Tests ---")
        
        result = self.test_endpoint('GET', '/', description="Root info endpoint")
        
        if 'error' not in result:
            required_fields = ['name', 'version', 'endpoints']
            for field in required_fields:
                if field in result:
                    logger.info(f"✓ Info contains {field}")
                    self.passed += 1
                else:
                    logger.error(f"✗ Info missing {field}")
                    self.failed += 1
    
    def test_debug_endpoint(self):
        """Test the debug info endpoint."""
        logger.info("\n--- Debug Endpoint Tests ---")
        
        result = self.test_endpoint('GET', '/debug', description="Debug info")
        
        if 'error' not in result:
            if result.get('models_loaded'):
                logger.info("✓ Models are loaded")
                self.passed += 1
            else:
                logger.error("✗ Models not loaded")
                self.failed += 1
                
            if result.get('dataset_loaded'):
                logger.info("✓ Dataset is loaded")
                self.passed += 1
            else:
                logger.error("✗ Dataset not loaded")
                self.failed += 1
    
    def test_schedule_endpoint(self):
        """Test the schedule endpoint."""
        logger.info("\n--- Schedule Endpoint Tests ---")
        
        result = self.test_endpoint('GET', '/schedule/next-week', description="Next week schedule")
        
        if 'error' not in result and isinstance(result, list):
            logger.info(f"✓ Schedule returned {len(result)} games")
            self.passed += 1
            
            if len(result) > 0:
                game = result[0]
                required_fields = ['season', 'week', 'home_team', 'away_team']
                for field in required_fields:
                    if field in game:
                        logger.info(f"✓ Schedule game contains {field}")
                        self.passed += 1
                    else:
                        logger.error(f"✗ Schedule game missing {field}")
                        self.failed += 1
        elif 'error' not in result:
            logger.error("✗ Schedule endpoint returned unexpected format")
            self.failed += 1
    
    def test_prediction_endpoint(self):
        """Test the prediction endpoint with valid data."""
        logger.info("\n--- Prediction Endpoint Tests ---")
        
        # Test with common team names
        test_cases = [
            {
                "home_team": "Kansas City Chiefs",
                "away_team": "Buffalo Bills",
                "season": 2024,
                "week": 1
            },
            {
                "home_team": "KC",
                "away_team": "BUF", 
                "season": 2024,
                "week": 2
            }
        ]
        
        for i, test_data in enumerate(test_cases, 1):
            result = self.test_endpoint(
                'POST', '/predict', 
                data=test_data,
                description=f"Prediction test case {i}"
            )
            
            if 'error' not in result:
                required_fields = ['home_score', 'away_score', 'home_win_probability', 'away_win_probability']
                for field in required_fields:
                    if field in result:
                        value = result[field]
                        if isinstance(value, (int, float)) and 0 <= value <= 100:
                            logger.info(f"✓ Prediction {field}: {value}")
                            self.passed += 1
                        else:
                            logger.error(f"✗ Prediction {field} invalid: {value}")
                            self.failed += 1
                    else:
                        logger.error(f"✗ Prediction missing {field}")
                        self.failed += 1
    
    def test_invalid_requests(self):
        """Test error handling with invalid requests."""
        logger.info("\n--- Error Handling Tests ---")
        
        # Test invalid prediction data
        invalid_cases = [
            ({}, "Empty request"),
            ({"home_team": "Invalid Team"}, "Invalid team name"),
            ({"home_team": "KC", "away_team": "BUF", "season": 2000, "week": 1}, "Invalid season"),
        ]
        
        for invalid_data, description in invalid_cases:
            result = self.test_endpoint(
                'POST', '/predict',
                expected_status=400,  # Expect bad request
                data=invalid_data,
                description=f"Invalid request: {description}"
            )
            
            # For error cases, we just check that we got the expected error status
            if result.get('error', '').startswith('Status 400'):
                self.passed += 1
                logger.info(f"✓ Properly handled invalid request: {description}")
    
    def run_all_tests(self):
        """Run the complete test suite."""
        logger.info("🏈 NFL Prediction API - Production Test Suite")
        logger.info(f"Testing API at: {self.base_url}")
        logger.info("=" * 60)
        
        # Run all test categories
        test_methods = [
            self.test_health_endpoint,
            self.test_info_endpoint,
            self.test_debug_endpoint,
            self.test_schedule_endpoint,
            self.test_prediction_endpoint,
            self.test_invalid_requests
        ]
        
        start_time = time.time()
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                logger.error(f"Test method {test_method.__name__} failed: {e}")
                self.failed += 1
        
        # Report results
        total_tests = self.passed + self.failed
        duration = time.time() - start_time
        
        logger.info("\n" + "=" * 60)
        logger.info(f"Test Results: {self.passed}/{total_tests} passed ({duration:.2f}s)")
        
        if self.failed == 0:
            logger.info("🎉 All tests passed! API is ready for production.")
            return True
        else:
            logger.error(f"💥 {self.failed} tests failed. Fix issues before deployment.")
            return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test NFL Prediction API")
    parser.add_argument('--host', default='localhost', help='API host')
    parser.add_argument('--port', type=int, default=8000, help='API port')
    parser.add_argument('--https', action='store_true', help='Use HTTPS')
    return parser.parse_args()

def main():
    """Main test runner."""
    args = parse_args()
    
    protocol = 'https' if args.https else 'http'
    base_url = f"{protocol}://{args.host}:{args.port}"
    
    tester = APITester(base_url)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()