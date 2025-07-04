#!/bin/bash
# Script to run all CORS security tests

echo "🔒 Running CORS Security Tests..."
echo "================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to project root
cd "$(dirname "$0")/.."

# Create test report directory
mkdir -p test_reports

echo -e "\n${YELLOW}1. Running CORS configuration tests...${NC}"
pytest tests/test_cors_config.py -v --tb=short \
    --junit-xml=test_reports/cors_config_junit.xml \
    --html=test_reports/cors_config_report.html \
    --self-contained-html

CORS_CONFIG_EXIT=$?

echo -e "\n${YELLOW}2. Running CORS security tests...${NC}"
pytest tests/test_cors_security.py -v --tb=short \
    --junit-xml=test_reports/cors_security_junit.xml \
    --html=test_reports/cors_security_report.html \
    --self-contained-html

CORS_SECURITY_EXIT=$?

echo -e "\n${YELLOW}3. Running CORS + Auth integration tests...${NC}"
pytest tests/test_cors_auth_integration.py -v --tb=short \
    --junit-xml=test_reports/cors_auth_junit.xml \
    --html=test_reports/cors_auth_report.html \
    --self-contained-html

CORS_AUTH_EXIT=$?

echo -e "\n${YELLOW}4. Running authentication middleware tests...${NC}"
pytest tests/test_auth_middleware.py -v --tb=short \
    --junit-xml=test_reports/auth_middleware_junit.xml \
    --html=test_reports/auth_middleware_report.html \
    --self-contained-html

AUTH_MIDDLEWARE_EXIT=$?

# Summary
echo -e "\n${YELLOW}================================${NC}"
echo -e "${YELLOW}Test Summary:${NC}"
echo -e "${YELLOW}================================${NC}"

if [ $CORS_CONFIG_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ CORS Configuration Tests: PASSED${NC}"
else
    echo -e "${RED}✗ CORS Configuration Tests: FAILED${NC}"
fi

if [ $CORS_SECURITY_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ CORS Security Tests: PASSED${NC}"
else
    echo -e "${RED}✗ CORS Security Tests: FAILED${NC}"
fi

if [ $CORS_AUTH_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ CORS + Auth Integration Tests: PASSED${NC}"
else
    echo -e "${RED}✗ CORS + Auth Integration Tests: FAILED${NC}"
fi

if [ $AUTH_MIDDLEWARE_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Authentication Middleware Tests: PASSED${NC}"
else
    echo -e "${RED}✗ Authentication Middleware Tests: FAILED${NC}"
fi

# Coverage report for CORS and Auth modules
echo -e "\n${YELLOW}5. Generating coverage report...${NC}"
pytest tests/test_cors_*.py tests/test_auth_middleware.py \
    --cov=music_gen.api.cors_config \
    --cov=music_gen.api.middleware.auth \
    --cov=music_gen.api.deps \
    --cov-report=html:test_reports/cors_auth_coverage \
    --cov-report=term

# Check if all tests passed
if [ $CORS_CONFIG_EXIT -eq 0 ] && [ $CORS_SECURITY_EXIT -eq 0 ] && [ $CORS_AUTH_EXIT -eq 0 ] && [ $AUTH_MIDDLEWARE_EXIT -eq 0 ]; then
    echo -e "\n${GREEN}✅ All CORS and Authentication tests passed!${NC}"
    echo -e "${GREEN}Reports generated in test_reports/${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed. Please check the reports in test_reports/${NC}"
    exit 1
fi