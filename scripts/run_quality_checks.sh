#!/bin/bash
set -e

echo "=========================================="
echo "Music Gen AI - Quality Assurance Suite"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check required tools
print_status "Checking required tools..."

if ! command_exists python; then
    print_error "Python is not installed"
    exit 1
fi

if ! command_exists pytest; then
    print_warning "pytest not found, installing..."
    pip install pytest pytest-cov pytest-asyncio
fi

print_success "All required tools are available"

# Set up paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
print_status "Project root: $PROJECT_ROOT"

# Create output directory
mkdir -p reports
REPORT_DIR="reports/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORT_DIR"

print_status "Reports will be saved to: $REPORT_DIR"

# 1. Code Quality Audit
echo ""
echo "=========================================="
echo "1. CODE QUALITY AUDIT"
echo "=========================================="

print_status "Running code quality analysis..."
python scripts/code_quality_audit.py . --output "$REPORT_DIR/code_quality_report.txt"

if [ $? -eq 0 ]; then
    print_success "Code quality audit completed"
    echo "Report saved to: $REPORT_DIR/code_quality_report.txt"
else
    print_error "Code quality audit failed"
fi

# 2. Unit Tests with Coverage
echo ""
echo "=========================================="
echo "2. UNIT TESTS WITH COVERAGE"
echo "=========================================="

print_status "Running unit tests..."
pytest tests/unit/ \
    --cov=music_gen \
    --cov-report=html:"$REPORT_DIR/coverage_html" \
    --cov-report=xml:"$REPORT_DIR/coverage.xml" \
    --cov-report=term-missing \
    --cov-fail-under=80 \
    --junitxml="$REPORT_DIR/junit_unit.xml" \
    -v

if [ $? -eq 0 ]; then
    print_success "Unit tests passed"
else
    print_error "Unit tests failed"
fi

# 3. Integration Tests
echo ""
echo "=========================================="
echo "3. INTEGRATION TESTS"
echo "=========================================="

print_status "Running integration tests..."
pytest tests/integration/ \
    --junitxml="$REPORT_DIR/junit_integration.xml" \
    -v

if [ $? -eq 0 ]; then
    print_success "Integration tests passed"
else
    print_warning "Integration tests failed or not available"
fi

# 4. Performance Tests
echo ""
echo "=========================================="
echo "4. PERFORMANCE TESTS"
echo "=========================================="

print_status "Running performance tests..."
pytest tests/performance/ \
    --junitxml="$REPORT_DIR/junit_performance.xml" \
    -v -s

if [ $? -eq 0 ]; then
    print_success "Performance tests passed"
else
    print_warning "Performance tests failed or not available"
fi

# 5. Type Checking (if mypy is available)
echo ""
echo "=========================================="
echo "5. TYPE CHECKING"
echo "=========================================="

if command_exists mypy; then
    print_status "Running type checking..."
    mypy music_gen --ignore-missing-imports --no-strict-optional > "$REPORT_DIR/mypy_report.txt" 2>&1
    
    if [ $? -eq 0 ]; then
        print_success "Type checking passed"
    else
        print_warning "Type checking found issues (see report)"
    fi
else
    print_warning "mypy not available, skipping type checking"
fi

# 6. Linting (if flake8 is available)
echo ""
echo "=========================================="
echo "6. LINTING"
echo "=========================================="

if command_exists flake8; then
    print_status "Running linting..."
    flake8 music_gen --max-line-length=100 --ignore=E203,W503 > "$REPORT_DIR/flake8_report.txt" 2>&1
    
    if [ $? -eq 0 ]; then
        print_success "Linting passed"
    else
        print_warning "Linting found issues (see report)"
    fi
else
    print_warning "flake8 not available, skipping linting"
fi

# 7. Security Check (if bandit is available)
echo ""
echo "=========================================="
echo "7. SECURITY SCAN"
echo "=========================================="

if command_exists bandit; then
    print_status "Running security scan..."
    bandit -r music_gen -f json -o "$REPORT_DIR/bandit_report.json" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        print_success "Security scan completed"
    else
        print_warning "Security scan found issues (see report)"
    fi
else
    print_warning "bandit not available, skipping security scan"
fi

# 8. Dependency Check
echo ""
echo "=========================================="
echo "8. DEPENDENCY ANALYSIS"
echo "=========================================="

print_status "Analyzing dependencies..."

# Create requirements analysis
pip list --format=freeze > "$REPORT_DIR/installed_packages.txt"

# Check for outdated packages
if command_exists pip-audit; then
    pip-audit --format=json --output="$REPORT_DIR/security_audit.json" 2>/dev/null || print_warning "pip-audit failed"
fi

print_success "Dependency analysis completed"

# 9. Generate Summary Report
echo ""
echo "=========================================="
echo "9. GENERATING SUMMARY REPORT"
echo "=========================================="

print_status "Generating summary report..."

cat > "$REPORT_DIR/summary_report.md" << EOF
# Music Gen AI - Quality Assurance Report

**Generated**: $(date)
**Project**: Music Gen AI
**Report Directory**: $REPORT_DIR

## Test Results Summary

### Unit Tests
- **Status**: $([ -f "$REPORT_DIR/junit_unit.xml" ] && echo "✅ Completed" || echo "❌ Failed")
- **Coverage Report**: [HTML Report](coverage_html/index.html)
- **JUnit XML**: junit_unit.xml

### Integration Tests  
- **Status**: $([ -f "$REPORT_DIR/junit_integration.xml" ] && echo "✅ Completed" || echo "❌ Failed")
- **JUnit XML**: junit_integration.xml

### Performance Tests
- **Status**: $([ -f "$REPORT_DIR/junit_performance.xml" ] && echo "✅ Completed" || echo "❌ Failed") 
- **JUnit XML**: junit_performance.xml

## Code Quality Analysis

### Quality Audit
- **Report**: [code_quality_report.txt](code_quality_report.txt)

### Type Checking
- **Status**: $([ -f "$REPORT_DIR/mypy_report.txt" ] && echo "✅ Completed" || echo "⏭️ Skipped")
- **Report**: mypy_report.txt

### Linting
- **Status**: $([ -f "$REPORT_DIR/flake8_report.txt" ] && echo "✅ Completed" || echo "⏭️ Skipped")
- **Report**: flake8_report.txt

### Security Scan
- **Status**: $([ -f "$REPORT_DIR/bandit_report.json" ] && echo "✅ Completed" || echo "⏭️ Skipped")
- **Report**: bandit_report.json

## Recommendations

1. **Review code quality report** for refactoring opportunities
2. **Maintain test coverage** above 90% 
3. **Address security findings** from bandit scan
4. **Fix type checking issues** for better code safety
5. **Resolve linting issues** for code consistency

## Files Generated

- \`code_quality_report.txt\` - Comprehensive code quality analysis
- \`coverage_html/\` - HTML coverage report
- \`coverage.xml\` - XML coverage report for CI/CD
- \`junit_*.xml\` - Test results in JUnit format
- \`mypy_report.txt\` - Type checking results
- \`flake8_report.txt\` - Linting results
- \`bandit_report.json\` - Security scan results
- \`installed_packages.txt\` - Current dependencies
- \`security_audit.json\` - Dependency security audit

EOF

print_success "Summary report generated"

# 10. Final Summary
echo ""
echo "=========================================="
echo "QUALITY ASSURANCE COMPLETE"
echo "=========================================="

print_success "All quality checks completed!"
echo ""
echo "📊 Reports available in: $REPORT_DIR"
echo ""
echo "Key files:"
echo "  • Summary: $REPORT_DIR/summary_report.md"
echo "  • Coverage: $REPORT_DIR/coverage_html/index.html"
echo "  • Quality: $REPORT_DIR/code_quality_report.txt"
echo ""

# Check if we can open the coverage report
if command_exists open && [ -f "$REPORT_DIR/coverage_html/index.html" ]; then
    echo "🚀 Opening coverage report..."
    open "$REPORT_DIR/coverage_html/index.html"
elif command_exists xdg-open && [ -f "$REPORT_DIR/coverage_html/index.html" ]; then
    echo "🚀 Opening coverage report..."
    xdg-open "$REPORT_DIR/coverage_html/index.html"
fi

print_success "Quality assurance suite completed successfully!"

# Return appropriate exit code
if [ -f "$REPORT_DIR/junit_unit.xml" ]; then
    exit 0
else
    exit 1
fi