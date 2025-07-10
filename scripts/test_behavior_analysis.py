#!/usr/bin/env python3
"""
Test Behavior Analysis

Analyzes existing tests to identify:
1. Which tests only verify coverage vs actual behavior
2. Where mocks hide real issues
3. Missing critical paths
4. Specific refactoring recommendations
"""

import re
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass


@dataclass
class TestAnalysis:
    """Analysis of a single test."""

    name: str
    test_type: str  # 'behavior', 'coverage_only', 'integration'
    issues: List[str]
    recommendations: List[str]
    complexity_score: int
    behavior_score: int
    mock_count: int
    assertion_count: int


class TestBehaviorAnalyzer:
    """Analyze test quality and behavior coverage."""

    def __init__(self):
        self.coverage_patterns = [
            r"assert\s+\w+\s+is\s+not\s+None",
            r"assert\s+len\([^)]+\)\s*==\s*\d+",
            r"assert\s+\w+\s+==\s+\w+",
            r"assert\s+hasattr\(",
        ]

        self.behavior_patterns = [
            r"assert.*called",
            r"pytest\.raises",
            r"assert.*is.*model",
            r"with\s+ThreadPoolExecutor",
            r"time\.sleep",
            r"gc\.collect",
        ]

        self.mock_issues = [
            r"Mock\([^)]*\).*Mock\([^)]*\)",  # Multiple mocks
            r"patch\([^)]*\).*patch\([^)]*\)",  # Multiple patches
            r"return_value.*return_value",  # Nested mocking
            r"side_effect.*Mock",  # Complex mock setup
        ]

    def analyze_test_file(self, file_path: Path) -> List[TestAnalysis]:
        """Analyze all tests in a file."""
        with open(file_path, "r") as f:
            content = f.read()

        # Parse AST
        tree = ast.parse(content)

        # Find test functions
        test_functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                test_functions.append((node.name, node))

        # Analyze each test
        analyses = []
        for test_name, test_node in test_functions:
            analysis = self._analyze_single_test(test_name, test_node, content)
            analyses.append(analysis)

        return analyses

    def _analyze_single_test(self, name: str, node: ast.FunctionDef, content: str) -> TestAnalysis:
        """Analyze a single test function."""
        # Extract function body
        func_start = node.lineno - 1
        func_end = node.end_lineno if node.end_lineno else len(content.split("\n"))
        func_lines = content.split("\n")[func_start:func_end]
        func_body = "\n".join(func_lines)

        # Count patterns
        coverage_count = sum(
            len(re.findall(pattern, func_body)) for pattern in self.coverage_patterns
        )
        behavior_count = sum(
            len(re.findall(pattern, func_body)) for pattern in self.behavior_patterns
        )
        mock_count = len(re.findall(r"\bMock\(|patch\(|@patch", func_body))
        assertion_count = len(re.findall(r"\bassert\s+|assert_\w+|pytest\.raises", func_body))

        # Determine test type
        test_type = self._classify_test(name, func_body, coverage_count, behavior_count, mock_count)

        # Identify issues
        issues = self._identify_issues(name, func_body, test_type, mock_count, assertion_count)

        # Generate recommendations
        recommendations = self._generate_recommendations(name, func_body, test_type, issues)

        # Calculate scores
        complexity_score = min(100, len(func_body.split("\n")) * 2)
        behavior_score = min(100, behavior_count * 20)

        return TestAnalysis(
            name=name,
            test_type=test_type,
            issues=issues,
            recommendations=recommendations,
            complexity_score=complexity_score,
            behavior_score=behavior_score,
            mock_count=mock_count,
            assertion_count=assertion_count,
        )

    def _classify_test(
        self, name: str, body: str, coverage_count: int, behavior_count: int, mock_count: int
    ) -> str:
        """Classify test as behavior, coverage_only, or integration."""

        # Integration test indicators
        if any(
            keyword in name.lower()
            for keyword in ["concurrent", "stress", "lifecycle", "integration"]
        ):
            return "integration"

        if "ThreadPoolExecutor" in body or "real" in name.lower():
            return "integration"

        # Behavior test indicators
        if behavior_count > coverage_count:
            return "behavior"

        if "pytest.raises" in body:
            return "behavior"

        if "assert_called" in body or "assert_not_called" in body:
            return "behavior"

        # Coverage-only indicators
        if coverage_count > 0 and behavior_count == 0:
            return "coverage_only"

        if mock_count > 3 and behavior_count == 0:
            return "coverage_only"

        if len(body.split("\n")) < 10 and "assert" in body:
            return "coverage_only"

        return "behavior"  # Default to behavior

    def _identify_issues(
        self, name: str, body: str, test_type: str, mock_count: int, assertion_count: int
    ) -> List[str]:
        """Identify specific issues with the test."""
        issues = []

        # No assertions
        if assertion_count == 0:
            issues.append("No assertions - test doesn't verify anything")

        # Over-mocking
        if mock_count > 5:
            issues.append(f"Excessive mocking ({mock_count} mocks) may hide real behavior")

        # Mock without verification
        if "Mock(" in body and "assert_called" not in body and "assert_not_called" not in body:
            issues.append("Uses mocks but doesn't verify interactions")

        # Coverage-only patterns
        if test_type == "coverage_only":
            if "assert.*is not None" in body:
                issues.append("Only checks for None values - doesn't verify correctness")

            if "assert len(" in body:
                issues.append("Only checks collection size - doesn't verify contents")

        # Missing error testing
        if "error" in name.lower() and "pytest.raises" not in body:
            issues.append("Tests error case but doesn't use pytest.raises")

        # Trivial assertions
        trivial_patterns = [r"assert True", r"assert 1 == 1", r'assert.*== ".*"']
        for pattern in trivial_patterns:
            if re.search(pattern, body):
                issues.append("Contains trivial assertions")
                break

        # Mock return value without behavior check
        if "return_value" in body and test_type == "coverage_only":
            issues.append("Sets mock return value but doesn't test resulting behavior")

        return issues

    def _generate_recommendations(
        self, name: str, body: str, test_type: str, issues: List[str]
    ) -> List[str]:
        """Generate specific recommendations for improvement."""
        recommendations = []

        if test_type == "coverage_only":
            recommendations.append("Refactor to test actual behavior instead of just state")

            if "get_model" in body:
                recommendations.append(
                    "Verify model loading behavior: caching, error handling, device placement"
                )

            if "cache" in name.lower():
                recommendations.append(
                    "Test cache behavior: hit/miss rates, memory usage, eviction"
                )

            if "concurrent" in name.lower():
                recommendations.append("Verify thread safety: race conditions, data consistency")

        # Specific improvements based on issues
        for issue in issues:
            if "No assertions" in issue:
                recommendations.append("Add assertions to verify expected outcomes")

            elif "Excessive mocking" in issue:
                recommendations.append("Reduce mocking - test real interactions where possible")

            elif "doesn't verify interactions" in issue:
                recommendations.append("Add assert_called_with() to verify mock interactions")

            elif "doesn't use pytest.raises" in issue:
                recommendations.append("Use pytest.raises to properly test exceptions")

            elif "Only checks for None" in issue:
                recommendations.append("Verify actual return values and their properties")

            elif "Only checks collection size" in issue:
                recommendations.append("Verify collection contents, not just size")

        # Method-specific recommendations
        if "unload_model" in body:
            recommendations.append("Verify memory is actually freed after unloading")

        if "clear_cache" in body:
            recommendations.append("Test that cache clearing actually frees resources")

        if "singleton" in name.lower():
            recommendations.append("Test singleton across threads and modules")

        return recommendations


def analyze_critical_paths(analyses: List[TestAnalysis]) -> Dict:
    """Analyze coverage of critical code paths."""

    critical_paths = {
        "model_loading": [],
        "caching": [],
        "error_handling": [],
        "memory_management": [],
        "concurrency": [],
        "device_management": [],
        "lifecycle": [],
    }

    path_keywords = {
        "model_loading": ["load", "get_model"],
        "caching": ["cache", "cached"],
        "error_handling": ["error", "exception", "corrupt", "fail"],
        "memory_management": ["memory", "cleanup", "gc"],
        "concurrency": ["concurrent", "thread", "parallel"],
        "device_management": ["device", "cuda", "cpu"],
        "lifecycle": ["lifecycle", "init", "destroy", "unload"],
    }

    for analysis in analyses:
        for path, keywords in path_keywords.items():
            if any(keyword in analysis.name.lower() for keyword in keywords):
                critical_paths[path].append(analysis)

    # Calculate coverage quality for each path
    path_quality = {}
    for path, tests in critical_paths.items():
        if not tests:
            path_quality[path] = {"coverage": 0, "quality": 0, "tests": []}
            continue

        behavior_tests = [t for t in tests if t.test_type == "behavior"]
        avg_behavior_score = sum(t.behavior_score for t in tests) / len(tests) if tests else 0

        path_quality[path] = {
            "coverage": len(tests),
            "quality": avg_behavior_score,
            "behavior_ratio": len(behavior_tests) / len(tests) if tests else 0,
            "tests": [t.name for t in tests],
        }

    return path_quality


def generate_refactoring_examples() -> Dict[str, str]:
    """Generate before/after examples for common test issues."""

    examples = {
        "coverage_only_to_behavior": {
            "before": '''
def test_get_model_returns_model(manager):
    """Coverage-only test."""
    model = manager.get_model("test", "optimized")
    assert model is not None  # Only checks it's not None
    assert len(manager._models) == 1  # Only checks count
''',
            "after": '''
def test_model_loading_caches_correctly(manager):
    """Behavior test."""
    with patch('...FastMusicGenerator') as mock:
        mock.return_value = Mock()
        
        # First load should call generator
        model1 = manager.get_model("test", "optimized")
        assert mock.call_count == 1
        
        # Second load should use cache
        model2 = manager.get_model("test", "optimized")
        assert mock.call_count == 1  # Not called again
        assert model1 is model2  # Same instance returned
        
        # Verify caching behavior, not just state
''',
        },
        "over_mocked_to_integration": {
            "before": '''
def test_model_operation(manager):
    """Over-mocked test."""
    with patch('manager._models') as mock_models:
        with patch('manager.get_model') as mock_get:
            with patch('gc.collect') as mock_gc:
                mock_get.return_value = Mock()
                # Everything is mocked - no real behavior tested
''',
            "after": '''
def test_model_cleanup_integration():
    """Integration test."""
    manager = ModelManager()
    
    # Load real model
    model = manager.get_model("test", "optimized")
    initial_memory = get_memory_usage()
    
    # Verify model is loaded
    assert "test_optimized_cuda" in manager._models
    
    # Unload and verify cleanup
    manager.unload_model("test")
    after_memory = get_memory_usage()
    
    # Test real cleanup behavior
    assert "test_optimized_cuda" not in manager._models
    assert after_memory < initial_memory + 10  # Memory freed
''',
        },
        "missing_error_handling": {
            "before": '''
def test_invalid_model_type(manager):
    """Incomplete error test."""
    try:
        manager.get_model("test", "invalid")
        # No assertion - test passes even if no error
    except ValueError:
        pass  # Catches error but doesn't verify message
''',
            "after": '''
def test_invalid_model_type_error(manager):
    """Complete error test."""
    with pytest.raises(ValueError, match="Unknown model type: invalid"):
        manager.get_model("test", "invalid")
    
    # Verify no partial state changes
    assert len(manager._models) == 0
    assert "test_invalid_cuda" not in manager._models
''',
        },
    }

    return examples


def main():
    """Run test behavior analysis."""
    project_root = Path(__file__).parent.parent
    test_file = project_root / "tests" / "unit" / "test_model_manager_comprehensive.py"

    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        print("Showing example refactoring patterns instead...\n")

        examples = generate_refactoring_examples()
        for title, example in examples.items():
            print(f"## {title.replace('_', ' ').title()}")
            print("### Before (Coverage-Only)")
            print("```python")
            print(example["before"])
            print("```")
            print("### After (Behavior Test)")
            print("```python")
            print(example["after"])
            print("```")
            print()
        return

    print("🔍 Analyzing Test Behavior Quality")
    print("=" * 60)

    analyzer = TestBehaviorAnalyzer()
    analyses = analyzer.analyze_test_file(test_file)

    # Categorize tests
    behavior_tests = [a for a in analyses if a.test_type == "behavior"]
    coverage_tests = [a for a in analyses if a.test_type == "coverage_only"]
    integration_tests = [a for a in analyses if a.test_type == "integration"]

    print(f"📊 Test Summary:")
    print(f"  Total Tests: {len(analyses)}")
    print(f"  Behavior Tests: {len(behavior_tests)} ({len(behavior_tests)/len(analyses)*100:.1f}%)")
    print(
        f"  Coverage-Only Tests: {len(coverage_tests)} ({len(coverage_tests)/len(analyses)*100:.1f}%)"
    )
    print(
        f"  Integration Tests: {len(integration_tests)} ({len(integration_tests)/len(analyses)*100:.1f}%)"
    )
    print()

    # Show coverage-only tests that need refactoring
    if coverage_tests:
        print("⚠️  Coverage-Only Tests (Need Refactoring)")
        print("-" * 40)
        for test in coverage_tests[:5]:
            print(f"  • {test.name}")
            if test.issues:
                print(f"    Issues: {test.issues[0]}")
            if test.recommendations:
                print(f"    Fix: {test.recommendations[0]}")
            print()

    # Analyze critical path coverage
    path_quality = analyze_critical_paths(analyses)

    print("🛤️  Critical Path Analysis")
    print("-" * 40)
    for path, quality in path_quality.items():
        status = "✅" if quality["quality"] > 70 else "⚠️" if quality["quality"] > 40 else "❌"
        print(
            f"  {status} {path}: {quality['coverage']} tests, {quality['quality']:.0f}% behavior quality"
        )
    print()

    # Highest priority refactoring targets
    priority_refactors = sorted(
        [a for a in analyses if a.test_type == "coverage_only"], key=lambda x: -x.complexity_score
    )

    if priority_refactors:
        print("🎯 Priority Refactoring Targets")
        print("-" * 40)
        for test in priority_refactors[:3]:
            print(f"  1. {test.name}")
            print(
                f"     Complexity: {test.complexity_score}, Behavior Score: {test.behavior_score}"
            )
            for rec in test.recommendations[:2]:
                print(f"     → {rec}")
            print()

    # Overall recommendations
    behavior_ratio = len(behavior_tests) / len(analyses) if analyses else 0

    print("📋 Overall Recommendations")
    print("-" * 40)

    if behavior_ratio < 0.6:
        print("1. 🚨 CRITICAL: Too many coverage-only tests")
        print("   → Focus on refactoring coverage tests to behavior tests")

    if len(integration_tests) < 3:
        print("2. ⚠️  Add more integration tests")
        print("   → Test component interactions, not just units")

    high_mock_tests = [a for a in analyses if a.mock_count > 5]
    if high_mock_tests:
        print("3. 🔧 Reduce over-mocking")
        print(f"   → {len(high_mock_tests)} tests have excessive mocks")

    print("4. ✅ Add mutation testing to verify test effectiveness")
    print("5. 🧪 Use property-based testing for edge cases")
    print()

    # Generate score
    overall_score = (
        behavior_ratio * 40
        + min(len(integration_tests) / 5, 1) * 30
        + (1 - len(coverage_tests) / len(analyses)) * 30
    )

    print(f"🎯 Overall Test Quality Score: {overall_score:.0f}/100")

    if overall_score >= 80:
        print("   ✅ Excellent - Tests focus on behavior")
    elif overall_score >= 60:
        print("   ✅ Good - Minor improvements needed")
    elif overall_score >= 40:
        print("   ⚠️  Fair - Significant refactoring needed")
    else:
        print("   ❌ Poor - Major test quality issues")


if __name__ == "__main__":
    main()
