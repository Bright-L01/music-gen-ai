#!/usr/bin/env python3
"""
Test Coverage Analysis Script

This script analyzes the ModelManager tests to identify:
1. Tests that verify actual behavior vs just coverage
2. Mock appropriateness
3. Missing critical path tests
4. Tests that would catch real bugs
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


class TestCoverageAnalyzer:
    """Analyze test quality beyond coverage metrics."""

    def __init__(self):
        self.behavior_tests = []
        self.coverage_only_tests = []
        self.mock_issues = []
        self.missing_tests = []
        self.critical_paths = set()

    def analyze_test_file(self, file_path: Path) -> Dict:
        """Analyze a test file for quality metrics."""
        with open(file_path, "r") as f:
            content = f.read()

        # Parse AST
        tree = ast.parse(content)

        # Analyze each test function
        test_functions = self._find_test_functions(tree)

        results = {
            "total_tests": len(test_functions),
            "behavior_tests": [],
            "coverage_only_tests": [],
            "mock_issues": [],
            "assertion_quality": {},
            "critical_paths_covered": set(),
            "missing_critical_paths": [],
        }

        for test_name, test_node in test_functions:
            analysis = self._analyze_test_function(test_name, test_node, content)

            if analysis["is_behavior_test"]:
                results["behavior_tests"].append(
                    {
                        "name": test_name,
                        "behaviors_tested": analysis["behaviors_tested"],
                        "assertion_count": analysis["assertion_count"],
                        "mock_count": analysis["mock_count"],
                    }
                )
            else:
                results["coverage_only_tests"].append(
                    {
                        "name": test_name,
                        "reason": analysis["coverage_only_reason"],
                        "suggestions": analysis["improvement_suggestions"],
                    }
                )

            results["assertion_quality"][test_name] = {
                "total_assertions": analysis["assertion_count"],
                "behavior_assertions": analysis["behavior_assertion_count"],
                "state_assertions": analysis["state_assertion_count"],
                "mock_assertions": analysis["mock_assertion_count"],
            }

            if analysis["mock_issues"]:
                results["mock_issues"].extend(analysis["mock_issues"])

            results["critical_paths_covered"].update(analysis["critical_paths"])

        # Identify missing critical paths
        results["missing_critical_paths"] = self._identify_missing_paths(
            results["critical_paths_covered"]
        )

        return results

    def _find_test_functions(self, tree: ast.AST) -> List[Tuple[str, ast.FunctionDef]]:
        """Find all test functions in the AST."""
        test_functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                test_functions.append((node.name, node))

        return test_functions

    def _analyze_test_function(self, name: str, node: ast.FunctionDef, content: str) -> Dict:
        """Analyze a single test function."""
        analysis = {
            "is_behavior_test": False,
            "behaviors_tested": [],
            "coverage_only_reason": None,
            "improvement_suggestions": [],
            "assertion_count": 0,
            "behavior_assertion_count": 0,
            "state_assertion_count": 0,
            "mock_assertion_count": 0,
            "mock_count": 0,
            "mock_issues": [],
            "critical_paths": set(),
        }

        # Extract function body
        func_start = node.lineno - 1
        func_end = node.end_lineno
        func_lines = content.split("\n")[func_start:func_end]
        func_body = "\n".join(func_lines)

        # Count assertions and analyze their quality
        assertions = self._analyze_assertions(func_body)
        analysis["assertion_count"] = assertions["total"]
        analysis["behavior_assertion_count"] = assertions["behavior"]
        analysis["state_assertion_count"] = assertions["state"]
        analysis["mock_assertion_count"] = assertions["mock"]

        # Count mocks
        analysis["mock_count"] = len(re.findall(r"\bMock\(|patch\(|@patch", func_body))

        # Analyze mock usage
        mock_issues = self._analyze_mock_usage(name, func_body)
        if mock_issues:
            analysis["mock_issues"] = mock_issues

        # Determine if it's a behavior test
        if analysis["behavior_assertion_count"] > 0:
            analysis["is_behavior_test"] = True
            analysis["behaviors_tested"] = self._extract_behaviors(name, func_body)
        else:
            analysis["coverage_only_reason"] = self._determine_coverage_only_reason(
                name, func_body, assertions
            )
            analysis["improvement_suggestions"] = self._suggest_improvements(
                name, func_body, assertions
            )

        # Identify critical paths
        analysis["critical_paths"] = self._identify_critical_paths(name, func_body)

        return analysis

    def _analyze_assertions(self, func_body: str) -> Dict[str, int]:
        """Analyze assertion quality."""
        assertions = {"total": 0, "behavior": 0, "state": 0, "mock": 0}

        # Count different types of assertions
        assertion_patterns = [
            (r"assert\s+.*\s+is\s+(not\s+)?None", "state"),
            (r"assert\s+len\(.*\)\s*[><=]=?\s*\d+", "state"),
            (r"assert\s+.*\s+in\s+", "state"),
            (r"assert\s+.*\s+==\s+", "behavior"),
            (r"assert\s+.*\s+!=\s+", "behavior"),
            (r"assert_called", "mock"),
            (r"assert_not_called", "mock"),
            (r"pytest\.raises", "behavior"),
        ]

        for pattern, assert_type in assertion_patterns:
            matches = len(re.findall(pattern, func_body))
            assertions[assert_type] += matches
            assertions["total"] += matches

        # General assert counting
        general_asserts = len(re.findall(r"\bassert\s+", func_body))
        unaccounted = general_asserts - sum(assertions[t] for t in ["state", "behavior"])
        assertions["behavior"] += unaccounted // 2  # Assume half are behavior
        assertions["state"] += unaccounted - (unaccounted // 2)

        return assertions

    def _analyze_mock_usage(self, test_name: str, func_body: str) -> List[str]:
        """Analyze mock usage for potential issues."""
        issues = []

        # Check for over-mocking
        mock_count = len(re.findall(r"\bMock\(|patch\(", func_body))
        if mock_count > 5:
            issues.append(
                f"{test_name}: Excessive mocking ({mock_count} mocks) may hide real issues"
            )

        # Check for mocking return values without behavior verification
        if "return_value" in func_body and "assert_called" not in func_body:
            issues.append(f"{test_name}: Mock return value set without verifying behavior")

        # Check for mocking entire classes/modules
        if re.search(r"patch\(['\"][\w\.]+['\"](?!\.)", func_body):
            issues.append(f"{test_name}: Mocking entire module may hide integration issues")

        return issues

    def _extract_behaviors(self, test_name: str, func_body: str) -> List[str]:
        """Extract behaviors being tested."""
        behaviors = []

        # Look for behavior indicators
        if "raises" in func_body:
            behaviors.append("error_handling")
        if "assert.*is.*model" in func_body:
            behaviors.append("return_value_correctness")
        if "cache" in test_name.lower():
            behaviors.append("caching_behavior")
        if "concurrent" in test_name.lower():
            behaviors.append("thread_safety")
        if "memory" in test_name.lower():
            behaviors.append("memory_management")

        return behaviors

    def _determine_coverage_only_reason(
        self, test_name: str, func_body: str, assertions: Dict
    ) -> str:
        """Determine why a test is coverage-only."""
        if assertions["total"] == 0:
            return "No assertions - test doesn't verify anything"
        elif assertions["mock"] == assertions["total"]:
            return "Only mock assertions - not testing actual behavior"
        elif assertions["state"] == assertions["total"]:
            return "Only state assertions - not verifying behavior"
        elif "pass" in func_body and len(func_body.split("\n")) < 5:
            return "Trivial test - likely just for coverage"
        else:
            return "Low behavior assertion ratio"

    def _suggest_improvements(self, test_name: str, func_body: str, assertions: Dict) -> List[str]:
        """Suggest improvements for coverage-only tests."""
        suggestions = []

        if assertions["total"] == 0:
            suggestions.append("Add assertions to verify expected behavior")

        if assertions["behavior"] == 0:
            suggestions.append("Add behavior assertions (verify outputs, side effects)")

        if "Mock" in func_body and assertions["mock"] == 0:
            suggestions.append("Verify mock interactions if using mocks")

        if "error" in test_name.lower() and "raises" not in func_body:
            suggestions.append("Use pytest.raises to test error conditions")

        return suggestions

    def _identify_critical_paths(self, test_name: str, func_body: str) -> Set[str]:
        """Identify critical paths being tested."""
        paths = set()

        critical_patterns = {
            "model_loading": r"get_model|load.*model",
            "cache_management": r"cache|clear_cache|unload",
            "error_recovery": r"raises|except|error",
            "concurrency": r"thread|concurrent|executor",
            "memory_management": r"memory|gc\.collect|empty_cache",
            "singleton": r"singleton|_instance",
            "device_handling": r"cuda|cpu|device",
        }

        for path_name, pattern in critical_patterns.items():
            if re.search(pattern, func_body, re.IGNORECASE):
                paths.add(path_name)

        return paths

    def _identify_missing_paths(self, covered_paths: Set[str]) -> List[str]:
        """Identify missing critical paths."""
        critical_paths = {
            "model_loading",
            "cache_management",
            "error_recovery",
            "concurrency",
            "memory_management",
            "singleton",
            "device_handling",
            "model_persistence",
            "performance_degradation",
            "resource_exhaustion",
            "model_validation",
            "hot_reload",
        }

        missing = list(critical_paths - covered_paths)
        return missing


def generate_behavior_test_examples():
    """Generate examples of behavior tests vs coverage tests."""
    examples = {
        "coverage_only": '''
# Coverage-only test - just checks state
def test_has_loaded_models(manager):
    """Test checking if models are loaded."""
    assert manager.has_loaded_models() is False
    manager._models["test"] = Mock()  
    assert manager.has_loaded_models() is True
    # This only tests the return value, not the behavior
''',
        "behavior_test": '''
# Behavior test - verifies actual functionality
def test_model_loading_caches_correctly(manager):
    """Test that models are cached after loading."""
    with patch('...FastMusicGenerator') as mock:
        mock.return_value = Mock()
        
        # First load
        model1 = manager.get_model("test", "optimized")
        assert mock.call_count == 1
        
        # Second load should use cache
        model2 = manager.get_model("test", "optimized") 
        assert mock.call_count == 1  # Not called again
        assert model1 is model2  # Same instance
        
        # Verifies caching BEHAVIOR, not just state
''',
        "mock_overuse": '''
# Over-mocked test - hides real issues
def test_model_operation(manager):
    """Test model operation."""
    with patch('manager._models') as mock_models:
        with patch('manager.get_model') as mock_get:
            with patch('manager.unload_model') as mock_unload:
                mock_get.return_value = Mock()
                # Too many mocks - not testing real behavior
''',
        "integration_test": '''
# Integration test - tests real behavior
def test_model_lifecycle():
    """Test complete model lifecycle."""
    manager = ModelManager()
    
    # Load a real model (or realistic mock)
    model = manager.get_model("test", "optimized")
    assert "test_optimized_cuda" in manager._models
    
    # Use the model
    cache_info = manager.get_cache_info()
    assert cache_info["loaded_models"] == 1
    
    # Unload and verify cleanup
    manager.unload_model("test")
    assert "test_optimized_cuda" not in manager._models
    
    # Tests real integration, not just units
''',
    }

    return examples


def analyze_all_tests():
    """Analyze all ModelManager tests."""
    analyzer = TestCoverageAnalyzer()

    # Analyze the comprehensive test file
    test_file = Path("tests/unit/test_model_manager_comprehensive.py")
    results = analyzer.analyze_test_file(test_file)

    # Generate report
    print("=" * 80)
    print("MODEL MANAGER TEST COVERAGE ANALYSIS")
    print("=" * 80)
    print()

    print(f"📊 Total Tests: {results['total_tests']}")
    print(f"✅ Behavior Tests: {len(results['behavior_tests'])}")
    print(f"⚠️  Coverage-Only Tests: {len(results['coverage_only_tests'])}")
    print(f"🔧 Mock Issues: {len(results['mock_issues'])}")
    print()

    # List behavior tests
    print("✅ BEHAVIOR TESTS (Good)")
    print("-" * 40)
    for test in results["behavior_tests"][:5]:  # Show first 5
        print(f"  • {test['name']}")
        print(f"    Behaviors: {', '.join(test['behaviors_tested'])}")
        print(f"    Assertions: {test['assertion_count']}")
    if len(results["behavior_tests"]) > 5:
        print(f"  ... and {len(results['behavior_tests']) - 5} more")
    print()

    # List coverage-only tests
    print("⚠️  COVERAGE-ONLY TESTS (Need Improvement)")
    print("-" * 40)
    for test in results["coverage_only_tests"][:5]:
        print(f"  • {test['name']}")
        print(f"    Reason: {test['reason']}")
        if test["suggestions"]:
            print(f"    Fix: {test['suggestions'][0]}")
    if len(results["coverage_only_tests"]) > 5:
        print(f"  ... and {len(results['coverage_only_tests']) - 5} more")
    print()

    # Mock issues
    if results["mock_issues"]:
        print("🔧 MOCK ISSUES")
        print("-" * 40)
        for issue in results["mock_issues"][:5]:
            print(f"  • {issue}")
        print()

    # Critical paths
    print("🛤️  CRITICAL PATH COVERAGE")
    print("-" * 40)
    print(f"Covered: {', '.join(sorted(results['critical_paths_covered']))}")
    if results["missing_critical_paths"]:
        print(f"Missing: {', '.join(sorted(results['missing_critical_paths']))}")
    print()

    # Recommendations
    print("📋 RECOMMENDATIONS")
    print("-" * 40)
    print("1. Refactor coverage-only tests to verify behavior")
    print("2. Reduce mock usage in integration points")
    print("3. Add tests for missing critical paths")
    print("4. Use integration tests for complex workflows")
    print("5. Add mutation testing to verify test effectiveness")

    return results


if __name__ == "__main__":
    # Check if test file exists
    test_file = Path("tests/unit/test_model_manager_comprehensive.py")
    if not test_file.exists():
        # Use the embedded test content for analysis
        print("Analyzing embedded test patterns...")

        examples = generate_behavior_test_examples()
        print("\n📚 TEST QUALITY EXAMPLES")
        print("=" * 80)

        for test_type, example in examples.items():
            print(f"\n{test_type.upper()}:")
            print(example)
    else:
        # Analyze actual test file
        analyze_all_tests()
