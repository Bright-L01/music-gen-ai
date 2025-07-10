#!/usr/bin/env python3
"""
Mutation Testing for ModelManager

This script introduces mutations (bugs) into the ModelManager code
and verifies that tests catch these bugs.
"""

import ast
import copy
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess
import json


class MutationOperator:
    """Base class for mutation operators."""

    def can_mutate(self, node: ast.AST) -> bool:
        """Check if this operator can mutate the given node."""
        raise NotImplementedError

    def mutate(self, node: ast.AST) -> ast.AST:
        """Apply mutation to the node."""
        raise NotImplementedError

    def describe(self) -> str:
        """Describe the mutation."""
        raise NotImplementedError


class ComparisonMutator(MutationOperator):
    """Mutate comparison operators."""

    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Compare)

    def mutate(self, node: ast.AST) -> ast.AST:
        """Swap comparison operators."""
        mutated = copy.deepcopy(node)

        # Mutation mappings
        mutations = {
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
            ast.Lt: ast.GtE,
            ast.Gt: ast.LtE,
            ast.LtE: ast.Gt,
            ast.GtE: ast.Lt,
            ast.Is: ast.IsNot,
            ast.IsNot: ast.Is,
            ast.In: ast.NotIn,
            ast.NotIn: ast.In,
        }

        for i, op in enumerate(mutated.ops):
            op_type = type(op)
            if op_type in mutations:
                mutated.ops[i] = mutations[op_type]()
                break

        return mutated

    def describe(self) -> str:
        return "Comparison operator mutation (== to !=, < to >=, etc.)"


class BooleanMutator(MutationOperator):
    """Mutate boolean values."""

    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, bool)

    def mutate(self, node: ast.AST) -> ast.AST:
        """Flip boolean values."""
        mutated = copy.deepcopy(node)
        mutated.value = not node.value
        return mutated

    def describe(self) -> str:
        return "Boolean value mutation (True to False, False to True)"


class ReturnValueMutator(MutationOperator):
    """Mutate return values."""

    def can_mutate(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Return):
            return False

        # Only mutate simple returns
        if node.value is None:
            return False

        if isinstance(node.value, ast.Constant):
            return True
        elif isinstance(node.value, ast.Name):
            return True

        return False

    def mutate(self, node: ast.AST) -> ast.AST:
        """Change return values."""
        mutated = copy.deepcopy(node)

        if isinstance(node.value, ast.Constant):
            if node.value.value is True:
                mutated.value = ast.Constant(value=False)
            elif node.value.value is False:
                mutated.value = ast.Constant(value=True)
            elif isinstance(node.value.value, (int, float)):
                mutated.value = ast.Constant(value=0)
            elif isinstance(node.value.value, str):
                mutated.value = ast.Constant(value="")

        elif isinstance(node.value, ast.Name):
            # Return None instead
            mutated.value = ast.Constant(value=None)

        return mutated

    def describe(self) -> str:
        return "Return value mutation"


class ConditionalMutator(MutationOperator):
    """Remove conditional checks."""

    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, ast.If)

    def mutate(self, node: ast.AST) -> ast.AST:
        """Always execute if-block (remove condition)."""
        mutated = copy.deepcopy(node)
        # Replace condition with True
        mutated.test = ast.Constant(value=True)
        return mutated

    def describe(self) -> str:
        return "Conditional mutation (always True)"


class DictionaryMutator(MutationOperator):
    """Mutate dictionary operations."""

    def can_mutate(self, node: ast.AST) -> bool:
        # Look for dict access patterns
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            return True
        return False

    def mutate(self, node: ast.AST) -> ast.AST:
        """Change dictionary key access."""
        mutated = copy.deepcopy(node)

        # If accessing with string key, change it
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            mutated.slice = ast.Constant(value=node.slice.value + "_mutated")

        return mutated

    def describe(self) -> str:
        return "Dictionary key mutation"


class ExceptionMutator(MutationOperator):
    """Remove exception handling."""

    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Try)

    def mutate(self, node: ast.AST) -> ast.AST:
        """Remove try-except, just execute body."""
        # Return just the try body without exception handling
        return node.body[0] if node.body else ast.Pass()

    def describe(self) -> str:
        return "Exception handling removal"


class MutationTester:
    """Apply mutations and test if they're caught."""

    def __init__(self, source_file: Path, test_command: List[str]):
        self.source_file = source_file
        self.test_command = test_command
        self.operators = [
            ComparisonMutator(),
            BooleanMutator(),
            ReturnValueMutator(),
            ConditionalMutator(),
            DictionaryMutator(),
            ExceptionMutator(),
        ]

    def find_mutation_sites(self, tree: ast.AST) -> List[Tuple[ast.AST, MutationOperator]]:
        """Find all possible mutation sites."""
        sites = []

        for node in ast.walk(tree):
            for operator in self.operators:
                if operator.can_mutate(node):
                    sites.append((node, operator))

        return sites

    def apply_mutation(self, tree: ast.AST, node: ast.AST, operator: MutationOperator) -> ast.AST:
        """Apply a single mutation to the tree."""

        class MutationTransformer(ast.NodeTransformer):
            def __init__(self, target_node, mutation_op):
                self.target_node = target_node
                self.mutation_op = mutation_op

            def visit(self, node):
                if node is self.target_node:
                    return self.mutation_op.mutate(node)
                return self.generic_visit(node)

        transformer = MutationTransformer(node, operator)
        mutated_tree = transformer.visit(copy.deepcopy(tree))
        return mutated_tree

    def run_tests(self) -> bool:
        """Run tests and return True if all pass."""
        try:
            result = subprocess.run(self.test_command, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    def test_mutation(self, mutated_code: str) -> bool:
        """Test if mutation is caught by tests."""
        # Save original file
        original_content = self.source_file.read_text()

        try:
            # Write mutated code
            self.source_file.write_text(mutated_code)

            # Run tests
            tests_pass = self.run_tests()

            # If tests still pass, mutation wasn't caught
            return not tests_pass

        finally:
            # Restore original file
            self.source_file.write_text(original_content)

    def run_mutation_testing(self) -> Dict:
        """Run mutation testing and return results."""
        # Parse source file
        source_code = self.source_file.read_text()
        tree = ast.parse(source_code)

        # Find mutation sites
        sites = self.find_mutation_sites(tree)

        results = {
            "total_mutations": len(sites),
            "caught_mutations": 0,
            "survived_mutations": 0,
            "mutation_score": 0.0,
            "survived_details": [],
        }

        print(f"Found {len(sites)} mutation sites in {self.source_file.name}")
        print("Running mutation testing...")

        for i, (node, operator) in enumerate(sites):
            print(f"\rTesting mutation {i+1}/{len(sites)}...", end="", flush=True)

            # Apply mutation
            mutated_tree = self.apply_mutation(tree, node, operator)
            mutated_code = ast.unparse(mutated_tree)

            # Test if caught
            caught = self.test_mutation(mutated_code)

            if caught:
                results["caught_mutations"] += 1
            else:
                results["survived_mutations"] += 1

                # Get line number for reporting
                line_no = getattr(node, "lineno", 0)

                results["survived_details"].append(
                    {
                        "line": line_no,
                        "mutation": operator.describe(),
                        "node_type": type(node).__name__,
                    }
                )

        # Calculate mutation score
        if results["total_mutations"] > 0:
            results["mutation_score"] = (
                results["caught_mutations"] / results["total_mutations"] * 100
            )

        print(f"\n\nMutation Testing Complete!")
        print(f"Total mutations: {results['total_mutations']}")
        print(f"Caught: {results['caught_mutations']}")
        print(f"Survived: {results['survived_mutations']}")
        print(f"Mutation Score: {results['mutation_score']:.1f}%")

        if results["survived_mutations"] > 0:
            print("\nSurvived Mutations (tests didn't catch these):")
            for detail in results["survived_details"][:10]:  # Show first 10
                print(f"  Line {detail['line']}: {detail['mutation']} on {detail['node_type']}")

            if len(results["survived_details"]) > 10:
                print(f"  ... and {len(results['survived_details']) - 10} more")

        return results


def create_minimal_test_mutations():
    """Create minimal mutations to test specific behaviors."""
    mutations = [
        {
            "name": "Singleton Pattern Break",
            "original": """
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
""",
            "mutated": """
    def __new__(cls):
        # Always create new instance (breaks singleton)
        instance = super().__new__(cls)
        instance._initialized = False
        return instance
""",
            "expected": "Tests should fail - singleton pattern broken",
        },
        {
            "name": "Cache Bypass",
            "original": """
    if model_key in self._models:
        logger.info(f"Using cached model: {model_key}")
        return self._models[model_key]
""",
            "mutated": """
    # Never use cache (always load new)
    if False:  # model_key in self._models:
        logger.info(f"Using cached model: {model_key}")
        return self._models[model_key]
""",
            "expected": "Tests should fail - caching behavior broken",
        },
        {
            "name": "Memory Cleanup Skip",
            "original": """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
""",
            "mutated": """
    # Skip memory cleanup
    pass
    # gc.collect()
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
""",
            "expected": "Memory management tests should fail",
        },
        {
            "name": "Wrong Model Type",
            "original": """
    if model_type == "optimized":
        model = FastMusicGenerator(...)
    elif model_type == "multi_instrument":
        model = FastMusicGenerator(...)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
""",
            "mutated": """
    if model_type == "optimized":
        model = FastMusicGenerator(...)
    elif model_type == "multi_instrument":
        model = FastMusicGenerator(...)
    else:
        # Don't raise error, just use default
        model = FastMusicGenerator(...)
""",
            "expected": "Error handling tests should fail",
        },
    ]

    return mutations


def generate_mutation_report(results: Dict) -> str:
    """Generate a detailed mutation testing report."""
    report = f"""
# Mutation Testing Report

## Summary
- **Total Mutations**: {results['total_mutations']}
- **Caught by Tests**: {results['caught_mutations']}
- **Survived**: {results['survived_mutations']}
- **Mutation Score**: {results['mutation_score']:.1f}%

## Analysis

### What is Mutation Testing?
Mutation testing introduces small bugs (mutations) into code and checks if tests catch them.
A high mutation score means tests are effective at catching bugs.

### Score Interpretation
- **90-100%**: Excellent - Tests catch almost all bugs
- **70-90%**: Good - Most bugs are caught
- **50-70%**: Fair - Many bugs slip through
- **<50%**: Poor - Tests miss most bugs

### Current Score: {results['mutation_score']:.1f}%
"""

    if results["mutation_score"] >= 90:
        report += "\n✅ **Excellent!** Your tests are highly effective at catching bugs.\n"
    elif results["mutation_score"] >= 70:
        report += "\n✅ **Good!** Your tests catch most bugs, but there's room for improvement.\n"
    elif results["mutation_score"] >= 50:
        report += (
            "\n⚠️  **Fair.** Many bugs could slip through. Consider adding more behavior tests.\n"
        )
    else:
        report += "\n❌ **Poor.** Most bugs aren't caught. Tests need significant improvement.\n"

    if results["survived_mutations"] > 0:
        report += "\n## Survived Mutations\nThese mutations weren't caught by tests:\n\n"

        # Group by mutation type
        by_type = {}
        for detail in results["survived_details"]:
            mut_type = detail["mutation"]
            if mut_type not in by_type:
                by_type[mut_type] = []
            by_type[mut_type].append(detail)

        for mut_type, details in by_type.items():
            report += f"### {mut_type} ({len(details)} instances)\n"
            for detail in details[:3]:  # Show first 3
                report += f"- Line {detail['line']}: {detail['node_type']}\n"
            if len(details) > 3:
                report += f"- ... and {len(details) - 3} more\n"
            report += "\n"

    report += """
## Recommendations

1. **Add Behavior Tests**: Focus on testing outcomes, not implementation
2. **Test Edge Cases**: Add tests for boundary conditions and error cases
3. **Reduce Mocking**: Over-mocking can hide real bugs
4. **Test Integration**: Add tests that verify component interactions
5. **Use Property Testing**: Generate random inputs to find edge cases
"""

    return report


def main():
    """Run mutation testing on ModelManager."""
    # Paths
    project_root = Path(__file__).parent.parent
    source_file = project_root / "music_gen" / "core" / "model_manager.py"

    # Check if source file exists
    if not source_file.exists():
        print(f"Source file not found: {source_file}")
        print("Creating example mutations instead...")

        mutations = create_minimal_test_mutations()
        print("\nExample Mutations That Tests Should Catch:\n")

        for mutation in mutations:
            print(f"**{mutation['name']}**")
            print(f"Expected: {mutation['expected']}")
            print("-" * 60)
            print()

        return

    # Test command
    test_command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/test_model_manager_comprehensive.py",
        "-xvs",
        "--tb=short",
    ]

    # Run mutation testing
    tester = MutationTester(source_file, test_command)
    results = tester.run_mutation_testing()

    # Generate report
    report = generate_mutation_report(results)

    # Save report
    report_file = project_root / "MUTATION_TEST_REPORT.md"
    report_file.write_text(report)

    print(f"\n\nReport saved to: {report_file}")

    # Return status based on score
    if results["mutation_score"] < 70:
        sys.exit(1)  # Fail if score is too low


if __name__ == "__main__":
    main()
