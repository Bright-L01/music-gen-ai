#!/usr/bin/env python3
"""
Code Quality Audit Script

Analyzes the codebase for:
- Code duplication
- Complex functions (too many lines, parameters)
- Unused imports and code
- Inconsistent naming conventions
- Missing documentation
- Code smells and anti-patterns
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import argparse


class CodeMetrics:
    """Collect code metrics and quality issues."""

    def __init__(self):
        self.files_analyzed = 0
        self.total_lines = 0
        self.total_functions = 0
        self.complex_functions = []
        self.long_functions = []
        self.functions_with_many_params = []
        self.duplicate_code_blocks = []
        self.unused_imports = []
        self.missing_docstrings = []
        self.naming_violations = []
        self.code_smells = []


class CodeAnalyzer(ast.NodeVisitor):
    """AST visitor for code analysis."""

    def __init__(self, filepath: str, source_lines: List[str]):
        self.filepath = filepath
        self.source_lines = source_lines
        self.issues = []
        self.imports = []
        self.functions = []
        self.classes = []
        self.current_class = None

    def visit_Import(self, node):
        """Visit import statements."""
        for alias in node.names:
            self.imports.append(
                {"name": alias.name, "asname": alias.asname, "line": node.lineno, "type": "import"}
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Visit from-import statements."""
        for alias in node.names:
            self.imports.append(
                {
                    "module": node.module,
                    "name": alias.name,
                    "asname": alias.asname,
                    "line": node.lineno,
                    "type": "from_import",
                }
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Visit function definitions."""
        func_info = self._analyze_function(node)
        self.functions.append(func_info)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definitions."""
        func_info = self._analyze_function(node, is_async=True)
        self.functions.append(func_info)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Visit class definitions."""
        old_class = self.current_class
        self.current_class = node.name

        class_info = {
            "name": node.name,
            "line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno),
            "docstring": ast.get_docstring(node),
            "methods": [],
            "is_public": not node.name.startswith("_"),
        }

        # Check naming convention
        if not self._is_valid_class_name(node.name):
            self.issues.append(
                {
                    "type": "naming_violation",
                    "message": f"Class '{node.name}' should use PascalCase",
                    "line": node.lineno,
                }
            )

        self.classes.append(class_info)
        self.generic_visit(node)
        self.current_class = old_class

    def _analyze_function(self, node, is_async=False):
        """Analyze a function node."""
        # Calculate function metrics
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", start_line)
        line_count = end_line - start_line + 1

        # Count parameters
        param_count = len(node.args.args)
        if node.args.vararg:
            param_count += 1
        if node.args.kwarg:
            param_count += 1
        param_count += len(node.args.kwonlyargs)

        # Get docstring
        docstring = ast.get_docstring(node)

        # Calculate complexity (simplified McCabe complexity)
        complexity = self._calculate_complexity(node)

        func_info = {
            "name": node.name,
            "line": start_line,
            "end_line": end_line,
            "line_count": line_count,
            "param_count": param_count,
            "complexity": complexity,
            "docstring": docstring,
            "is_async": is_async,
            "is_public": not node.name.startswith("_"),
            "is_method": self.current_class is not None,
            "class_name": self.current_class,
        }

        # Check for issues
        if line_count > 50:
            self.issues.append(
                {
                    "type": "long_function",
                    "message": f"Function '{node.name}' is too long ({line_count} lines)",
                    "line": start_line,
                    "data": func_info,
                }
            )

        if param_count > 7:
            self.issues.append(
                {
                    "type": "many_parameters",
                    "message": f"Function '{node.name}' has too many parameters ({param_count})",
                    "line": start_line,
                    "data": func_info,
                }
            )

        if complexity > 10:
            self.issues.append(
                {
                    "type": "complex_function",
                    "message": f"Function '{node.name}' is too complex (complexity: {complexity})",
                    "line": start_line,
                    "data": func_info,
                }
            )

        if func_info["is_public"] and not docstring:
            self.issues.append(
                {
                    "type": "missing_docstring",
                    "message": f"Public function '{node.name}' is missing docstring",
                    "line": start_line,
                    "data": func_info,
                }
            )

        # Check naming convention
        if not self._is_valid_function_name(node.name):
            self.issues.append(
                {
                    "type": "naming_violation",
                    "message": f"Function '{node.name}' should use snake_case",
                    "line": start_line,
                }
            )

        return func_info

    def _calculate_complexity(self, node):
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Decision points increase complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # Each additional boolean operation adds complexity
                complexity += len(child.values) - 1

        return complexity

    def _is_valid_function_name(self, name):
        """Check if function name follows snake_case convention."""
        if name.startswith("__") and name.endswith("__"):
            return True  # Magic methods are OK
        return name.islower() and "_" in name or name.islower()

    def _is_valid_class_name(self, name):
        """Check if class name follows PascalCase convention."""
        return name[0].isupper() and "_" not in name


def analyze_file(filepath: Path) -> Tuple[CodeAnalyzer, List[str]]:
    """Analyze a single Python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
            source_lines = source.splitlines()

        tree = ast.parse(source, filename=str(filepath))
        analyzer = CodeAnalyzer(str(filepath), source_lines)
        analyzer.visit(tree)

        return analyzer, source_lines

    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return None, []
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None, []


def find_duplicate_code(analyzers: List[CodeAnalyzer], min_lines: int = 5) -> List[Dict]:
    """Find duplicate code blocks across files."""
    # This is a simplified implementation
    # In practice, you'd use more sophisticated algorithms

    line_groups = defaultdict(list)

    # Group files by similar line patterns
    for analyzer in analyzers:
        if not analyzer:
            continue

        lines = analyzer.source_lines
        for i in range(len(lines) - min_lines + 1):
            block = "\n".join(lines[i : i + min_lines])
            # Normalize whitespace and remove comments
            normalized = normalize_code_block(block)
            if normalized.strip():
                line_groups[normalized].append(
                    {
                        "file": analyzer.filepath,
                        "start_line": i + 1,
                        "end_line": i + min_lines,
                        "block": block,
                    }
                )

    # Find groups with multiple occurrences
    duplicates = []
    for normalized_block, occurrences in line_groups.items():
        if len(occurrences) > 1:
            duplicates.append(
                {
                    "block": normalized_block,
                    "occurrences": occurrences,
                    "count": len(occurrences),
                }
            )

    return duplicates


def normalize_code_block(block: str) -> str:
    """Normalize code block for duplicate detection."""
    lines = []
    for line in block.split("\n"):
        # Remove leading/trailing whitespace
        line = line.strip()
        # Skip empty lines and comments
        if line and not line.startswith("#"):
            lines.append(line)
    return "\n".join(lines)


def find_unused_imports(analyzer: CodeAnalyzer) -> List[Dict]:
    """Find potentially unused imports."""
    used_names = set()

    # Simple heuristic: collect all names used in the code
    try:
        with open(analyzer.filepath, "r") as f:
            content = f.read()

        # This is a very basic check - could be improved with AST analysis
        for imp in analyzer.imports:
            name = imp.get("asname") or imp["name"]
            if imp["type"] == "from_import":
                # For from imports, check if the imported name is used
                if name != "*" and name not in content:
                    yield {
                        "type": "unused_import",
                        "name": name,
                        "line": imp["line"],
                        "import_info": imp,
                    }
            else:
                # For regular imports, check if module name is used
                if name not in content:
                    yield {
                        "type": "unused_import",
                        "name": name,
                        "line": imp["line"],
                        "import_info": imp,
                    }

    except Exception:
        pass


def analyze_codebase(root_path: Path, exclude_patterns: List[str] = None) -> CodeMetrics:
    """Analyze entire codebase."""
    if exclude_patterns is None:
        exclude_patterns = [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "env",
            ".pytest_cache",
            "htmlcov",
            ".coverage",
            "dist",
            "build",
            "*.egg-info",
        ]

    metrics = CodeMetrics()
    analyzers = []

    # Find all Python files
    python_files = []
    for path in root_path.rglob("*.py"):
        # Skip excluded patterns
        skip = False
        for pattern in exclude_patterns:
            if pattern in str(path):
                skip = True
                break
        if not skip:
            python_files.append(path)

    print(f"Analyzing {len(python_files)} Python files...")

    # Analyze each file
    for filepath in python_files:
        analyzer, source_lines = analyze_file(filepath)
        if analyzer:
            analyzers.append(analyzer)
            metrics.files_analyzed += 1
            metrics.total_lines += len(source_lines)
            metrics.total_functions += len(analyzer.functions)

            # Collect issues
            for issue in analyzer.issues:
                if issue["type"] == "long_function":
                    metrics.long_functions.append(
                        {
                            "file": filepath,
                            "function": issue["data"]["name"],
                            "lines": issue["data"]["line_count"],
                            "line": issue["line"],
                        }
                    )
                elif issue["type"] == "complex_function":
                    metrics.complex_functions.append(
                        {
                            "file": filepath,
                            "function": issue["data"]["name"],
                            "complexity": issue["data"]["complexity"],
                            "line": issue["line"],
                        }
                    )
                elif issue["type"] == "many_parameters":
                    metrics.functions_with_many_params.append(
                        {
                            "file": filepath,
                            "function": issue["data"]["name"],
                            "params": issue["data"]["param_count"],
                            "line": issue["line"],
                        }
                    )
                elif issue["type"] == "missing_docstring":
                    metrics.missing_docstrings.append(
                        {
                            "file": filepath,
                            "function": issue["data"]["name"],
                            "line": issue["line"],
                        }
                    )
                elif issue["type"] == "naming_violation":
                    metrics.naming_violations.append(
                        {
                            "file": filepath,
                            "message": issue["message"],
                            "line": issue["line"],
                        }
                    )

            # Check for unused imports
            for unused in find_unused_imports(analyzer):
                metrics.unused_imports.append(
                    {
                        "file": filepath,
                        "name": unused["name"],
                        "line": unused["line"],
                    }
                )

    # Find duplicate code
    print("Analyzing code duplication...")
    duplicates = find_duplicate_code(analyzers)
    metrics.duplicate_code_blocks = duplicates

    return metrics


def generate_report(metrics: CodeMetrics) -> str:
    """Generate quality report."""
    report = []

    report.append("=" * 60)
    report.append("CODE QUALITY AUDIT REPORT")
    report.append("=" * 60)

    # Summary
    report.append(f"\nSUMMARY:")
    report.append(f"  Files analyzed: {metrics.files_analyzed}")
    report.append(f"  Total lines: {metrics.total_lines:,}")
    report.append(f"  Total functions: {metrics.total_functions}")

    # Issues summary
    total_issues = (
        len(metrics.long_functions)
        + len(metrics.complex_functions)
        + len(metrics.functions_with_many_params)
        + len(metrics.missing_docstrings)
        + len(metrics.naming_violations)
        + len(metrics.unused_imports)
        + len(metrics.duplicate_code_blocks)
    )
    report.append(f"  Total issues found: {total_issues}")

    # Detailed issues
    if metrics.long_functions:
        report.append(f"\nLONG FUNCTIONS ({len(metrics.long_functions)}):")
        for func in sorted(metrics.long_functions, key=lambda x: x["lines"], reverse=True)[:10]:
            report.append(
                f"  {func['file']}:{func['line']} - {func['function']} ({func['lines']} lines)"
            )

    if metrics.complex_functions:
        report.append(f"\nCOMPLEX FUNCTIONS ({len(metrics.complex_functions)}):")
        for func in sorted(metrics.complex_functions, key=lambda x: x["complexity"], reverse=True)[
            :10
        ]:
            report.append(
                f"  {func['file']}:{func['line']} - {func['function']} (complexity: {func['complexity']})"
            )

    if metrics.functions_with_many_params:
        report.append(
            f"\nFUNCTIONS WITH TOO MANY PARAMETERS ({len(metrics.functions_with_many_params)}):"
        )
        for func in sorted(
            metrics.functions_with_many_params, key=lambda x: x["params"], reverse=True
        )[:10]:
            report.append(
                f"  {func['file']}:{func['line']} - {func['function']} ({func['params']} params)"
            )

    if metrics.missing_docstrings:
        report.append(f"\nMISSING DOCSTRINGS ({len(metrics.missing_docstrings)}):")
        for func in metrics.missing_docstrings[:20]:
            report.append(f"  {func['file']}:{func['line']} - {func['function']}")

    if metrics.naming_violations:
        report.append(f"\nNAMING VIOLATIONS ({len(metrics.naming_violations)}):")
        for violation in metrics.naming_violations[:20]:
            report.append(f"  {violation['file']}:{violation['line']} - {violation['message']}")

    if metrics.unused_imports:
        report.append(f"\nPOTENTIALLY UNUSED IMPORTS ({len(metrics.unused_imports)}):")
        for imp in metrics.unused_imports[:20]:
            report.append(f"  {imp['file']}:{imp['line']} - {imp['name']}")

    if metrics.duplicate_code_blocks:
        report.append(f"\nDUPLICATE CODE BLOCKS ({len(metrics.duplicate_code_blocks)}):")
        for dup in sorted(metrics.duplicate_code_blocks, key=lambda x: x["count"], reverse=True)[
            :10
        ]:
            report.append(f"  Found in {dup['count']} places:")
            for occ in dup["occurrences"][:3]:
                report.append(f"    {occ['file']}:{occ['start_line']}-{occ['end_line']}")
            if len(dup["occurrences"]) > 3:
                report.append(f"    ... and {len(dup['occurrences']) - 3} more")

    # Recommendations
    report.append(f"\nRECOMMENDATIONS:")
    if metrics.long_functions:
        report.append("  • Break down long functions into smaller, focused functions")
    if metrics.complex_functions:
        report.append("  • Reduce complexity by extracting helper functions or simplifying logic")
    if metrics.functions_with_many_params:
        report.append(
            "  • Use parameter objects or configuration classes for functions with many parameters"
        )
    if metrics.missing_docstrings:
        report.append("  • Add docstrings to all public functions and classes")
    if metrics.naming_violations:
        report.append("  • Follow Python naming conventions (PEP 8)")
    if metrics.unused_imports:
        report.append("  • Remove unused imports to reduce clutter")
    if metrics.duplicate_code_blocks:
        report.append("  • Extract duplicate code into shared functions or modules")

    return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Code Quality Audit")
    parser.add_argument("path", nargs="?", default=".", help="Path to analyze")
    parser.add_argument("--output", "-o", help="Output file for report")
    parser.add_argument("--exclude", "-e", action="append", help="Patterns to exclude")

    args = parser.parse_args()

    root_path = Path(args.path)
    if not root_path.exists():
        print(f"Error: Path {root_path} does not exist")
        sys.exit(1)

    exclude_patterns = args.exclude or []

    # Run analysis
    print(f"Starting code quality audit of {root_path}...")
    metrics = analyze_codebase(root_path, exclude_patterns)

    # Generate report
    report = generate_report(metrics)

    # Output report
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
