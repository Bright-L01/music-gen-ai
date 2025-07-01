#!/usr/bin/env python3
"""
Build comprehensive test suite to achieve 90%+ test coverage.
Analyzes code and generates appropriate tests.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
MUSIC_GEN_DIR = PROJECT_ROOT / "music_gen"
TESTS_DIR = PROJECT_ROOT / "tests"


class TestBuilder:
    """Builds test files for uncovered modules."""

    def __init__(self):
        self.test_templates = {
            "api": self._generate_api_test,
            "model": self._generate_model_test,
            "utils": self._generate_utils_test,
            "audio": self._generate_audio_test,
            "data": self._generate_data_test,
            "streaming": self._generate_streaming_test,
            "training": self._generate_training_test,
            "config": self._generate_config_test,
            "default": self._generate_default_test,
        }

    def analyze_module(self, module_path: Path) -> Dict:
        """Analyze module to understand what needs testing."""
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()
            try:
                tree = ast.parse(content)
            except:
                return {"type": "invalid", "content": content}

        analysis = {
            "imports": [],
            "classes": [],
            "functions": [],
            "constants": [],
            "has_main": False,
            "decorators": set(),
            "is_api": False,
            "is_model": False,
            "complexity": 0,
        }

        # Analyze imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    analysis["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    analysis["imports"].append(f"{module}.{alias.name}")

        # Analyze structure
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(
                            {
                                "name": item.name,
                                "args": [arg.arg for arg in item.args.args],
                                "decorators": [
                                    self._get_decorator_name(d) for d in item.decorator_list
                                ],
                            }
                        )
                analysis["classes"].append(
                    {
                        "name": node.name,
                        "methods": methods,
                        "bases": [self._get_name(base) for base in node.bases],
                    }
                )
            elif isinstance(node, ast.FunctionDef):
                analysis["functions"].append(
                    {
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
                    }
                )
                if node.name == "main":
                    analysis["has_main"] = True
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        analysis["constants"].append(target.id)

        # Detect module type
        if "fastapi" in str(analysis["imports"]) or "FastAPI" in content:
            analysis["is_api"] = True
        if "torch.nn" in str(analysis["imports"]) or "nn.Module" in content:
            analysis["is_model"] = True

        # Calculate complexity
        analysis["complexity"] = (
            len(analysis["classes"]) * 3
            + len(analysis["functions"])
            + sum(len(c["methods"]) for c in analysis["classes"])
        )

        return analysis

    def _get_decorator_name(self, node):
        """Extract decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return "unknown"

    def _get_name(self, node):
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "unknown"

    def determine_test_type(self, module_path: Path, analysis: Dict) -> str:
        """Determine which type of test to generate."""
        path_str = str(module_path)

        if analysis.get("is_api") or "/api/" in path_str:
            return "api"
        elif analysis.get("is_model") or "/models/" in path_str:
            return "model"
        elif "/utils/" in path_str:
            return "utils"
        elif "/audio/" in path_str:
            return "audio"
        elif "/data/" in path_str:
            return "data"
        elif "/streaming/" in path_str:
            return "streaming"
        elif "/training/" in path_str:
            return "training"
        elif "/config" in path_str:
            return "config"
        else:
            return "default"

    def generate_test(self, module_path: Path) -> str:
        """Generate test for a module."""
        analysis = self.analyze_module(module_path)
        test_type = self.determine_test_type(module_path, analysis)

        generator = self.test_templates.get(test_type, self._generate_default_test)
        return generator(module_path, analysis)

    def _generate_api_test(self, module_path: Path, analysis: Dict) -> str:
        """Generate API endpoint tests."""
        module_name = module_path.stem
        import_path = self._get_import_path(module_path)

        test_content = f'''"""
Tests for {import_path}
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from {import_path} import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_auth():
    """Mock authentication."""
    with patch("{import_path}.get_current_user") as mock:
        mock.return_value = {{"id": "test_user", "email": "test@example.com"}}
        yield mock


class Test{self._to_class_name(module_name)}API:
    """Test {module_name} API endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "ok"]

'''

        # Add tests for each endpoint found
        for func in analysis.get("functions", []):
            if any(
                d in ["get", "post", "put", "delete", "patch"] for d in func.get("decorators", [])
            ):
                test_content += self._generate_endpoint_test(func)

        return test_content

    def _generate_endpoint_test(self, func: Dict) -> str:
        """Generate test for a specific endpoint."""
        name = func["name"]
        return f'''    def test_{name}(self, client, mock_auth):
        """Test {name} endpoint."""
        # TODO: Implement test for {name}
        pass

'''

    def _generate_model_test(self, module_path: Path, analysis: Dict) -> str:
        """Generate model tests."""
        module_name = module_path.stem
        import_path = self._get_import_path(module_path)

        test_content = f'''"""
Tests for {import_path}
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from {import_path} import *


class Test{self._to_class_name(module_name)}Model:
    """Test {module_name} model components."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

'''

        # Add tests for each class
        for cls in analysis.get("classes", []):
            if any("nn.Module" in base for base in cls.get("bases", [])):
                test_content += self._generate_model_class_test(cls)

        return test_content

    def _generate_model_class_test(self, cls: Dict) -> str:
        """Generate test for a model class."""
        name = cls["name"]
        return f'''    def test_{name.lower()}_creation(self, device):
        """Test {name} model creation."""
        model = {name}()
        assert isinstance(model, nn.Module)

    def test_{name.lower()}_forward(self, device):
        """Test {name} forward pass."""
        model = {name}().to(device)
        # TODO: Create appropriate input tensor
        # input_tensor = torch.randn(1, 128).to(device)
        # output = model(input_tensor)
        # assert output.shape == expected_shape
        pass

'''

    def _generate_utils_test(self, module_path: Path, analysis: Dict) -> str:
        """Generate utility function tests."""
        module_name = module_path.stem
        import_path = self._get_import_path(module_path)

        test_content = f'''"""
Tests for {import_path}
"""

import pytest
from unittest.mock import patch, MagicMock

from {import_path} import *


class Test{self._to_class_name(module_name)}:
    """Test {module_name} utilities."""

'''

        # Add tests for each function
        for func in analysis.get("functions", []):
            if not func["name"].startswith("_"):
                test_content += self._generate_function_test(func)

        # Add tests for each class
        for cls in analysis.get("classes", []):
            test_content += self._generate_class_tests(cls)

        return test_content

    def _generate_function_test(self, func: Dict) -> str:
        """Generate test for a function."""
        name = func["name"]
        args = [a for a in func["args"] if a != "self"]

        test = f'''    def test_{name}(self):
        """Test {name} function."""
'''

        if not args:
            test += f"""        result = {name}()
        # TODO: Add assertions
        assert result is not None
"""
        else:
            test += f"""        # TODO: Provide test arguments
        # result = {name}({", ".join(args)})
        # assert result == expected_value
        pass
"""

        test += "\n"
        return test

    def _generate_class_tests(self, cls: Dict) -> str:
        """Generate tests for a class."""
        name = cls["name"]
        tests = f'''
class Test{name}:
    """Test {name} class."""

    def test_creation(self):
        """Test {name} instantiation."""
        instance = {name}()
        assert instance is not None

'''

        # Add test for each public method
        for method in cls.get("methods", []):
            if not method["name"].startswith("_") or method["name"] == "__init__":
                continue
            tests += f'''    def test_{method["name"]}(self):
        """Test {name}.{method["name"]} method."""
        instance = {name}()
        # TODO: Test {method["name"]} method
        pass

'''

        return tests

    def _generate_audio_test(self, module_path: Path, analysis: Dict) -> str:
        """Generate audio processing tests."""
        module_name = module_path.stem
        import_path = self._get_import_path(module_path)

        return f'''"""
Tests for {import_path}
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from {import_path} import *


class Test{self._to_class_name(module_name)}:
    """Test {module_name} audio processing."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        # 1 second of audio at 16kHz
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Generate 440Hz sine wave
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        return audio, sample_rate

    @pytest.fixture
    def sample_tensor(self):
        """Create sample audio tensor."""
        # Batch of 2, 1 channel, 16000 samples
        return torch.randn(2, 1, 16000)

    # TODO: Add specific tests for audio processing functions

'''

    def _generate_data_test(self, module_path: Path, analysis: Dict) -> str:
        """Generate data loading/processing tests."""
        module_name = module_path.stem
        import_path = self._get_import_path(module_path)

        return f'''"""
Tests for {import_path}
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open

from {import_path} import *


class Test{self._to_class_name(module_name)}:
    """Test {module_name} data handling."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {{
            "id": "test_001",
            "text": "Generate upbeat jazz music",
            "audio_path": "/path/to/audio.wav",
            "duration": 10.0,
            "metadata": {{"genre": "jazz", "mood": "upbeat"}}
        }}

    @pytest.fixture
    def mock_dataset_path(self, tmp_path):
        """Create temporary dataset directory."""
        dataset_dir = tmp_path / "test_dataset"
        dataset_dir.mkdir()

        # Create sample files
        (dataset_dir / "metadata.json").write_text('[{{"id": "1", "text": "test"}}]')
        (dataset_dir / "audio").mkdir()

        return dataset_dir

    # TODO: Add specific tests for data loading

'''

    def _generate_streaming_test(self, module_path: Path, analysis: Dict) -> str:
        """Generate streaming tests."""
        module_name = module_path.stem
        import_path = self._get_import_path(module_path)

        return f'''"""
Tests for {import_path}
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from {import_path} import *


class Test{self._to_class_name(module_name)}:
    """Test {module_name} streaming functionality."""

    @pytest.mark.asyncio
    async def test_streaming_setup(self):
        """Test streaming setup."""
        # TODO: Implement streaming tests
        pass

    @pytest.fixture
    def mock_websocket(self):
        """Create mock websocket."""
        ws = AsyncMock()
        ws.send_text = AsyncMock()
        ws.receive_text = AsyncMock(return_value='{{"type": "test"}}')
        return ws

'''

    def _generate_training_test(self, module_path: Path, analysis: Dict) -> str:
        """Generate training tests."""
        module_name = module_path.stem
        import_path = self._get_import_path(module_path)

        return f'''"""
Tests for {import_path}
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from {import_path} import *


class Test{self._to_class_name(module_name)}:
    """Test {module_name} training components."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model for training."""
        model = MagicMock()
        model.parameters.return_value = [torch.randn(10, 10)]
        model.train.return_value = model
        model.eval.return_value = model
        return model

    @pytest.fixture
    def mock_dataloader(self):
        """Create mock dataloader."""
        dataloader = MagicMock()
        dataloader.__iter__.return_value = iter([
            {{"input": torch.randn(4, 128), "target": torch.randn(4, 10)}}
            for _ in range(2)
        ])
        dataloader.__len__.return_value = 2
        return dataloader

    # TODO: Add specific training tests

'''

    def _generate_config_test(self, module_path: Path, analysis: Dict) -> str:
        """Generate configuration tests."""
        module_name = module_path.stem
        import_path = self._get_import_path(module_path)

        return f'''"""
Tests for {import_path}
"""

import pytest
from unittest.mock import patch, mock_open
import yaml
import json

from {import_path} import *


class Test{self._to_class_name(module_name)}:
    """Test {module_name} configuration."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {{
            "model": {{
                "name": "test_model",
                "hidden_size": 512,
                "num_layers": 6
            }},
            "training": {{
                "batch_size": 32,
                "learning_rate": 0.001
            }}
        }}

    def test_config_loading(self, sample_config, tmp_path):
        """Test configuration loading."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(sample_config))

        # TODO: Test config loading
        pass

'''

    def _generate_default_test(self, module_path: Path, analysis: Dict) -> str:
        """Generate default test template."""
        module_name = module_path.stem
        import_path = self._get_import_path(module_path)

        test_content = f'''"""
Tests for {import_path}
"""

import pytest
from unittest.mock import patch, MagicMock

from {import_path} import *


class Test{self._to_class_name(module_name)}:
    """Test {module_name} module."""

'''

        # Add basic tests for functions and classes
        for func in analysis.get("functions", []):
            if not func["name"].startswith("_"):
                test_content += f'''    def test_{func["name"]}(self):
        """Test {func["name"]} function."""
        # TODO: Implement test
        pass

'''

        for cls in analysis.get("classes", []):
            test_content += f'''    def test_{cls["name"].lower()}_creation(self):
        """Test {cls["name"]} creation."""
        # TODO: Implement test
        pass

'''

        return test_content

    def _get_import_path(self, module_path: Path) -> str:
        """Get import path for a module."""
        rel_path = module_path.relative_to(PROJECT_ROOT)
        import_path = str(rel_path).replace("/", ".").replace(".py", "")
        return import_path

    def _to_class_name(self, name: str) -> str:
        """Convert module name to class name."""
        return "".join(word.capitalize() for word in name.split("_"))


def build_comprehensive_tests():
    """Build comprehensive test suite."""
    print("=== Building Comprehensive Test Suite ===\n")

    builder = TestBuilder()

    # Get all Python modules
    modules = list(MUSIC_GEN_DIR.rglob("*.py"))
    modules = [m for m in modules if "__pycache__" not in str(m)]

    # Filter out __init__.py files with no content
    modules = [
        m for m in modules if not (m.name == "__init__.py" and len(m.read_text().strip()) == 0)
    ]

    # Analyze which modules need tests
    modules_needing_tests = []
    for module in modules:
        test_file = find_test_file(module)
        if not test_file:
            modules_needing_tests.append(module)

    print(f"Found {len(modules_needing_tests)} modules without tests\n")

    # Generate tests for high-priority modules
    generated_count = 0
    max_to_generate = 20  # Limit to avoid overwhelming

    # Sort by complexity (analyze each module)
    print("Analyzing modules...")
    module_complexities = []
    for module in modules_needing_tests:
        analysis = builder.analyze_module(module)
        complexity = analysis.get("complexity", 0)
        module_complexities.append((module, complexity, analysis))

    module_complexities.sort(key=lambda x: x[1], reverse=True)

    print("\nGenerating tests for high-priority modules...\n")

    for module, complexity, analysis in module_complexities[:max_to_generate]:
        if complexity < 3:  # Skip trivial modules
            continue

        rel_path = module.relative_to(MUSIC_GEN_DIR)
        test_type = builder.determine_test_type(module, analysis)

        print(f"Generating {test_type} test for: {rel_path} (complexity: {complexity})")

        # Determine test location
        test_dir = TESTS_DIR / "unit"
        if test_type in ["api"]:
            test_dir = TESTS_DIR / "integration"

        test_file = test_dir / f"test_{module.stem}.py"

        if not test_file.exists():
            test_content = builder.generate_test(module)

            test_dir.mkdir(parents=True, exist_ok=True)
            test_file.write_text(test_content)

            print(f"  ✓ Created: {test_file.relative_to(PROJECT_ROOT)}")
            generated_count += 1

    print(f"\n✓ Generated {generated_count} test files")
    print("\nNext steps:")
    print("1. Review and complete the generated test files")
    print("2. Run: pytest --cov=music_gen --cov-report=term-missing")
    print("3. Focus on implementing TODOs in the generated tests")
    print("4. Add integration tests for complete workflows")


def find_test_file(module_path: Path) -> Path:
    """Find test file for a module."""
    relative_path = module_path.relative_to(MUSIC_GEN_DIR)
    test_name = f"test_{relative_path.stem}.py"

    possible_locations = [
        TESTS_DIR / "unit" / test_name,
        TESTS_DIR / "integration" / test_name,
        TESTS_DIR / test_name,
    ]

    for location in possible_locations:
        if location.exists():
            return location

    return None


if __name__ == "__main__":
    build_comprehensive_tests()
