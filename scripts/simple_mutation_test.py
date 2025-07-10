#!/usr/bin/env python3
"""
Simple Mutation Testing for ModelManager

Tests specific mutations that should be caught by good tests.
"""

import sys
import time
import gc
from pathlib import Path
from unittest.mock import Mock, patch


# Mock dependencies
sys.modules["torch"] = Mock()
sys.modules["torch.cuda"] = Mock()
sys.modules["psutil"] = Mock()

torch_mock = sys.modules["torch"]
torch_mock.cuda.is_available.return_value = True
torch_mock.cuda.memory_allocated.return_value = 1024**3
torch_mock.cuda.empty_cache = Mock()
torch_mock.cuda.OutOfMemoryError = Exception

# Mock fast generator
fast_generator_mock = Mock()
sys.modules["music_gen.optimization.fast_generator"] = fast_generator_mock

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Read and exec the model_manager.py file directly
model_manager_path = project_root / "music_gen" / "core" / "model_manager.py"

model_manager_namespace = {
    "__name__": "music_gen.core.model_manager",
    "__file__": str(model_manager_path),
    "gc": gc,
    "logging": Mock(),
    "Path": Path,
    "torch": torch_mock,
}

with open(model_manager_path, "r") as f:
    model_manager_code = f.read()

exec(model_manager_code, model_manager_namespace)
ModelManager = model_manager_namespace["ModelManager"]


class MutationTest:
    """A single mutation test."""

    def __init__(self, name: str, mutated_code: str, expected_failure: str):
        self.name = name
        self.mutated_code = mutated_code
        self.expected_failure = expected_failure
        self.caught = False
        self.error = None


def create_mutation_tests():
    """Create specific mutation tests."""

    original_code = model_manager_code

    mutations = [
        MutationTest(
            name="Break Singleton Pattern",
            mutated_code=original_code.replace(
                "if cls._instance is None:", "if False:  # Always create new instance"
            ),
            expected_failure="Singleton behavior broken - should create same instance",
        ),
        MutationTest(
            name="Disable Caching",
            mutated_code=original_code.replace(
                "if model_key in self._models:", "if False:  # Never use cache"
            ),
            expected_failure="Caching disabled - should reuse cached models",
        ),
        MutationTest(
            name="Skip Memory Cleanup",
            mutated_code=original_code.replace("gc.collect()", "pass  # Skip cleanup"),
            expected_failure="Memory cleanup skipped",
        ),
        MutationTest(
            name="Wrong Model Key",
            mutated_code=original_code.replace(
                'model_key = f"{model_name}_{model_type}_{device}"',
                'model_key = f"{model_name}_wrong_{device}"',
            ),
            expected_failure="Model key generation broken",
        ),
        MutationTest(
            name="Always Return True for Unload",
            mutated_code=original_code.replace(
                "return unloaded", "return True  # Always return True"
            ),
            expected_failure="Unload should return False when model doesn't exist",
        ),
        MutationTest(
            name="Ignore Device Parameter",
            mutated_code=original_code.replace(
                "if device is None:", "if True:  # Always use default device"
            ),
            expected_failure="Device parameter ignored",
        ),
    ]

    return mutations


def test_mutation(mutation: MutationTest):
    """Test if a mutation is caught by our tests."""

    # Create new namespace with mutated code
    mutated_namespace = {
        "__name__": "music_gen.core.model_manager",
        "__file__": str(model_manager_path),
        "gc": gc,
        "logging": Mock(),
        "Path": Path,
        "torch": torch_mock,
    }

    try:
        # Execute mutated code
        exec(mutation.mutated_code, mutated_namespace)
        MutatedModelManager = mutated_namespace["ModelManager"]

        # Run behavior tests
        caught = run_behavior_tests(MutatedModelManager)
        mutation.caught = caught

    except Exception as e:
        # If code fails to execute, consider it caught
        mutation.caught = True
        mutation.error = str(e)


def run_behavior_tests(ModelManagerClass):
    """Run focused behavior tests that should catch mutations."""

    try:
        # Test 1: Singleton behavior
        ModelManagerClass._instance = None
        manager1 = ModelManagerClass()
        manager2 = ModelManagerClass()

        if manager1 is not manager2:
            return True  # Mutation caught - singleton broken

        # Test 2: Caching behavior
        ModelManagerClass._instance = None
        manager = ModelManagerClass()

        # Mock FastMusicGenerator
        mock_generator = Mock()
        mock_model = Mock()
        mock_generator.return_value = mock_model
        mutated_namespace["FastMusicGenerator"] = mock_generator

        # Load same model twice
        model1 = manager.get_model("test", "optimized")
        mock_generator.reset_mock()
        model2 = manager.get_model("test", "optimized")

        # Should not call generator second time (caching)
        if mock_generator.called:
            return True  # Mutation caught - caching broken

        if model1 is not model2:
            return True  # Mutation caught - should return same instance

        # Test 3: Device handling
        model_cpu = manager.get_model("test-cpu", "optimized", device="cpu")
        expected_key_cpu = "test-cpu_optimized_cpu"

        if expected_key_cpu not in manager._models:
            return True  # Mutation caught - device parameter ignored

        # Test 4: Unload behavior
        result = manager.unload_model("nonexistent")
        if result is True:
            return True  # Mutation caught - should return False for nonexistent

        # Test 5: Model key generation
        manager.get_model("key-test", "optimized")
        expected_key = "key-test_optimized_cuda"

        if expected_key not in manager._models:
            return True  # Mutation caught - wrong key generated

        # Test 6: Memory cleanup
        initial_models = dict(manager._models)
        manager.clear_cache()

        if len(manager._models) > 0:
            return True  # Mutation caught - cache not cleared

        return False  # Mutation survived - tests didn't catch it

    except Exception:
        return True  # Any exception means mutation was caught


def main():
    """Run mutation testing."""
    print("🧬 Running Mutation Testing")
    print("=" * 50)

    mutations = create_mutation_tests()

    caught_count = 0
    total_count = len(mutations)

    for mutation in mutations:
        print(f"\n🔬 Testing: {mutation.name}")
        test_mutation(mutation)

        if mutation.caught:
            print(f"   ✅ CAUGHT - Tests detected this mutation")
            caught_count += 1
        else:
            print(f"   ❌ SURVIVED - Tests did not catch this mutation")
            print(f"   Expected: {mutation.expected_failure}")

        if mutation.error:
            print(f"   Error: {mutation.error}")

    # Calculate mutation score
    mutation_score = (caught_count / total_count) * 100 if total_count > 0 else 0

    print(f"\n📊 MUTATION TESTING RESULTS")
    print("=" * 50)
    print(f"Total Mutations: {total_count}")
    print(f"Caught: {caught_count}")
    print(f"Survived: {total_count - caught_count}")
    print(f"Mutation Score: {mutation_score:.1f}%")

    if mutation_score >= 90:
        print("\n🎉 EXCELLENT - Tests catch almost all bugs!")
        status = "excellent"
    elif mutation_score >= 70:
        print("\n✅ GOOD - Tests catch most bugs")
        status = "good"
    elif mutation_score >= 50:
        print("\n⚠️  FAIR - Many bugs could slip through")
        status = "fair"
    else:
        print("\n❌ POOR - Most bugs aren't caught by tests")
        status = "poor"

    # Show survived mutations
    survived = [m for m in mutations if not m.caught]
    if survived:
        print(f"\n🔍 SURVIVED MUTATIONS ({len(survived)} bugs not caught):")
        for mutation in survived:
            print(f"  • {mutation.name}")
            print(f"    Impact: {mutation.expected_failure}")

    print(f"\n📋 RECOMMENDATIONS:")
    if mutation_score < 70:
        print("  1. Add more behavior-focused tests")
        print("  2. Test edge cases and error conditions")
        print("  3. Verify actual outcomes, not just state")

    if survived:
        print("  4. Add specific tests for survived mutations")
        print("  5. Use integration tests for complex behaviors")

    print("  6. Consider property-based testing")
    print("  7. Add performance regression tests")

    return mutation_score >= 70


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
