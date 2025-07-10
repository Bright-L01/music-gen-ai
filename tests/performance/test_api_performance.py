"""
Performance tests for API endpoints.

These tests measure response times, throughput, and resource usage
to ensure the API meets performance requirements.
"""

import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock
from injector import Injector, Module, provider, singleton

from music_gen.api.endpoints.generation_refactored import router as generation_router
from music_gen.core.interfaces.services import GenerationService
from music_gen.core.interfaces.repositories import TaskRepository, AudioRepository
from music_gen.core.container import Container


class FastMockModule(Module):
    """Fast mock module for performance testing."""

    @singleton
    @provider
    def provide_generation_service(self) -> GenerationService:
        """Provide fast mock generation service."""
        service = AsyncMock(spec=GenerationService)

        async def fast_generate(request):
            # Simulate fast generation
            await asyncio.sleep(0.01)  # 10ms simulation
            from music_gen.core.interfaces.services import GenerationResult
            import torch

            return GenerationResult(
                audio=torch.randn(1, 24000),  # 1 second audio
                sample_rate=24000,
                duration=1.0,
                metadata={"prompt": request.prompt},
            )

        service.generate.side_effect = fast_generate
        return service

    @singleton
    @provider
    def provide_task_repository(self) -> TaskRepository:
        """Provide fast in-memory task repository."""
        repo = AsyncMock(spec=TaskRepository)
        repo._tasks = {}

        async def create_task(task_id, data):
            repo._tasks[task_id] = data

        async def get_task(task_id):
            return repo._tasks.get(task_id)

        async def update_task(task_id, updates):
            if task_id in repo._tasks:
                repo._tasks[task_id].update(updates)

        repo.create_task.side_effect = create_task
        repo.get_task.side_effect = get_task
        repo.update_task.side_effect = update_task

        return repo

    @singleton
    @provider
    def provide_audio_repository(self) -> AudioRepository:
        """Provide fast mock audio repository."""
        repo = AsyncMock(spec=AudioRepository)
        repo.save_audio.return_value = "/tmp/fast_audio.wav"
        return repo


@pytest.fixture(scope="module")
def performance_app():
    """Create performance test app."""
    from fastapi import FastAPI

    # Setup fast mocks
    Container.reset()
    container = Injector([FastMockModule()])
    Container._instance = container

    app = FastAPI()
    app.include_router(generation_router, prefix="/api/v1/generate")

    yield app

    Container.reset()


@pytest.fixture(scope="module")
def performance_client(performance_app):
    """Create performance test client."""
    return TestClient(performance_app)


@pytest.mark.performance
class TestAPIPerformance:
    """Performance tests for API endpoints."""

    def test_single_request_latency(self, performance_client):
        """Test latency of single generation request."""
        request_data = {"prompt": "Test music", "duration": 10.0}

        # Warm up
        performance_client.post("/api/v1/generate/", json=request_data)

        # Measure latency
        start_time = time.time()
        response = performance_client.post("/api/v1/generate/", json=request_data)
        end_time = time.time()

        latency = end_time - start_time

        assert response.status_code == 200
        assert latency < 0.1  # Should respond within 100ms

        print(f"Single request latency: {latency:.3f}s")

    def test_concurrent_request_performance(self, performance_client):
        """Test performance under concurrent load."""

        def make_request(i):
            start_time = time.time()
            response = performance_client.post(
                "/api/v1/generate/",
                json={
                    "prompt": f"Test music {i}",
                    "duration": 10.0,
                },
            )
            end_time = time.time()
            return {
                "status_code": response.status_code,
                "latency": end_time - start_time,
                "request_id": i,
            }

        # Test with increasing concurrency
        concurrency_levels = [1, 5, 10, 20]

        for concurrency in concurrency_levels:
            print(f"\nTesting with {concurrency} concurrent requests:")

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                start_time = time.time()
                futures = [executor.submit(make_request, i) for i in range(concurrency)]
                results = [f.result() for f in futures]
                total_time = time.time() - start_time

            # Analyze results
            latencies = [r["latency"] for r in results]
            success_count = sum(1 for r in results if r["status_code"] == 200)

            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            throughput = concurrency / total_time

            print(f"  Success rate: {success_count}/{concurrency}")
            print(f"  Average latency: {avg_latency:.3f}s")
            print(f"  95th percentile latency: {p95_latency:.3f}s")
            print(f"  Throughput: {throughput:.1f} req/s")

            # Performance assertions
            assert success_count == concurrency  # All requests should succeed
            assert avg_latency < 0.5  # Average latency under 500ms
            assert p95_latency < 1.0  # 95th percentile under 1s

    def test_batch_request_performance(self, performance_client):
        """Test batch request performance."""
        batch_sizes = [1, 2, 5]

        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")

            batch_request = {
                "requests": [{"prompt": f"Music {i}", "duration": 10.0} for i in range(batch_size)]
            }

            start_time = time.time()
            response = performance_client.post("/api/v1/generate/batch", json=batch_request)
            end_time = time.time()

            latency = end_time - start_time
            per_request_latency = latency / batch_size

            assert response.status_code == 200
            print(f"  Total latency: {latency:.3f}s")
            print(f"  Per-request latency: {per_request_latency:.3f}s")

            # Batch should be more efficient than individual requests
            assert per_request_latency < 0.1

    def test_status_check_performance(self, performance_client):
        """Test performance of status checking."""
        # Create some tasks
        task_ids = []
        for i in range(10):
            response = performance_client.post(
                "/api/v1/generate/",
                json={
                    "prompt": f"Test {i}",
                    "duration": 10.0,
                },
            )
            task_ids.append(response.json()["task_id"])

        # Measure status check performance
        latencies = []
        for task_id in task_ids:
            start_time = time.time()
            response = performance_client.get(f"/api/v1/generate/{task_id}")
            end_time = time.time()

            latencies.append(end_time - start_time)
            assert response.status_code == 200

        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)

        print(f"\nStatus check performance:")
        print(f"  Average latency: {avg_latency:.3f}s")
        print(f"  Max latency: {max_latency:.3f}s")

        # Status checks should be very fast
        assert avg_latency < 0.05  # Under 50ms
        assert max_latency < 0.1  # Under 100ms

    def test_sustained_load_performance(self, performance_client):
        """Test performance under sustained load."""
        duration_seconds = 10
        requests_per_second = 5
        total_requests = duration_seconds * requests_per_second

        print(f"\nSustained load test: {requests_per_second} req/s for {duration_seconds}s")

        def make_sustained_request(i):
            # Spread requests evenly over time
            delay = i / requests_per_second
            time.sleep(delay)

            start_time = time.time()
            response = performance_client.post(
                "/api/v1/generate/",
                json={
                    "prompt": f"Sustained test {i}",
                    "duration": 5.0,
                },
            )
            end_time = time.time()

            return {
                "status_code": response.status_code,
                "latency": end_time - start_time,
                "timestamp": start_time,
            }

        # Run sustained load
        test_start = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_sustained_request, i) for i in range(total_requests)]
            results = [f.result() for f in futures]
        test_duration = time.time() - test_start

        # Analyze results
        success_count = sum(1 for r in results if r["status_code"] == 200)
        latencies = [r["latency"] for r in results]

        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        actual_throughput = total_requests / test_duration

        print(f"  Success rate: {success_count}/{total_requests}")
        print(f"  Average latency: {avg_latency:.3f}s")
        print(f"  Max latency: {max_latency:.3f}s")
        print(f"  Actual throughput: {actual_throughput:.1f} req/s")

        # Performance assertions
        assert success_count >= total_requests * 0.95  # 95% success rate
        assert avg_latency < 1.0  # Average under 1s
        assert actual_throughput >= requests_per_second * 0.8  # 80% of target throughput

    def test_memory_usage_stability(self, performance_client):
        """Test that memory usage doesn't grow significantly under load."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"\nMemory usage test (initial: {initial_memory:.1f} MB)")

        # Make many requests
        for batch in range(5):
            requests = []
            for i in range(20):
                response = performance_client.post(
                    "/api/v1/generate/",
                    json={
                        "prompt": f"Memory test {batch}-{i}",
                        "duration": 5.0,
                    },
                )
                requests.append(response)

            # Check memory after each batch
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory

            print(f"  Batch {batch + 1}: {current_memory:.1f} MB (+{memory_growth:.1f} MB)")

            # Memory growth should be reasonable
            assert memory_growth < 100  # Less than 100MB growth

            # All requests should succeed
            assert all(r.status_code == 200 for r in requests)


@pytest.mark.performance
class TestRepositoryPerformance:
    """Performance tests for repository implementations."""

    def test_task_repository_performance(self):
        """Test task repository performance."""
        from music_gen.infrastructure.repositories import InMemoryTaskRepository

        repo = InMemoryTaskRepository(max_tasks=1000)

        async def run_task_performance_test():
            # Test creation performance
            start_time = time.time()
            for i in range(100):
                await repo.create_task(f"task_{i}", {"status": "pending", "data": i})
            creation_time = time.time() - start_time

            # Test retrieval performance
            start_time = time.time()
            for i in range(100):
                task = await repo.get_task(f"task_{i}")
                assert task is not None
            retrieval_time = time.time() - start_time

            # Test update performance
            start_time = time.time()
            for i in range(100):
                await repo.update_task(f"task_{i}", {"status": "completed"})
            update_time = time.time() - start_time

            # Test list performance
            start_time = time.time()
            tasks = await repo.list_tasks()
            list_time = time.time() - start_time

            print(f"\nTask repository performance (100 operations):")
            print(f"  Creation: {creation_time:.3f}s ({creation_time*10:.1f}ms/op)")
            print(f"  Retrieval: {retrieval_time:.3f}s ({retrieval_time*10:.1f}ms/op)")
            print(f"  Update: {update_time:.3f}s ({update_time*10:.1f}ms/op)")
            print(f"  List: {list_time:.3f}s")

            # Performance assertions
            assert creation_time < 0.1  # Under 100ms for 100 creates
            assert retrieval_time < 0.05  # Under 50ms for 100 retrievals
            assert update_time < 0.1  # Under 100ms for 100 updates
            assert list_time < 0.01  # Under 10ms for listing

        asyncio.run(run_task_performance_test())

    def test_model_repository_performance(self):
        """Test model repository performance."""
        import tempfile
        from pathlib import Path
        from music_gen.infrastructure.repositories import FileSystemModelRepository

        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = FileSystemModelRepository(Path(tmp_dir))

            async def run_model_performance_test():
                import torch

                # Create test model states
                model_states = {}
                for i in range(10):
                    model_states[f"model_{i}"] = {
                        "state_dict": {"weight": torch.randn(100, 100)},
                        "config": {"hidden_size": 768, "index": i},
                    }

                # Test save performance
                start_time = time.time()
                for model_id, state in model_states.items():
                    await repo.save_model(model_id, state)
                save_time = time.time() - start_time

                # Test load performance
                start_time = time.time()
                for model_id in model_states:
                    loaded = await repo.load_model(model_id)
                    assert loaded is not None
                load_time = time.time() - start_time

                # Test list performance
                start_time = time.time()
                models = await repo.list_models()
                list_time = time.time() - start_time

                print(f"\nModel repository performance (10 models):")
                print(f"  Save: {save_time:.3f}s ({save_time*100:.1f}ms/model)")
                print(f"  Load: {load_time:.3f}s ({load_time*100:.1f}ms/model)")
                print(f"  List: {list_time:.3f}s")

                # Performance assertions
                assert save_time < 2.0  # Under 2s for 10 saves
                assert load_time < 1.0  # Under 1s for 10 loads
                assert list_time < 0.1  # Under 100ms for listing

            asyncio.run(run_model_performance_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
