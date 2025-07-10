#!/usr/bin/env python3
"""
Performance Profiler for Music Gen AI API

This script profiles API endpoints to identify performance bottlenecks
and provides recommendations for optimization.
"""

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import psutil
import numpy as np
from datetime import datetime


@dataclass
class ProfileResult:
    """Single profiling result for an API call."""

    endpoint: str
    method: str
    status_code: int
    response_time: float
    memory_before: float
    memory_after: float
    memory_delta: float
    payload_size: int
    response_size: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EndpointStats:
    """Aggregated statistics for an endpoint."""

    endpoint: str
    method: str
    total_calls: int = 0
    avg_response_time: float = 0.0
    min_response_time: float = float("inf")
    max_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    total_memory_delta: float = 0.0
    avg_memory_delta: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    response_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class APIProfiler:
    """Profiles API endpoints for performance analysis."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.results: List[ProfileResult] = []
        self.stats: Dict[str, EndpointStats] = defaultdict(lambda: EndpointStats("", ""))

    async def profile_endpoint(
        self,
        endpoint: str,
        method: str = "GET",
        payload: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        num_requests: int = 10,
        concurrent: int = 1,
    ) -> List[ProfileResult]:
        """Profile a single endpoint with multiple requests."""

        print(f"Profiling {method} {endpoint} ({num_requests} requests, {concurrent} concurrent)")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrent)

        async def make_request(session: aiohttp.ClientSession) -> ProfileResult:
            async with semaphore:
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                start_time = time.time()

                try:
                    url = f"{self.base_url}{endpoint}"

                    if method.upper() == "GET":
                        async with session.get(url, headers=headers) as response:
                            response_data = await response.read()
                            status_code = response.status
                            response_size = len(response_data)

                    elif method.upper() == "POST":
                        json_payload = json.dumps(payload) if payload else "{}"
                        payload_size = len(json_payload.encode())

                        async with session.post(url, json=payload, headers=headers) as response:
                            response_data = await response.read()
                            status_code = response.status
                            response_size = len(response_data)

                    else:
                        raise ValueError(f"Unsupported method: {method}")

                except Exception as e:
                    status_code = 0
                    response_size = 0
                    payload_size = len(json.dumps(payload).encode()) if payload else 0
                    print(f"Error in request: {e}")

                end_time = time.time()
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                return ProfileResult(
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    response_time=end_time - start_time,
                    memory_before=memory_before,
                    memory_after=memory_after,
                    memory_delta=memory_after - memory_before,
                    payload_size=payload_size if payload else 0,
                    response_size=response_size,
                )

        # Execute requests
        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session) for _ in range(num_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, ProfileResult)]
        self.results.extend(valid_results)

        # Update statistics
        self._update_stats(valid_results)

        return valid_results

    def _update_stats(self, results: List[ProfileResult]):
        """Update endpoint statistics with new results."""

        for result in results:
            key = f"{result.method}:{result.endpoint}"
            stats = self.stats[key]

            if not stats.endpoint:
                stats.endpoint = result.endpoint
                stats.method = result.method

            stats.total_calls += 1
            stats.response_times.append(result.response_time)
            stats.total_memory_delta += result.memory_delta

            # Update min/max
            stats.min_response_time = min(stats.min_response_time, result.response_time)
            stats.max_response_time = max(stats.max_response_time, result.response_time)

            # Track errors
            if result.status_code >= 400:
                stats.errors.append(f"HTTP {result.status_code}")

        # Calculate derived statistics
        for stats in self.stats.values():
            if stats.response_times:
                stats.avg_response_time = np.mean(stats.response_times)
                stats.p95_response_time = np.percentile(stats.response_times, 95)
                stats.p99_response_time = np.percentile(stats.response_times, 99)
                stats.avg_memory_delta = stats.total_memory_delta / len(stats.response_times)
                stats.error_rate = len(stats.errors) / stats.total_calls * 100

                # Calculate throughput (requests per second)
                if stats.response_times:
                    total_time = sum(stats.response_times)
                    stats.throughput = stats.total_calls / total_time if total_time > 0 else 0

    async def profile_critical_paths(self):
        """Profile all critical API endpoints."""

        print("Starting comprehensive API profiling...")

        # Health check endpoints (lightweight)
        await self.profile_endpoint("/health", "GET", num_requests=20, concurrent=5)
        await self.profile_endpoint("/health/status", "GET", num_requests=20, concurrent=5)

        # Authentication endpoints
        await self.profile_endpoint("/api/auth/csrf-token", "GET", num_requests=10, concurrent=2)

        # Generation endpoints (compute-intensive)
        generation_payload = {
            "prompt": "A short upbeat jazz melody",
            "duration": 5,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.9,
        }
        await self.profile_endpoint(
            "/api/v1/generate/text",
            "POST",
            payload=generation_payload,
            num_requests=5,
            concurrent=1,
        )

        # Model management endpoints
        await self.profile_endpoint("/api/v1/models", "GET", num_requests=10, concurrent=3)
        await self.profile_endpoint("/api/v1/models/status", "GET", num_requests=10, concurrent=3)

        # Resource monitoring endpoints
        await self.profile_endpoint("/api/v1/resources", "GET", num_requests=15, concurrent=3)
        await self.profile_endpoint("/api/v1/task-status", "GET", num_requests=15, concurrent=3)

        # Static file serving
        await self.profile_endpoint("/", "GET", num_requests=10, concurrent=5)

        print("Profiling completed!")

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance results and identify bottlenecks."""

        analysis = {
            "summary": {
                "total_requests": len(self.results),
                "total_endpoints": len(self.stats),
                "overall_avg_response_time": np.mean([r.response_time for r in self.results]),
                "overall_error_rate": len([r for r in self.results if r.status_code >= 400])
                / len(self.results)
                * 100,
                "total_memory_usage": sum([r.memory_delta for r in self.results]),
            },
            "bottlenecks": [],
            "recommendations": [],
            "endpoint_rankings": {"slowest": [], "memory_intensive": [], "error_prone": []},
        }

        # Identify bottlenecks
        for key, stats in self.stats.items():
            # Slow endpoints (>1 second average)
            if stats.avg_response_time > 1.0:
                analysis["bottlenecks"].append(
                    {
                        "type": "slow_response",
                        "endpoint": f"{stats.method} {stats.endpoint}",
                        "avg_response_time": stats.avg_response_time,
                        "p99_response_time": stats.p99_response_time,
                    }
                )

            # Memory-intensive endpoints (>10MB average)
            if stats.avg_memory_delta > 10:
                analysis["bottlenecks"].append(
                    {
                        "type": "high_memory_usage",
                        "endpoint": f"{stats.method} {stats.endpoint}",
                        "avg_memory_delta": stats.avg_memory_delta,
                        "total_memory_delta": stats.total_memory_delta,
                    }
                )

            # Error-prone endpoints (>5% error rate)
            if stats.error_rate > 5:
                analysis["bottlenecks"].append(
                    {
                        "type": "high_error_rate",
                        "endpoint": f"{stats.method} {stats.endpoint}",
                        "error_rate": stats.error_rate,
                        "errors": stats.errors,
                    }
                )

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations()

        # Rank endpoints
        sorted_by_time = sorted(
            self.stats.values(), key=lambda s: s.avg_response_time, reverse=True
        )
        sorted_by_memory = sorted(
            self.stats.values(), key=lambda s: s.avg_memory_delta, reverse=True
        )
        sorted_by_errors = sorted(self.stats.values(), key=lambda s: s.error_rate, reverse=True)

        analysis["endpoint_rankings"]["slowest"] = [
            {
                "endpoint": f"{s.method} {s.endpoint}",
                "avg_response_time": s.avg_response_time,
                "p99_response_time": s.p99_response_time,
            }
            for s in sorted_by_time[:5]
        ]

        analysis["endpoint_rankings"]["memory_intensive"] = [
            {
                "endpoint": f"{s.method} {s.endpoint}",
                "avg_memory_delta": s.avg_memory_delta,
                "total_memory_delta": s.total_memory_delta,
            }
            for s in sorted_by_memory[:5]
        ]

        analysis["endpoint_rankings"]["error_prone"] = [
            {
                "endpoint": f"{s.method} {s.endpoint}",
                "error_rate": s.error_rate,
                "total_errors": len(s.errors),
            }
            for s in sorted_by_errors[:5]
            if s.error_rate > 0
        ]

        return analysis

    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on analysis."""

        recommendations = []

        # Check for slow endpoints
        slow_endpoints = [s for s in self.stats.values() if s.avg_response_time > 1.0]
        if slow_endpoints:
            recommendations.append(
                {
                    "category": "Response Time",
                    "priority": "High",
                    "recommendation": "Implement caching for slow endpoints",
                    "endpoints": [f"{s.method} {s.endpoint}" for s in slow_endpoints],
                    "implementation": "Add Redis caching middleware, implement database query optimization",
                }
            )

        # Check for memory-intensive endpoints
        memory_endpoints = [s for s in self.stats.values() if s.avg_memory_delta > 10]
        if memory_endpoints:
            recommendations.append(
                {
                    "category": "Memory Usage",
                    "priority": "High",
                    "recommendation": "Optimize memory usage for intensive endpoints",
                    "endpoints": [f"{s.method} {s.endpoint}" for s in memory_endpoints],
                    "implementation": "Implement streaming responses, lazy loading, memory profiling",
                }
            )

        # Check for generation endpoints
        gen_endpoints = [s for s in self.stats.values() if "generate" in s.endpoint.lower()]
        if gen_endpoints:
            recommendations.append(
                {
                    "category": "AI Model Performance",
                    "priority": "High",
                    "recommendation": "Optimize AI model inference",
                    "endpoints": [f"{s.method} {s.endpoint}" for s in gen_endpoints],
                    "implementation": "Model quantization, batch processing, async task queues",
                }
            )

        # Database optimization recommendations
        recommendations.append(
            {
                "category": "Database",
                "priority": "Medium",
                "recommendation": "Implement database query optimization",
                "implementation": "Add indexes, optimize N+1 queries, implement connection pooling",
            }
        )

        # File handling optimization
        recommendations.append(
            {
                "category": "File Handling",
                "priority": "Medium",
                "recommendation": "Optimize audio file processing",
                "implementation": "Streaming file uploads, async processing, temporary file cleanup",
            }
        )

        return recommendations

    def save_report(self, output_path: Path):
        """Save detailed performance report."""

        analysis = self.analyze_performance()

        # Prepare detailed data
        report = {
            "generated_at": datetime.now().isoformat(),
            "analysis": analysis,
            "detailed_stats": {
                key: {
                    "endpoint": stats.endpoint,
                    "method": stats.method,
                    "total_calls": stats.total_calls,
                    "avg_response_time": stats.avg_response_time,
                    "min_response_time": stats.min_response_time,
                    "max_response_time": stats.max_response_time,
                    "p95_response_time": stats.p95_response_time,
                    "p99_response_time": stats.p99_response_time,
                    "avg_memory_delta": stats.avg_memory_delta,
                    "error_rate": stats.error_rate,
                    "throughput": stats.throughput,
                }
                for key, stats in self.stats.items()
            },
            "raw_results": [
                {
                    "endpoint": r.endpoint,
                    "method": r.method,
                    "status_code": r.status_code,
                    "response_time": r.response_time,
                    "memory_delta": r.memory_delta,
                    "payload_size": r.payload_size,
                    "response_size": r.response_size,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self.results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Performance report saved to {output_path}")

    def print_summary(self):
        """Print a summary of performance analysis."""

        analysis = self.analyze_performance()

        print("\n" + "=" * 60)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\nTotal Requests: {analysis['summary']['total_requests']}")
        print(f"Total Endpoints: {analysis['summary']['total_endpoints']}")
        print(f"Overall Avg Response Time: {analysis['summary']['overall_avg_response_time']:.3f}s")
        print(f"Overall Error Rate: {analysis['summary']['overall_error_rate']:.1f}%")
        print(f"Total Memory Usage: {analysis['summary']['total_memory_usage']:.1f}MB")

        print("\n" + "-" * 40)
        print("PERFORMANCE BOTTLENECKS")
        print("-" * 40)

        if analysis["bottlenecks"]:
            for bottleneck in analysis["bottlenecks"]:
                print(f"\n{bottleneck['type'].upper()}: {bottleneck['endpoint']}")
                for key, value in bottleneck.items():
                    if key not in ["type", "endpoint"]:
                        print(f"  {key}: {value}")
        else:
            print("No significant bottlenecks detected!")

        print("\n" + "-" * 40)
        print("TOP OPTIMIZATION RECOMMENDATIONS")
        print("-" * 40)

        for i, rec in enumerate(analysis["recommendations"][:3], 1):
            print(f"\n{i}. {rec['recommendation']} (Priority: {rec['priority']})")
            print(f"   Implementation: {rec['implementation']}")
            if "endpoints" in rec:
                print(f"   Affected endpoints: {', '.join(rec['endpoints'])}")


async def main():
    """Main profiling function."""

    # Create profiler
    profiler = APIProfiler()

    try:
        # Run profiling
        await profiler.profile_critical_paths()

        # Analyze and report
        profiler.print_summary()

        # Save detailed report
        output_path = Path("performance_report.json")
        profiler.save_report(output_path)

    except Exception as e:
        print(f"Profiling failed: {e}")
        print("Make sure the API server is running on http://localhost:8000")


if __name__ == "__main__":
    asyncio.run(main())
