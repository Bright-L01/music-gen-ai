#!/usr/bin/env python3
"""
Comprehensive load testing suite runner for Music Gen AI.

Executes all load tests in sequence and generates a comprehensive
performance analysis report.
"""

import asyncio
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=1800  # 30 minute timeout
        )

        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
            return True
        else:
            print(f"❌ FAILED: {description}")
            print("Error:", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱️ TIMEOUT: {description} exceeded 30 minutes")
        return False
    except Exception as e:
        print(f"💥 ERROR: {description} - {str(e)}")
        return False


def check_prerequisites():
    """Check if prerequisites are installed."""
    print("🔍 Checking prerequisites...")

    prerequisites = [
        ("python", "python --version"),
        ("locust", "locust --version"),
        ("psutil", "python -c 'import psutil; print(psutil.__version__)'"),
        ("websockets", "python -c 'import websockets; print(websockets.__version__)'"),
        ("aiohttp", "python -c 'import aiohttp; print(aiohttp.__version__)'"),
    ]

    all_good = True
    for name, command in prerequisites:
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {name}: {result.stdout.strip()}")
            else:
                print(f"❌ {name}: Not found or error")
                all_good = False
        except Exception as e:
            print(f"❌ {name}: Error checking - {str(e)}")
            all_good = False

    return all_good


def run_locust_tests():
    """Run Locust load tests."""
    print("\n🚀 Starting Locust Load Tests...")

    # Test different scenarios
    test_scenarios = [
        {
            "name": "Basic Load Test",
            "command": "locust -f locustfile.py --host=http://localhost:8000 --headless -u 50 -r 5 -t 300s --html=basic_load_report.html",
            "description": "50 users, 5 minute duration",
        },
        {
            "name": "WebSocket Focus Test",
            "command": "locust -f locustfile.py --host=http://localhost:8000 --headless -u 20 -r 2 -t 120s --tags websocket",
            "description": "WebSocket streaming focus",
        },
        {
            "name": "Database Stress Test",
            "command": "locust -f locustfile.py --host=http://localhost:8000 --headless -u 30 -r 3 -t 180s --tags database",
            "description": "Database operations focus",
        },
    ]

    results = []
    for scenario in test_scenarios:
        success = run_command(scenario["command"], f"Locust: {scenario['name']}")
        results.append(
            {"test": scenario["name"], "success": success, "description": scenario["description"]}
        )

        if not success:
            print(f"⚠️ Continuing despite failure in {scenario['name']}")

        time.sleep(10)  # Brief pause between tests

    return results


def run_websocket_tests():
    """Run WebSocket load tests."""
    print("\n🌐 Starting WebSocket Load Tests...")
    return run_command("python websocket_load_test.py", "WebSocket Load Testing")


def run_database_tests():
    """Run database pool tests."""
    print("\n🗄️ Starting Database Pool Tests...")
    return run_command("python database_pool_test.py", "Database Connection Pool Testing")


def run_redis_tests():
    """Run Redis pool tests."""
    print("\n🔴 Starting Redis Pool Tests...")
    return run_command("python redis_pool_test.py", "Redis Connection Pool and Caching Testing")


def run_bottleneck_analysis():
    """Run bottleneck analysis."""
    print("\n🔍 Starting Bottleneck Analysis...")
    return run_command("python bottleneck_analyzer.py", "Performance Bottleneck Analysis")


def generate_baseline_report():
    """Generate baseline metrics report."""
    print("\n📊 Generating Baseline Metrics Report...")
    return run_command("python baseline_metrics_report.py", "Baseline Performance Metrics Report")


def generate_summary_report(test_results: Dict[str, Any]):
    """Generate a summary report of all tests."""
    print("\n📋 Generating Summary Report...")

    # Load all generated reports
    reports = {}
    report_files = [
        ("performance_report.json", "locust"),
        ("websocket_load_report.json", "websocket"),
        ("database_pool_report.json", "database"),
        ("redis_pool_report.json", "redis"),
        ("bottleneck_analysis_report.json", "bottlenecks"),
        ("baseline_metrics_report.json", "baseline"),
    ]

    for filename, key in report_files:
        file_path = Path(filename)
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    reports[key] = json.load(f)
                print(f"✅ Loaded {filename}")
            except Exception as e:
                print(f"⚠️ Could not load {filename}: {e}")
                reports[key] = None
        else:
            print(f"⚠️ {filename} not found")
            reports[key] = None

    # Create summary
    summary = {
        "test_execution_summary": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_tests_run": len(test_results),
            "successful_tests": sum(1 for result in test_results.values() if result),
            "failed_tests": sum(1 for result in test_results.values() if not result),
            "test_results": test_results,
        },
        "performance_highlights": {},
        "key_findings": [],
        "recommendations": [],
        "reports_generated": {k: v is not None for k, v in reports.items()},
    }

    # Extract key metrics from reports
    if reports["locust"]:
        locust_summary = reports["locust"].get("test_summary", {})
        summary["performance_highlights"]["api"] = {
            "avg_response_time_ms": locust_summary.get("avg_response_time_ms", 0),
            "throughput_rps": locust_summary.get("throughput_rps", 0),
            "error_rate_percent": locust_summary.get("error_rate_percent", 0),
            "max_concurrent_requests": locust_summary.get("max_concurrent_requests", 0),
        }

    if reports["websocket"]:
        ws_summary = (
            reports["websocket"].get("websocket_load_test_report", {}).get("test_summary", {})
        )
        summary["performance_highlights"]["websocket"] = {
            "success_rate_percent": ws_summary.get("success_rate_percent", 0),
            "max_concurrent_connections": ws_summary.get("max_concurrent_connections", 0),
        }

    if reports["database"]:
        db_summary = reports["database"].get("database_pool_load_test", {}).get("test_summary", {})
        summary["performance_highlights"]["database"] = {
            "max_queries_per_second": db_summary.get("max_queries_per_second", 0),
            "avg_response_time_seconds": db_summary.get("avg_response_time_seconds", 0),
        }

    if reports["redis"]:
        redis_summary = reports["redis"].get("redis_pool_load_test", {}).get("test_summary", {})
        summary["performance_highlights"]["redis"] = {
            "max_commands_per_second": redis_summary.get("max_commands_per_second", 0),
            "avg_cache_hit_rate_percent": redis_summary.get("avg_cache_hit_rate_percent", 0),
        }

    # Extract key findings from bottleneck analysis
    if reports["bottlenecks"]:
        bottlenecks = reports["bottlenecks"]
        summary["key_findings"] = [
            f"Total bottlenecks identified: {bottlenecks.get('analysis_metadata', {}).get('total_bottlenecks_identified', 0)}",
            f"Critical issues: {len(bottlenecks.get('critical_bottlenecks', []))}",
            f"High priority issues: {len(bottlenecks.get('high_priority_bottlenecks', []))}",
        ]

        # Add executive summary
        if "executive_summary" in bottlenecks:
            summary["key_findings"].append(bottlenecks["executive_summary"])

    # Extract recommendations from baseline report
    if reports["baseline"]:
        baseline = reports["baseline"]
        if "production_recommendations" in baseline:
            summary["recommendations"] = baseline["production_recommendations"][:5]  # Top 5

        # Add performance score
        if "executive_summary" in baseline:
            exec_summary = baseline["executive_summary"]
            summary["performance_highlights"]["overall_score"] = exec_summary.get(
                "overall_performance_score", 0
            )
            summary["key_findings"].append(
                f"Production readiness: {exec_summary.get('readiness_assessment', 'Unknown')}"
            )

    # Save summary report
    with open("load_test_summary_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Summary report saved to: load_test_summary_report.json")
    return summary


def print_final_summary(test_results: Dict[str, Any], summary: Dict[str, Any]):
    """Print final test summary to console."""
    print("\n" + "=" * 80)
    print("🎯 LOAD TESTING COMPLETE - FINAL SUMMARY")
    print("=" * 80)

    # Test execution summary
    exec_summary = summary["test_execution_summary"]
    print(f"\n📊 Test Execution Results:")
    print(f"   • Total Tests: {exec_summary['total_tests_run']}")
    print(f"   • Successful: {exec_summary['successful_tests']} ✅")
    print(f"   • Failed: {exec_summary['failed_tests']} ❌")
    print(f"   • Timestamp: {exec_summary['timestamp']}")

    # Individual test results
    print(f"\n🔍 Individual Test Results:")
    for test_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   • {test_name}: {status}")

    # Performance highlights
    if "performance_highlights" in summary:
        perf = summary["performance_highlights"]
        print(f"\n⚡ Performance Highlights:")

        if "overall_score" in perf:
            print(f"   • Overall Performance Score: {perf['overall_score']:.1f}/100")

        if "api" in perf:
            api = perf["api"]
            print(f"   • API Avg Response Time: {api['avg_response_time_ms']:.1f}ms")
            print(f"   • API Throughput: {api['throughput_rps']:.1f} RPS")
            print(f"   • API Error Rate: {api['error_rate_percent']:.1f}%")

        if "websocket" in perf:
            ws = perf["websocket"]
            print(f"   • WebSocket Success Rate: {ws['success_rate_percent']:.1f}%")
            print(f"   • Max Concurrent WS: {ws['max_concurrent_connections']}")

        if "database" in perf:
            db = perf["database"]
            print(f"   • Database Max QPS: {db['max_queries_per_second']:.1f}")
            print(f"   • Database Avg Latency: {db['avg_response_time_seconds']*1000:.1f}ms")

        if "redis" in perf:
            redis = perf["redis"]
            print(f"   • Redis Max CPS: {redis['max_commands_per_second']:.1f}")
            print(f"   • Redis Cache Hit Rate: {redis['avg_cache_hit_rate_percent']:.1f}%")

    # Key findings
    if summary.get("key_findings"):
        print(f"\n🔍 Key Findings:")
        for finding in summary["key_findings"]:
            print(f"   • {finding}")

    # Top recommendations
    if summary.get("recommendations"):
        print(f"\n💡 Top Recommendations:")
        for i, rec in enumerate(summary["recommendations"][:3], 1):
            print(f"   {i}. {rec}")

    # Generated reports
    print(f"\n📄 Generated Reports:")
    for report_name, generated in summary["reports_generated"].items():
        status = "✅" if generated else "❌"
        print(f"   • {report_name}: {status}")

    print(f"\n🎉 Load testing suite execution complete!")
    print(f"📁 All reports saved in current directory")
    print(f"📋 Summary report: load_test_summary_report.json")


def main():
    """Main execution function."""
    print("🚀 Music Gen AI - Comprehensive Load Testing Suite")
    print("=" * 60)

    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Please install missing dependencies.")
        sys.exit(1)

    # Start time
    start_time = time.time()

    # Run all tests
    test_results = {}

    try:
        # 1. Locust Tests
        locust_results = run_locust_tests()
        test_results["Locust Load Tests"] = any(result["success"] for result in locust_results)

        # 2. WebSocket Tests
        test_results["WebSocket Load Tests"] = run_websocket_tests()

        # 3. Database Tests
        test_results["Database Pool Tests"] = run_database_tests()

        # 4. Redis Tests
        test_results["Redis Pool Tests"] = run_redis_tests()

        # 5. Analysis Tests
        test_results["Bottleneck Analysis"] = run_bottleneck_analysis()
        test_results["Baseline Metrics Report"] = generate_baseline_report()

        # Generate summary
        summary = generate_summary_report(test_results)

        # Calculate total time
        total_time = time.time() - start_time
        summary["test_execution_summary"]["total_duration_minutes"] = total_time / 60

        # Print final summary
        print_final_summary(test_results, summary)

        print(f"\n⏱️ Total execution time: {total_time/60:.1f} minutes")

        # Exit with appropriate code
        if all(test_results.values()):
            print("🎉 All tests completed successfully!")
            sys.exit(0)
        else:
            print("⚠️ Some tests failed. Check individual test outputs.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
