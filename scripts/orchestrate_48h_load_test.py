#!/usr/bin/env python3
"""
48-Hour Sustained Load Test Orchestration for Music Gen AI Staging
Comprehensive orchestration of load testing, monitoring, and reporting
"""

import os
import json
import time
import subprocess
import threading
import signal
import sys
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import psutil
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/results/48h_load_test.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class LoadTestOrchestrator:
    def __init__(self):
        self.base_url = os.getenv("TARGET_HOST", "http://nginx-staging")
        self.test_duration = int(os.getenv("TEST_DURATION", 172800))  # 48 hours in seconds
        self.results_dir = Path("/app/results/48h_load_test")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Test configuration
        self.test_phases = [
            {"name": "warmup", "duration": 1800, "users": 2, "description": "30-minute warmup"},
            {"name": "ramp_up", "duration": 3600, "users": 5, "description": "1-hour ramp up"},
            {
                "name": "sustained_low",
                "duration": 14400,
                "users": 8,
                "description": "4-hour sustained low load",
            },
            {
                "name": "sustained_medium",
                "duration": 28800,
                "users": 12,
                "description": "8-hour sustained medium load",
            },
            {
                "name": "sustained_high",
                "duration": 57600,
                "users": 15,
                "description": "16-hour sustained high load",
            },
            {
                "name": "peak_test",
                "duration": 7200,
                "users": 20,
                "description": "2-hour peak load test",
            },
            {
                "name": "stress_test",
                "duration": 3600,
                "users": 25,
                "description": "1-hour stress test",
            },
            {"name": "cooldown", "duration": 1800, "users": 5, "description": "30-minute cooldown"},
            {
                "name": "endurance",
                "duration": 57600,
                "users": 10,
                "description": "16-hour endurance test",
            },
        ]

        # Running processes
        self.running_processes = {}
        self.monitoring_active = False
        self.test_start_time = None
        self.test_results = {"phases": [], "system_metrics": [], "alerts": [], "incidents": []}

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_all_tests()
        self.generate_final_report()
        sys.exit(0)

    def pre_test_validation(self) -> bool:
        """Validate system before starting 48-hour test"""
        logger.info("Running pre-test validation...")

        validation_results = {
            "system_health": False,
            "dependencies": False,
            "disk_space": False,
            "memory": False,
            "network": False,
        }

        try:
            # 1. Check system health
            logger.info("Checking system health...")
            health_response = requests.get(f"{self.base_url}/health", timeout=30)
            validation_results["system_health"] = health_response.status_code == 200
            logger.info(f"System health: {'✅' if validation_results['system_health'] else '❌'}")

            # 2. Check dependencies (database, redis, etc.)
            logger.info("Checking dependencies...")
            detailed_health = requests.get(f"{self.base_url}/health/detailed", timeout=30)
            if detailed_health.status_code == 200:
                health_data = detailed_health.json()
                db_ok = health_data.get("checks", {}).get("database", {}).get("status") == "healthy"
                redis_ok = health_data.get("checks", {}).get("redis", {}).get("status") == "healthy"
                validation_results["dependencies"] = db_ok and redis_ok
            logger.info(f"Dependencies: {'✅' if validation_results['dependencies'] else '❌'}")

            # 3. Check disk space (need at least 50GB free)
            logger.info("Checking disk space...")
            disk_usage = psutil.disk_usage("/")
            free_gb = disk_usage.free / (1024**3)
            validation_results["disk_space"] = free_gb > 50
            logger.info(
                f"Disk space: {free_gb:.1f}GB free {'✅' if validation_results['disk_space'] else '❌'}"
            )

            # 4. Check memory (need at least 8GB available)
            logger.info("Checking memory...")
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            validation_results["memory"] = available_gb > 8
            logger.info(
                f"Memory: {available_gb:.1f}GB available {'✅' if validation_results['memory'] else '❌'}"
            )

            # 5. Test network connectivity to all services
            logger.info("Testing network connectivity...")
            services = [
                "http://prometheus-staging:9090/api/v1/query?query=up",
                "http://grafana-staging:3000/api/health",
                "http://postgres-staging:5432",  # Will fail but connection attempt is what matters
                "http://redis-staging:6379",  # Will fail but connection attempt is what matters
            ]

            network_ok = True
            for service in services[:2]:  # Only test HTTP services
                try:
                    response = requests.get(service, timeout=10)
                    if response.status_code not in [200, 404]:  # 404 is ok for some endpoints
                        network_ok = False
                except:
                    network_ok = False

            validation_results["network"] = network_ok
            logger.info(f"Network connectivity: {'✅' if validation_results['network'] else '❌'}")

            # Overall validation
            all_valid = all(validation_results.values())

            logger.info("\nPre-test validation summary:")
            for check, result in validation_results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"  {check}: {status}")

            if not all_valid:
                logger.error("Pre-test validation failed. Cannot proceed with 48-hour test.")
                return False

            logger.info("✅ All pre-test validations passed. Ready for 48-hour test.")
            return True

        except Exception as e:
            logger.error(f"Pre-test validation error: {e}")
            return False

    def start_system_monitoring(self):
        """Start comprehensive system monitoring"""
        logger.info("Starting system monitoring...")
        self.monitoring_active = True

        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Collect system metrics
                    metrics = {
                        "timestamp": datetime.now().isoformat(),
                        "cpu_percent": psutil.cpu_percent(interval=1),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_usage_percent": psutil.disk_usage("/").percent,
                        "load_average": psutil.getloadavg(),
                        "network_io": psutil.net_io_counters()._asdict(),
                        "running_processes": len(psutil.pids()),
                    }

                    # Get application metrics from Prometheus
                    try:
                        prom_response = requests.get(
                            "http://prometheus-staging:9090/api/v1/query",
                            params={"query": "up"},
                            timeout=5,
                        )
                        if prom_response.status_code == 200:
                            metrics["prometheus_up"] = True
                    except:
                        metrics["prometheus_up"] = False

                    self.test_results["system_metrics"].append(metrics)

                    # Check for alerts
                    if metrics["cpu_percent"] > 90:
                        self.log_alert(
                            "High CPU usage", f"CPU: {metrics['cpu_percent']:.1f}%", "warning"
                        )

                    if metrics["memory_percent"] > 90:
                        self.log_alert(
                            "High memory usage",
                            f"Memory: {metrics['memory_percent']:.1f}%",
                            "warning",
                        )

                    if metrics["disk_usage_percent"] > 90:
                        self.log_alert(
                            "High disk usage",
                            f"Disk: {metrics['disk_usage_percent']:.1f}%",
                            "critical",
                        )

                    # Log periodic status
                    if len(self.test_results["system_metrics"]) % 60 == 0:  # Every 10 minutes
                        elapsed = (
                            (datetime.now() - self.test_start_time).total_seconds()
                            if self.test_start_time
                            else 0
                        )
                        logger.info(
                            f"Status ({elapsed/3600:.1f}h): CPU {metrics['cpu_percent']:.1f}%, "
                            f"Memory {metrics['memory_percent']:.1f}%, "
                            f"Disk {metrics['disk_usage_percent']:.1f}%"
                        )

                except Exception as e:
                    logger.error(f"Monitoring error: {e}")

                time.sleep(10)  # Collect metrics every 10 seconds

        monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitoring_thread.start()

        logger.info("System monitoring started")

    def log_alert(self, title: str, message: str, severity: str):
        """Log an alert during testing"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "message": message,
            "severity": severity,
        }
        self.test_results["alerts"].append(alert)

        severity_icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(severity, "📝")
        logger.warning(f"{severity_icon} ALERT: {title} - {message}")

    def run_load_test_phase(self, phase: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single load test phase"""
        phase_name = phase["name"]
        duration = phase["duration"]
        users = phase["users"]
        description = phase["description"]

        logger.info(f"Starting phase: {phase_name} ({description})")
        logger.info(f"Duration: {duration}s ({duration/3600:.1f}h), Users: {users}")

        phase_start_time = time.time()
        phase_results = {
            "name": phase_name,
            "start_time": datetime.fromtimestamp(phase_start_time).isoformat(),
            "duration": duration,
            "users": users,
            "description": description,
            "success": False,
            "metrics": {},
            "errors": [],
        }

        try:
            # Start load testing process
            load_test_cmd = [
                "python",
                "/app/scripts/comprehensive_load_test.py",
                self.base_url,
                str(users),
                str(duration),
            ]

            logger.info(f"Executing: {' '.join(load_test_cmd)}")

            load_test_process = subprocess.Popen(
                load_test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            self.running_processes[phase_name] = load_test_process

            # Monitor the phase
            while True:
                poll_result = load_test_process.poll()

                if poll_result is not None:
                    # Process completed
                    stdout, stderr = load_test_process.communicate()

                    if poll_result == 0:
                        phase_results["success"] = True
                        logger.info(f"Phase {phase_name} completed successfully")
                    else:
                        phase_results["errors"].append(f"Load test failed with code {poll_result}")
                        logger.error(f"Phase {phase_name} failed with exit code {poll_result}")
                        if stderr:
                            logger.error(f"Error output: {stderr}")

                    break

                # Check if we've exceeded the phase duration
                elapsed = time.time() - phase_start_time
                if elapsed > duration + 300:  # 5-minute grace period
                    logger.warning(f"Phase {phase_name} exceeded duration, terminating...")
                    load_test_process.terminate()
                    time.sleep(10)
                    if load_test_process.poll() is None:
                        load_test_process.kill()
                    phase_results["errors"].append("Phase exceeded duration limit")
                    break

                time.sleep(30)  # Check every 30 seconds

            # Cleanup
            if phase_name in self.running_processes:
                del self.running_processes[phase_name]

            phase_end_time = time.time()
            actual_duration = phase_end_time - phase_start_time
            phase_results["actual_duration"] = actual_duration
            phase_results["end_time"] = datetime.fromtimestamp(phase_end_time).isoformat()

            logger.info(f"Phase {phase_name} completed in {actual_duration:.1f}s")

        except Exception as e:
            phase_results["errors"].append(str(e))
            logger.error(f"Phase {phase_name} error: {e}")

        self.test_results["phases"].append(phase_results)
        return phase_results

    def run_periodic_health_checks(self):
        """Run periodic health checks during the test"""

        def health_check_loop():
            while self.monitoring_active:
                try:
                    # Run integration tests every hour
                    logger.info("Running periodic health check...")

                    integration_cmd = ["python", "/app/scripts/integration_test_suite.py"]
                    integration_result = subprocess.run(
                        integration_cmd,
                        capture_output=True,
                        text=True,
                        timeout=600,  # 10 minutes timeout
                    )

                    if integration_result.returncode != 0:
                        self.log_alert(
                            "Integration test failure",
                            f"Integration tests failed during load test",
                            "critical",
                        )

                        # Log as incident
                        incident = {
                            "timestamp": datetime.now().isoformat(),
                            "type": "integration_test_failure",
                            "description": "Integration tests failed during sustained load",
                            "returncode": integration_result.returncode,
                            "stderr": integration_result.stderr[:1000],  # Truncate long errors
                        }
                        self.test_results["incidents"].append(incident)

                    # Check backup systems every 6 hours
                    elapsed = (datetime.now() - self.test_start_time).total_seconds()
                    if elapsed % 21600 < 3600:  # Every 6 hours with 1-hour window
                        logger.info("Running backup system test...")
                        backup_cmd = ["python", "/app/scripts/backup_restore_test.py"]
                        backup_result = subprocess.run(
                            backup_cmd,
                            capture_output=True,
                            text=True,
                            timeout=1800,  # 30 minutes timeout
                        )

                        if backup_result.returncode != 0:
                            self.log_alert(
                                "Backup system failure", "Backup/restore tests failed", "warning"
                            )

                except Exception as e:
                    logger.error(f"Health check error: {e}")

                time.sleep(3600)  # Run every hour

        health_thread = threading.Thread(target=health_check_loop, daemon=True)
        health_thread.start()

    def run_48_hour_test(self):
        """Execute the complete 48-hour sustained load test"""
        logger.info("=" * 80)
        logger.info("STARTING 48-HOUR SUSTAINED LOAD TEST")
        logger.info("=" * 80)

        # Pre-test validation
        if not self.pre_test_validation():
            logger.error("Pre-test validation failed. Aborting 48-hour test.")
            return False

        self.test_start_time = datetime.now()
        logger.info(f"Test started at: {self.test_start_time.isoformat()}")

        # Start monitoring
        self.start_system_monitoring()
        self.run_periodic_health_checks()

        # Execute test phases
        total_planned_duration = sum(phase["duration"] for phase in self.test_phases)
        logger.info(
            f"Total planned duration: {total_planned_duration}s ({total_planned_duration/3600:.1f}h)"
        )

        for i, phase in enumerate(self.test_phases, 1):
            logger.info(f"\n📊 Phase {i}/{len(self.test_phases)}: {phase['name']}")

            phase_result = self.run_load_test_phase(phase)

            if not phase_result["success"]:
                self.log_alert(
                    f"Phase {phase['name']} failed",
                    f"Load test phase failed: {phase_result.get('errors', [])}",
                    "critical",
                )

            # Brief pause between phases
            if i < len(self.test_phases):
                logger.info("Pausing 2 minutes between phases...")
                time.sleep(120)

        # Stop monitoring
        self.monitoring_active = False

        test_end_time = datetime.now()
        total_duration = (test_end_time - self.test_start_time).total_seconds()

        logger.info(f"\n48-hour test completed at: {test_end_time.isoformat()}")
        logger.info(f"Total actual duration: {total_duration:.1f}s ({total_duration/3600:.1f}h)")

        # Generate final report
        self.generate_final_report()

        return True

    def stop_all_tests(self):
        """Stop all running test processes"""
        logger.info("Stopping all running tests...")

        for phase_name, process in self.running_processes.items():
            try:
                logger.info(f"Terminating {phase_name}...")
                process.terminate()
                time.sleep(5)
                if process.poll() is None:
                    process.kill()
            except Exception as e:
                logger.error(f"Error stopping {phase_name}: {e}")

        self.running_processes.clear()
        self.monitoring_active = False

    def generate_final_report(self):
        """Generate comprehensive final report"""
        logger.info("Generating final 48-hour test report...")

        test_end_time = datetime.now()
        total_duration = (
            (test_end_time - self.test_start_time).total_seconds() if self.test_start_time else 0
        )

        # Calculate statistics
        successful_phases = [p for p in self.test_results["phases"] if p["success"]]
        failed_phases = [p for p in self.test_results["phases"] if not p["success"]]

        # System metrics analysis
        if self.test_results["system_metrics"]:
            cpu_values = [m["cpu_percent"] for m in self.test_results["system_metrics"]]
            memory_values = [m["memory_percent"] for m in self.test_results["system_metrics"]]

            system_stats = {
                "avg_cpu": sum(cpu_values) / len(cpu_values),
                "max_cpu": max(cpu_values),
                "avg_memory": sum(memory_values) / len(memory_values),
                "max_memory": max(memory_values),
                "metrics_collected": len(self.test_results["system_metrics"]),
            }
        else:
            system_stats = {}

        final_report = {
            "test_summary": {
                "start_time": self.test_start_time.isoformat() if self.test_start_time else None,
                "end_time": test_end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "total_duration_hours": total_duration / 3600,
                "planned_phases": len(self.test_phases),
                "completed_phases": len(self.test_results["phases"]),
                "successful_phases": len(successful_phases),
                "failed_phases": len(failed_phases),
                "success_rate": len(successful_phases) / len(self.test_results["phases"]) * 100
                if self.test_results["phases"]
                else 0,
            },
            "system_performance": system_stats,
            "alerts_summary": {
                "total_alerts": len(self.test_results["alerts"]),
                "critical_alerts": len(
                    [a for a in self.test_results["alerts"] if a["severity"] == "critical"]
                ),
                "warning_alerts": len(
                    [a for a in self.test_results["alerts"] if a["severity"] == "warning"]
                ),
                "info_alerts": len(
                    [a for a in self.test_results["alerts"] if a["severity"] == "info"]
                ),
            },
            "incidents": self.test_results["incidents"],
            "phase_results": self.test_results["phases"],
            "full_system_metrics": self.test_results["system_metrics"],
            "all_alerts": self.test_results["alerts"],
            "recommendations": self._generate_recommendations(),
        }

        # Save detailed report
        report_file = self.results_dir / f"48h_load_test_report_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(final_report, f, indent=2)

        # Save summary report
        summary_file = self.results_dir / "48h_test_summary.json"
        summary_report = {
            "test_summary": final_report["test_summary"],
            "system_performance": final_report["system_performance"],
            "alerts_summary": final_report["alerts_summary"],
            "recommendations": final_report["recommendations"],
        }
        with open(summary_file, "w") as f:
            json.dump(summary_report, f, indent=2)

        logger.info(f"Final report saved to: {report_file}")
        logger.info(f"Summary report saved to: {summary_file}")

        self.print_final_summary(final_report)

        return final_report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        # Analyze system performance
        if self.test_results["system_metrics"]:
            cpu_values = [m["cpu_percent"] for m in self.test_results["system_metrics"]]
            memory_values = [m["memory_percent"] for m in self.test_results["system_metrics"]]

            avg_cpu = sum(cpu_values) / len(cpu_values)
            max_cpu = max(cpu_values)
            avg_memory = sum(memory_values) / len(memory_values)
            max_memory = max(memory_values)

            if avg_cpu > 70:
                recommendations.append(
                    f"High average CPU usage ({avg_cpu:.1f}%) - consider scaling out or optimizing performance"
                )

            if max_cpu > 95:
                recommendations.append(
                    f"CPU peaked at {max_cpu:.1f}% - investigate CPU bottlenecks"
                )

            if avg_memory > 80:
                recommendations.append(
                    f"High average memory usage ({avg_memory:.1f}%) - consider increasing memory or optimizing memory usage"
                )

            if max_memory > 95:
                recommendations.append(
                    f"Memory peaked at {max_memory:.1f}% - risk of out-of-memory errors"
                )

        # Analyze alerts
        critical_alerts = [a for a in self.test_results["alerts"] if a["severity"] == "critical"]
        if critical_alerts:
            recommendations.append(
                f"Resolve {len(critical_alerts)} critical alerts that occurred during testing"
            )

        # Analyze phase failures
        failed_phases = [p for p in self.test_results["phases"] if not p["success"]]
        if failed_phases:
            recommendations.append(f"Investigate {len(failed_phases)} failed test phases")

        # Analyze incidents
        if self.test_results["incidents"]:
            recommendations.append(
                f"Address {len(self.test_results['incidents'])} incidents that occurred during testing"
            )

        # General recommendations
        if not recommendations:
            recommendations.append(
                "System performed well during 48-hour test - maintain current configuration"
            )

        recommendations.append("Schedule regular 48-hour load tests to validate system reliability")
        recommendations.append("Monitor identified bottlenecks in production")

        return recommendations

    def print_final_summary(self, report: Dict[str, Any]):
        """Print final test summary"""
        print("\n" + "=" * 80)
        print("48-HOUR SUSTAINED LOAD TEST - FINAL SUMMARY")
        print("=" * 80)

        summary = report["test_summary"]
        print(f"Duration: {summary['total_duration_hours']:.1f} hours")
        print(f"Phases: {summary['completed_phases']}/{summary['planned_phases']} completed")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print()

        if report["system_performance"]:
            perf = report["system_performance"]
            print("SYSTEM PERFORMANCE:")
            print(
                f"  Average CPU: {perf.get('avg_cpu', 0):.1f}% (Peak: {perf.get('max_cpu', 0):.1f}%)"
            )
            print(
                f"  Average Memory: {perf.get('avg_memory', 0):.1f}% (Peak: {perf.get('max_memory', 0):.1f}%)"
            )
            print(f"  Metrics Collected: {perf.get('metrics_collected', 0)}")
            print()

        alerts = report["alerts_summary"]
        print(f"ALERTS: {alerts['total_alerts']} total")
        if alerts["critical_alerts"]:
            print(f"  🚨 Critical: {alerts['critical_alerts']}")
        if alerts["warning_alerts"]:
            print(f"  ⚠️  Warning: {alerts['warning_alerts']}")
        if alerts["info_alerts"]:
            print(f"  ℹ️  Info: {alerts['info_alerts']}")
        print()

        if report["incidents"]:
            print(f"INCIDENTS: {len(report['incidents'])} occurred")
            for incident in report["incidents"][:5]:  # Show first 5
                print(f"  - {incident['type']}: {incident['description']}")
            if len(report["incidents"]) > 5:
                print(f"  ... and {len(report['incidents']) - 5} more")
            print()

        print("RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")

        print("=" * 80)


def main():
    """Main orchestration execution"""
    orchestrator = LoadTestOrchestrator()

    try:
        success = orchestrator.run_48_hour_test()
        exit_code = 0 if success else 1

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        orchestrator.stop_all_tests()
        orchestrator.generate_final_report()
        exit_code = 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"48-hour test orchestration failed: {e}")
        orchestrator.stop_all_tests()
        exit_code = 1

    finally:
        orchestrator.stop_all_tests()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
