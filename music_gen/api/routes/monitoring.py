from datetime import timedelta

"""
Resource monitoring API endpoints.

Provides real-time resource usage information and health status.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query

from ...core.container import get_container
from ...core.resource_manager import ResourceManager, ResourceOptimizer
from ..schemas.monitoring import (
    OptimizationSuggestionResponse,
    ResourceAlertResponse,
    ResourceReportResponse,
    ResourceStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/monitoring", tags=["monitoring"])


def get_resource_manager() -> ResourceManager:
    """Get resource manager from DI container."""
    container = get_container()
    # Get from model service which has the resource manager
    model_service = container.get("model_service")
    if hasattr(model_service, "resource_manager"):
        return model_service.resource_manager
    else:
        raise HTTPException(status_code=500, detail="Resource manager not available")


@router.get("/status", response_model=ResourceStatusResponse)
async def get_resource_status(
    resource_manager: ResourceManager = Depends(get_resource_manager),
) -> ResourceStatusResponse:
    """Get current resource status."""
    try:
        snapshot = resource_manager.monitor.get_current_snapshot()

        return ResourceStatusResponse(
            timestamp=snapshot.timestamp,
            cpu={
                "usage_percent": snapshot.cpu_percent,
                "memory_used_gb": round(snapshot.cpu_memory_used_gb, 2),
                "memory_available_gb": round(snapshot.cpu_memory_available_gb, 2),
                "memory_percent": round(snapshot.cpu_memory_percent, 1),
            },
            gpu={
                "available": snapshot.gpu_available,
                "memory_used_gb": round(snapshot.gpu_memory_used_gb, 2),
                "memory_total_gb": round(snapshot.gpu_memory_total_gb, 2),
                "memory_percent": round(snapshot.gpu_memory_percent, 1),
                "utilization": round(snapshot.gpu_utilization, 1),
                "temperature": round(snapshot.gpu_temperature, 1),
            },
            process={
                "memory_gb": round(snapshot.process_memory_gb, 2),
                "gpu_memory_gb": round(snapshot.process_gpu_memory_gb, 2),
            },
        )
    except Exception as e:
        logger.error(f"Failed to get resource status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report", response_model=ResourceReportResponse)
async def get_resource_report(
    resource_manager: ResourceManager = Depends(get_resource_manager),
) -> ResourceReportResponse:
    """Get comprehensive resource report."""
    try:
        report = resource_manager.get_resource_report()

        return ResourceReportResponse(
            timestamp=datetime.fromisoformat(report["timestamp"]),
            current=report["current"],
            average_5min=report["average_5min"],
            allocated_resources=report["allocated_resources"],
            cached_models=report["cached_models"],
            alerts=report["alerts"],
            health_status=report["health_status"],
        )
    except Exception as e:
        logger.error(f"Failed to get resource report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=List[ResourceAlertResponse])
async def get_resource_alerts(
    severity: str = Query(None, description="Filter by severity (warning, critical, error)"),
    limit: int = Query(20, description="Maximum number of alerts to return"),
    resource_manager: ResourceManager = Depends(get_resource_manager),
) -> List[ResourceAlertResponse]:
    """Get recent resource alerts."""
    try:
        alerts = resource_manager.monitor.get_recent_alerts(severity=severity)

        # Convert to response format
        alert_responses = []
        for alert in alerts[:limit]:
            alert_responses.append(
                ResourceAlertResponse(
                    timestamp=alert.timestamp,
                    severity=alert.severity,
                    resource_type=alert.resource_type,
                    message=alert.message,
                    current_value=alert.current_value,
                    threshold=alert.threshold,
                    recommendations=alert.recommendations,
                )
            )

        return alert_responses
    except Exception as e:
        logger.error(f"Failed to get resource alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimization-suggestions", response_model=OptimizationSuggestionResponse)
async def get_optimization_suggestions(
    resource_manager: ResourceManager = Depends(get_resource_manager),
) -> OptimizationSuggestionResponse:
    """Get resource optimization suggestions."""
    try:
        # Get current resource report
        report = resource_manager.get_resource_report()

        # Get optimization suggestions
        suggestions = ResourceOptimizer.get_optimization_suggestions(report)

        # Get optimal batch sizes for different model sizes
        snapshot = resource_manager.monitor.get_current_snapshot()
        available_gpu_memory = 0
        if snapshot.gpu_available:
            available_gpu_memory = snapshot.gpu_memory_total_gb - snapshot.gpu_memory_used_gb

        optimal_batch_sizes = {
            "small": ResourceOptimizer.get_optimal_batch_size("small", available_gpu_memory),
            "medium": ResourceOptimizer.get_optimal_batch_size("medium", available_gpu_memory),
            "large": ResourceOptimizer.get_optimal_batch_size("large", available_gpu_memory),
        }

        return OptimizationSuggestionResponse(
            suggestions=suggestions,
            optimal_batch_sizes=optimal_batch_sizes,
            current_health=report["health_status"],
        )
    except Exception as e:
        logger.error(f"Failed to get optimization suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup")
async def trigger_cleanup(
    force: bool = Query(False, description="Force aggressive cleanup"),
    resource_manager: ResourceManager = Depends(get_resource_manager),
) -> Dict[str, Any]:
    """Trigger manual resource cleanup."""
    try:
        logger.info(f"Manual cleanup triggered (force={force})")

        # Get before snapshot
        before = resource_manager.monitor.get_current_snapshot()

        if force:
            # Aggressive cleanup
            resource_manager._emergency_cleanup()
            resource_manager._emergency_gpu_cleanup()
        else:
            # Normal cleanup
            import gc

            gc.collect()
            if torch.cuda.is_available():
                import torch

                torch.cuda.empty_cache()

        # Get after snapshot
        after = resource_manager.monitor.get_current_snapshot()

        # Calculate freed resources
        cpu_freed = before.cpu_memory_used_gb - after.cpu_memory_used_gb
        gpu_freed = before.gpu_memory_used_gb - after.gpu_memory_used_gb

        return {
            "success": True,
            "cpu_memory_freed_gb": round(max(0, cpu_freed), 2),
            "gpu_memory_freed_gb": round(max(0, gpu_freed), 2),
            "message": f"Cleanup completed. CPU: {cpu_freed:.2f}GB freed, GPU: {gpu_freed:.2f}GB freed",
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_resource_history(
    window_seconds: int = Query(300, description="Time window in seconds (default 5 minutes)"),
    resource_manager: ResourceManager = Depends(get_resource_manager),
) -> Dict[str, Any]:
    """Get resource usage history."""
    try:
        # Get snapshots from history
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        history_data = []

        for snapshot in resource_manager.monitor.history:
            if snapshot.timestamp > cutoff_time:
                history_data.append(
                    {
                        "timestamp": snapshot.timestamp.isoformat(),
                        "cpu_percent": snapshot.cpu_percent,
                        "cpu_memory_percent": snapshot.cpu_memory_percent,
                        "gpu_memory_percent": snapshot.gpu_memory_percent,
                        "gpu_utilization": snapshot.gpu_utilization,
                    }
                )

        # Calculate statistics
        if history_data:
            cpu_percents = [d["cpu_percent"] for d in history_data]
            cpu_mem_percents = [d["cpu_memory_percent"] for d in history_data]
            gpu_mem_percents = [d["gpu_memory_percent"] for d in history_data]
            gpu_utils = [d["gpu_utilization"] for d in history_data]

            stats = {
                "cpu_percent": {
                    "avg": round(sum(cpu_percents) / len(cpu_percents), 1),
                    "max": round(max(cpu_percents), 1),
                    "min": round(min(cpu_percents), 1),
                },
                "cpu_memory_percent": {
                    "avg": round(sum(cpu_mem_percents) / len(cpu_mem_percents), 1),
                    "max": round(max(cpu_mem_percents), 1),
                    "min": round(min(cpu_mem_percents), 1),
                },
                "gpu_memory_percent": {
                    "avg": round(sum(gpu_mem_percents) / len(gpu_mem_percents), 1),
                    "max": round(max(gpu_mem_percents), 1),
                    "min": round(min(gpu_mem_percents), 1),
                },
                "gpu_utilization": {
                    "avg": round(sum(gpu_utils) / len(gpu_utils), 1),
                    "max": round(max(gpu_utils), 1),
                    "min": round(min(gpu_utils), 1),
                },
            }
        else:
            stats = {}

        return {
            "window_seconds": window_seconds,
            "data_points": len(history_data),
            "statistics": stats,
            "history": history_data[-100:],  # Limit to last 100 points
        }
    except Exception as e:
        logger.error(f"Failed to get resource history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
