"""
Schemas for resource monitoring API.
"""

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class CPUResourceInfo(BaseModel):
    """CPU resource information."""

    usage_percent: float = Field(..., description="CPU usage percentage")
    memory_used_gb: float = Field(..., description="Used memory in GB")
    memory_available_gb: float = Field(..., description="Available memory in GB")
    memory_percent: float = Field(..., description="Memory usage percentage")


class GPUResourceInfo(BaseModel):
    """GPU resource information."""

    available: bool = Field(..., description="Whether GPU is available")
    memory_used_gb: float = Field(0.0, description="Used GPU memory in GB")
    memory_total_gb: float = Field(0.0, description="Total GPU memory in GB")
    memory_percent: float = Field(0.0, description="GPU memory usage percentage")
    utilization: float = Field(0.0, description="GPU compute utilization percentage")
    temperature: float = Field(0.0, description="GPU temperature in Celsius")


class ProcessResourceInfo(BaseModel):
    """Process-specific resource information."""

    memory_gb: float = Field(..., description="Process memory usage in GB")
    gpu_memory_gb: float = Field(0.0, description="Process GPU memory usage in GB")


class ResourceStatusResponse(BaseModel):
    """Current resource status response."""

    timestamp: datetime
    cpu: CPUResourceInfo
    gpu: GPUResourceInfo
    process: ProcessResourceInfo


class ResourceAlertInfo(BaseModel):
    """Resource alert information."""

    timestamp: datetime
    severity: str
    type: str
    message: str
    value: float
    threshold: float


class ResourceAlertsInfo(BaseModel):
    """Resource alerts summary."""

    total: int
    critical: int
    warnings: int
    recent: List[ResourceAlertInfo]


class ResourceReportResponse(BaseModel):
    """Comprehensive resource report response."""

    timestamp: datetime
    current: Dict[str, Any]
    average_5min: Dict[str, float]
    allocated_resources: Dict[str, Any]
    cached_models: Dict[str, Any]
    alerts: ResourceAlertsInfo
    health_status: str


class ResourceAlertResponse(BaseModel):
    """Individual resource alert response."""

    timestamp: datetime
    severity: str = Field(..., description="Alert severity (warning, critical, error)")
    resource_type: str = Field(..., description="Type of resource (cpu_memory, gpu_memory, etc)")
    message: str = Field(..., description="Alert message")
    current_value: float = Field(..., description="Current value that triggered the alert")
    threshold: float = Field(..., description="Threshold that was exceeded")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")


class OptimizationSuggestionResponse(BaseModel):
    """Resource optimization suggestions response."""

    suggestions: List[str] = Field(..., description="List of optimization suggestions")
    optimal_batch_sizes: Dict[str, int] = Field(
        ..., description="Optimal batch sizes for different model sizes"
    )
    current_health: str = Field(..., description="Current system health status")
