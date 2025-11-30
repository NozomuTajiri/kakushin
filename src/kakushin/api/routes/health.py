"""Health check endpoints."""

from datetime import datetime
from typing import Any

import torch
from fastapi import APIRouter
from pydantic import BaseModel

from kakushin.core.config import get_settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    timestamp: str
    gpu_available: bool
    gpu_name: str | None = None
    gpu_memory_total: float | None = None
    gpu_memory_used: float | None = None
    gpu_memory_usage_percent: float | None = None


class GPUInfo(BaseModel):
    """GPU information model."""

    available: bool
    device_count: int
    devices: list[dict[str, Any]]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and GPU status."""
    settings = get_settings()

    gpu_available = torch.cuda.is_available()
    gpu_name = None
    gpu_memory_total = None
    gpu_memory_used = None
    gpu_memory_usage_percent = None

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Get current memory usage
        gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_memory_usage_percent = (gpu_memory_used / gpu_memory_total) * 100

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        timestamp=datetime.utcnow().isoformat(),
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_total=round(gpu_memory_total, 2) if gpu_memory_total else None,
        gpu_memory_used=round(gpu_memory_used, 2) if gpu_memory_used else None,
        gpu_memory_usage_percent=(
            round(gpu_memory_usage_percent, 2) if gpu_memory_usage_percent else None
        ),
    )


@router.get("/health/gpu", response_model=GPUInfo)
async def gpu_info() -> GPUInfo:
    """Get detailed GPU information."""
    available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if available else 0

    devices = []
    if available:
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)

            devices.append(
                {
                    "index": i,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_gb": round(total_memory, 2),
                    "allocated_memory_gb": round(allocated, 2),
                    "reserved_memory_gb": round(reserved, 2),
                    "usage_percent": round((allocated / total_memory) * 100, 2),
                }
            )

    return GPUInfo(
        available=available,
        device_count=device_count,
        devices=devices,
    )


@router.get("/ready")
async def readiness_check() -> dict[str, str]:
    """Kubernetes readiness probe."""
    return {"status": "ready"}


@router.get("/live")
async def liveness_check() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "alive"}
