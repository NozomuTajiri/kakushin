"""GPU monitoring utilities."""

from dataclasses import dataclass
from typing import Any

import structlog
import torch

logger = structlog.get_logger()


@dataclass
class GPUStats:
    """GPU statistics."""

    available: bool
    device_index: int
    device_name: str
    total_memory_gb: float
    allocated_memory_gb: float
    reserved_memory_gb: float
    free_memory_gb: float
    usage_percent: float

    @property
    def is_under_threshold(self) -> bool:
        """Check if usage is under 92% threshold."""
        return self.usage_percent < 92.0


class GPUMonitor:
    """Monitor GPU memory usage and provide optimization suggestions."""

    def __init__(self, device_index: int = 0, threshold: float = 0.92):
        self.device_index = device_index
        self.threshold = threshold
        self.logger = logger.bind(component="GPUMonitor")

    @property
    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()

    def get_stats(self) -> GPUStats:
        """Get current GPU statistics."""
        if not self.is_available:
            return GPUStats(
                available=False,
                device_index=self.device_index,
                device_name="N/A",
                total_memory_gb=0.0,
                allocated_memory_gb=0.0,
                reserved_memory_gb=0.0,
                free_memory_gb=0.0,
                usage_percent=0.0,
            )

        props = torch.cuda.get_device_properties(self.device_index)
        total = props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(self.device_index) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device_index) / (1024**3)
        free = total - reserved

        return GPUStats(
            available=True,
            device_index=self.device_index,
            device_name=props.name,
            total_memory_gb=round(total, 2),
            allocated_memory_gb=round(allocated, 2),
            reserved_memory_gb=round(reserved, 2),
            free_memory_gb=round(free, 2),
            usage_percent=round((reserved / total) * 100, 2),
        )

    def check_threshold(self) -> tuple[bool, GPUStats]:
        """Check if GPU usage is under threshold.

        Returns:
            Tuple of (is_safe, stats)
        """
        stats = self.get_stats()
        is_safe = stats.usage_percent < (self.threshold * 100)
        return is_safe, stats

    def should_pause(self) -> bool:
        """Check if operations should pause due to memory pressure."""
        is_safe, stats = self.check_threshold()
        if not is_safe:
            self.logger.warning(
                "GPU memory threshold exceeded",
                usage_percent=stats.usage_percent,
                threshold_percent=self.threshold * 100,
            )
        return not is_safe

    def get_recommended_batch_size(
        self,
        model_memory_gb: float = 4.0,
        base_batch_size: int = 1,
    ) -> int:
        """Get recommended batch size based on available memory.

        Args:
            model_memory_gb: Estimated memory per model/operation
            base_batch_size: Minimum batch size

        Returns:
            Recommended batch size
        """
        stats = self.get_stats()
        if not stats.available:
            return base_batch_size

        # Leave 8% headroom (92% threshold)
        available_for_batching = stats.free_memory_gb - (stats.total_memory_gb * 0.08)

        if available_for_batching <= 0:
            return base_batch_size

        recommended = int(available_for_batching / model_memory_gb)
        return max(base_batch_size, recommended)

    def clear_cache(self) -> None:
        """Clear CUDA cache to free up memory."""
        if self.is_available:
            torch.cuda.empty_cache()
            self.logger.info("CUDA cache cleared")

    def synchronize(self) -> None:
        """Synchronize CUDA operations."""
        if self.is_available:
            torch.cuda.synchronize(self.device_index)

    def log_stats(self) -> None:
        """Log current GPU statistics."""
        stats = self.get_stats()
        self.logger.info(
            "GPU Statistics",
            device=stats.device_name,
            total_gb=stats.total_memory_gb,
            allocated_gb=stats.allocated_memory_gb,
            free_gb=stats.free_memory_gb,
            usage_percent=stats.usage_percent,
        )


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
