"""MemoryAgent - GPUメモリを効率的に管理するエージェント."""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum

import structlog
import torch
import torch.nn as nn

from kakushin.agents.base import BaseAgent, AgentResult
from kakushin.core.config import Settings
from kakushin.utils.gpu_monitor import GPUMonitor, GPUStats

logger = structlog.get_logger()


class MemoryAction(Enum):
    """Actions that can be taken for memory management."""

    CONTINUE = "continue"  # Safe to continue
    REDUCE_BATCH = "reduce_batch"  # Reduce batch size
    OFFLOAD = "offload"  # Offload models to CPU
    PAUSE = "pause"  # Pause operations
    CLEAR_CACHE = "clear_cache"  # Clear CUDA cache


@dataclass
class MemoryInput:
    """Input for MemoryAgent."""

    check_only: bool = False  # Just check status
    model_to_load: str | None = None
    model_to_offload: nn.Module | None = None
    target_batch_size: int | None = None


@dataclass
class MemoryResult:
    """Result from MemoryAgent."""

    stats: GPUStats
    recommended_action: MemoryAction
    recommended_batch_size: int
    is_safe: bool
    loaded_models: list[str] = field(default_factory=list)
    message: str = ""


class MemoryAgent(BaseAgent[MemoryInput, MemoryResult]):
    """Agent that manages GPU memory for efficient video generation."""

    def __init__(self, settings: Settings | None = None):
        super().__init__(name="MemoryAgent", settings=settings)
        self.monitor = GPUMonitor(
            threshold=settings.gpu_memory_limit if settings else 0.92
        )
        self._loaded_models: dict[str, nn.Module] = {}
        self._offloaded_models: dict[str, nn.Module] = {}

    async def execute(self, input_data: MemoryInput) -> AgentResult[MemoryResult]:
        """Execute memory management."""
        try:
            # Get current stats
            stats = self.monitor.get_stats()
            is_safe = stats.is_under_threshold

            # Determine recommended action
            action = self._determine_action(stats, input_data)

            # Calculate recommended batch size
            recommended_batch = self.monitor.get_recommended_batch_size(
                model_memory_gb=4.0,  # Estimated memory per batch
                base_batch_size=1,
            )

            # Execute action if not just checking
            if not input_data.check_only:
                await self._execute_action(action, input_data)

            result = MemoryResult(
                stats=stats,
                recommended_action=action,
                recommended_batch_size=recommended_batch,
                is_safe=is_safe,
                loaded_models=list(self._loaded_models.keys()),
                message=self._get_message(action, stats),
            )

            self.logger.info(
                "Memory check completed",
                usage_percent=stats.usage_percent,
                action=action.value,
                is_safe=is_safe,
            )

            return AgentResult(
                success=True,
                data=result,
                metadata={
                    "gpu_usage_percent": stats.usage_percent,
                    "free_memory_gb": stats.free_memory_gb,
                },
            )

        except Exception as e:
            self.logger.exception("Memory management failed", error=str(e))
            return AgentResult(success=False, error=str(e))

    def _determine_action(
        self, stats: GPUStats, input_data: MemoryInput
    ) -> MemoryAction:
        """Determine the best action based on current memory state."""
        if not stats.available:
            return MemoryAction.CONTINUE  # No GPU, nothing to manage

        usage = stats.usage_percent

        if usage >= 95:
            return MemoryAction.PAUSE
        elif usage >= 92:
            return MemoryAction.OFFLOAD
        elif usage >= 85:
            return MemoryAction.REDUCE_BATCH
        elif usage >= 80:
            return MemoryAction.CLEAR_CACHE
        else:
            return MemoryAction.CONTINUE

    async def _execute_action(
        self, action: MemoryAction, input_data: MemoryInput
    ) -> None:
        """Execute the recommended action."""
        if action == MemoryAction.CLEAR_CACHE:
            self.clear_cache()

        elif action == MemoryAction.OFFLOAD:
            if input_data.model_to_offload:
                self.offload_model("temp_model", input_data.model_to_offload)
            else:
                # Offload least recently used models
                self._offload_lru_models()

        elif action == MemoryAction.PAUSE:
            self.logger.warning("GPU memory critical - operations should pause")

    def _get_message(self, action: MemoryAction, stats: GPUStats) -> str:
        """Generate a human-readable message about memory state."""
        messages = {
            MemoryAction.CONTINUE: f"Memory usage OK ({stats.usage_percent:.1f}%)",
            MemoryAction.REDUCE_BATCH: f"High memory ({stats.usage_percent:.1f}%) - reduce batch size",
            MemoryAction.OFFLOAD: f"Memory critical ({stats.usage_percent:.1f}%) - offloading models",
            MemoryAction.PAUSE: f"Memory exceeded ({stats.usage_percent:.1f}%) - pausing operations",
            MemoryAction.CLEAR_CACHE: f"Memory elevated ({stats.usage_percent:.1f}%) - clearing cache",
        }
        return messages.get(action, "Unknown state")

    def clear_cache(self) -> None:
        """Clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("CUDA cache cleared")

    def offload_model(self, name: str, model: nn.Module) -> None:
        """Offload a model from GPU to CPU."""
        if name in self._loaded_models:
            del self._loaded_models[name]

        model.cpu()
        self._offloaded_models[name] = model
        self.clear_cache()

        self.logger.info(f"Model '{name}' offloaded to CPU")

    def load_model(self, name: str) -> nn.Module | None:
        """Load a model from CPU to GPU."""
        if name in self._offloaded_models:
            model = self._offloaded_models[name]
            del self._offloaded_models[name]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            self._loaded_models[name] = model

            self.logger.info(f"Model '{name}' loaded to {device}")
            return model

        return self._loaded_models.get(name)

    def _offload_lru_models(self) -> None:
        """Offload least recently used models."""
        if self._loaded_models:
            # Offload first model (simple LRU implementation)
            name = next(iter(self._loaded_models))
            model = self._loaded_models[name]
            self.offload_model(name, model)

    def register_model(self, name: str, model: nn.Module) -> None:
        """Register a model for memory management."""
        self._loaded_models[name] = model
        self.logger.info(f"Model '{name}' registered for memory management")

    def get_status(self) -> dict[str, Any]:
        """Get current memory management status."""
        stats = self.monitor.get_stats()
        return {
            "gpu_available": stats.available,
            "usage_percent": stats.usage_percent,
            "free_memory_gb": stats.free_memory_gb,
            "loaded_models": list(self._loaded_models.keys()),
            "offloaded_models": list(self._offloaded_models.keys()),
            "is_safe": stats.is_under_threshold,
        }

    async def should_pause(self) -> bool:
        """Check if operations should pause due to memory pressure."""
        result = await self.run(MemoryInput(check_only=True))
        if result.success and result.data:
            return result.data.recommended_action == MemoryAction.PAUSE
        return False

    async def get_recommended_batch_size(self) -> int:
        """Get recommended batch size based on current memory."""
        result = await self.run(MemoryInput(check_only=True))
        if result.success and result.data:
            return result.data.recommended_batch_size
        return 1
