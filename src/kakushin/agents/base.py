"""Base agent class for all Kakushin agents."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from datetime import datetime

import structlog

from kakushin.core.config import Settings, get_settings

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type


@dataclass
class AgentResult(Generic[R]):
    """Result wrapper for agent execution."""

    success: bool
    data: R | None = None
    error: str | None = None
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC, Generic[T, R]):
    """Abstract base class for all agents."""

    def __init__(
        self,
        name: str,
        settings: Settings | None = None,
    ):
        self.id: UUID = uuid4()
        self.name = name
        self.settings = settings or get_settings()
        self.logger = structlog.get_logger().bind(agent=name)
        self._created_at = datetime.utcnow()

    @abstractmethod
    async def execute(self, input_data: T) -> AgentResult[R]:
        """Execute the agent's main task.

        Args:
            input_data: Input data for the agent

        Returns:
            AgentResult containing the execution result
        """
        pass

    async def validate_input(self, input_data: T) -> bool:
        """Validate input data before execution.

        Override this method to add custom validation.
        """
        return True

    async def pre_execute(self, input_data: T) -> None:
        """Hook called before execution.

        Override for setup/initialization.
        """
        pass

    async def post_execute(self, result: AgentResult[R]) -> None:
        """Hook called after execution.

        Override for cleanup/logging.
        """
        pass

    async def run(self, input_data: T) -> AgentResult[R]:
        """Run the agent with full lifecycle.

        This method handles validation, pre/post hooks, and error handling.
        """
        start_time = datetime.utcnow()

        try:
            # Validate input
            if not await self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    error="Input validation failed",
                )

            # Pre-execute hook
            await self.pre_execute(input_data)

            # Execute main task
            self.logger.info("Executing agent", input_type=type(input_data).__name__)
            result = await self.execute(input_data)

            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time = execution_time

            # Post-execute hook
            await self.post_execute(result)

            self.logger.info(
                "Agent execution completed",
                success=result.success,
                execution_time=execution_time,
            )

            return result

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.exception("Agent execution failed", error=str(e))

            return AgentResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, id={self.id})"
