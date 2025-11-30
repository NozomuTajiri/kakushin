"""VideoGenAgent - 動画セグメントを生成するエージェント."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
import numpy as np
from PIL import Image

from kakushin.agents.base import BaseAgent, AgentResult
from kakushin.agents.memory_agent import MemoryAgent, MemoryInput
from kakushin.core.config import Settings
from kakushin.models.video import VideoSegment, Frame
from kakushin.models.storyboard import Shot, StyleGuide, Character, Location

logger = structlog.get_logger()


class VideoModel(Enum):
    """Available video generation models."""

    WAN21 = "wan21"
    COGVIDEOX = "cogvideox"
    SORA = "sora"  # Commercial API
    VEO = "veo"  # Commercial API
    RUNWAY = "runway"  # Commercial API


@dataclass
class VideoGenInput:
    """Input for VideoGenAgent."""

    shot: Shot
    characters: dict[str, Character]
    locations: dict[str, Location]
    style_guide: StyleGuide
    prev_frame: Frame | None = None
    model: VideoModel = VideoModel.WAN21
    seed: int | None = None


class VideoGenAgent(BaseAgent[VideoGenInput, VideoSegment]):
    """Agent that generates video segments from shot descriptions."""

    def __init__(self, settings: Settings | None = None):
        super().__init__(name="VideoGenAgent", settings=settings)
        self.memory_agent = MemoryAgent(settings=settings)
        self._models: dict[VideoModel, Any] = {}
        self._current_model: VideoModel | None = None

    async def execute(self, input_data: VideoGenInput) -> AgentResult[VideoSegment]:
        """Execute video generation for a single shot."""
        self.logger.info(
            "Generating video segment",
            shot_index=input_data.shot.index,
            model=input_data.model.value,
            duration=input_data.shot.duration,
        )

        try:
            # Check memory before generation
            memory_result = await self.memory_agent.run(MemoryInput(check_only=True))
            if memory_result.success and memory_result.data:
                if not memory_result.data.is_safe:
                    self.logger.warning("GPU memory high, clearing cache")
                    self.memory_agent.clear_cache()

            # Build full prompt
            prompt = self._build_prompt(input_data)

            # Generate video using selected model
            segment = await self._generate_with_model(
                model=input_data.model,
                prompt=prompt,
                duration=input_data.shot.duration,
                prev_frame=input_data.prev_frame,
                seed=input_data.seed,
            )

            # Update segment metadata
            segment.shot_index = input_data.shot.index
            segment.prompt = prompt
            segment.metadata.update({
                "shot_id": str(input_data.shot.id),
                "model": input_data.model.value,
            })

            self.logger.info(
                "Video segment generated",
                shot_index=input_data.shot.index,
                frames=segment.frame_count,
                duration=segment.duration,
            )

            return AgentResult(
                success=True,
                data=segment,
                metadata={
                    "shot_index": input_data.shot.index,
                    "frame_count": segment.frame_count,
                    "duration": segment.duration,
                },
            )

        except Exception as e:
            self.logger.exception("Video generation failed", error=str(e))
            return AgentResult(success=False, error=str(e))

    def _build_prompt(self, input_data: VideoGenInput) -> str:
        """Build complete prompt from shot and context."""
        shot = input_data.shot
        parts = []

        # Shot type and camera movement
        parts.append(f"{shot.shot_type.value} shot")
        if shot.camera_movement.value != "static":
            parts.append(shot.camera_movement.value.replace("_", " "))

        # Location
        if shot.location_id:
            loc_id_str = str(shot.location_id)
            if loc_id_str in input_data.locations:
                parts.append(input_data.locations[loc_id_str].to_prompt())

        # Characters
        for char_id in shot.characters:
            char_id_str = str(char_id)
            if char_id_str in input_data.characters:
                parts.append(input_data.characters[char_id_str].to_prompt())

        # Main shot prompt
        if shot.prompt:
            parts.append(shot.prompt)

        # Action description
        if shot.action_description:
            parts.append(shot.action_description)

        # Style guide
        if input_data.style_guide:
            parts.append(input_data.style_guide.to_prompt_suffix())

        return ", ".join(parts)

    async def _generate_with_model(
        self,
        model: VideoModel,
        prompt: str,
        duration: float,
        prev_frame: Frame | None = None,
        seed: int | None = None,
    ) -> VideoSegment:
        """Generate video using the specified model."""
        if model == VideoModel.WAN21:
            return await self._generate_wan21(prompt, duration, prev_frame, seed)
        elif model == VideoModel.COGVIDEOX:
            return await self._generate_cogvideox(prompt, duration, prev_frame, seed)
        elif model in (VideoModel.SORA, VideoModel.VEO, VideoModel.RUNWAY):
            return await self._generate_commercial(model, prompt, duration, seed)
        else:
            raise ValueError(f"Unsupported model: {model}")

    async def _generate_wan21(
        self,
        prompt: str,
        duration: float,
        prev_frame: Frame | None = None,
        seed: int | None = None,
    ) -> VideoSegment:
        """Generate using Wan2.1 model."""
        from kakushin.integrations.wan21 import Wan21Integration, Wan21Config

        # Lazy load model
        if VideoModel.WAN21 not in self._models:
            config = Wan21Config(
                offload=self.settings.offload_to_cpu,
            )
            self._models[VideoModel.WAN21] = Wan21Integration(
                config=config,
                settings=self.settings,
            )

        integration = self._models[VideoModel.WAN21]

        # Calculate frames from duration
        num_frames = int(duration * 16)  # Wan2.1 default is 16fps

        if prev_frame is not None:
            # Use image-to-video for continuation
            prev_image = Image.fromarray(prev_frame.data)
            segment = integration.image_to_video(
                image=prev_image,
                prompt=prompt,
                num_frames=num_frames,
                seed=seed,
            )
        else:
            # Text-to-video for first segment
            segment = integration.text_to_video(
                prompt=prompt,
                num_frames=num_frames,
                seed=seed,
            )

        return segment

    async def _generate_cogvideox(
        self,
        prompt: str,
        duration: float,
        prev_frame: Frame | None = None,
        seed: int | None = None,
    ) -> VideoSegment:
        """Generate using CogVideoX model."""
        from kakushin.integrations.cogvideox import CogVideoXIntegration, CogVideoXConfig

        # Lazy load model
        if VideoModel.COGVIDEOX not in self._models:
            config = CogVideoXConfig(
                offload=self.settings.offload_to_cpu,
            )
            self._models[VideoModel.COGVIDEOX] = CogVideoXIntegration(
                config=config,
                settings=self.settings,
            )

        integration = self._models[VideoModel.COGVIDEOX]

        # Calculate frames from duration
        num_frames = int(duration * 8)  # CogVideoX default is 8fps

        if prev_frame is not None:
            # Create a temporary segment for continuation
            temp_segment = VideoSegment(frames=[prev_frame])
            segment = integration.continue_video(
                video=temp_segment,
                prompt=prompt,
                num_frames=num_frames,
                seed=seed,
            )
        else:
            segment = integration.text_to_video(
                prompt=prompt,
                num_frames=num_frames,
                seed=seed,
            )

        return segment

    async def _generate_commercial(
        self,
        model: VideoModel,
        prompt: str,
        duration: float,
        seed: int | None = None,
    ) -> VideoSegment:
        """Generate using commercial API (placeholder)."""
        self.logger.warning(
            f"Commercial API {model.value} not yet implemented, using mock"
        )

        # Return mock segment for now
        fps = 24
        num_frames = int(duration * fps)
        frames = []

        for i in range(num_frames):
            # Create placeholder frame
            frame_data = np.zeros((480, 854, 3), dtype=np.uint8)
            frame = Frame(
                data=frame_data,
                timestamp=i / fps,
                frame_number=i,
            )
            frames.append(frame)

        return VideoSegment(
            frames=frames,
            duration=duration,
            fps=fps,
            prompt=prompt,
            metadata={"model": model.value, "mock": True},
        )

    def select_model(self, quality_tier: str = "standard") -> VideoModel:
        """Select the best model based on quality requirements.

        Args:
            quality_tier: "draft", "standard", or "high"

        Returns:
            Selected VideoModel
        """
        if quality_tier == "draft":
            return VideoModel.WAN21  # Fastest, lower quality
        elif quality_tier == "high":
            # Prefer commercial APIs for highest quality
            if self.settings.openai_api_key:
                return VideoModel.SORA
            return VideoModel.COGVIDEOX
        else:
            return VideoModel.WAN21  # Default to OSS

    def unload_models(self) -> None:
        """Unload all loaded models to free memory."""
        for model_type, integration in self._models.items():
            if hasattr(integration, "unload"):
                integration.unload()
                self.logger.info(f"Unloaded model: {model_type.value}")

        self._models.clear()
        self.memory_agent.clear_cache()

    async def validate_input(self, input_data: VideoGenInput) -> bool:
        """Validate input data."""
        if not input_data.shot:
            self.logger.error("Shot is required")
            return False

        if input_data.shot.duration <= 0:
            self.logger.error("Shot duration must be positive")
            return False

        return True
