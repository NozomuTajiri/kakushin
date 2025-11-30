"""Pipeline Orchestrator - 全エージェントを統合し動画生成パイプラインを制御."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

import structlog

from kakushin.agents.storyboard_agent import StoryboardAgent, StoryboardInput
from kakushin.agents.videogen_agent import VideoGenAgent, VideoGenInput, VideoModel
from kakushin.agents.memory_agent import MemoryAgent, MemoryInput
from kakushin.agents.consistency_agent import ConsistencyAgent, ConsistencyInput
from kakushin.agents.transition_agent import TransitionAgent, TransitionInput
from kakushin.agents.quality_agent import QualityAgent, QualityInput
from kakushin.core.config import Settings, get_settings
from kakushin.models.video import VideoSegment, FinalVideo
from kakushin.models.storyboard import Storyboard

logger = structlog.get_logger()


class PipelineStatus(Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineProgress:
    """Progress tracking for pipeline execution."""

    status: PipelineStatus = PipelineStatus.PENDING
    current_phase: str = ""
    current_shot: int = 0
    total_shots: int = 0
    segments_completed: int = 0
    segments_failed: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None

    @property
    def progress_percent(self) -> float:
        if self.total_shots == 0:
            return 0.0
        return (self.segments_completed / self.total_shots) * 100


@dataclass
class Checkpoint:
    """Checkpoint for resuming pipeline execution."""

    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    scenario: str = ""
    storyboard: Storyboard | None = None
    completed_segments: list[VideoSegment] = field(default_factory=list)
    current_shot_index: int = 0
    progress: PipelineProgress = field(default_factory=PipelineProgress)

    def save(self, path: Path) -> None:
        """Save checkpoint to file."""
        data = {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "scenario": self.scenario,
            "current_shot_index": self.current_shot_index,
            "segments_completed": len(self.completed_segments),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        """Load checkpoint from file."""
        data = json.loads(path.read_text())
        return cls(
            id=UUID(data["id"]),
            scenario=data["scenario"],
            current_shot_index=data["current_shot_index"],
        )


class PipelineOrchestrator:
    """Orchestrates the complete video generation pipeline."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.logger = logger.bind(component="Orchestrator")

        # Initialize agents
        self.storyboard_agent = StoryboardAgent(settings=self.settings)
        self.videogen_agent = VideoGenAgent(settings=self.settings)
        self.memory_agent = MemoryAgent(settings=self.settings)
        self.consistency_agent = ConsistencyAgent(settings=self.settings)
        self.transition_agent = TransitionAgent(settings=self.settings)
        self.quality_agent = QualityAgent(settings=self.settings)

        # State
        self._progress = PipelineProgress()
        self._checkpoint: Checkpoint | None = None
        self._is_cancelled = False
        self._progress_callback: Callable[[PipelineProgress], None] | None = None

    @property
    def progress(self) -> PipelineProgress:
        return self._progress

    def set_progress_callback(
        self, callback: Callable[[PipelineProgress], None]
    ) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _update_progress(self, **kwargs) -> None:
        """Update progress and notify callback."""
        for key, value in kwargs.items():
            if hasattr(self._progress, key):
                setattr(self._progress, key, value)

        if self._progress_callback:
            self._progress_callback(self._progress)

    async def run(
        self,
        scenario: str,
        duration: float = 600.0,
        output_path: Path | None = None,
        model: VideoModel = VideoModel.WAN21,
        checkpoint: Checkpoint | None = None,
    ) -> FinalVideo:
        """Run the complete video generation pipeline.

        Args:
            scenario: Text description of the video to generate
            duration: Target duration in seconds (default: 10 minutes)
            output_path: Path to save the final video
            model: Video generation model to use
            checkpoint: Optional checkpoint to resume from

        Returns:
            FinalVideo object containing the generated video
        """
        self.logger.info(
            "Starting pipeline",
            duration=duration,
            model=model.value,
            has_checkpoint=checkpoint is not None,
        )

        self._is_cancelled = False
        self._progress = PipelineProgress(
            status=PipelineStatus.RUNNING,
            start_time=datetime.utcnow(),
        )

        try:
            # Phase 1: Storyboard Generation
            self._update_progress(current_phase="storyboard")

            if checkpoint and checkpoint.storyboard:
                storyboard = checkpoint.storyboard
                self.logger.info("Using storyboard from checkpoint")
            else:
                storyboard = await self._generate_storyboard(scenario, duration)

            self._update_progress(total_shots=storyboard.shot_count)
            self.logger.info(
                "Storyboard ready",
                shots=storyboard.shot_count,
                characters=len(storyboard.characters),
            )

            # Register characters with consistency agent
            for char in storyboard.characters.values():
                self.consistency_agent.register_character(char)
            self.consistency_agent.set_style_guide(storyboard.style_guide)

            # Phase 2: Video Segment Generation
            self._update_progress(current_phase="generation")

            start_index = 0
            completed_segments = []

            if checkpoint:
                start_index = checkpoint.current_shot_index
                completed_segments = checkpoint.completed_segments
                self._update_progress(segments_completed=len(completed_segments))

            segments = await self._generate_segments(
                storyboard=storyboard,
                model=model,
                start_index=start_index,
                existing_segments=completed_segments,
            )

            # Phase 3: Transitions and Final Assembly
            self._update_progress(current_phase="assembly")

            output_path = output_path or (
                self.settings.output_dir / f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )

            final_video = await self._assemble_video(
                segments=segments,
                storyboard=storyboard,
                output_path=output_path,
            )

            # Phase 4: Final Quality Check
            self._update_progress(current_phase="quality_check")

            quality_result = await self.quality_agent.run(
                QualityInput(video=final_video)
            )

            if quality_result.success and quality_result.data:
                final_video.quality_score = quality_result.data.score.overall
                self.logger.info(
                    "Final quality score",
                    score=final_video.quality_score,
                )

            # Complete
            self._update_progress(
                status=PipelineStatus.COMPLETED,
                current_phase="complete",
                end_time=datetime.utcnow(),
            )

            self.logger.info(
                "Pipeline completed",
                duration=final_video.total_duration,
                quality_score=final_video.quality_score,
            )

            return final_video

        except Exception as e:
            self.logger.exception("Pipeline failed", error=str(e))
            self._update_progress(
                status=PipelineStatus.FAILED,
                error=str(e),
                end_time=datetime.utcnow(),
            )
            raise

    async def _generate_storyboard(
        self, scenario: str, duration: float
    ) -> Storyboard:
        """Generate storyboard from scenario."""
        input_data = StoryboardInput(
            scenario=scenario,
            target_duration=duration,
            style="cinematic",
        )

        result = await self.storyboard_agent.run(input_data)

        if not result.success or not result.data:
            raise RuntimeError(f"Storyboard generation failed: {result.error}")

        return result.data

    async def _generate_segments(
        self,
        storyboard: Storyboard,
        model: VideoModel,
        start_index: int = 0,
        existing_segments: list[VideoSegment] | None = None,
    ) -> list[VideoSegment]:
        """Generate video segments for all shots."""
        segments = list(existing_segments or [])
        prev_frame = None

        if segments:
            # Get last frame from existing segments for continuity
            prev_frame = segments[-1].last_frame

        for i, shot in enumerate(storyboard.shots[start_index:], start=start_index):
            if self._is_cancelled:
                self.logger.info("Pipeline cancelled")
                break

            self._update_progress(current_shot=i)

            # Check memory before generation
            memory_result = await self.memory_agent.run(MemoryInput(check_only=True))
            if memory_result.success and memory_result.data:
                if not memory_result.data.is_safe:
                    self.logger.warning("Memory pressure, pausing briefly")
                    self.memory_agent.clear_cache()
                    await asyncio.sleep(1)

            # Generate segment
            segment = await self._generate_single_segment(
                shot=shot,
                storyboard=storyboard,
                model=model,
                prev_frame=prev_frame,
            )

            if segment:
                segments.append(segment)
                prev_frame = segment.last_frame

                # Update consistency references
                self.consistency_agent.update_reference_frame(
                    segment, shot.characters
                )

                self._update_progress(segments_completed=len(segments))

                # Save checkpoint periodically
                if i % 5 == 0:
                    self._save_checkpoint(storyboard, segments, i + 1)

            else:
                self._update_progress(
                    segments_failed=self._progress.segments_failed + 1
                )

        return segments

    async def _generate_single_segment(
        self,
        shot,
        storyboard: Storyboard,
        model: VideoModel,
        prev_frame=None,
        max_retries: int = 2,
    ) -> VideoSegment | None:
        """Generate a single video segment with quality checking and retry."""
        for attempt in range(max_retries + 1):
            self.logger.info(
                f"Generating shot {shot.index}",
                attempt=attempt + 1,
                max_retries=max_retries + 1,
            )

            # Convert character/location dicts for videogen input
            chars_dict = {
                str(k): v for k, v in storyboard.characters.items()
            }
            locs_dict = {
                str(k): v for k, v in storyboard.locations.items()
            }

            input_data = VideoGenInput(
                shot=shot,
                characters=chars_dict,
                locations=locs_dict,
                style_guide=storyboard.style_guide,
                prev_frame=prev_frame,
                model=model,
            )

            result = await self.videogen_agent.run(input_data)

            if not result.success or not result.data:
                self.logger.warning(
                    f"Generation failed for shot {shot.index}",
                    error=result.error,
                )
                continue

            segment = result.data

            # Quality check
            quality_result = await self.quality_agent.run(
                QualityInput(segment=segment)
            )

            if quality_result.success and quality_result.data:
                if not quality_result.data.should_regenerate:
                    return segment
                else:
                    self.logger.warning(
                        f"Quality too low for shot {shot.index}",
                        score=quality_result.data.score.overall,
                    )

        self.logger.error(f"Failed to generate acceptable segment for shot {shot.index}")
        return None

    async def _assemble_video(
        self,
        segments: list[VideoSegment],
        storyboard: Storyboard,
        output_path: Path,
    ) -> FinalVideo:
        """Assemble segments into final video."""
        input_data = TransitionInput(
            segments=segments,
            shots=storyboard.shots,
            output_path=output_path,
            fps=self.settings.default_fps,
        )

        result = await self.transition_agent.run(input_data)

        if not result.success or not result.data:
            raise RuntimeError(f"Video assembly failed: {result.error}")

        return result.data.final_video

    def _save_checkpoint(
        self,
        storyboard: Storyboard,
        segments: list[VideoSegment],
        current_index: int,
    ) -> None:
        """Save checkpoint for resumption."""
        checkpoint = Checkpoint(
            scenario=storyboard.synopsis,
            storyboard=storyboard,
            completed_segments=segments,
            current_shot_index=current_index,
            progress=self._progress,
        )

        checkpoint_path = (
            self.settings.checkpoint_dir / f"checkpoint_{checkpoint.id}.json"
        )
        checkpoint.save(checkpoint_path)
        self._checkpoint = checkpoint

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def save_checkpoint(self) -> Checkpoint | None:
        """Get the current checkpoint."""
        return self._checkpoint

    def cancel(self) -> None:
        """Cancel the running pipeline."""
        self._is_cancelled = True
        self._update_progress(status=PipelineStatus.CANCELLED)
        self.logger.info("Pipeline cancellation requested")

    def pause(self) -> None:
        """Pause the pipeline (saves checkpoint)."""
        self._update_progress(status=PipelineStatus.PAUSED)
        self.logger.info("Pipeline paused")

    def get_status(self) -> dict[str, Any]:
        """Get current pipeline status."""
        return {
            "status": self._progress.status.value,
            "phase": self._progress.current_phase,
            "progress_percent": self._progress.progress_percent,
            "current_shot": self._progress.current_shot,
            "total_shots": self._progress.total_shots,
            "segments_completed": self._progress.segments_completed,
            "segments_failed": self._progress.segments_failed,
            "start_time": (
                self._progress.start_time.isoformat()
                if self._progress.start_time
                else None
            ),
            "error": self._progress.error,
        }


async def generate_video(
    scenario: str,
    duration: float = 600.0,
    output_path: str | Path | None = None,
    model: str = "wan21",
    settings: Settings | None = None,
) -> FinalVideo:
    """Convenience function to generate a video.

    Args:
        scenario: Text description of the video
        duration: Target duration in seconds
        output_path: Path to save the video
        model: Model to use ("wan21" or "cogvideox")
        settings: Optional settings override

    Returns:
        FinalVideo object
    """
    orchestrator = PipelineOrchestrator(settings=settings)

    video_model = VideoModel.WAN21 if model == "wan21" else VideoModel.COGVIDEOX

    return await orchestrator.run(
        scenario=scenario,
        duration=duration,
        output_path=Path(output_path) if output_path else None,
        model=video_model,
    )
