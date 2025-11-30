"""TransitionAgent - セグメント間をシームレスに結合するエージェント."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import subprocess
import tempfile

import structlog
import numpy as np
from PIL import Image

from kakushin.agents.base import BaseAgent, AgentResult
from kakushin.core.config import Settings
from kakushin.models.video import VideoSegment, Frame, FinalVideo
from kakushin.models.storyboard import TransitionType, Shot

logger = structlog.get_logger()


@dataclass
class TransitionInput:
    """Input for TransitionAgent."""

    segments: list[VideoSegment]
    shots: list[Shot] | None = None  # For transition type decisions
    output_path: Path | None = None
    fps: int = 24
    resolution: tuple[int, int] = (1080, 1920)  # (height, width)


@dataclass
class TransitionResult:
    """Result from TransitionAgent."""

    final_video: FinalVideo
    output_path: Path | None = None


class TransitionAgent(BaseAgent[TransitionInput, TransitionResult]):
    """Agent that blends video segments together seamlessly."""

    def __init__(self, settings: Settings | None = None):
        super().__init__(name="TransitionAgent", settings=settings)

    async def execute(
        self, input_data: TransitionInput
    ) -> AgentResult[TransitionResult]:
        """Concatenate and blend video segments."""
        self.logger.info(
            "Processing transitions",
            segment_count=len(input_data.segments),
            target_fps=input_data.fps,
        )

        try:
            if not input_data.segments:
                return AgentResult(success=False, error="No segments provided")

            # Process segments with transitions
            all_frames = []
            prev_segment = None

            for i, segment in enumerate(input_data.segments):
                # Determine transition type
                transition_type = TransitionType.CUT
                if input_data.shots and i > 0 and i - 1 < len(input_data.shots):
                    transition_type = input_data.shots[i - 1].transition_out

                # Apply transition from previous segment
                if prev_segment is not None:
                    transition_frames = await self._create_transition(
                        prev_segment,
                        segment,
                        transition_type,
                        input_data.fps,
                    )
                    all_frames.extend(transition_frames)

                # Add current segment frames (skip first few if transition)
                start_idx = 0 if prev_segment is None else 0
                for frame in segment.frames[start_idx:]:
                    # Normalize frame timestamp
                    new_frame = Frame(
                        data=frame.data,
                        timestamp=len(all_frames) / input_data.fps,
                        frame_number=len(all_frames),
                    )
                    all_frames.append(new_frame)

                prev_segment = segment

            # Create final video
            total_duration = len(all_frames) / input_data.fps

            final_video = FinalVideo(
                segments=input_data.segments,
                total_duration=total_duration,
                fps=input_data.fps,
                resolution=input_data.resolution,
            )

            # Export to file if output path specified
            output_path = None
            if input_data.output_path:
                output_path = await self._export_video(
                    all_frames,
                    input_data.output_path,
                    input_data.fps,
                )
                final_video.file_path = output_path

            result = TransitionResult(
                final_video=final_video,
                output_path=output_path,
            )

            self.logger.info(
                "Transitions completed",
                total_frames=len(all_frames),
                duration=total_duration,
            )

            return AgentResult(
                success=True,
                data=result,
                metadata={
                    "total_frames": len(all_frames),
                    "duration": total_duration,
                },
            )

        except Exception as e:
            self.logger.exception("Transition processing failed", error=str(e))
            return AgentResult(success=False, error=str(e))

    async def _create_transition(
        self,
        segment_a: VideoSegment,
        segment_b: VideoSegment,
        transition_type: TransitionType,
        fps: int,
    ) -> list[Frame]:
        """Create transition frames between two segments."""
        if transition_type == TransitionType.CUT:
            return []  # No transition frames needed

        elif transition_type == TransitionType.CROSSFADE:
            return self._create_crossfade(segment_a, segment_b, fps)

        elif transition_type == TransitionType.FADE_TO_BLACK:
            return self._create_fade_to_black(segment_a, segment_b, fps)

        elif transition_type == TransitionType.DISSOLVE:
            return self._create_dissolve(segment_a, segment_b, fps)

        else:
            return []  # Default to cut

    def _create_crossfade(
        self,
        segment_a: VideoSegment,
        segment_b: VideoSegment,
        fps: int,
        duration: float = 0.5,
    ) -> list[Frame]:
        """Create crossfade transition frames."""
        if not segment_a.frames or not segment_b.frames:
            return []

        num_frames = int(duration * fps)
        frames = []

        frame_a = segment_a.last_frame
        frame_b = segment_b.first_frame

        if frame_a is None or frame_b is None:
            return []

        # Resize if needed
        img_a = Image.fromarray(frame_a.data)
        img_b = Image.fromarray(frame_b.data)

        if img_a.size != img_b.size:
            img_b = img_b.resize(img_a.size)

        arr_a = np.array(img_a).astype(np.float32)
        arr_b = np.array(img_b).astype(np.float32)

        for i in range(num_frames):
            alpha = i / (num_frames - 1)
            blended = (1 - alpha) * arr_a + alpha * arr_b
            blended = blended.astype(np.uint8)

            frame = Frame(
                data=blended,
                timestamp=i / fps,
                frame_number=i,
            )
            frames.append(frame)

        return frames

    def _create_fade_to_black(
        self,
        segment_a: VideoSegment,
        segment_b: VideoSegment,
        fps: int,
        duration: float = 0.5,
    ) -> list[Frame]:
        """Create fade to black transition."""
        if not segment_a.frames or not segment_b.frames:
            return []

        num_frames = int(duration * fps)
        half_frames = num_frames // 2
        frames = []

        frame_a = segment_a.last_frame
        frame_b = segment_b.first_frame

        if frame_a is None or frame_b is None:
            return []

        arr_a = np.array(Image.fromarray(frame_a.data)).astype(np.float32)
        arr_b = np.array(Image.fromarray(frame_b.data)).astype(np.float32)

        # Fade out to black
        for i in range(half_frames):
            alpha = 1 - (i / half_frames)
            faded = (arr_a * alpha).astype(np.uint8)
            frames.append(Frame(data=faded, timestamp=i / fps, frame_number=i))

        # Fade in from black
        for i in range(half_frames):
            alpha = i / half_frames
            faded = (arr_b * alpha).astype(np.uint8)
            frames.append(
                Frame(
                    data=faded,
                    timestamp=(half_frames + i) / fps,
                    frame_number=half_frames + i,
                )
            )

        return frames

    def _create_dissolve(
        self,
        segment_a: VideoSegment,
        segment_b: VideoSegment,
        fps: int,
        duration: float = 0.75,
    ) -> list[Frame]:
        """Create dissolve transition (similar to crossfade but longer)."""
        return self._create_crossfade(segment_a, segment_b, fps, duration)

    def interpolate_frames(
        self,
        frame_a: Frame,
        frame_b: Frame,
        num_intermediate: int = 2,
    ) -> list[Frame]:
        """Generate intermediate frames between two frames."""
        frames = []

        arr_a = frame_a.data.astype(np.float32)
        arr_b = frame_b.data.astype(np.float32)

        for i in range(1, num_intermediate + 1):
            alpha = i / (num_intermediate + 1)
            interpolated = ((1 - alpha) * arr_a + alpha * arr_b).astype(np.uint8)
            frames.append(
                Frame(
                    data=interpolated,
                    timestamp=0,  # Will be set later
                    frame_number=0,
                )
            )

        return frames

    def decide_transition(
        self,
        shot_a: Shot,
        shot_b: Shot,
    ) -> TransitionType:
        """Decide the best transition type between two shots."""
        # Same location -> cut or crossfade
        if shot_a.location_id == shot_b.location_id:
            return TransitionType.CUT

        # Different location with time jump -> fade to black
        # (simplified heuristic)
        if shot_a.shot_type != shot_b.shot_type:
            return TransitionType.CROSSFADE

        # Default to cut
        return TransitionType.CUT

    async def _export_video(
        self,
        frames: list[Frame],
        output_path: Path,
        fps: int,
    ) -> Path:
        """Export frames to video file using FFmpeg."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save frames as images
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame.data)
                img.save(temp_path / f"frame_{i:06d}.png")

            # Use FFmpeg to create video
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-framerate",
                str(fps),
                "-i",
                str(temp_path / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(output_path),
            ]

            self.logger.info(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")

            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        self.logger.info(f"Video exported to {output_path}")
        return output_path

    async def concatenate_videos(
        self,
        video_paths: list[Path],
        output_path: Path,
    ) -> Path:
        """Concatenate multiple video files using FFmpeg."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for path in video_paths:
                f.write(f"file '{path}'\n")
            list_file = f.name

        try:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file,
                "-c",
                "copy",
                str(output_path),
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

        finally:
            Path(list_file).unlink()

        return output_path
