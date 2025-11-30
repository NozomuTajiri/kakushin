"""Video-related data models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray


@dataclass
class Frame:
    """Single video frame."""

    data: NDArray[np.uint8]  # Shape: (H, W, 3) or (H, W, 4)
    timestamp: float  # seconds
    frame_number: int

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def channels(self) -> int:
        return self.data.shape[2] if len(self.data.shape) > 2 else 1


@dataclass
class VideoSegment:
    """A segment of generated video (typically 5-10 seconds)."""

    id: UUID = field(default_factory=uuid4)
    shot_index: int = 0
    frames: list[Frame] = field(default_factory=list)
    duration: float = 0.0  # seconds
    fps: int = 24
    resolution: tuple[int, int] = (480, 854)  # (height, width)
    prompt: str = ""
    file_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def first_frame(self) -> Frame | None:
        return self.frames[0] if self.frames else None

    @property
    def last_frame(self) -> Frame | None:
        return self.frames[-1] if self.frames else None

    def get_frame_at(self, timestamp: float) -> Frame | None:
        """Get frame closest to the given timestamp."""
        if not self.frames:
            return None

        frame_index = int(timestamp * self.fps)
        frame_index = max(0, min(frame_index, len(self.frames) - 1))
        return self.frames[frame_index]


@dataclass
class FinalVideo:
    """Complete generated video."""

    id: UUID = field(default_factory=uuid4)
    segments: list[VideoSegment] = field(default_factory=list)
    total_duration: float = 0.0
    fps: int = 24
    resolution: tuple[int, int] = (1080, 1920)  # (height, width)
    file_path: Path | None = None
    quality_score: float = 0.0
    consistency_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    def calculate_duration(self) -> float:
        """Calculate total duration from segments."""
        return sum(seg.duration for seg in self.segments)


@dataclass
class QualityScore:
    """Quality assessment for a video segment or final video."""

    overall: float = 0.0  # 0-100
    sharpness: float = 0.0
    temporal_consistency: float = 0.0
    artifact_score: float = 0.0  # Lower is better
    motion_smoothness: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_acceptable(self) -> bool:
        """Check if quality meets minimum threshold (70)."""
        return self.overall >= 70


@dataclass
class ConsistencyScore:
    """Consistency assessment across video segments."""

    overall: float = 0.0  # 0-100
    character_consistency: float = 0.0
    style_consistency: float = 0.0
    scene_consistency: float = 0.0
    color_consistency: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_acceptable(self) -> bool:
        """Check if consistency meets minimum threshold (70)."""
        return self.overall >= 70
