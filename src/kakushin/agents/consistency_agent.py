"""ConsistencyAgent - キャラクター・スタイルの一貫性を維持するエージェント."""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import structlog
import numpy as np
from PIL import Image

from kakushin.agents.base import BaseAgent, AgentResult
from kakushin.core.config import Settings
from kakushin.models.video import VideoSegment, ConsistencyScore
from kakushin.models.storyboard import Character, StyleGuide

logger = structlog.get_logger()


@dataclass
class CharacterProfile:
    """Stored profile for a character with consistency tracking."""

    character: Character
    embedding: np.ndarray | None = None
    reference_frames: list[np.ndarray] = field(default_factory=list)
    appearance_count: int = 0
    consistency_history: list[float] = field(default_factory=list)

    @property
    def average_consistency(self) -> float:
        """Get average consistency score."""
        if not self.consistency_history:
            return 100.0
        return sum(self.consistency_history) / len(self.consistency_history)


@dataclass
class SceneGraphNode:
    """Node in the scene graph tracking spatial relationships."""

    entity_id: str
    entity_type: str  # "character", "object", "location"
    position: tuple[float, float] | None = None  # normalized (0-1)
    size: tuple[float, float] | None = None
    last_seen_frame: int = 0


@dataclass
class ConsistencyInput:
    """Input for ConsistencyAgent."""

    segment: VideoSegment
    character_ids: list[UUID] = field(default_factory=list)
    check_style: bool = True
    check_characters: bool = True
    check_scene: bool = True


@dataclass
class ConsistencyResult:
    """Result from ConsistencyAgent."""

    score: ConsistencyScore
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class ConsistencyAgent(BaseAgent[ConsistencyInput, ConsistencyResult]):
    """Agent that maintains visual consistency across video segments."""

    def __init__(self, settings: Settings | None = None):
        super().__init__(name="ConsistencyAgent", settings=settings)

        # Character memory
        self._character_profiles: dict[UUID, CharacterProfile] = {}

        # Scene graph for spatial tracking
        self._scene_graph: dict[str, SceneGraphNode] = {}

        # Style guide reference
        self._style_guide: StyleGuide | None = None
        self._style_reference_frames: list[np.ndarray] = []

        # Color histogram reference for style consistency
        self._color_reference: np.ndarray | None = None

    async def execute(
        self, input_data: ConsistencyInput
    ) -> AgentResult[ConsistencyResult]:
        """Check consistency of a video segment."""
        self.logger.info(
            "Checking consistency",
            shot_index=input_data.segment.shot_index,
            frame_count=input_data.segment.frame_count,
        )

        try:
            issues = []
            recommendations = []

            # Initialize scores
            character_score = 100.0
            style_score = 100.0
            scene_score = 100.0
            color_score = 100.0

            # Check character consistency
            if input_data.check_characters and input_data.character_ids:
                char_result = await self._check_character_consistency(
                    input_data.segment, input_data.character_ids
                )
                character_score = char_result["score"]
                issues.extend(char_result["issues"])

            # Check style consistency
            if input_data.check_style and self._style_guide:
                style_result = await self._check_style_consistency(input_data.segment)
                style_score = style_result["score"]
                color_score = style_result["color_score"]
                issues.extend(style_result["issues"])

            # Check scene consistency
            if input_data.check_scene:
                scene_result = await self._check_scene_consistency(input_data.segment)
                scene_score = scene_result["score"]
                issues.extend(scene_result["issues"])

            # Calculate overall score
            overall_score = (
                character_score * 0.4
                + style_score * 0.2
                + scene_score * 0.2
                + color_score * 0.2
            )

            # Generate recommendations if score is low
            if overall_score < 70:
                recommendations.append("Consider regenerating this segment")
            if character_score < 70:
                recommendations.append(
                    "Character appearance differs significantly from reference"
                )
            if style_score < 70:
                recommendations.append("Visual style deviates from style guide")
            if color_score < 70:
                recommendations.append("Color palette inconsistent with previous segments")

            score = ConsistencyScore(
                overall=overall_score,
                character_consistency=character_score,
                style_consistency=style_score,
                scene_consistency=scene_score,
                color_consistency=color_score,
            )

            result = ConsistencyResult(
                score=score,
                issues=issues,
                recommendations=recommendations,
            )

            self.logger.info(
                "Consistency check completed",
                overall_score=overall_score,
                issues_count=len(issues),
            )

            return AgentResult(
                success=True,
                data=result,
                metadata={"overall_score": overall_score},
            )

        except Exception as e:
            self.logger.exception("Consistency check failed", error=str(e))
            return AgentResult(success=False, error=str(e))

    def register_character(self, character: Character) -> CharacterProfile:
        """Register a character for consistency tracking."""
        profile = CharacterProfile(character=character)
        self._character_profiles[character.id] = profile
        self.logger.info(f"Registered character: {character.name}")
        return profile

    def get_character_profile(self, char_id: UUID) -> CharacterProfile | None:
        """Get a character profile by ID."""
        return self._character_profiles.get(char_id)

    def set_style_guide(self, style_guide: StyleGuide) -> None:
        """Set the style guide for consistency checking."""
        self._style_guide = style_guide
        self.logger.info("Style guide set")

    def update_reference_frame(
        self,
        segment: VideoSegment,
        character_ids: list[UUID] | None = None,
    ) -> None:
        """Update reference frames from a good segment."""
        if not segment.frames:
            return

        # Use middle frame as reference
        mid_idx = len(segment.frames) // 2
        ref_frame = segment.frames[mid_idx].data

        # Update style reference
        self._style_reference_frames.append(ref_frame)
        if len(self._style_reference_frames) > 10:
            self._style_reference_frames.pop(0)

        # Update color reference
        self._update_color_reference(ref_frame)

        # Update character references
        if character_ids:
            for char_id in character_ids:
                if char_id in self._character_profiles:
                    profile = self._character_profiles[char_id]
                    profile.reference_frames.append(ref_frame)
                    if len(profile.reference_frames) > 5:
                        profile.reference_frames.pop(0)
                    profile.appearance_count += 1

        self.logger.info("Reference frames updated")

    async def _check_character_consistency(
        self,
        segment: VideoSegment,
        character_ids: list[UUID],
    ) -> dict[str, Any]:
        """Check if characters appear consistent with their profiles."""
        issues = []
        scores = []

        for char_id in character_ids:
            profile = self._character_profiles.get(char_id)
            if not profile:
                continue

            if not profile.reference_frames:
                # First appearance, use as reference
                scores.append(100.0)
                continue

            # Compare current frames with reference
            # This is a simplified check - in production, use proper
            # face/person detection and embedding comparison
            similarity = self._compute_frame_similarity(
                segment.frames[len(segment.frames) // 2].data,
                profile.reference_frames[-1],
            )

            score = min(100.0, similarity * 100)
            scores.append(score)
            profile.consistency_history.append(score)

            if score < 70:
                issues.append(
                    f"Character '{profile.character.name}' appearance inconsistent "
                    f"(score: {score:.1f})"
                )

        avg_score = sum(scores) / len(scores) if scores else 100.0
        return {"score": avg_score, "issues": issues}

    async def _check_style_consistency(
        self, segment: VideoSegment
    ) -> dict[str, Any]:
        """Check if visual style is consistent."""
        issues = []
        style_score = 100.0
        color_score = 100.0

        if not segment.frames:
            return {"score": style_score, "color_score": color_score, "issues": issues}

        current_frame = segment.frames[len(segment.frames) // 2].data

        # Check color consistency
        if self._color_reference is not None:
            current_hist = self._compute_color_histogram(current_frame)
            color_similarity = self._compare_histograms(
                current_hist, self._color_reference
            )
            color_score = min(100.0, color_similarity * 100)

            if color_score < 70:
                issues.append(f"Color palette deviation (score: {color_score:.1f})")

        # Check style reference similarity
        if self._style_reference_frames:
            similarities = []
            for ref_frame in self._style_reference_frames[-3:]:
                sim = self._compute_frame_similarity(current_frame, ref_frame)
                similarities.append(sim)

            style_score = min(100.0, max(similarities) * 100)

            if style_score < 70:
                issues.append(f"Visual style deviation (score: {style_score:.1f})")

        return {"score": style_score, "color_score": color_score, "issues": issues}

    async def _check_scene_consistency(
        self, segment: VideoSegment
    ) -> dict[str, Any]:
        """Check scene/spatial consistency."""
        # Simplified scene consistency check
        # In production, track objects and their positions across frames

        issues = []
        score = 100.0

        if not segment.frames or len(segment.frames) < 2:
            return {"score": score, "issues": issues}

        # Check for sudden jumps between consecutive frames
        prev_frame = segment.frames[0].data
        jump_count = 0

        for frame in segment.frames[1:]:
            diff = self._compute_frame_difference(prev_frame, frame.data)
            if diff > 0.5:  # Large difference threshold
                jump_count += 1
            prev_frame = frame.data

        jump_ratio = jump_count / len(segment.frames)
        score = max(0, 100 - jump_ratio * 200)

        if score < 70:
            issues.append(f"Scene instability detected (jump ratio: {jump_ratio:.2f})")

        return {"score": score, "issues": issues}

    def _compute_frame_similarity(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> float:
        """Compute similarity between two frames (0-1)."""
        # Resize to same size if needed
        if frame1.shape != frame2.shape:
            frame2 = np.array(
                Image.fromarray(frame2).resize(
                    (frame1.shape[1], frame1.shape[0])
                )
            )

        # Compute normalized cross-correlation
        f1 = frame1.astype(np.float32) / 255.0
        f2 = frame2.astype(np.float32) / 255.0

        diff = np.abs(f1 - f2).mean()
        similarity = 1.0 - diff

        return max(0.0, min(1.0, similarity))

    def _compute_frame_difference(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> float:
        """Compute difference between frames (0-1)."""
        return 1.0 - self._compute_frame_similarity(frame1, frame2)

    def _compute_color_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute color histogram for a frame."""
        # Convert to HSV and compute histogram
        from PIL import Image

        img = Image.fromarray(frame).convert("HSV")
        hsv = np.array(img)

        # Compute histograms for H, S, V channels
        h_hist, _ = np.histogram(hsv[:, :, 0], bins=18, range=(0, 180))
        s_hist, _ = np.histogram(hsv[:, :, 1], bins=8, range=(0, 256))
        v_hist, _ = np.histogram(hsv[:, :, 2], bins=8, range=(0, 256))

        # Normalize
        hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        hist = hist / (hist.sum() + 1e-7)

        return hist

    def _compare_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compare two histograms (0-1 similarity)."""
        # Use histogram intersection
        intersection = np.minimum(hist1, hist2).sum()
        return float(intersection)

    def _update_color_reference(self, frame: np.ndarray) -> None:
        """Update color reference histogram."""
        new_hist = self._compute_color_histogram(frame)

        if self._color_reference is None:
            self._color_reference = new_hist
        else:
            # Exponential moving average
            alpha = 0.3
            self._color_reference = (
                alpha * new_hist + (1 - alpha) * self._color_reference
            )

    def get_style_guide(self) -> StyleGuide | None:
        """Get current style guide."""
        return self._style_guide

    def update_scene_graph(self, segment: VideoSegment) -> None:
        """Update scene graph with entities from segment."""
        # Placeholder for proper object tracking
        if segment.frames:
            last_frame_num = segment.frames[-1].frame_number
            for entity_id, node in self._scene_graph.items():
                node.last_seen_frame = last_frame_num

    def get_status(self) -> dict[str, Any]:
        """Get current consistency tracking status."""
        return {
            "registered_characters": len(self._character_profiles),
            "style_references": len(self._style_reference_frames),
            "scene_entities": len(self._scene_graph),
            "has_style_guide": self._style_guide is not None,
            "has_color_reference": self._color_reference is not None,
        }
