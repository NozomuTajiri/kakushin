"""Tests for data models."""

import numpy as np
import pytest

from kakushin.models.video import (
    Frame,
    VideoSegment,
    FinalVideo,
    QualityScore,
    ConsistencyScore,
)
from kakushin.models.storyboard import (
    Character,
    Location,
    Shot,
    Storyboard,
    StyleGuide,
    ShotType,
    CameraMovement,
    TransitionType,
)


class TestFrame:
    """Tests for Frame model."""

    def test_frame_creation(self):
        """Test creating a frame."""
        data = np.zeros((480, 854, 3), dtype=np.uint8)
        frame = Frame(data=data, timestamp=0.0, frame_number=0)

        assert frame.height == 480
        assert frame.width == 854
        assert frame.channels == 3

    def test_frame_single_channel(self):
        """Test frame with single channel."""
        data = np.zeros((480, 854), dtype=np.uint8)
        frame = Frame(data=data, timestamp=0.0, frame_number=0)

        assert frame.channels == 1


class TestVideoSegment:
    """Tests for VideoSegment model."""

    def test_segment_creation(self):
        """Test creating a video segment."""
        segment = VideoSegment(
            shot_index=0,
            duration=5.0,
            fps=24,
            resolution=(480, 854),
            prompt="Test prompt",
        )

        assert segment.shot_index == 0
        assert segment.duration == 5.0
        assert segment.frame_count == 0

    def test_segment_with_frames(self):
        """Test segment with frames."""
        segment = VideoSegment()
        data = np.zeros((480, 854, 3), dtype=np.uint8)

        for i in range(24):  # 1 second of frames
            frame = Frame(data=data, timestamp=i / 24, frame_number=i)
            segment.frames.append(frame)

        assert segment.frame_count == 24
        assert segment.first_frame is not None
        assert segment.last_frame is not None

    def test_get_frame_at(self):
        """Test getting frame at timestamp."""
        segment = VideoSegment(fps=24)
        data = np.zeros((480, 854, 3), dtype=np.uint8)

        for i in range(24):
            frame = Frame(data=data, timestamp=i / 24, frame_number=i)
            segment.frames.append(frame)

        # Get frame at 0.5 seconds (frame 12)
        frame = segment.get_frame_at(0.5)
        assert frame is not None
        assert frame.frame_number == 12


class TestStoryboard:
    """Tests for Storyboard model."""

    def test_storyboard_creation(self):
        """Test creating a storyboard."""
        storyboard = Storyboard(
            title="Test Movie",
            synopsis="A test scenario",
            target_duration=600.0,
        )

        assert storyboard.shot_count == 0
        assert storyboard.estimated_duration == 0.0

    def test_add_character(self):
        """Test adding characters."""
        storyboard = Storyboard()
        character = Character(
            name="John",
            description="A tall man with brown hair",
            clothing="blue suit",
        )

        storyboard.add_character(character)
        assert character.id in storyboard.characters

    def test_add_shot(self):
        """Test adding shots."""
        storyboard = Storyboard()
        shot = Shot(
            prompt="A wide shot of the city",
            duration=5.0,
            shot_type=ShotType.WIDE,
        )

        storyboard.add_shot(shot)
        assert storyboard.shot_count == 1
        assert shot.index == 0
        assert storyboard.estimated_duration == 5.0


class TestCharacter:
    """Tests for Character model."""

    def test_character_to_prompt(self):
        """Test character prompt generation."""
        character = Character(
            name="Jane",
            description="A young woman",
            clothing="red dress",
            distinctive_features=["glasses", "short hair"],
        )

        prompt = character.to_prompt()
        assert "young woman" in prompt
        assert "red dress" in prompt
        assert "glasses" in prompt


class TestStyleGuide:
    """Tests for StyleGuide model."""

    def test_style_guide_prompt_suffix(self):
        """Test style guide prompt generation."""
        style = StyleGuide(
            visual_style="cinematic",
            lighting_style="dramatic",
            mood="tense",
            film_grain=True,
        )

        suffix = style.to_prompt_suffix()
        assert "cinematic" in suffix
        assert "dramatic" in suffix
        assert "film grain" in suffix


class TestQualityScore:
    """Tests for QualityScore model."""

    def test_quality_acceptable(self):
        """Test quality threshold check."""
        good_score = QualityScore(overall=80)
        assert good_score.is_acceptable

        bad_score = QualityScore(overall=60)
        assert not bad_score.is_acceptable


class TestConsistencyScore:
    """Tests for ConsistencyScore model."""

    def test_consistency_acceptable(self):
        """Test consistency threshold check."""
        good_score = ConsistencyScore(overall=75)
        assert good_score.is_acceptable

        bad_score = ConsistencyScore(overall=65)
        assert not bad_score.is_acceptable
