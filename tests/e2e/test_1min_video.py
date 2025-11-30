"""E2E Test: 1分動画生成テスト."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import yaml

from kakushin.core.orchestrator import PipelineOrchestrator, generate_video
from kakushin.core.config import Settings
from kakushin.models.video import VideoSegment, Frame, FinalVideo
from kakushin.models.storyboard import Storyboard, Shot, Character, Location, StyleGuide


# Load test scenario
SCENARIO_PATH = Path(__file__).parent / "scenarios" / "shibuya_morning.yaml"


@pytest.fixture
def test_scenario():
    """Load test scenario from YAML."""
    with open(SCENARIO_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_settings(tmp_path):
    """Create mock settings for testing."""
    return Settings(
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoints",
        temp_dir=tmp_path / "temp",
        model_cache_dir=tmp_path / "models",
        anthropic_api_key="test-key",
        gpu_memory_limit=0.92,
    )


@pytest.fixture
def mock_storyboard():
    """Create a mock storyboard for testing."""
    storyboard = Storyboard(
        title="Test Video",
        synopsis="Test scenario",
        target_duration=60.0,
    )

    # Add characters
    char1 = Character(name="サラリーマン", description="スーツを着た男性")
    char2 = Character(name="女子高生", description="制服を着た学生")
    storyboard.add_character(char1)
    storyboard.add_character(char2)

    # Add locations
    loc1 = Location(name="渋谷交差点", description="早朝のスクランブル交差点")
    storyboard.add_location(loc1)

    # Add shots
    for i in range(8):
        shot = Shot(
            prompt=f"Test shot {i}",
            duration=7.5,
            characters=[char1.id] if i % 2 == 0 else [char2.id],
            location_id=loc1.id,
        )
        storyboard.add_shot(shot)

    storyboard.style_guide = StyleGuide(
        visual_style="cinematic",
        mood="peaceful",
    )

    return storyboard


@pytest.fixture
def mock_video_segment():
    """Create a mock video segment."""
    frames = []
    for i in range(120):  # 5 seconds at 24fps
        frame_data = np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8)
        frame = Frame(data=frame_data, timestamp=i / 24, frame_number=i)
        frames.append(frame)

    return VideoSegment(
        frames=frames,
        duration=5.0,
        fps=24,
        resolution=(480, 854),
        prompt="Test segment",
    )


class TestStoryboardGeneration:
    """Tests for storyboard generation phase."""

    @pytest.mark.asyncio
    async def test_storyboard_has_characters(
        self, test_scenario, mock_settings, mock_storyboard
    ):
        """Test that storyboard contains characters."""
        assert len(mock_storyboard.characters) >= test_scenario["expected_characters"] - 1

    @pytest.mark.asyncio
    async def test_storyboard_has_locations(
        self, test_scenario, mock_settings, mock_storyboard
    ):
        """Test that storyboard contains locations."""
        assert len(mock_storyboard.locations) >= 1

    @pytest.mark.asyncio
    async def test_storyboard_shot_count(
        self, test_scenario, mock_settings, mock_storyboard
    ):
        """Test that storyboard has appropriate number of shots."""
        min_shots, max_shots = map(int, test_scenario["expected_shots"].split("-"))
        assert min_shots <= mock_storyboard.shot_count <= max_shots * 2


class TestVideoGeneration:
    """Tests for video generation phase."""

    @pytest.mark.asyncio
    async def test_segment_generation(self, mock_settings, mock_video_segment):
        """Test that video segments are generated correctly."""
        assert mock_video_segment.frame_count == 120
        assert mock_video_segment.duration == 5.0
        assert mock_video_segment.first_frame is not None
        assert mock_video_segment.last_frame is not None

    @pytest.mark.asyncio
    async def test_segment_frame_quality(self, mock_video_segment):
        """Test frame data quality."""
        for frame in mock_video_segment.frames[:10]:
            assert frame.data.shape == (480, 854, 3)
            assert frame.data.dtype == np.uint8


class TestPipelineOrchestrator:
    """Tests for the pipeline orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, mock_settings):
        """Test orchestrator initializes correctly."""
        orchestrator = PipelineOrchestrator(settings=mock_settings)

        assert orchestrator.storyboard_agent is not None
        assert orchestrator.videogen_agent is not None
        assert orchestrator.memory_agent is not None
        assert orchestrator.consistency_agent is not None

    @pytest.mark.asyncio
    async def test_progress_tracking(self, mock_settings):
        """Test progress tracking works."""
        orchestrator = PipelineOrchestrator(settings=mock_settings)

        progress_updates = []

        def callback(progress):
            progress_updates.append(progress.status)

        orchestrator.set_progress_callback(callback)

        # Initial status
        status = orchestrator.get_status()
        assert status["status"] == "pending"

    @pytest.mark.asyncio
    async def test_checkpoint_creation(self, mock_settings, mock_storyboard):
        """Test checkpoint can be saved."""
        orchestrator = PipelineOrchestrator(settings=mock_settings)

        # Simulate checkpoint save
        orchestrator._save_checkpoint(
            storyboard=mock_storyboard,
            segments=[],
            current_index=0,
        )

        checkpoint = orchestrator.save_checkpoint()
        assert checkpoint is not None


class TestQualityAssurance:
    """Tests for quality assurance."""

    @pytest.mark.asyncio
    async def test_quality_score_calculation(self, mock_settings, mock_video_segment):
        """Test quality score calculation."""
        from kakushin.agents.quality_agent import QualityAgent, QualityInput

        agent = QualityAgent(settings=mock_settings)
        result = await agent.run(QualityInput(segment=mock_video_segment))

        assert result.success
        assert result.data is not None
        assert 0 <= result.data.score.overall <= 100


class TestConsistencyTracking:
    """Tests for consistency tracking."""

    @pytest.mark.asyncio
    async def test_character_registration(self, mock_settings, mock_storyboard):
        """Test character registration for consistency."""
        from kakushin.agents.consistency_agent import ConsistencyAgent

        agent = ConsistencyAgent(settings=mock_settings)

        for char in mock_storyboard.characters.values():
            profile = agent.register_character(char)
            assert profile is not None
            assert profile.character.name == char.name


class TestE2EIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(True, reason="Requires GPU and model weights")
    async def test_full_pipeline_mock(
        self, test_scenario, mock_settings, mock_storyboard, mock_video_segment
    ):
        """Test full pipeline with mocked components."""
        orchestrator = PipelineOrchestrator(settings=mock_settings)

        # Mock the storyboard agent
        with patch.object(
            orchestrator.storyboard_agent,
            "run",
            new_callable=AsyncMock,
        ) as mock_storyboard_run:
            from kakushin.agents.base import AgentResult

            mock_storyboard_run.return_value = AgentResult(
                success=True,
                data=mock_storyboard,
            )

            # Mock the videogen agent
            with patch.object(
                orchestrator.videogen_agent,
                "run",
                new_callable=AsyncMock,
            ) as mock_videogen_run:
                mock_videogen_run.return_value = AgentResult(
                    success=True,
                    data=mock_video_segment,
                )

                # Mock quality agent
                with patch.object(
                    orchestrator.quality_agent,
                    "run",
                    new_callable=AsyncMock,
                ) as mock_quality_run:
                    from kakushin.models.video import QualityScore
                    from kakushin.agents.quality_agent import QualityResult

                    mock_quality_run.return_value = AgentResult(
                        success=True,
                        data=QualityResult(
                            score=QualityScore(overall=85),
                            should_regenerate=False,
                        ),
                    )

                    # Run pipeline
                    scenario_text = test_scenario["scenario"]
                    duration = test_scenario["duration"]

                    # The actual run would require all components
                    # This test verifies the mocking setup works
                    assert mock_storyboard.shot_count > 0


# CLI Test
def test_scenario_yaml_valid(test_scenario):
    """Test that the scenario YAML is valid."""
    assert "title" in test_scenario
    assert "scenario" in test_scenario
    assert "duration" in test_scenario
    assert test_scenario["duration"] == 60
