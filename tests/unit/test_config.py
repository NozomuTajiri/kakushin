"""Tests for configuration module."""

import pytest
from kakushin.core.config import Settings, get_settings


def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()

    assert settings.app_name == "kakushin"
    assert settings.app_version == "0.1.0"
    assert settings.gpu_memory_limit == 0.92
    assert settings.default_video_model == "wan21"
    assert settings.min_quality_score == 70


def test_settings_gpu_memory_limit_bounds():
    """Test GPU memory limit bounds validation."""
    # Valid value
    settings = Settings(gpu_memory_limit=0.85)
    assert settings.gpu_memory_limit == 0.85

    # Test minimum bound
    with pytest.raises(ValueError):
        Settings(gpu_memory_limit=0.4)  # Below 0.5

    # Test maximum bound
    with pytest.raises(ValueError):
        Settings(gpu_memory_limit=1.1)  # Above 1.0


def test_get_settings_cached():
    """Test that get_settings returns cached instance."""
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2


def test_settings_directories_created(tmp_path, monkeypatch):
    """Test that settings creates required directories."""
    monkeypatch.chdir(tmp_path)

    settings = Settings(
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoints",
        temp_dir=tmp_path / "temp",
        model_cache_dir=tmp_path / "models",
    )

    assert settings.output_dir.exists()
    assert settings.checkpoint_dir.exists()
    assert settings.temp_dir.exists()
    assert settings.model_cache_dir.exists()
