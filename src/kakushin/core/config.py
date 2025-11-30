"""Configuration management for Kakushin."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "kakushin"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # API Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # GPU Settings
    gpu_memory_limit: float = Field(
        default=0.92,
        ge=0.5,
        le=1.0,
        description="Maximum GPU memory usage ratio (0.92 = 92%)",
    )
    device: str = "cuda"
    offload_to_cpu: bool = True

    # Model Settings
    default_video_model: Literal["wan21", "cogvideox"] = "wan21"
    model_cache_dir: Path = Path("./models")

    # Video Generation
    default_duration: float = 5.0  # seconds per segment
    default_resolution: tuple[int, int] = (480, 854)  # height, width
    default_fps: int = 24
    max_parallel_segments: int = 2

    # LLM Settings
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"

    # Quality Settings
    min_quality_score: int = 70
    min_consistency_score: int = 70

    # Paths
    output_dir: Path = Path("./output")
    checkpoint_dir: Path = Path("./checkpoints")
    temp_dir: Path = Path("./temp")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
