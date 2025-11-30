"""Wan2.1 Video Generation Model Integration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import torch
import numpy as np
from PIL import Image

from kakushin.core.config import Settings, get_settings
from kakushin.models.video import VideoSegment, Frame
from kakushin.utils.gpu_monitor import GPUMonitor, get_device

logger = structlog.get_logger()


@dataclass
class Wan21Config:
    """Configuration for Wan2.1 model."""

    model_path: str = "Wan-AI/Wan2.1-T2V-1.3B"
    device: str = "cuda"
    offload: bool = True
    dtype: str = "float16"
    num_frames: int = 81  # ~5 seconds at 16fps
    height: int = 480
    width: int = 854
    fps: int = 16
    guidance_scale: float = 6.0
    num_inference_steps: int = 50


class Wan21Integration:
    """Integration with Wan2.1 video generation model."""

    def __init__(
        self,
        config: Wan21Config | None = None,
        settings: Settings | None = None,
    ):
        self.config = config or Wan21Config()
        self.settings = settings or get_settings()
        self.logger = logger.bind(component="Wan21")
        self.gpu_monitor = GPUMonitor()

        self._pipeline = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def load(self) -> None:
        """Load the Wan2.1 model."""
        if self._is_loaded:
            self.logger.info("Model already loaded")
            return

        self.logger.info("Loading Wan2.1 model", model=self.config.model_path)

        try:
            # Import diffusers components
            from diffusers import DiffusionPipeline
            from diffusers.utils import export_to_video

            # Determine dtype
            dtype = torch.float16 if self.config.dtype == "float16" else torch.float32

            # Load pipeline
            self._pipeline = DiffusionPipeline.from_pretrained(
                self.config.model_path,
                torch_dtype=dtype,
            )

            # Enable memory optimizations
            if self.config.offload:
                self._pipeline.enable_model_cpu_offload()
            else:
                self._pipeline.to(get_device())

            # Enable memory efficient attention if available
            try:
                self._pipeline.enable_xformers_memory_efficient_attention()
                self.logger.info("xformers memory efficient attention enabled")
            except Exception:
                self.logger.info("xformers not available, using default attention")

            self._is_loaded = True
            self.logger.info("Wan2.1 model loaded successfully")

        except ImportError as e:
            self.logger.error(f"Failed to import required modules: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._is_loaded = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info("Wan2.1 model unloaded")

    def text_to_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        guidance_scale: float | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
    ) -> VideoSegment:
        """Generate video from text prompt.

        Args:
            prompt: Text description of the video
            negative_prompt: What to avoid in the video
            num_frames: Number of frames to generate
            height: Video height
            width: Video width
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility

        Returns:
            VideoSegment containing the generated video
        """
        if not self._is_loaded:
            self.load()

        # Use config defaults if not specified
        num_frames = num_frames or self.config.num_frames
        height = height or self.config.height
        width = width or self.config.width
        guidance_scale = guidance_scale or self.config.guidance_scale
        num_inference_steps = num_inference_steps or self.config.num_inference_steps

        self.logger.info(
            "Generating video",
            prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
            num_frames=num_frames,
            resolution=f"{width}x{height}",
        )

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=get_device()).manual_seed(seed)

        # Check GPU memory before generation
        if self.gpu_monitor.should_pause():
            self.logger.warning("GPU memory critical, clearing cache before generation")
            torch.cuda.empty_cache()

        # Generate video
        output = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "blurry, low quality, distorted",
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        # Convert to VideoSegment
        frames = self._convert_output_to_frames(output.frames[0])
        duration = len(frames) / self.config.fps

        segment = VideoSegment(
            frames=frames,
            duration=duration,
            fps=self.config.fps,
            resolution=(height, width),
            prompt=prompt,
            metadata={
                "model": "wan2.1",
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
            },
        )

        self.logger.info(
            "Video generated",
            frames=len(frames),
            duration=duration,
        )

        return segment

    def image_to_video(
        self,
        image: Image.Image | np.ndarray | str | Path,
        prompt: str,
        num_frames: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
    ) -> VideoSegment:
        """Generate video from an input image.

        Args:
            image: Input image (PIL Image, numpy array, or path)
            prompt: Text description for video motion
            num_frames: Number of frames to generate
            guidance_scale: Classifier-free guidance scale
            seed: Random seed

        Returns:
            VideoSegment containing the generated video
        """
        if not self._is_loaded:
            self.load()

        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        num_frames = num_frames or self.config.num_frames
        guidance_scale = guidance_scale or self.config.guidance_scale

        self.logger.info(
            "Generating video from image",
            prompt=prompt[:50] + "...",
            num_frames=num_frames,
        )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=get_device()).manual_seed(seed)

        # Note: I2V support depends on the specific Wan2.1 variant
        # This is a placeholder for the actual I2V implementation
        output = self._pipeline(
            prompt=prompt,
            image=image,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        frames = self._convert_output_to_frames(output.frames[0])
        duration = len(frames) / self.config.fps

        return VideoSegment(
            frames=frames,
            duration=duration,
            fps=self.config.fps,
            prompt=prompt,
            metadata={"model": "wan2.1-i2v", "seed": seed},
        )

    def _convert_output_to_frames(
        self, output_frames: list[Image.Image] | np.ndarray
    ) -> list[Frame]:
        """Convert model output to Frame objects."""
        frames = []

        if isinstance(output_frames, np.ndarray):
            # Shape: (num_frames, H, W, C)
            for i, frame_data in enumerate(output_frames):
                frame = Frame(
                    data=frame_data.astype(np.uint8),
                    timestamp=i / self.config.fps,
                    frame_number=i,
                )
                frames.append(frame)
        else:
            # List of PIL Images
            for i, pil_image in enumerate(output_frames):
                frame_data = np.array(pil_image)
                frame = Frame(
                    data=frame_data,
                    timestamp=i / self.config.fps,
                    frame_number=i,
                )
                frames.append(frame)

        return frames

    def save_video(
        self,
        segment: VideoSegment,
        output_path: str | Path,
        fps: int | None = None,
    ) -> Path:
        """Save VideoSegment to file.

        Args:
            segment: VideoSegment to save
            output_path: Output file path
            fps: Frames per second (uses segment fps if not specified)

        Returns:
            Path to saved video file
        """
        from diffusers.utils import export_to_video

        fps = fps or segment.fps
        output_path = Path(output_path)

        # Convert frames to PIL images
        pil_frames = [Image.fromarray(f.data) for f in segment.frames]

        # Export to video
        export_to_video(pil_frames, str(output_path), fps=fps)

        self.logger.info(f"Video saved to {output_path}")
        return output_path

    def get_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "model_name": "Wan2.1",
            "model_path": self.config.model_path,
            "is_loaded": self._is_loaded,
            "config": {
                "device": self.config.device,
                "offload": self.config.offload,
                "dtype": self.config.dtype,
                "default_resolution": f"{self.config.width}x{self.config.height}",
                "default_fps": self.config.fps,
            },
        }
