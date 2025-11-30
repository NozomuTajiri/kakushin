"""QualityAgent - 生成品質を評価し再生成判定を行うエージェント."""

from dataclasses import dataclass, field
from typing import Any

import structlog
import numpy as np
from PIL import Image
import cv2

from kakushin.agents.base import BaseAgent, AgentResult
from kakushin.core.config import Settings
from kakushin.models.video import VideoSegment, FinalVideo, QualityScore

logger = structlog.get_logger()


@dataclass
class Artifact:
    """Detected artifact in a video frame."""

    frame_number: int
    artifact_type: str  # "blur", "noise", "compression", "distortion"
    severity: float  # 0-1
    location: tuple[int, int, int, int] | None = None  # (x, y, w, h)
    description: str = ""


@dataclass
class QualityInput:
    """Input for QualityAgent."""

    segment: VideoSegment | None = None
    video: FinalVideo | None = None
    check_sharpness: bool = True
    check_artifacts: bool = True
    check_motion: bool = True


@dataclass
class QualityResult:
    """Result from QualityAgent."""

    score: QualityScore
    artifacts: list[Artifact] = field(default_factory=list)
    should_regenerate: bool = False
    frame_scores: list[float] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


class QualityAgent(BaseAgent[QualityInput, QualityResult]):
    """Agent that evaluates video quality and determines if regeneration is needed."""

    def __init__(self, settings: Settings | None = None):
        super().__init__(name="QualityAgent", settings=settings)
        self.min_score = settings.min_quality_score if settings else 70

    async def execute(self, input_data: QualityInput) -> AgentResult[QualityResult]:
        """Evaluate video quality."""
        self.logger.info("Evaluating quality")

        try:
            if input_data.segment:
                result = await self._evaluate_segment(input_data)
            elif input_data.video:
                result = await self._evaluate_video(input_data)
            else:
                return AgentResult(success=False, error="No segment or video provided")

            result.should_regenerate = result.score.overall < self.min_score

            self.logger.info(
                "Quality evaluation completed",
                overall_score=result.score.overall,
                should_regenerate=result.should_regenerate,
                artifacts_found=len(result.artifacts),
            )

            return AgentResult(
                success=True,
                data=result,
                metadata={
                    "overall_score": result.score.overall,
                    "should_regenerate": result.should_regenerate,
                },
            )

        except Exception as e:
            self.logger.exception("Quality evaluation failed", error=str(e))
            return AgentResult(success=False, error=str(e))

    async def _evaluate_segment(self, input_data: QualityInput) -> QualityResult:
        """Evaluate quality of a single segment."""
        segment = input_data.segment
        if not segment or not segment.frames:
            return QualityResult(
                score=QualityScore(overall=0),
                report={"error": "No frames to evaluate"},
            )

        artifacts = []
        frame_scores = []
        sharpness_scores = []
        artifact_scores = []
        motion_scores = []

        prev_frame = None
        for frame in segment.frames:
            # Calculate individual metrics
            sharpness = self._calculate_sharpness(frame.data)
            sharpness_scores.append(sharpness)

            if input_data.check_artifacts:
                frame_artifacts = self._detect_artifacts(frame)
                artifacts.extend(frame_artifacts)
                artifact_score = 100 - (len(frame_artifacts) * 20)
                artifact_scores.append(max(0, artifact_score))

            if input_data.check_motion and prev_frame is not None:
                motion = self._evaluate_motion(prev_frame.data, frame.data)
                motion_scores.append(motion)

            # Overall frame score
            frame_score = sharpness
            if artifact_scores:
                frame_score = (frame_score + artifact_scores[-1]) / 2
            frame_scores.append(frame_score)

            prev_frame = frame

        # Calculate aggregate scores
        avg_sharpness = sum(sharpness_scores) / len(sharpness_scores)
        avg_artifact = (
            sum(artifact_scores) / len(artifact_scores) if artifact_scores else 100
        )
        avg_motion = sum(motion_scores) / len(motion_scores) if motion_scores else 80

        # Overall score
        overall = (avg_sharpness * 0.4 + avg_artifact * 0.3 + avg_motion * 0.3)

        score = QualityScore(
            overall=overall,
            sharpness=avg_sharpness,
            artifact_score=100 - avg_artifact,  # Lower is better for artifacts
            motion_smoothness=avg_motion,
            temporal_consistency=avg_motion,  # Reuse motion score
        )

        report = {
            "frame_count": len(segment.frames),
            "avg_sharpness": round(avg_sharpness, 2),
            "avg_artifact_score": round(avg_artifact, 2),
            "avg_motion_smoothness": round(avg_motion, 2),
            "artifacts_detected": len(artifacts),
        }

        return QualityResult(
            score=score,
            artifacts=artifacts,
            frame_scores=frame_scores,
            report=report,
        )

    async def _evaluate_video(self, input_data: QualityInput) -> QualityResult:
        """Evaluate quality of a complete video."""
        video = input_data.video
        if not video or not video.segments:
            return QualityResult(
                score=QualityScore(overall=0),
                report={"error": "No segments to evaluate"},
            )

        all_artifacts = []
        all_frame_scores = []
        segment_scores = []

        for segment in video.segments:
            segment_input = QualityInput(
                segment=segment,
                check_sharpness=input_data.check_sharpness,
                check_artifacts=input_data.check_artifacts,
                check_motion=input_data.check_motion,
            )
            segment_result = await self._evaluate_segment(segment_input)

            all_artifacts.extend(segment_result.artifacts)
            all_frame_scores.extend(segment_result.frame_scores)
            segment_scores.append(segment_result.score.overall)

        # Calculate video-level scores
        avg_overall = sum(segment_scores) / len(segment_scores)
        avg_sharpness = sum(s.sharpness for s in [QualityScore()] + segment_scores) / (
            len(segment_scores) + 1
        )

        score = QualityScore(
            overall=avg_overall,
            sharpness=video.quality_score if video.quality_score else avg_overall,
            temporal_consistency=video.consistency_score
            if video.consistency_score
            else avg_overall,
        )

        report = {
            "segment_count": len(video.segments),
            "total_duration": video.total_duration,
            "segment_scores": [round(s, 2) for s in segment_scores],
            "total_artifacts": len(all_artifacts),
        }

        return QualityResult(
            score=score,
            artifacts=all_artifacts,
            frame_scores=all_frame_scores,
            report=report,
        )

    def _calculate_sharpness(self, frame_data: np.ndarray) -> float:
        """Calculate sharpness score using Laplacian variance."""
        # Convert to grayscale if needed
        if len(frame_data.shape) == 3:
            gray = cv2.cvtColor(frame_data, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame_data

        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize to 0-100 scale
        # Typical sharp image variance is > 500
        sharpness = min(100, (variance / 500) * 100)

        return sharpness

    def _detect_artifacts(self, frame) -> list[Artifact]:
        """Detect visual artifacts in a frame."""
        artifacts = []
        frame_data = frame.data

        # Detect blur
        sharpness = self._calculate_sharpness(frame_data)
        if sharpness < 30:
            artifacts.append(
                Artifact(
                    frame_number=frame.frame_number,
                    artifact_type="blur",
                    severity=(30 - sharpness) / 30,
                    description=f"Blurry frame (sharpness: {sharpness:.1f})",
                )
            )

        # Detect noise (simplified)
        gray = cv2.cvtColor(frame_data, cv2.COLOR_RGB2GRAY)
        noise_level = self._estimate_noise(gray)
        if noise_level > 20:
            artifacts.append(
                Artifact(
                    frame_number=frame.frame_number,
                    artifact_type="noise",
                    severity=min(1.0, noise_level / 50),
                    description=f"Noisy frame (noise: {noise_level:.1f})",
                )
            )

        # Detect compression artifacts (blocking)
        blocking = self._detect_blocking(gray)
        if blocking > 0.3:
            artifacts.append(
                Artifact(
                    frame_number=frame.frame_number,
                    artifact_type="compression",
                    severity=blocking,
                    description=f"Compression artifacts (severity: {blocking:.2f})",
                )
            )

        return artifacts

    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Estimate noise level in grayscale image."""
        # Use median filter to estimate noise
        median = cv2.medianBlur(gray, 5)
        diff = np.abs(gray.astype(np.float32) - median.astype(np.float32))
        noise = np.median(diff)
        return float(noise)

    def _detect_blocking(self, gray: np.ndarray) -> float:
        """Detect JPEG-like blocking artifacts."""
        # Simple blocking detection using gradients at 8x8 boundaries
        h, w = gray.shape

        # Calculate horizontal gradients at 8-pixel boundaries
        h_grads = []
        for x in range(8, w - 8, 8):
            grad = np.abs(
                gray[:, x].astype(np.float32) - gray[:, x - 1].astype(np.float32)
            )
            h_grads.append(np.mean(grad))

        # Calculate vertical gradients
        v_grads = []
        for y in range(8, h - 8, 8):
            grad = np.abs(
                gray[y, :].astype(np.float32) - gray[y - 1, :].astype(np.float32)
            )
            v_grads.append(np.mean(grad))

        if not h_grads or not v_grads:
            return 0.0

        # Compare boundary gradients to non-boundary gradients
        avg_boundary = (np.mean(h_grads) + np.mean(v_grads)) / 2

        # Calculate non-boundary gradients
        non_boundary_h = []
        for x in range(1, w - 1):
            if x % 8 not in (0, 7):
                grad = np.abs(
                    gray[:, x].astype(np.float32) - gray[:, x - 1].astype(np.float32)
                )
                non_boundary_h.append(np.mean(grad))

        if non_boundary_h:
            avg_non_boundary = np.mean(non_boundary_h)
            blocking_ratio = avg_boundary / (avg_non_boundary + 1e-7)
            return min(1.0, max(0.0, (blocking_ratio - 1) / 0.5))

        return 0.0

    def _evaluate_motion(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray
    ) -> float:
        """Evaluate motion smoothness between frames."""
        # Calculate optical flow magnitude
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

        # Use absolute difference as proxy for motion
        diff = cv2.absdiff(prev_gray, curr_gray)
        motion_magnitude = np.mean(diff)

        # Score based on reasonable motion range
        # Too little motion (static) or too much (jumpy) is penalized
        if motion_magnitude < 2:
            score = 90  # Very smooth / static
        elif motion_magnitude < 10:
            score = 100  # Ideal motion
        elif motion_magnitude < 30:
            score = 80  # Acceptable motion
        elif motion_magnitude < 50:
            score = 60  # High motion
        else:
            score = max(20, 100 - motion_magnitude)  # Excessive motion

        return score

    def should_regenerate(self, score: QualityScore) -> bool:
        """Determine if a segment should be regenerated."""
        return score.overall < self.min_score

    def generate_report(self, video: FinalVideo) -> dict[str, Any]:
        """Generate a comprehensive quality report for a video."""
        return {
            "video_id": str(video.id),
            "duration": video.total_duration,
            "segment_count": video.segment_count,
            "quality_score": video.quality_score,
            "consistency_score": video.consistency_score,
            "resolution": video.resolution,
            "fps": video.fps,
        }
