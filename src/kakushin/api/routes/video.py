"""Video generation API endpoints."""

from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class VideoGenerationRequest(BaseModel):
    """Request model for video generation."""

    scenario: str = Field(..., min_length=10, description="シナリオ/ストーリーの説明")
    duration: int = Field(default=60, ge=10, le=600, description="動画の長さ（秒）")
    resolution: str = Field(default="1080p", description="解像度")
    style: str | None = Field(default=None, description="映像スタイル")


class VideoGenerationResponse(BaseModel):
    """Response model for video generation."""

    job_id: UUID
    status: str
    message: str
    estimated_duration_minutes: float | None = None


class JobStatusResponse(BaseModel):
    """Response model for job status."""

    job_id: UUID
    status: str
    progress: float = Field(ge=0, le=100)
    current_step: str | None = None
    segments_completed: int = 0
    segments_total: int = 0
    output_url: str | None = None
    error: str | None = None


# In-memory job storage (will be replaced with Redis/DB)
_jobs: dict[UUID, JobStatusResponse] = {}


@router.post("/video/generate", response_model=VideoGenerationResponse)
async def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks,
) -> VideoGenerationResponse:
    """Start a new video generation job."""
    job_id = uuid4()

    # Calculate estimated duration (rough estimate)
    # Assume ~2 minutes per 10 seconds of video on RTX 4090
    estimated_minutes = (request.duration / 10) * 2

    # Create job status
    _jobs[job_id] = JobStatusResponse(
        job_id=job_id,
        status="queued",
        progress=0,
        current_step="Initializing",
        segments_total=request.duration // 5,  # 5 seconds per segment
    )

    # TODO: Add to Celery queue
    # background_tasks.add_task(process_video_generation, job_id, request)

    return VideoGenerationResponse(
        job_id=job_id,
        status="queued",
        message="Video generation job created",
        estimated_duration_minutes=estimated_minutes,
    )


@router.get("/video/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: UUID) -> JobStatusResponse:
    """Get the status of a video generation job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return _jobs[job_id]


@router.delete("/video/cancel/{job_id}")
async def cancel_job(job_id: UUID) -> dict[str, str]:
    """Cancel a video generation job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    if job.status in ("completed", "failed", "cancelled"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status}",
        )

    job.status = "cancelled"
    # TODO: Cancel Celery task

    return {"message": f"Job {job_id} cancelled"}


@router.get("/video/jobs")
async def list_jobs(
    status: str | None = None,
    limit: int = 10,
) -> list[JobStatusResponse]:
    """List video generation jobs."""
    jobs = list(_jobs.values())

    if status:
        jobs = [j for j in jobs if j.status == status]

    return jobs[:limit]
