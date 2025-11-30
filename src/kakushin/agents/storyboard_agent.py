"""StoryboardAgent - シナリオをショット単位に分解するエージェント."""

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import structlog
from anthropic import Anthropic

from kakushin.agents.base import BaseAgent, AgentResult
from kakushin.core.config import Settings
from kakushin.models.storyboard import (
    Storyboard,
    Shot,
    Character,
    Location,
    StyleGuide,
    ShotType,
    CameraMovement,
    TransitionType,
)

logger = structlog.get_logger()


@dataclass
class StoryboardInput:
    """Input for StoryboardAgent."""

    scenario: str
    target_duration: float = 600.0  # 10 minutes default
    style: str = "cinematic"
    language: str = "ja"  # ja or en


STORYBOARD_SYSTEM_PROMPT = """You are a professional film director and screenwriter AI.
Your task is to break down a scenario into detailed shots for AI video generation.

Rules:
1. Each shot should be 5-10 seconds long
2. Total duration should match the target duration
3. Maintain visual and narrative consistency
4. Describe each shot with specific visual details
5. Include camera movements and shot types
6. Define all characters with consistent appearances
7. Define all locations with atmospheric details

Output Format (JSON):
{
  "title": "Movie Title",
  "characters": [
    {
      "name": "Character Name",
      "description": "Physical appearance",
      "clothing": "What they wear",
      "age_range": "20s/30s/etc",
      "distinctive_features": ["feature1", "feature2"]
    }
  ],
  "locations": [
    {
      "name": "Location Name",
      "description": "Visual description",
      "time_of_day": "day/night/dawn/dusk",
      "weather": "clear/cloudy/rainy",
      "lighting": "natural/dramatic/soft"
    }
  ],
  "style_guide": {
    "visual_style": "cinematic/anime/realistic",
    "color_palette": ["color1", "color2"],
    "lighting_style": "natural/dramatic",
    "mood": "tense/calm/exciting"
  },
  "shots": [
    {
      "index": 0,
      "prompt": "Detailed visual description",
      "duration": 5.0,
      "shot_type": "wide/medium/close_up/etc",
      "camera_movement": "static/pan_left/zoom_in/etc",
      "characters": ["Character Name"],
      "location": "Location Name",
      "action": "What happens in this shot",
      "transition_out": "cut/crossfade"
    }
  ]
}
"""


class StoryboardAgent(BaseAgent[StoryboardInput, Storyboard]):
    """Agent that breaks down scenarios into detailed storyboards."""

    def __init__(self, settings: Settings | None = None):
        super().__init__(name="StoryboardAgent", settings=settings)
        self.client = Anthropic(api_key=settings.anthropic_api_key if settings else "")

    async def execute(self, input_data: StoryboardInput) -> AgentResult[Storyboard]:
        """Execute storyboard generation."""
        self.logger.info(
            "Generating storyboard",
            scenario_length=len(input_data.scenario),
            target_duration=input_data.target_duration,
        )

        try:
            # Calculate target number of shots
            avg_shot_duration = 7.5  # seconds
            target_shots = int(input_data.target_duration / avg_shot_duration)

            # Generate storyboard using LLM
            user_prompt = f"""
Scenario:
{input_data.scenario}

Requirements:
- Target duration: {input_data.target_duration} seconds ({input_data.target_duration / 60:.1f} minutes)
- Target number of shots: {target_shots} (approximately)
- Visual style: {input_data.style}
- Language for prompts: {"Japanese" if input_data.language == "ja" else "English"}

Please create a detailed storyboard with all shots, characters, and locations.
Return ONLY valid JSON, no additional text.
"""

            response = self.client.messages.create(
                model=self.settings.llm_model,
                max_tokens=8000,
                system=STORYBOARD_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Parse response
            response_text = response.content[0].text
            storyboard = self._parse_response(response_text, input_data)

            self.logger.info(
                "Storyboard generated",
                shot_count=storyboard.shot_count,
                character_count=len(storyboard.characters),
                location_count=len(storyboard.locations),
            )

            return AgentResult(
                success=True,
                data=storyboard,
                metadata={
                    "shot_count": storyboard.shot_count,
                    "estimated_duration": storyboard.estimated_duration,
                },
            )

        except Exception as e:
            self.logger.exception("Failed to generate storyboard", error=str(e))
            return AgentResult(success=False, error=str(e))

    def _parse_response(
        self, response_text: str, input_data: StoryboardInput
    ) -> Storyboard:
        """Parse LLM response into Storyboard object."""
        import json

        # Clean response text (remove markdown code blocks if present)
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data = json.loads(text)

        # Create storyboard
        storyboard = Storyboard(
            title=data.get("title", "Untitled"),
            synopsis=input_data.scenario,
            target_duration=input_data.target_duration,
        )

        # Parse style guide
        if "style_guide" in data:
            sg = data["style_guide"]
            storyboard.style_guide = StyleGuide(
                visual_style=sg.get("visual_style", "cinematic"),
                color_palette=sg.get("color_palette", []),
                lighting_style=sg.get("lighting_style", "natural"),
                mood=sg.get("mood", "neutral"),
            )

        # Parse characters
        char_name_to_id = {}
        for char_data in data.get("characters", []):
            char = Character(
                id=uuid4(),
                name=char_data.get("name", "Unknown"),
                description=char_data.get("description", ""),
                clothing=char_data.get("clothing", ""),
                age_range=char_data.get("age_range", ""),
                distinctive_features=char_data.get("distinctive_features", []),
            )
            storyboard.add_character(char)
            char_name_to_id[char.name] = char.id

        # Parse locations
        loc_name_to_id = {}
        for loc_data in data.get("locations", []):
            loc = Location(
                id=uuid4(),
                name=loc_data.get("name", "Unknown"),
                description=loc_data.get("description", ""),
                time_of_day=loc_data.get("time_of_day", "day"),
                weather=loc_data.get("weather", "clear"),
                lighting=loc_data.get("lighting", "natural"),
            )
            storyboard.add_location(loc)
            loc_name_to_id[loc.name] = loc.id

        # Parse shots
        for shot_data in data.get("shots", []):
            # Map character names to IDs
            char_ids = []
            for char_name in shot_data.get("characters", []):
                if char_name in char_name_to_id:
                    char_ids.append(char_name_to_id[char_name])

            # Map location name to ID
            loc_name = shot_data.get("location", "")
            loc_id = loc_name_to_id.get(loc_name)

            shot = Shot(
                id=uuid4(),
                prompt=shot_data.get("prompt", ""),
                duration=shot_data.get("duration", 5.0),
                shot_type=self._parse_shot_type(shot_data.get("shot_type", "medium")),
                camera_movement=self._parse_camera_movement(
                    shot_data.get("camera_movement", "static")
                ),
                transition_out=self._parse_transition(
                    shot_data.get("transition_out", "cut")
                ),
                characters=char_ids,
                location_id=loc_id,
                action_description=shot_data.get("action", ""),
            )
            storyboard.add_shot(shot)

        return storyboard

    def _parse_shot_type(self, value: str) -> ShotType:
        """Parse shot type string to enum."""
        mapping = {
            "extreme_wide": ShotType.EXTREME_WIDE,
            "wide": ShotType.WIDE,
            "full": ShotType.FULL,
            "medium": ShotType.MEDIUM,
            "medium_close": ShotType.MEDIUM_CLOSE,
            "close_up": ShotType.CLOSE_UP,
            "extreme_close_up": ShotType.EXTREME_CLOSE_UP,
            "over_shoulder": ShotType.OVER_SHOULDER,
            "pov": ShotType.POV,
        }
        return mapping.get(value.lower(), ShotType.MEDIUM)

    def _parse_camera_movement(self, value: str) -> CameraMovement:
        """Parse camera movement string to enum."""
        mapping = {
            "static": CameraMovement.STATIC,
            "pan_left": CameraMovement.PAN_LEFT,
            "pan_right": CameraMovement.PAN_RIGHT,
            "tilt_up": CameraMovement.TILT_UP,
            "tilt_down": CameraMovement.TILT_DOWN,
            "zoom_in": CameraMovement.ZOOM_IN,
            "zoom_out": CameraMovement.ZOOM_OUT,
            "dolly_in": CameraMovement.DOLLY_IN,
            "dolly_out": CameraMovement.DOLLY_OUT,
            "tracking": CameraMovement.TRACKING,
            "crane": CameraMovement.CRANE,
        }
        return mapping.get(value.lower(), CameraMovement.STATIC)

    def _parse_transition(self, value: str) -> TransitionType:
        """Parse transition type string to enum."""
        mapping = {
            "cut": TransitionType.CUT,
            "crossfade": TransitionType.CROSSFADE,
            "fade_to_black": TransitionType.FADE_TO_BLACK,
            "dissolve": TransitionType.DISSOLVE,
            "wipe": TransitionType.WIPE,
        }
        return mapping.get(value.lower(), TransitionType.CUT)

    async def validate_input(self, input_data: StoryboardInput) -> bool:
        """Validate input data."""
        if not input_data.scenario or len(input_data.scenario) < 10:
            self.logger.error("Scenario too short")
            return False

        if input_data.target_duration < 10 or input_data.target_duration > 1200:
            self.logger.error("Invalid target duration")
            return False

        return True
