"""Storyboard and scene-related data models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class TransitionType(Enum):
    """Types of transitions between shots."""

    CUT = "cut"
    CROSSFADE = "crossfade"
    FADE_TO_BLACK = "fade_to_black"
    DISSOLVE = "dissolve"
    WIPE = "wipe"


class CameraMovement(Enum):
    """Types of camera movements."""

    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    TRACKING = "tracking"
    CRANE = "crane"


class ShotType(Enum):
    """Types of camera shots."""

    EXTREME_WIDE = "extreme_wide"
    WIDE = "wide"
    FULL = "full"
    MEDIUM = "medium"
    MEDIUM_CLOSE = "medium_close"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"
    OVER_SHOULDER = "over_shoulder"
    POV = "pov"


@dataclass
class Character:
    """Character definition for consistency tracking."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""  # Physical appearance description
    clothing: str = ""
    age_range: str = ""
    gender: str = ""
    distinctive_features: list[str] = field(default_factory=list)
    embedding: list[float] | None = None  # Feature vector for consistency

    def to_prompt(self) -> str:
        """Convert character to prompt-friendly description."""
        parts = [self.description]
        if self.clothing:
            parts.append(f"wearing {self.clothing}")
        if self.distinctive_features:
            parts.append(", ".join(self.distinctive_features))
        return ", ".join(parts)


@dataclass
class Location:
    """Location/setting definition."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    time_of_day: str = "day"  # day, night, dawn, dusk
    weather: str = "clear"
    lighting: str = ""
    props: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert location to prompt-friendly description."""
        parts = [self.description]
        if self.time_of_day:
            parts.append(f"{self.time_of_day}time")
        if self.weather != "clear":
            parts.append(self.weather)
        if self.lighting:
            parts.append(f"{self.lighting} lighting")
        return ", ".join(parts)


@dataclass
class StyleGuide:
    """Visual style guide for consistency."""

    color_palette: list[str] = field(default_factory=list)
    lighting_style: str = "natural"
    mood: str = "neutral"
    film_grain: bool = False
    aspect_ratio: str = "16:9"
    visual_style: str = "cinematic"  # cinematic, anime, realistic, etc.
    reference_images: list[str] = field(default_factory=list)

    def to_prompt_suffix(self) -> str:
        """Generate style prompt suffix."""
        parts = [self.visual_style]
        if self.lighting_style:
            parts.append(f"{self.lighting_style} lighting")
        if self.mood:
            parts.append(f"{self.mood} mood")
        if self.film_grain:
            parts.append("film grain")
        return ", ".join(parts)


@dataclass
class Shot:
    """Single shot in the storyboard."""

    id: UUID = field(default_factory=uuid4)
    index: int = 0
    prompt: str = ""
    duration: float = 5.0  # seconds
    shot_type: ShotType = ShotType.MEDIUM
    camera_movement: CameraMovement = CameraMovement.STATIC
    transition_in: TransitionType = TransitionType.CUT
    transition_out: TransitionType = TransitionType.CUT
    characters: list[UUID] = field(default_factory=list)  # Character IDs
    location_id: UUID | None = None
    action_description: str = ""
    dialogue: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def build_full_prompt(
        self,
        characters: dict[UUID, Character],
        locations: dict[UUID, Location],
        style_guide: StyleGuide | None = None,
    ) -> str:
        """Build complete prompt including all context."""
        parts = []

        # Shot type
        parts.append(f"{self.shot_type.value} shot")

        # Camera movement
        if self.camera_movement != CameraMovement.STATIC:
            parts.append(self.camera_movement.value.replace("_", " "))

        # Location
        if self.location_id and self.location_id in locations:
            parts.append(locations[self.location_id].to_prompt())

        # Characters
        for char_id in self.characters:
            if char_id in characters:
                parts.append(characters[char_id].to_prompt())

        # Main prompt
        parts.append(self.prompt)

        # Action
        if self.action_description:
            parts.append(self.action_description)

        # Style
        if style_guide:
            parts.append(style_guide.to_prompt_suffix())

        return ", ".join(parts)


@dataclass
class Storyboard:
    """Complete storyboard for video generation."""

    id: UUID = field(default_factory=uuid4)
    title: str = ""
    synopsis: str = ""  # Original scenario
    shots: list[Shot] = field(default_factory=list)
    characters: dict[UUID, Character] = field(default_factory=dict)
    locations: dict[UUID, Location] = field(default_factory=dict)
    style_guide: StyleGuide = field(default_factory=StyleGuide)
    target_duration: float = 600.0  # 10 minutes in seconds
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shot_count(self) -> int:
        return len(self.shots)

    @property
    def estimated_duration(self) -> float:
        return sum(shot.duration for shot in self.shots)

    def get_shot(self, index: int) -> Shot | None:
        """Get shot by index."""
        if 0 <= index < len(self.shots):
            return self.shots[index]
        return None

    def add_character(self, character: Character) -> None:
        """Add a character to the storyboard."""
        self.characters[character.id] = character

    def add_location(self, location: Location) -> None:
        """Add a location to the storyboard."""
        self.locations[location.id] = location

    def add_shot(self, shot: Shot) -> None:
        """Add a shot to the storyboard."""
        shot.index = len(self.shots)
        self.shots.append(shot)
