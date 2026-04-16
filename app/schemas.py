from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
GuidanceMethod = Literal["cfg", "apg"]


class SpeechRequest(BaseModel):
    """OpenAI-compatible `/v1/audio/speech` request (voice cloning)."""

    model: Optional[str] = Field(default=None, description="Accepted for OpenAI compatibility; ignored.")
    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(..., description="Voice id matching a file pair in the voices directory.")
    response_format: ResponseFormat = Field(default="mp3")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    duration: Optional[float] = Field(default=None, ge=0.5, le=60.0, description="Optional target duration in seconds; overrides auto-estimate.")
    steps: Optional[int] = Field(default=None, ge=1, le=100)
    cfg_strength: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    guidance_method: Optional[GuidanceMethod] = Field(default=None)
    seed: Optional[int] = Field(default=None)


class ZeroShotRequest(BaseModel):
    """`/v1/audio/zeroshot` request — no reference audio, pure text-to-speech."""

    model: Optional[str] = Field(default=None)
    input: str = Field(..., description="Text to synthesize.")
    response_format: ResponseFormat = Field(default="mp3")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    duration: Optional[float] = Field(default=None, ge=0.5, le=60.0)
    steps: Optional[int] = Field(default=None, ge=1, le=100)
    cfg_strength: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    guidance_method: Optional[GuidanceMethod] = Field(default=None)
    seed: Optional[int] = Field(default=None)


class VoiceInfo(BaseModel):
    id: str
    preview_url: str
    prompt_text: str


class VoiceList(BaseModel):
    object: Literal["list"] = "list"
    data: list[VoiceInfo]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model: str
    device: Optional[str] = None
    dtype: Optional[str] = None
    sample_rate: Optional[int] = None
