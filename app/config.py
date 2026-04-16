from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False, extra="ignore")

    longcat_model: str = Field(default="meituan-longcat/LongCat-AudioDiT-1B")
    longcat_device: Literal["auto", "cuda", "cpu"] = Field(default="auto")
    longcat_cuda_index: int = Field(default=0)
    longcat_dtype: Literal["float16", "bfloat16", "float32"] = Field(default="float32")
    longcat_cache_dir: Optional[str] = Field(default=None)

    longcat_voices_dir: str = Field(default="/voices")
    longcat_prompt_cache_size: int = Field(default=32, ge=0)

    longcat_steps: int = Field(default=16, ge=1, le=100)
    longcat_cfg_strength: float = Field(default=4.0, ge=0.0, le=20.0)
    longcat_guidance_method_clone: Literal["cfg", "apg"] = Field(default="apg")
    longcat_guidance_method_zeroshot: Literal["cfg", "apg"] = Field(default="cfg")
    longcat_seed: Optional[int] = Field(default=None)

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")
    max_input_chars: int = Field(default=8000)
    default_response_format: Literal[
        "mp3", "opus", "aac", "flac", "wav", "pcm"
    ] = Field(default="mp3")

    @property
    def voices_path(self) -> Path:
        return Path(self.longcat_voices_dir)

    @property
    def resolved_device(self) -> str:
        import torch

        if self.longcat_device == "auto":
            if torch.cuda.is_available():
                return f"cuda:{self.longcat_cuda_index}"
            return "cpu"
        if self.longcat_device == "cuda":
            return f"cuda:{self.longcat_cuda_index}"
        return self.longcat_device

    @property
    def resolved_dtype(self):
        import torch

        device = self.resolved_device
        if device.startswith("cpu") and self.longcat_dtype == "float16":
            return torch.float32
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.longcat_dtype]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
