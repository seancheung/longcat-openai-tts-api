from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_LONGCAT_ROOT = Path(os.environ.get("LONGCAT_SRC", "/opt/api/LongCat-AudioDiT"))
if _LONGCAT_ROOT.exists():
    sys.path.insert(0, str(_LONGCAT_ROOT))


class TTSEngine:
    def __init__(self, settings):
        import torch
        import torch.nn.functional as F
        import audiodit  # noqa: F401  registers AudioDiTModel with transformers
        from audiodit import AudioDiTModel
        from transformers import AutoTokenizer
        from utils import approx_duration_from_text, load_audio, normalize_text

        self._torch = torch
        self._F = F
        self._load_audio = load_audio
        self._normalize_text = normalize_text
        self._approx_duration_from_text = approx_duration_from_text

        self.settings = settings

        if settings.longcat_cache_dir:
            os.environ.setdefault("HF_HOME", settings.longcat_cache_dir)

        device = settings.resolved_device
        dtype = settings.resolved_dtype

        log.info(
            "loading LongCat-AudioDiT model=%s device=%s dtype=%s",
            settings.longcat_model, device, dtype,
        )

        self.model = AudioDiTModel.from_pretrained(
            settings.longcat_model, torch_dtype=dtype
        ).to(device)
        self.model.vae.to_half()
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder_model)

        self.device = device
        self.dtype = str(dtype).replace("torch.", "")
        self.sample_rate = int(self.model.config.sampling_rate)
        self.latent_hop = int(self.model.config.latent_hop)
        self.max_wav_duration = float(self.model.config.max_wav_duration)

        self._lock = asyncio.Lock()
        self._prompt_cache: "OrderedDict[Tuple[str, float], Tuple]" = OrderedDict()
        self._prompt_cache_size = int(settings.longcat_prompt_cache_size)

    # ------------------------------------------------------------------
    # prompt audio LRU
    # ------------------------------------------------------------------
    def _encode_prompt(self, wav_path: str):
        torch = self._torch
        F = self._F
        full_hop = self.latent_hop

        # load_audio returns (1, T); model() expects (B, 1, T).
        prompt_wav = self._load_audio(wav_path, self.sample_rate).unsqueeze(0)

        # Separate pad+encode pass mirroring LongCat-AudioDiT/inference.py.
        off = 3
        pw = self._load_audio(wav_path, self.sample_rate)  # (1, T)
        if pw.shape[-1] % full_hop != 0:
            pw = F.pad(pw, (0, full_hop - pw.shape[-1] % full_hop))
        pw = F.pad(pw, (0, full_hop * off))
        with torch.no_grad():
            plt = self.model.vae.encode(pw.unsqueeze(0).to(self.device))
        if off:
            plt = plt[..., :-off]
        prompt_dur = int(plt.shape[-1])
        return prompt_wav, prompt_dur

    def _get_prompt(self, wav_path: str, mtime: float):
        if self._prompt_cache_size <= 0:
            return self._encode_prompt(wav_path)
        key = (wav_path, mtime)
        if key in self._prompt_cache:
            self._prompt_cache.move_to_end(key)
            return self._prompt_cache[key]
        value = self._encode_prompt(wav_path)
        self._prompt_cache[key] = value
        while len(self._prompt_cache) > self._prompt_cache_size:
            self._prompt_cache.popitem(last=False)
        return value

    # ------------------------------------------------------------------
    # seed
    # ------------------------------------------------------------------
    def _apply_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = self.settings.longcat_seed
        if seed is None:
            return
        torch = self._torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed))

    # ------------------------------------------------------------------
    # duration estimation (mirrors inference.py)
    # ------------------------------------------------------------------
    def _estimate_duration_frames(
        self,
        gen_text: str,
        *,
        prompt_text: Optional[str],
        prompt_dur_frames: int,
        override_seconds: Optional[float],
        speed: float,
    ) -> int:
        prompt_time = prompt_dur_frames * self.latent_hop / self.sample_rate
        if override_seconds is not None:
            dur_sec = float(override_seconds)
        else:
            dur_sec = self._approx_duration_from_text(
                gen_text, max_duration=max(0.5, self.max_wav_duration - prompt_time)
            )
            if prompt_text:
                approx_pd = self._approx_duration_from_text(
                    prompt_text, max_duration=self.max_wav_duration
                )
                if approx_pd > 0:
                    ratio = float(np.clip(prompt_time / approx_pd, 1.0, 1.5))
                    dur_sec = dur_sec * ratio

        if speed and speed > 0:
            dur_sec = dur_sec / float(speed)

        frames = int(dur_sec * self.sample_rate // self.latent_hop)
        frames = min(
            frames + prompt_dur_frames,
            int(self.max_wav_duration * self.sample_rate // self.latent_hop),
        )
        return max(1, frames)

    # ------------------------------------------------------------------
    # inference entrypoints
    # ------------------------------------------------------------------
    def _run_sync(
        self,
        *,
        full_text: str,
        prompt_wav,
        duration_frames: int,
        steps: int,
        cfg_strength: float,
        guidance_method: str,
        seed: Optional[int],
    ) -> np.ndarray:
        torch = self._torch

        self._apply_seed(seed)
        inputs = self.tokenizer([full_text], padding="longest", return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        if prompt_wav is not None:
            prompt_wav = prompt_wav.to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_audio=prompt_wav,
                duration=duration_frames,
                steps=steps,
                cfg_strength=cfg_strength,
                guidance_method=guidance_method,
            )

        wav = output.waveform.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
        return np.ascontiguousarray(wav)

    async def synthesize_clone(
        self,
        text: str,
        *,
        prompt_wav_path: str,
        prompt_mtime: float,
        prompt_text: str,
        duration: Optional[float] = None,
        speed: float = 1.0,
        steps: Optional[int] = None,
        cfg_strength: Optional[float] = None,
        guidance_method: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = self.settings
        gen_text = self._normalize_text(text)
        ref_text = self._normalize_text(prompt_text)
        full_text = f"{ref_text} {gen_text}"

        prompt_wav, prompt_dur = self._get_prompt(prompt_wav_path, prompt_mtime)
        duration_frames = self._estimate_duration_frames(
            gen_text,
            prompt_text=ref_text,
            prompt_dur_frames=prompt_dur,
            override_seconds=duration,
            speed=speed,
        )

        async with self._lock:
            return await asyncio.to_thread(
                self._run_sync,
                full_text=full_text,
                prompt_wav=prompt_wav,
                duration_frames=duration_frames,
                steps=steps if steps is not None else s.longcat_steps,
                cfg_strength=cfg_strength if cfg_strength is not None else s.longcat_cfg_strength,
                guidance_method=guidance_method or s.longcat_guidance_method_clone,
                seed=seed,
            )

    async def synthesize_zeroshot(
        self,
        text: str,
        *,
        duration: Optional[float] = None,
        speed: float = 1.0,
        steps: Optional[int] = None,
        cfg_strength: Optional[float] = None,
        guidance_method: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = self.settings
        gen_text = self._normalize_text(text)

        duration_frames = self._estimate_duration_frames(
            gen_text,
            prompt_text=None,
            prompt_dur_frames=0,
            override_seconds=duration,
            speed=speed,
        )

        async with self._lock:
            return await asyncio.to_thread(
                self._run_sync,
                full_text=gen_text,
                prompt_wav=None,
                duration_frames=duration_frames,
                steps=steps if steps is not None else s.longcat_steps,
                cfg_strength=cfg_strength if cfg_strength is not None else s.longcat_cfg_strength,
                guidance_method=guidance_method or s.longcat_guidance_method_zeroshot,
                seed=seed,
            )
