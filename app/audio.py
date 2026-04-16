from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import soundfile as sf


CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "application/octet-stream",
}


def _to_mono_float32(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[1] else arr.mean(axis=1)
    elif arr.ndim > 2:
        arr = arr.reshape(-1)
    return np.clip(arr.astype(np.float32, copy=False), -1.0, 1.0)


def encode(samples: np.ndarray, sample_rate: int, fmt: str) -> Tuple[bytes, str]:
    fmt = fmt.lower()
    if fmt not in CONTENT_TYPES:
        raise ValueError(f"unsupported response_format: {fmt}")

    samples = _to_mono_float32(samples)

    if fmt == "wav":
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), CONTENT_TYPES[fmt]

    if fmt == "flac":
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="FLAC")
        return buf.getvalue(), CONTENT_TYPES[fmt]

    if fmt == "pcm":
        pcm = (samples * 32767.0).astype("<i2").tobytes()
        return pcm, CONTENT_TYPES[fmt]

    return _encode_compressed(samples, sample_rate, fmt), CONTENT_TYPES[fmt]


def _encode_compressed(samples: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    import av

    codec_map = {
        "mp3": ("mp3", "mp3", "s16p"),
        "opus": ("libopus", "ogg", "flt"),
        "aac": ("aac", "adts", "fltp"),
    }
    codec, container_fmt, sample_fmt = codec_map[fmt]

    buf = io.BytesIO()
    with av.open(buf, mode="w", format=container_fmt) as container:
        stream = container.add_stream(codec, rate=sample_rate)
        stream.layout = "mono"
        if hasattr(stream, "format"):
            try:
                stream.format = sample_fmt
            except Exception:
                pass

        layout = "mono"
        data = samples.reshape(1, -1)

        if sample_fmt.startswith("s16"):
            pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
            frame = av.AudioFrame.from_ndarray(pcm16.reshape(1, -1), format=sample_fmt, layout=layout)
        else:
            frame = av.AudioFrame.from_ndarray(data, format=sample_fmt, layout=layout)
        frame.sample_rate = sample_rate
        frame.pts = None

        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    return buf.getvalue()
