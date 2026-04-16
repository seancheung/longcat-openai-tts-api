# LongCat-AudioDiT OpenAI-TTS API

**English** · [中文](./README.zh.md)

An [OpenAI TTS](https://platform.openai.com/docs/api-reference/audio/createSpeech)-compatible HTTP service wrapping [LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT) — Meituan's diffusion transformer (DiT) TTS that operates directly in the waveform latent space, supporting zero-shot Chinese/English synthesis and voice cloning via a short reference audio.

## Features

- **OpenAI TTS compatible** — `POST /v1/audio/speech` with the same request shape as the OpenAI SDK
- **Voice cloning** — each voice is a `xxx.wav` + `xxx.txt` pair in a mounted directory; the DiT continues the reference to reproduce timbre and prosody
- **Zero-shot TTS** — extra `POST /v1/audio/zeroshot` endpoint synthesizes from text only, no reference audio needed
- **2 images** — `cuda` and `cpu`
- **Model weights downloaded at runtime** — nothing heavy baked into the image; HuggingFace cache is mounted for reuse
- **Multiple output formats** — `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm` (24 kHz mono)

## Available images

| Image | Device |
|---|---|
| `ghcr.io/seancheung/longcat-openai-tts-api:cuda-latest` | CUDA 12.4 |
| `ghcr.io/seancheung/longcat-openai-tts-api:latest`      | CPU |

Images are built for `linux/amd64`.

## Quick start

### 1. Prepare the voices directory

```
voices/
├── alice.wav     # reference audio, mono, 24kHz recommended (anything ≥16kHz is resampled), ~3-20s
├── alice.txt     # UTF-8 text: the exact transcript of alice.wav
├── bob.wav
└── bob.txt
```

**Rules**: a voice is valid only when both files with the same stem exist; the stem is the voice id; unpaired or extra files are ignored. Voices are used by `/v1/audio/speech` (voice cloning); `/v1/audio/zeroshot` does not need the `voices/` directory.

### 2. Run the container

GPU (recommended):

```bash
docker run --rm -p 8000:8000 --gpus all \
  -v $PWD/voices:/voices:ro \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  ghcr.io/seancheung/longcat-openai-tts-api:cuda-latest
```

CPU:

```bash
docker run --rm -p 8000:8000 \
  -v $PWD/voices:/voices:ro \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  ghcr.io/seancheung/longcat-openai-tts-api:latest
```

Model weights (≈4 GB for `LongCat-AudioDiT-1B`, ≈14 GB for the 3.5B variant) are pulled from HuggingFace on first start. Mounting `/root/.cache/huggingface` persists them across container restarts.

> **GPU prerequisites**: NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on Linux. On Windows use Docker Desktop + WSL2 + NVIDIA Windows driver; no host CUDA toolkit required. The 1B model needs ≈6 GB VRAM.

### 3. docker-compose

See [`docker/docker-compose.example.yml`](./docker/docker-compose.example.yml).

## API usage

The service listens on port `8000` by default.

### GET `/v1/audio/voices`

List all usable voices.

```bash
curl -s http://localhost:8000/v1/audio/voices | jq
```

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "alice",
      "preview_url": "http://localhost:8000/v1/audio/voices/preview?id=alice",
      "prompt_text": "Hello, this is a reference audio sample."
    }
  ]
}
```

### GET `/v1/audio/voices/preview?id={id}`

Returns the raw reference wav (`audio/wav`), suitable for a browser `<audio>` element.

### POST `/v1/audio/speech`

OpenAI TTS-compatible endpoint — **voice cloning** mode. The voice's `wav` is encoded into the DiT's latent space once (LRU-cached) and the generation continues from that latent prefix, producing speech that matches the reference speaker's timbre and delivery.

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "longcat-audiodit",
    "input": "Hello world, this is a test.",
    "voice": "alice",
    "response_format": "mp3"
  }' \
  -o out.mp3
```

Request fields:

| Field | Type | Description |
|---|---|---|
| `model` | string | Accepted but ignored (for OpenAI SDK compatibility) |
| `input` | string | Text to synthesize, up to 8000 characters |
| `voice` | string | Voice id — must match an entry from `/v1/audio/voices` |
| `response_format` | string | `mp3` (default) / `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed` | float | `0.25 - 4.0`, default `1.0`. Implemented by scaling the estimated target duration (shorter = faster) |
| `duration` | float | Optional target duration in seconds (`0.5 - 60.0`). Overrides both auto-estimate and `speed` |
| `steps` | int | Optional ODE solver steps (`1 - 100`, default `16`); higher is slower but smoother |
| `cfg_strength` | float | Optional classifier-free guidance scale (`0.0 - 20.0`, default `4.0`) |
| `guidance_method` | string | `cfg` or `apg` (default `apg` for cloning — better speaker similarity) |
| `seed` | int | Optional random seed for reproducible output |

Output audio is mono 24 kHz; `pcm` is raw s16le.

### POST `/v1/audio/zeroshot`

**Zero-shot TTS** — no reference audio; the DiT generates speech directly from text. Same request shape as `/v1/audio/speech` minus `voice`.

```bash
curl -s http://localhost:8000/v1/audio/zeroshot \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Today is sunny turning cloudy with rain.",
    "response_format": "mp3"
  }' \
  -o out_zeroshot.mp3
```

Default `guidance_method` is `cfg` for zero-shot (fastest); flip to `apg` for higher fidelity at the cost of speed.

### Using the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-noop")

with client.audio.speech.with_streaming_response.create(
    model="longcat-audiodit",
    voice="alice",
    input="Hello world",
    response_format="mp3",
) as resp:
    resp.stream_to_file("out.mp3")
```

Extensions (`duration`, `steps`, `cfg_strength`, `guidance_method`, `seed`) can be passed through `extra_body={...}`.

### GET `/healthz`

Returns model name, device, dtype, sample rate and status for health checks.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LONGCAT_MODEL` | `meituan-longcat/LongCat-AudioDiT-1B` | HuggingFace repo id or local path. Set to `meituan-longcat/LongCat-AudioDiT-3.5B` for the larger model. |
| `LONGCAT_DEVICE` | `auto` | `auto` → CUDA > CPU. Or `cuda` / `cpu` |
| `LONGCAT_CUDA_INDEX` | `0` | Selects `cuda:N` when device is `cuda` or `auto` |
| `LONGCAT_DTYPE` | `float32` | `float16` / `bfloat16` / `float32` for the DiT backbone. VAE always runs in fp16. On CPU, `float16` silently falls back to `float32`. |
| `LONGCAT_CACHE_DIR` | — | Sets `HF_HOME` before model load |
| `LONGCAT_VOICES_DIR` | `/voices` | Voices directory |
| `LONGCAT_PROMPT_CACHE_SIZE` | `32` | LRU size for pre-encoded voice latents (`0` disables) |
| `LONGCAT_STEPS` | `16` | Default ODE solver steps |
| `LONGCAT_CFG_STRENGTH` | `4.0` | Default guidance strength |
| `LONGCAT_GUIDANCE_METHOD_CLONE` | `apg` | Default guidance method for `/v1/audio/speech` |
| `LONGCAT_GUIDANCE_METHOD_ZEROSHOT` | `cfg` | Default guidance method for `/v1/audio/zeroshot` |
| `LONGCAT_SEED` | — | Default random seed (unset = non-deterministic) |
| `MAX_INPUT_CHARS` | `8000` | Upper bound for the `input` field |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | |

## Building images locally

Initialize the submodule first (the workflow does this automatically).

```bash
git submodule update --init --recursive

# CUDA image
docker buildx build -f docker/Dockerfile.cuda \
  -t longcat-openai-tts-api:cuda .

# CPU image
docker buildx build -f docker/Dockerfile.cpu \
  -t longcat-openai-tts-api:cpu .
```

## Caveats

- **Zero-shot works but cloning is the recommended path.** LongCat-AudioDiT achieves state-of-the-art speaker similarity on the Seed benchmark, which is the headline feature; zero-shot output is generic.
- **CPU inference is impractical.** A single 10-second clip can take several minutes on CPU. The CPU image ships mainly for CI/smoke-test parity and small-scale experimentation. Use the CUDA image in production.
- **Concurrency**: a single model instance is not thread-safe; the service serializes inference with an asyncio Lock. Scale out by running more containers behind a load balancer.
- **Long text**: requests whose `input` exceeds `MAX_INPUT_CHARS` (default 8000) return 413. The underlying model has a hard duration ceiling (`max_wav_duration`, typically 30 s); long prompts are clipped at that ceiling regardless of the auto-estimate.
- **Streaming is not supported** on the HTTP layer — the endpoint returns the complete audio when generation finishes. LongCat-AudioDiT is a non-autoregressive diffusion model, so true token-level streaming is not a natural fit.
- **Voice prompt caching**: the reference wav is VAE-encoded into latents once per `(wav_path, mtime)` and cached in an in-memory LRU. Updating a voice's `.wav` on disk invalidates the entry automatically.
- **Duration estimation** is character-based (Chinese vs ASCII) and then scaled by `speed`. If the result sounds truncated or over-stretched, pass an explicit `duration` in seconds.
- **`guidance_method` tradeoff**: `cfg` is the standard classifier-free guidance (faster); `apg` (Adaptive Projected Guidance) yields better quality, especially for cloning, at ~1.5× cost.
- **No built-in auth** — deploy behind a reverse proxy (Nginx, Cloudflare, etc.) if you need token-based access control.

## Project layout

```
.
├── LongCat-AudioDiT/           # read-only submodule, never modified
├── app/                        # FastAPI application
│   ├── server.py
│   ├── engine.py               # model loading + prompt latent LRU + inference
│   ├── voices.py               # voices directory scanner
│   ├── audio.py                # multi-format encoder
│   ├── config.py
│   └── schemas.py
├── docker/
│   ├── Dockerfile.cuda
│   ├── Dockerfile.cpu
│   ├── requirements.api.txt
│   ├── entrypoint.sh
│   └── docker-compose.example.yml
├── .github/workflows/
│   └── build-images.yml        # cuda + cpu matrix build
├── voices/                     # mounted at runtime
└── README.md
```

## Acknowledgements

Built on top of [meituan-longcat/LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT) (MIT).
