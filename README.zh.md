# LongCat-AudioDiT OpenAI-TTS API

[English](./README.md) · **中文**

一个 [OpenAI TTS](https://platform.openai.com/docs/api-reference/audio/createSpeech) 兼容的 HTTP 服务，对 [LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT)（美团开源的扩散式 Transformer TTS，直接在波形潜空间上做生成）进行封装，支持中英文 Zero-shot 合成以及通过参考音频进行音色克隆。

## 特性

- **OpenAI TTS 兼容**：`POST /v1/audio/speech`，请求体格式与 OpenAI SDK 一致
- **音色克隆**：挂载 `voices/` 目录下的 `xxx.wav` + `xxx.txt` 对，DiT 会基于该参考续写，还原目标说话人的音色与韵律
- **Zero-shot TTS**：额外提供 `POST /v1/audio/zeroshot`，无需参考音频，仅凭文本生成
- **2 个镜像**：`cuda` 与 `cpu`
- **模型运行时下载**：不打包进镜像，HuggingFace 缓存目录挂载后可复用
- **多种输出格式**：`mp3`、`opus`、`aac`、`flac`、`wav`、`pcm`（单声道 24 kHz）

## 可用镜像

| 镜像 | 设备 |
|---|---|
| `ghcr.io/seancheung/longcat-openai-tts-api:cuda-latest` | CUDA 12.4 |
| `ghcr.io/seancheung/longcat-openai-tts-api:latest`      | CPU |

镜像仅构建 `linux/amd64`。

## 快速开始

### 1. 准备音色目录

```
voices/
├── alice.wav     # 参考音频，单声道，建议 24kHz（≥16kHz 会自动重采样），时长 3-20 秒
├── alice.txt     # UTF-8 纯文本，内容为 alice.wav 中说出的原文
├── bob.wav
└── bob.txt
```

**规则**：必须同时存在同名的 `.wav` 和 `.txt` 才会被识别为有效音色；文件名（不含后缀）即音色 id；多余或缺对的文件会被忽略。`/v1/audio/speech`（音色克隆）会用到 `voices/`；`/v1/audio/zeroshot` 不需要该目录。

### 2. 运行容器

GPU 版本（推荐）：

```bash
docker run --rm -p 8000:8000 --gpus all \
  -v $PWD/voices:/voices:ro \
  -v $PWD/cache:/root/.cache \
  ghcr.io/seancheung/longcat-openai-tts-api:cuda-latest
```

CPU 版本：

```bash
docker run --rm -p 8000:8000 \
  -v $PWD/voices:/voices:ro \
  -v $PWD/cache:/root/.cache \
  ghcr.io/seancheung/longcat-openai-tts-api:latest
```

首次启动会从 HuggingFace 下载模型权重（`LongCat-AudioDiT-1B` 约 4 GB，`LongCat-AudioDiT-3.5B` 约 14 GB）。挂载 `/root/.cache` 可让权重在容器重启后复用。

> **GPU 要求**：宿主机需安装 NVIDIA 驱动与 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。Windows 需 Docker Desktop + WSL2 + NVIDIA Windows 驱动。1B 模型约需 6 GB 显存。

### 3. docker-compose

参考 [`docker/docker-compose.example.yml`](./docker/docker-compose.example.yml)。

## API 用法

服务默认监听 `8000` 端口。

### GET `/v1/audio/voices`

列出所有可用音色。

```bash
curl -s http://localhost:8000/v1/audio/voices | jq
```

返回：

```json
{
  "object": "list",
  "data": [
    {
      "id": "alice",
      "preview_url": "http://localhost:8000/v1/audio/voices/preview?id=alice",
      "prompt_text": "你好，这是一段参考音频。"
    }
  ]
}
```

### GET `/v1/audio/voices/preview?id={id}`

返回参考音频本体（`audio/wav`），可用于浏览器 `<audio>` 试听。

### POST `/v1/audio/speech`

OpenAI TTS 兼容接口——**音色克隆**模式。音色的 wav 会一次性编码到 DiT 的潜空间（LRU 缓存），生成时从该潜空间前缀续写，还原目标说话人的音色与语气。

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "longcat-audiodit",
    "input": "你好世界，这是一段测试语音。",
    "voice": "alice",
    "response_format": "mp3"
  }' \
  -o out.mp3
```

请求字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | string | 接受但忽略（为了与 OpenAI SDK 兼容） |
| `input` | string | 要合成的文本，最长 8000 字符 |
| `voice` | string | 音色 id，必须匹配 `/v1/audio/voices` 中的某一项 |
| `response_format` | string | `mp3`（默认） / `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed` | float | `0.25 - 4.0`，默认 `1.0`。通过缩放估算出的目标时长实现（时长更短 = 语速更快） |
| `duration` | float | 可选目标时长（秒，`0.5 - 60.0`），会同时覆盖自动估算与 `speed` |
| `steps` | int | 可选 ODE 求解器步数（`1 - 100`，默认 `16`），越高越慢但更平滑 |
| `cfg_strength` | float | 可选 classifier-free guidance 强度（`0.0 - 20.0`，默认 `4.0`） |
| `guidance_method` | string | `cfg` 或 `apg`（克隆默认 `apg`，音色相似度更高） |
| `seed` | int | 可选随机种子，用于复现 |

输出为单声道 24 kHz；`pcm` 为裸 s16le 数据。

### POST `/v1/audio/zeroshot`

**Zero-shot TTS**——无需参考音频，DiT 直接从文本生成。请求体与 `/v1/audio/speech` 相同，去掉 `voice`。

```bash
curl -s http://localhost:8000/v1/audio/zeroshot \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "今天天气多云转雨。",
    "response_format": "mp3"
  }' \
  -o out_zeroshot.mp3
```

Zero-shot 默认 `guidance_method` 为 `cfg`（更快）；换成 `apg` 可略微提升质量，代价是约 1.5× 的耗时。

### 使用 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-noop")

with client.audio.speech.with_streaming_response.create(
    model="longcat-audiodit",
    voice="alice",
    input="你好世界",
    response_format="mp3",
) as resp:
    resp.stream_to_file("out.mp3")
```

`duration`、`steps`、`cfg_strength`、`guidance_method`、`seed` 等扩展字段可通过 `extra_body={...}` 传入。

### GET `/healthz`

返回模型名、设备、精度、采样率与状态，用于健康检查。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LONGCAT_MODEL` | `meituan-longcat/LongCat-AudioDiT-1B` | HuggingFace 仓库 id 或本地路径。设为 `meituan-longcat/LongCat-AudioDiT-3.5B` 可切到大模型 |
| `LONGCAT_DEVICE` | `auto` | `auto` 按 CUDA > CPU 优先级，也可强制 `cuda` / `cpu` |
| `LONGCAT_CUDA_INDEX` | `0` | `cuda` / `auto` 时选择的 `cuda:N` |
| `LONGCAT_DTYPE` | `float32` | DiT 主干的精度，`float16` / `bfloat16` / `float32`；VAE 固定 fp16。CPU 上 `float16` 自动回退为 `float32` |
| `LONGCAT_CACHE_DIR` | — | 加载模型前写入 `HF_HOME` |
| `LONGCAT_VOICES_DIR` | `/voices` | 音色目录 |
| `LONGCAT_PROMPT_CACHE_SIZE` | `32` | 预编码音色潜变量的 LRU 大小（`0` 表示关闭） |
| `LONGCAT_STEPS` | `16` | 默认 ODE 求解器步数 |
| `LONGCAT_CFG_STRENGTH` | `4.0` | 默认 guidance 强度 |
| `LONGCAT_GUIDANCE_METHOD_CLONE` | `apg` | `/v1/audio/speech` 的默认 guidance 方法 |
| `LONGCAT_GUIDANCE_METHOD_ZEROSHOT` | `cfg` | `/v1/audio/zeroshot` 的默认 guidance 方法 |
| `LONGCAT_SEED` | — | 默认随机种子（不设则非确定） |
| `MAX_INPUT_CHARS` | `8000` | `input` 字段上限 |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | |

## 本地构建镜像

构建前需先初始化 submodule（workflow 已处理）。

```bash
git submodule update --init --recursive

# CUDA 镜像
docker buildx build -f docker/Dockerfile.cuda \
  -t longcat-openai-tts-api:cuda .

# CPU 镜像
docker buildx build -f docker/Dockerfile.cpu \
  -t longcat-openai-tts-api:cpu .
```

## 局限 / 注意事项

- **Zero-shot 可用，但克隆才是主推路径**：LongCat-AudioDiT 在 Seed 基准上的亮点是音色相似度 SOTA，克隆场景最能发挥其优势；zero-shot 生成的是通用音色
- **CPU 推理极慢**：10 秒音频在 CPU 上可能需要数分钟。CPU 镜像主要用于 CI / 冒烟测试与小规模实验，生产环境请用 CUDA 镜像
- **并发**：单模型实例非线程安全，服务内部用 asyncio Lock 串行化。并发请求依赖横向扩容（多容器 + 负载均衡）
- **长文本**：超过 `MAX_INPUT_CHARS`（默认 8000）返回 413。底层模型有硬时长上限（`max_wav_duration`，通常 30 秒），超长文本的自动估算会被该上限截断
- **不支持 HTTP 层流式返回**：生成完成后一次性返回。LongCat-AudioDiT 是非自回归扩散模型，token 级流式不自然
- **音色 prompt 缓存**：参考 wav 会按 `(wav_path, mtime)` 在内存 LRU 中缓存一次 VAE 编码结果。磁盘上 wav 被更新后缓存会自动失效
- **时长估算**：基于字符数（中文/英文权重不同）再按 `speed` 缩放。若结果被截断或过度拉伸，可显式传 `duration`（秒）
- **`guidance_method` 权衡**：`cfg` 是标准 classifier-free guidance（更快）；`apg`（Adaptive Projected Guidance）在克隆场景下质量更好，耗时约 1.5×
- **无内置鉴权**：如需 token 访问控制，请在反向代理层（Nginx、Cloudflare 等）做

## 目录结构

```
.
├── LongCat-AudioDiT/           # 只读 submodule，不修改
├── app/                        # FastAPI 应用
│   ├── server.py
│   ├── engine.py               # 模型加载 + 音色潜变量 LRU + 推理
│   ├── voices.py               # 音色扫描
│   ├── audio.py                # 多格式编码
│   ├── config.py
│   └── schemas.py
├── docker/
│   ├── Dockerfile.cuda
│   ├── Dockerfile.cpu
│   ├── requirements.api.txt
│   ├── entrypoint.sh
│   └── docker-compose.example.yml
├── .github/workflows/
│   └── build-images.yml        # cuda + cpu 矩阵构建
└── README.md
```

## 致谢

基于 [meituan-longcat/LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT)（MIT）。
