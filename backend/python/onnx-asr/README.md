# onnx-asr LocalAI backend

This directory contains a LocalAI v4-compatible gRPC backend that uses `onnx-asr` for speech-to-text workloads.

## Quick start

```bash
bash install.sh
bash run.sh --addr 0.0.0.0:50051
```

`install.sh` creates a local virtual environment, installs shared dependencies from `requirements.txt`, installs a runtime profile, and generates the gRPC Python stubs from `backend/backend.proto`.

## Runtime profiles

- `cpu` (default): installs `onnxruntime`
- `gpu`: installs `onnxruntime-gpu`
- `directml`: installs `onnxruntime-windowsml`

Select a non-default profile with `BUILD_PROFILE`, for example:

```bash
BUILD_PROFILE=gpu bash install.sh
```

## Supported backend options

Pass model options through `ModelOptions.Options` as `key:value` strings.

- `providers:cpu,cuda,tensorrt`
- `provider_options:{"CUDAExecutionProvider":{"device_id":0}}`
- `language:en`
- `target_language:en`
- `quantization:int8`
- `timestamps:true`
- `pnc:true`
- `vad:true`
- `vad_model:silero`
- `vad_path:/path/to/vad`
- `vad_providers:cpu`
- `vad.min_silence_duration_ms:500`
- `vad.max_speech_duration_s:30`
- `threads:4`
- `inter_op_threads:2`
- `execution_mode:parallel`

If `TranscriptRequest.translate` is enabled and `target_language` is not set, the backend defaults to `en`.

## Notes

- The backend converts `onnx-asr` timestamps from seconds to milliseconds before filling `TranscriptSegment.start` and `TranscriptSegment.end`.
- Token strings from `onnx-asr` are not copied into `TranscriptSegment.tokens` because LocalAI expects integer token ids there.
- Diarization and prompt injection are not implemented by `onnx-asr`; requests using those fields are accepted but those fields are ignored.

## Container build

Build a CPU image from the repo root:

```bash
docker build -t localai-onnx-asr:latest-cpu --build-arg BUILD_PROFILE=cpu .
```

The Docker build runs `install.sh` and `test.sh`, so a successful image build also validates the backend inside the container.

If you want to experiment with other dependency profiles, you can override `BUILD_PROFILE`, for example:

```bash
docker build -t localai-onnx-asr:latest-gpu --build-arg BUILD_PROFILE=gpu .
```

Only the CPU image is wired into the bundled gallery file, because that path was validated end-to-end here.

Run the backend container directly:

```bash
docker run --rm -p 50051:50051 localai-onnx-asr:latest-cpu
```

The image exposes a top-level `/run.sh` entrypoint so it matches LocalAI backend container expectations.

## LocalAI gallery setup

This repo includes `gallery/index.yaml`, a minimal backend gallery that references `localai-onnx-asr:latest-cpu`.

Register it in LocalAI with an absolute path or URL. For a local file path:

```bash
local-ai run --backend-galleries '[{"name":"localai-onnx","url":"file:///ABSOLUTE/PATH/TO/gallery/index.yaml"}]'
```

Or persist it with the environment variable:

```bash
export LOCALAI_BACKEND_GALLERIES='[{"name":"localai-onnx","url":"file:///ABSOLUTE/PATH/TO/gallery/index.yaml"}]'
local-ai run
```

If you publish the image to a registry, update `gallery/index.yaml` so `uri` points at the pushed image names.

## GHCR publishing

This repo now includes `.github/workflows/publish-ghcr.yml`, which builds and pushes the CPU backend image to GitHub Container Registry whenever you push a tag that starts with `v`.

Published image name:

```text
ghcr.io/<your-github-owner>/localai-onnx-asr
```

Published tags include:

- the git tag itself, for example `v0.1.0`
- the plain semver version, for example `0.1.0`
- the major/minor tag, for example `0.1`
- `latest-cpu`

Example release flow:

```bash
git tag v0.1.0
git push origin v0.1.0
```

For a GHCR-backed LocalAI gallery, start from `gallery/index.ghcr.example.yaml` and replace `YOUR_GITHUB_OWNER` with your GitHub owner or org.

## Example LocalAI model config

```yaml
name: onnx-parakeet
backend: onnx-asr

parameters:
  model: nemo-parakeet-tdt-0.6b-v3

known_usecases:
  - transcript

options:
  - timestamps:true
  - vad:true
  - vad_model:silero
  - language:en
```
