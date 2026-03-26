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
