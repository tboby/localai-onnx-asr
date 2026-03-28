#!/usr/bin/env python3
"""LocalAI gRPC backend for onnx-asr speech transcription."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import subprocess
import struct
import sys
import time
from concurrent import futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import grpc

import backend_pb2
import backend_pb2_grpc


_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_MESSAGE_LIMIT_BYTES = 50 * 1024 * 1024
MAX_WORKERS = int(os.environ.get("PYTHON_GRPC_MAX_WORKERS", "1"))
OPENAI_AUDIO_EXTENSIONS = {
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".wav",
    ".webm",
}
AZURE_PROVIDER = "AzureExecutionProvider"
GPU_PROVIDERS = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "CoreMLExecutionProvider",
    "WebGPUExecutionProvider",
]
PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "trt": "TensorrtExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "webgpu": "WebGPUExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
}

MODEL_ALIASES = {
    "parakeet-v3": "nemo-parakeet-tdt-0.6b-v3",
    "parakeet-v2": "nemo-parakeet-tdt-0.6b-v2",
    "parakeet-ctc": "nemo-parakeet-ctc-0.6b",
    "parakeet-rnnt": "nemo-parakeet-rnnt-0.6b",
    "canary-1b-v2": "nemo-canary-1b-v2",
}


@dataclass
class RuntimeConfig:
    model_name: str
    model_path: str | None = None
    language: str | None = None
    target_language: str | None = None
    pnc: bool = True
    use_vad: bool = False
    timestamps: bool = True
    options: dict[str, Any] = field(default_factory=dict)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _load_onnx_asr_module() -> Any:
    return importlib.import_module("onnx_asr")


def _load_onnxruntime_module() -> Any:
    return importlib.import_module("onnxruntime")


def _parse_option_value(raw_value: str) -> Any:
    value = raw_value.strip()
    lowered = value.lower()

    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    if value.startswith("{") or value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def _parse_options(option_items: Iterable[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for item in option_items:
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        options[key.strip()] = _parse_option_value(value)
    return options


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _as_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _seconds_to_ms(value: Any) -> int:
    if value is None:
        return 0
    return max(0, int(round(float(value) * 1000)))


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_mono_float32(waveform: Any) -> Any:
    np = importlib.import_module("numpy")

    array = np.asarray(waveform)
    if array.ndim == 0:
        return array.astype(np.float32).reshape(1)
    if array.ndim > 1:
        if array.ndim == 2 and array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
            array = array.mean(axis=0)
        else:
            array = array.mean(axis=-1)

    if array.dtype.kind == "f":
        return array.astype(np.float32, copy=False)

    if array.dtype.kind == "u":
        info = np.iinfo(array.dtype)
        midpoint = (info.max + 1) / 2.0
        return ((array.astype(np.float32) - midpoint) / midpoint).astype(np.float32, copy=False)

    if array.dtype.kind == "i":
        info = np.iinfo(array.dtype)
        scale = float(max(abs(info.min), info.max))
        return (array.astype(np.float32) / scale).astype(np.float32, copy=False)

    return array.astype(np.float32)


def _read_wav_with_fallback(audio_path: Path) -> tuple[Any, int]:
    np = importlib.import_module("numpy")

    with audio_path.open("rb") as handle:
        if handle.read(4) != b"RIFF":
            raise ValueError(f"Unsupported WAV container: {audio_path}")
        handle.read(4)
        if handle.read(4) != b"WAVE":
            raise ValueError(f"Unsupported WAV header: {audio_path}")

        audio_format = None
        channels = None
        sample_rate = None
        bits_per_sample = None
        data = None

        while True:
            chunk_header = handle.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            chunk_data = handle.read(chunk_size)
            if chunk_size % 2:
                handle.read(1)

            if chunk_id == b"fmt ":
                if len(chunk_data) < 16:
                    raise ValueError("Invalid WAV fmt chunk")
                audio_format, channels, sample_rate, _, _, bits_per_sample = struct.unpack(
                    "<HHIIHH", chunk_data[:16]
                )
            elif chunk_id == b"data":
                data = chunk_data

        if audio_format is None or channels is None or sample_rate is None or bits_per_sample is None or data is None:
            raise ValueError("Incomplete WAV file")

        if audio_format == 3:
            if bits_per_sample == 32:
                waveform = np.frombuffer(data, dtype="<f4")
            elif bits_per_sample == 64:
                waveform = np.frombuffer(data, dtype="<f8").astype(np.float32)
            else:
                raise ValueError(f"Unsupported IEEE float WAV bit depth: {bits_per_sample}")
        elif audio_format == 1:
            if bits_per_sample == 8:
                waveform = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif bits_per_sample == 16:
                waveform = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            elif bits_per_sample == 24:
                raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
                ints = (
                    raw[:, 0].astype(np.int32)
                    | (raw[:, 1].astype(np.int32) << 8)
                    | (raw[:, 2].astype(np.int32) << 16)
                )
                ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
                waveform = ints.astype(np.float32) / 8388608.0
            elif bits_per_sample == 32:
                waveform = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported PCM WAV bit depth: {bits_per_sample}")
        else:
            raise ValueError(f"Unsupported WAV format: {audio_format}")

        if channels > 1:
            waveform = waveform.reshape(-1, channels).mean(axis=1)

        return waveform.astype(np.float32, copy=False), int(sample_rate)


def _decode_audio_with_soundfile(audio_path: Path) -> tuple[Any, int]:
    sf = importlib.import_module("soundfile")

    waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    return _to_mono_float32(waveform), int(sample_rate)


def _decode_audio_with_ffmpeg(audio_path: Path) -> tuple[Any, int]:
    imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
    np = importlib.import_module("numpy")

    sample_rate = 16000
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="ignore").strip() or "ffmpeg decode failed"
        raise ValueError(error)

    waveform = np.frombuffer(result.stdout, dtype="<f4")
    if waveform.size == 0:
        raise ValueError(f"ffmpeg produced no audio samples for {audio_path}")

    return waveform.astype(np.float32, copy=False), sample_rate


def _should_try_generic_audio_decoders(audio_path: Path, error: Exception) -> bool:
    suffix = audio_path.suffix.lower()
    if suffix in OPENAI_AUDIO_EXTENSIONS:
        return True

    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "riff",
            "wave",
            "format",
            "could not",
            "failed to open",
            "invalid data",
            "unsupported",
        )
    )


def _decode_audio_fallback(audio_path: Path) -> tuple[Any, int]:
    errors: list[str] = []

    for name, loader in (
        ("soundfile", _decode_audio_with_soundfile),
        ("ffmpeg", _decode_audio_with_ffmpeg),
    ):
        try:
            waveform, sample_rate = loader(audio_path)
            _log(f"Falling back to {name} decoder for input: {audio_path}")
            return waveform, sample_rate
        except Exception as err:
            errors.append(f"{name}: {err}")

    raise ValueError("; ".join(errors))


def _recognize_audio(model: Any, audio_path: Path, recognize_kwargs: dict[str, Any]) -> Any:
    try:
        return model.recognize(str(audio_path), **recognize_kwargs)
    except Exception as err:
        if "unknown format: 3" in str(err):
            waveform, sample_rate = _read_wav_with_fallback(audio_path)
            _log(f"Falling back to manual WAV loader for IEEE float input: {audio_path}")
            return model.recognize(waveform, sample_rate=sample_rate, **recognize_kwargs)

        if not _should_try_generic_audio_decoders(audio_path, err):
            raise

        waveform, sample_rate = _decode_audio_fallback(audio_path)
        return model.recognize(waveform, sample_rate=sample_rate, **recognize_kwargs)


def _resolve_model_name(model_name: str | None) -> str | None:
    if not model_name:
        return None
    return MODEL_ALIASES.get(model_name.strip().lower(), model_name)


def _looks_like_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / filename).exists() for filename in ("config.json", "preprocessor_config.json", "model.onnx"))


def _resolve_model_path(model_name: str | None, model_path: str | None) -> str | None:
    if not model_path:
        return None

    root = Path(model_path)
    if _looks_like_model_dir(root):
        return str(root)

    candidates: list[Path] = []
    if model_name:
        raw_names = {
            model_name,
            model_name.replace("/", "--"),
            model_name.replace("/", "___"),
            model_name.replace("/", os.sep),
        }
        for raw_name in raw_names:
            candidate = root / raw_name
            if _looks_like_model_dir(candidate):
                return str(candidate)
            candidates.extend(path for path in root.glob(f"**/{raw_name}") if _looks_like_model_dir(path))

    config_matches = [path.parent for path in root.glob("**/config.json") if path.is_file()]
    if len(config_matches) == 1 and _looks_like_model_dir(config_matches[0]):
        return str(config_matches[0])

    for candidate in candidates:
        if _looks_like_model_dir(candidate):
            return str(candidate)

    return None


def _normalize_provider_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        return cleaned
    alias = PROVIDER_ALIASES.get(cleaned.lower())
    if alias:
        return alias
    return cleaned


def _extract_provider_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [item for item in str(value).split(",") if item.strip()]
    return [_normalize_provider_name(str(item)) for item in items if str(item).strip()]


def _available_providers(ort: Any) -> list[str]:
    return [name for name in ort.get_available_providers() if name != AZURE_PROVIDER]


def _select_provider_names(explicit: Any, request_cuda: bool, available: list[str]) -> list[str]:
    explicit_names = _extract_provider_names(explicit)
    if explicit_names:
        missing = [name for name in explicit_names if name not in available]
        if missing:
            raise ValueError(
                "Requested providers are unavailable: "
                + ", ".join(missing)
                + ". Available providers: "
                + ", ".join(available)
            )
        return explicit_names

    if request_cuda:
        selected = [provider for provider in GPU_PROVIDERS if provider in available]
        if not selected:
            raise ValueError(
                "CUDA requested but no GPU execution provider is available. "
                f"Available providers: {', '.join(available)}"
            )
        if "CPUExecutionProvider" in available:
            selected.append("CPUExecutionProvider")
        return selected

    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    return available[:1]


def _build_provider_payload(provider_names: list[str], provider_options: Any) -> list[Any]:
    if not isinstance(provider_options, dict):
        return provider_names

    payload: list[Any] = []
    for name in provider_names:
        options = provider_options.get(name)
        if options is None:
            options = provider_options.get(name.replace("ExecutionProvider", ""))
        if options is None:
            payload.append(name)
            continue
        if not isinstance(options, dict):
            raise ValueError(f"Provider options for {name} must be an object")
        payload.append((name, options))
    return payload


def _extract_prefixed_options(options: dict[str, Any], prefix: str) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    for key, value in options.items():
        if key.startswith(prefix):
            extracted[key[len(prefix) :]] = value
    return extracted


def _make_session_options(ort: Any, request_threads: int, options: dict[str, Any]) -> Any | None:
    intra_threads = request_threads or int(options.get("threads", 0) or 0)
    inter_threads = int(options.get("inter_op_threads", 0) or 0)
    execution_mode = _as_string(options.get("execution_mode"))

    if not intra_threads and not inter_threads and not execution_mode:
        return None

    session_options = ort.SessionOptions()
    if intra_threads:
        session_options.intra_op_num_threads = intra_threads
    if inter_threads:
        session_options.inter_op_num_threads = inter_threads
    if execution_mode and execution_mode.lower() == "parallel":
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    return session_options


def _segment_bounds(result: Any) -> tuple[int, int]:
    start = getattr(result, "start", None)
    end = getattr(result, "end", None)
    if start is not None or end is not None:
        start_ms = _seconds_to_ms(start)
        end_ms = _seconds_to_ms(end)
        return start_ms, max(start_ms, end_ms)

    timestamps = getattr(result, "timestamps", None) or []
    values = [float(item) for item in timestamps if item is not None]
    if not values:
        return 0, 0

    start_ms = _seconds_to_ms(values[0])
    end_ms = _seconds_to_ms(values[-1])
    return start_ms, max(start_ms, end_ms)


def _normalize_results(raw_result: Any, use_vad: bool) -> tuple[list[backend_pb2.TranscriptSegment], str]:
    if raw_result is None:
        return [], ""

    if use_vad:
        items = list(raw_result)
    elif isinstance(raw_result, (list, tuple)):
        items = list(raw_result)
    else:
        items = [raw_result]

    segments: list[backend_pb2.TranscriptSegment] = []
    texts: list[str] = []
    for index, item in enumerate(items):
        if item is None:
            continue
        text = _clean_text(getattr(item, "text", ""))
        start_ms, end_ms = _segment_bounds(item)
        segments.append(
            backend_pb2.TranscriptSegment(
                id=index,
                start=start_ms,
                end=end_ms,
                text=text,
            )
        )
        if text:
            texts.append(text)

    transcript_text = " ".join(texts).strip()
    if not segments and transcript_text:
        segments.append(backend_pb2.TranscriptSegment(id=0, start=0, end=0, text=transcript_text))
    return segments, transcript_text


class BackendServicer(backend_pb2_grpc.BackendServicer):
    """Implements the LocalAI backend contract used for ASR."""

    def __init__(self) -> None:
        self.model = None
        self.runtime: RuntimeConfig | None = None
        self.state = backend_pb2.StatusResponse.UNINITIALIZED
        self.last_error: str | None = None

    def Health(self, request: backend_pb2.HealthMessage, context: grpc.ServicerContext) -> backend_pb2.Reply:
        return backend_pb2.Reply(message=b"OK")

    def Status(
        self, request: backend_pb2.HealthMessage, context: grpc.ServicerContext
    ) -> backend_pb2.StatusResponse:
        return backend_pb2.StatusResponse(
            state=self.state,
            memory=backend_pb2.MemoryUsageData(total=0),
        )

    def Free(self, request: backend_pb2.HealthMessage, context: grpc.ServicerContext) -> backend_pb2.Result:
        self.model = None
        self.runtime = None
        self.state = backend_pb2.StatusResponse.UNINITIALIZED
        self.last_error = None
        return backend_pb2.Result(success=True, message="Model unloaded")

    def LoadModel(self, request: backend_pb2.ModelOptions, context: grpc.ServicerContext) -> backend_pb2.Result:
        self.state = backend_pb2.StatusResponse.BUSY
        self.last_error = None

        try:
            options = _parse_options(request.Options)
            requested_model_name = _as_string(request.Model)
            model_name = _resolve_model_name(requested_model_name)
            model_path = _as_string(options.get("model_path")) or _as_string(options.get("path")) or _as_string(request.ModelPath)
            resolved_model_path = _resolve_model_path(model_name, model_path)
            load_target = model_name or resolved_model_path or model_path
            if not load_target:
                raise ValueError("Model or ModelPath is required")

            onnx_asr = _load_onnx_asr_module()
            ort = _load_onnxruntime_module()
            available = _available_providers(ort)
            providers = _select_provider_names(options.get("providers"), request.CUDA, available)
            provider_payload = _build_provider_payload(providers, options.get("provider_options"))
            session_options = _make_session_options(ort, request.Threads, options)
            quantization = _as_string(options.get("quantization"))

            load_kwargs: dict[str, Any] = {
                "quantization": quantization,
                "sess_options": session_options,
                "providers": provider_payload,
            }
            if resolved_model_path:
                load_kwargs["path"] = resolved_model_path

            if requested_model_name and model_name != requested_model_name:
                _log(f"Resolved model alias {requested_model_name} -> {model_name}")
            if model_path and not resolved_model_path and model_name:
                _log(
                    "Ignoring ModelPath because it does not look like a concrete model directory: "
                    f"{model_path}"
                )

            _log(f"Loading onnx-asr model: {load_target}")
            model = onnx_asr.load_model(load_target, **load_kwargs)

            use_vad = _as_bool(options.get("vad"), False)
            if use_vad:
                vad_name = _as_string(options.get("vad_model")) or "silero"
                vad_path = _as_string(options.get("vad_path"))
                vad_quantization = _as_string(options.get("vad_quantization"))
                vad_providers = _select_provider_names(options.get("vad_providers"), request.CUDA, available)
                vad_provider_payload = _build_provider_payload(vad_providers, options.get("vad_provider_options"))
                vad_options = _extract_prefixed_options(options, "vad.")
                vad = onnx_asr.load_vad(
                    vad_name,
                    path=vad_path,
                    quantization=vad_quantization,
                    sess_options=session_options,
                    providers=vad_provider_payload,
                )
                model = model.with_vad(vad, **vad_options)

            timestamps = _as_bool(options.get("timestamps"), True)
            if timestamps:
                model = model.with_timestamps()

            self.model = model
            self.runtime = RuntimeConfig(
                model_name=load_target,
                model_path=resolved_model_path or model_path,
                language=_as_string(options.get("language")),
                target_language=_as_string(options.get("target_language")),
                pnc=_as_bool(options.get("pnc"), True),
                use_vad=use_vad,
                timestamps=timestamps,
                options=options,
            )
            self.state = backend_pb2.StatusResponse.READY
            _log(f"Model loaded successfully with providers: {', '.join(providers)}")
            return backend_pb2.Result(success=True, message="Model loaded successfully")
        except Exception as err:
            self.model = None
            self.runtime = None
            self.state = backend_pb2.StatusResponse.ERROR
            self.last_error = str(err)
            _log(f"[ERROR] LoadModel failed: {err}")
            return backend_pb2.Result(success=False, message=str(err))

    def AudioTranscription(
        self, request: backend_pb2.TranscriptRequest, context: grpc.ServicerContext
    ) -> backend_pb2.TranscriptResult:
        if self.model is None or self.runtime is None:
            _log("AudioTranscription called before LoadModel")
            return backend_pb2.TranscriptResult(segments=[], text="")

        audio_path = Path(request.dst)
        if not audio_path.exists():
            _log(f"Audio file not found: {audio_path}")
            return backend_pb2.TranscriptResult(segments=[], text="")

        try:
            recognize_kwargs: dict[str, Any] = {"pnc": self.runtime.pnc}
            language = _as_string(request.language) or self.runtime.language
            if language:
                recognize_kwargs["language"] = language

            target_language = self.runtime.target_language
            if request.translate and not target_language:
                target_language = "en"
            if target_language:
                recognize_kwargs["target_language"] = target_language

            if request.diarize:
                _log("Diarization requested but not supported by onnx-asr; continuing without diarization")
            if request.prompt:
                _log("Prompt provided but onnx-asr does not expose prompt injection; ignoring prompt")

            raw_result = _recognize_audio(self.model, audio_path, recognize_kwargs)
            segments, text = _normalize_results(raw_result, self.runtime.use_vad)
            return backend_pb2.TranscriptResult(segments=segments, text=text)
        except Exception as err:
            _log(f"[ERROR] AudioTranscription failed: {err}")
            return backend_pb2.TranscriptResult(segments=[], text="")


def serve(address: str) -> None:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_message_length", _MESSAGE_LIMIT_BYTES),
            ("grpc.max_send_message_length", _MESSAGE_LIMIT_BYTES),
            ("grpc.max_receive_message_length", _MESSAGE_LIMIT_BYTES),
        ],
    )
    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)
    server.add_insecure_port(address)
    server.start()
    _log(f"Server started. Listening on: {address}")

    def signal_handler(sig: int, frame: Any) -> None:
        _log("Received termination signal. Shutting down...")
        server.stop(0)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the onnx-asr LocalAI backend.")
    parser.add_argument("--addr", default="localhost:50051", help="Address to bind the server to.")
    args = parser.parse_args()
    serve(args.addr)
