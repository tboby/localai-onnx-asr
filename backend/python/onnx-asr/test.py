"""Tests for the onnx-asr LocalAI backend."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import grpc
import numpy as np

import backend
import backend_pb2
import backend_pb2_grpc


HERE = Path(__file__).resolve().parent


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class FakeAdapter:
    def __init__(self) -> None:
        self.vad_calls: list[tuple[object, dict[str, object]]] = []
        self.with_timestamps_calls = 0
        self.recognize_result = None
        self.recognize_side_effects: list[object] = []

    def with_vad(self, vad: object, **kwargs: object) -> "FakeAdapter":
        self.vad_calls.append((vad, kwargs))
        return self

    def with_timestamps(self) -> "FakeAdapter":
        self.with_timestamps_calls += 1
        return self

    def recognize(self, audio_path: str, **kwargs: object):
        self.last_recognize_call = (audio_path, kwargs)
        if self.recognize_side_effects:
            effect = self.recognize_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect
        return self.recognize_result


class FakeOnnxAsr:
    def __init__(self) -> None:
        self.adapter = FakeAdapter()
        self.load_model_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.load_vad_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load_model(self, *args: object, **kwargs: object) -> FakeAdapter:
        self.load_model_calls.append((args, kwargs))
        return self.adapter

    def load_vad(self, *args: object, **kwargs: object) -> object:
        self.load_vad_calls.append((args, kwargs))
        return object()


class FakeOnnxRuntime:
    class SessionOptions:
        def __init__(self) -> None:
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0
            self.execution_mode = None

    class ExecutionMode:
        ORT_PARALLEL = "parallel"

    @staticmethod
    def get_available_providers() -> list[str]:
        return ["CPUExecutionProvider"]


class BackendServicerTests(unittest.TestCase):
    def test_load_model_builds_vad_pipeline(self) -> None:
        servicer = backend.BackendServicer()
        fake_asr = FakeOnnxAsr()

        with mock.patch.object(backend, "_load_onnx_asr_module", return_value=fake_asr), mock.patch.object(
            backend, "_load_onnxruntime_module", return_value=FakeOnnxRuntime
        ):
            response = servicer.LoadModel(
                backend_pb2.ModelOptions(
                    Model="nemo-parakeet-tdt-0.6b-v3",
                    Options=[
                        "providers:cpu",
                        "vad:true",
                        "vad.min_silence_duration_ms:300",
                        "target_language:en",
                    ],
                ),
                None,
            )

        self.assertTrue(response.success)
        self.assertEqual(response.message, "Model loaded successfully")
        self.assertEqual(fake_asr.load_model_calls[0][0][0], "nemo-parakeet-tdt-0.6b-v3")
        self.assertEqual(fake_asr.load_model_calls[0][1]["providers"], ["CPUExecutionProvider"])
        self.assertEqual(len(fake_asr.load_vad_calls), 1)
        self.assertEqual(fake_asr.adapter.with_timestamps_calls, 1)
        self.assertEqual(fake_asr.adapter.vad_calls[0][1]["min_silence_duration_ms"], 300)

    def test_load_model_rejects_missing_gpu_provider(self) -> None:
        servicer = backend.BackendServicer()
        fake_asr = FakeOnnxAsr()

        with mock.patch.object(backend, "_load_onnx_asr_module", return_value=fake_asr), mock.patch.object(
            backend, "_load_onnxruntime_module", return_value=FakeOnnxRuntime
        ):
            response = servicer.LoadModel(
                backend_pb2.ModelOptions(Model="whisper-base", CUDA=True),
                None,
            )

        self.assertFalse(response.success)
        self.assertIn("CUDA requested", response.message)

    def test_load_model_ignores_non_model_cache_root(self) -> None:
        servicer = backend.BackendServicer()
        fake_asr = FakeOnnxAsr()

        with tempfile.TemporaryDirectory() as model_cache:
            with mock.patch.object(backend, "_load_onnx_asr_module", return_value=fake_asr), mock.patch.object(
                backend, "_load_onnxruntime_module", return_value=FakeOnnxRuntime
            ):
                response = servicer.LoadModel(
                    backend_pb2.ModelOptions(
                        Model="nemo-parakeet-tdt-0.6b-v3",
                        ModelPath=model_cache,
                    ),
                    None,
                )

        self.assertTrue(response.success)
        self.assertEqual(fake_asr.load_model_calls[0][0][0], "nemo-parakeet-tdt-0.6b-v3")
        self.assertNotIn("path", fake_asr.load_model_calls[0][1])

    def test_load_model_resolves_common_aliases(self) -> None:
        servicer = backend.BackendServicer()
        fake_asr = FakeOnnxAsr()

        with mock.patch.object(backend, "_load_onnx_asr_module", return_value=fake_asr), mock.patch.object(
            backend, "_load_onnxruntime_module", return_value=FakeOnnxRuntime
        ):
            response = servicer.LoadModel(
                backend_pb2.ModelOptions(Model="parakeet-v3"),
                None,
            )

        self.assertTrue(response.success)
        self.assertEqual(fake_asr.load_model_calls[0][0][0], "nemo-parakeet-tdt-0.6b-v3")

    def test_audio_transcription_normalizes_vad_segments(self) -> None:
        servicer = backend.BackendServicer()
        fake_model = FakeAdapter()
        fake_model.recognize_result = [
            SimpleNamespace(start=0.0, end=0.5, text="hello"),
            SimpleNamespace(start=0.75, end=1.25, text="world"),
        ]
        servicer.model = fake_model
        servicer.runtime = backend.RuntimeConfig(model_name="test", use_vad=True)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            audio_path = handle.name

        try:
            response = servicer.AudioTranscription(backend_pb2.TranscriptRequest(dst=audio_path), None)
        finally:
            os.unlink(audio_path)

        self.assertEqual(response.text, "hello world")
        self.assertEqual(len(response.segments), 2)
        self.assertEqual(response.segments[0].start, 0)
        self.assertEqual(response.segments[0].end, 500)
        self.assertEqual(response.segments[1].start, 750)
        self.assertEqual(response.segments[1].end, 1250)

    def test_audio_transcription_normalizes_single_result(self) -> None:
        servicer = backend.BackendServicer()
        fake_model = FakeAdapter()
        fake_model.recognize_result = SimpleNamespace(text="hello world", timestamps=[0.1, 0.4])
        servicer.model = fake_model
        servicer.runtime = backend.RuntimeConfig(model_name="test", use_vad=False, pnc=False)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            audio_path = handle.name

        try:
            response = servicer.AudioTranscription(
                backend_pb2.TranscriptRequest(dst=audio_path, translate=True),
                None,
            )
        finally:
            os.unlink(audio_path)

        self.assertEqual(response.text, "hello world")
        self.assertEqual(len(response.segments), 1)
        self.assertEqual(response.segments[0].start, 100)
        self.assertEqual(response.segments[0].end, 400)
        self.assertEqual(fake_model.last_recognize_call[1]["target_language"], "en")
        self.assertFalse(fake_model.last_recognize_call[1]["pnc"])

    def test_audio_transcription_falls_back_for_float_wav(self) -> None:
        servicer = backend.BackendServicer()
        fake_model = FakeAdapter()
        fake_model.recognize_side_effects = [
            ValueError("unknown format: 3"),
            SimpleNamespace(text="hello float", timestamps=[0.0, 0.5]),
        ]
        servicer.model = fake_model
        servicer.runtime = backend.RuntimeConfig(model_name="test", use_vad=False)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            audio_path = Path(handle.name)

        try:
            samples = np.linspace(-0.5, 0.5, 160, dtype=np.float32)
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(4)
                wav_file.setframerate(16000)
                wav_file.setcomptype("NONE", "not compressed")
                wav_file.writeframes(samples.tobytes())

            with audio_path.open("r+b") as wav_handle:
                wav_handle.seek(20)
                wav_handle.write((3).to_bytes(2, byteorder="little", signed=False))

            response = servicer.AudioTranscription(backend_pb2.TranscriptRequest(dst=str(audio_path)), None)
        finally:
            audio_path.unlink()

        self.assertEqual(response.text, "hello float")
        self.assertIsInstance(fake_model.last_recognize_call[0], np.ndarray)
        self.assertEqual(fake_model.last_recognize_call[0].dtype, np.float32)
        self.assertEqual(fake_model.last_recognize_call[1]["sample_rate"], 16000)

    def test_audio_transcription_falls_back_for_openai_audio_types(self) -> None:
        servicer = backend.BackendServicer()
        fake_model = FakeAdapter()
        fake_model.recognize_side_effects = [
            ValueError("file does not start with RIFF id"),
            SimpleNamespace(text="hello mp3", timestamps=[0.0, 0.25]),
        ]
        servicer.model = fake_model
        servicer.runtime = backend.RuntimeConfig(model_name="test", use_vad=False)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
            audio_path = Path(handle.name)

        try:
            with mock.patch.object(
                backend,
                "_decode_audio_fallback",
                return_value=(np.array([0.1, -0.1], dtype=np.float32), 22050),
            ) as decode_fallback:
                response = servicer.AudioTranscription(backend_pb2.TranscriptRequest(dst=str(audio_path)), None)
        finally:
            audio_path.unlink()

        self.assertEqual(response.text, "hello mp3")
        decode_fallback.assert_called_once_with(audio_path)
        self.assertIsInstance(fake_model.last_recognize_call[0], np.ndarray)
        self.assertEqual(fake_model.last_recognize_call[1]["sample_rate"], 22050)


class GrpcSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.address = f"127.0.0.1:{_find_free_port()}"
        self.process = subprocess.Popen(
            [sys.executable, "backend.py", "--addr", self.address],
            cwd=HERE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with grpc.insecure_channel(self.address) as channel:
            grpc.channel_ready_future(channel).result(timeout=10)

    def tearDown(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        finally:
            if self.process.stdout is not None:
                self.process.stdout.close()
            if self.process.stderr is not None:
                self.process.stderr.close()

    def test_health_endpoint(self) -> None:
        with grpc.insecure_channel(self.address) as channel:
            stub = backend_pb2_grpc.BackendStub(channel)
            response = stub.Health(backend_pb2.HealthMessage())
        self.assertEqual(response.message, b"OK")


if __name__ == "__main__":
    unittest.main(verbosity=2)
