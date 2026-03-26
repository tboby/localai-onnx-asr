# Backend gallery

This directory contains a minimal LocalAI backend gallery for the `onnx-asr` backend.

`index.yaml` points at a local CPU Docker image tag by default:

- `localai-onnx-asr:latest-cpu`

For local-only usage, build that exact tag and then register this gallery with LocalAI.

The entry is intentionally a single concrete backend definition named `onnx-asr` with a direct `uri`. That matches how LocalAI resolves external backend installs by name.

If you plan to publish the backend, update `gallery/index.yaml` to point at your pushed OCI image names, for example `ghcr.io/<user>/localai-onnx-asr:latest-cpu`.

There is also a starter file at `gallery/index.ghcr.example.yaml` for GHCR-based publishing.
