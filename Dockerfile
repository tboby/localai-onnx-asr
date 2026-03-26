ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE}

ARG BUILD_PROFILE=cpu
ARG BACKEND_PORT=50051

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV BUILD_PROFILE=${BUILD_PROFILE}
ENV LOCALAI_BACKEND_DIR=/opt/localai/backend/python/onnx-asr

RUN apt-get update && \
    apt-get install -y --no-install-recommends bash ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/localai

COPY backend/backend.proto /opt/localai/backend/backend.proto
COPY backend/python/common/libbackend.sh /opt/localai/backend/python/common/libbackend.sh
COPY backend/python/onnx-asr /opt/localai/backend/python/onnx-asr
COPY container/run.sh /run.sh

RUN bash /opt/localai/backend/python/onnx-asr/install.sh && \
    bash /opt/localai/backend/python/onnx-asr/test.sh && \
    chmod +x /run.sh

EXPOSE ${BACKEND_PORT}

ENTRYPOINT ["/run.sh"]
CMD []
