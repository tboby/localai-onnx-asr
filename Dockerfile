ARG BASE_IMAGE=ubuntu:24.04

FROM ${BASE_IMAGE} AS builder

ARG BACKEND=onnx-asr
ARG BUILD_PROFILE=cpu

ENV DEBIAN_FRONTEND=noninteractive
ENV BUILD_PROFILE=${BUILD_PROFILE}
ENV PORTABLE_PYTHON=true
ENV USE_PIP=true

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        make \
        python3 \
        python3-pip \
        python3-venv \
        python-is-python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY backend/python/${BACKEND} /${BACKEND}
COPY backend/backend.proto /${BACKEND}/backend.proto
COPY backend/python/common /${BACKEND}/common

RUN cd /${BACKEND} && make && bash test.sh

FROM scratch

ARG BACKEND=onnx-asr
COPY --from=builder /${BACKEND}/ /

ENTRYPOINT ["/run.sh"]
CMD []
