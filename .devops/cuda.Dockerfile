ARG UBUNTU_VERSION=24.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=12.8.1
ARG GCC_VERSION=14
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG BASE_CUDA_RUN_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A

ARG NODE_VERSION=24

FROM docker.io/node:$NODE_VERSION AS web

ARG APP_VERSION

WORKDIR /app/tools/ui

COPY tools/ui/package.json tools/ui/package-lock.json ./
RUN npm ci

COPY tools/ui/ ./
RUN LLAMA_BUILD_NUMBER="$APP_VERSION" npm run build

FROM ${BASE_CUDA_DEV_CONTAINER} AS build

ARG GCC_VERSION
# CUDA architecture to build for (defaults to all supported archs)
ARG CUDA_DOCKER_ARCH=default
# Git commit info passed from host (avoids needing .git in build context)
ARG GIT_COMMIT=unknown
ARG GIT_COUNT=0

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y gcc-${GCC_VERSION} g++-${GCC_VERSION} build-essential cmake python3 python3-pip git libssl-dev libgomp1

ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}

WORKDIR /app

COPY . .

COPY --from=web /app/tools/ui/dist tools/ui/dist

RUN --mount=type=bind,source=.buildcache,target=/app/build,rw \
    if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
    export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build \
    -DGGML_NATIVE=ON -DGGML_CUDA=ON -DGGML_BACKEND_DL=OFF -DGGML_CPU_ALL_VARIANTS=OFF \
    -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_FA_USEFUL_QUANTS=ON \
    -DLLAMA_USE_PREBUILT_UI=ON \
    -DLLAMA_BUILD_TESTS=OFF ${CMAKE_ARGS} -DLLAMA_BUILD_RPC=ON -DGGML_RPC=ON \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    -DLLAMA_BUILD_COMMIT=${GIT_COMMIT} -DLLAMA_BUILD_NUMBER=${GIT_COUNT} . && \
    cmake --build build --config Release -j$(nproc) && \
    mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

RUN --mount=type=bind,source=.buildcache,target=/app/build,rw \
    mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && find /app/full -name "*.so*" -delete \
    && cp *.py /app/full \
    && cp -r conversion /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

## Base image
FROM ${BASE_CUDA_RUN_CONTAINER} AS base

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG IMAGE_URL=https://github.com/ggml-org/llama.cpp
ARG IMAGE_SOURCE=https://github.com/ggml-org/llama.cpp
LABEL org.opencontainers.image.created=$BUILD_DATE \
    org.opencontainers.image.version=$APP_VERSION \
    org.opencontainers.image.revision=$APP_REVISION \
    org.opencontainers.image.title="llama.cpp" \
    org.opencontainers.image.description="LLM inference in C/C++" \
    org.opencontainers.image.url=$IMAGE_URL \
    org.opencontainers.image.source=$IMAGE_SOURCE

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && apt autoremove -y \
    && rm -rf /tmp/* /var/tmp/*

### Full
FROM base AS full

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-wheel \
    ffmpeg \
    && apt autoremove -y \
    && rm -rf /tmp/* /var/tmp/*

# copy pip requirements separately first so `pip install` will be cached unless they're edited, and own't re-run any time other files in /app change
COPY requirements.txt requirements.txt
COPY requirements/ requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages --upgrade setuptools \
    && pip install --break-system-packages -r requirements.txt

# COPY --from=build happens last in each downstream stage (not in `base`) so that
# recompiling from a source change doesn't bust the apt/pip cache above.
COPY --from=build /app/lib/ /app
COPY --from=build /app/full /app


ENTRYPOINT ["/app/tools.sh"]

### Light, CLI only
FROM base AS light

COPY --from=build /app/lib/ /app
COPY --from=build /app/full/llama /app/full/llama-cli /app/full/llama-completion /app

WORKDIR /app

ENTRYPOINT [ "/app/llama-cli" ]

### Server, Server only
FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/lib/ /app
COPY --from=build /app/full/llama /app/full/llama-server /app

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
