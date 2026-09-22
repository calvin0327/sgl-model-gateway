# syntax=docker/dockerfile:1.6
######################### BUILD IMAGE #########################
ARG RUST_IMAGE=rust:1.93.1-bookworm
ARG RUNTIME_IMAGE=debian:bookworm-slim
FROM ${RUST_IMAGE} AS builder
ENV RUSTUP_TOOLCHAIN=1.93.1
ARG BUILD_PROFILE=ci

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libssl-dev \
        pkg-config \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/build/target \
    cargo build --locked --profile "${BUILD_PROFILE}" --bin sgl-model-gateway --features vendored-openssl \
    && cp "target/${BUILD_PROFILE}/sgl-model-gateway" /usr/local/bin/sgl-model-gateway

######################### RUNTIME IMAGE #########################
FROM ${RUNTIME_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/sgl-model-gateway /usr/local/bin/sgl-model-gateway

# The config is mounted by the deployment; the same image supports all policies.
EXPOSE 8000 29000
ENTRYPOINT ["sgl-model-gateway"]
