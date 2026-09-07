# syntax=docker/dockerfile:1.6

######################### BUILD IMAGE #########################
FROM rust:1.93-bookworm AS builder
ENV RUSTUP_TOOLCHAIN=1.93

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
    cargo build --locked --release --bin sgl-model-gateway --features vendored-openssl \
    && cp target/release/sgl-model-gateway /usr/local/bin/sgl-model-gateway

######################### RUNTIME IMAGE #########################
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/sgl-model-gateway /usr/local/bin/sgl-model-gateway

ENTRYPOINT ["sgl-model-gateway"]
