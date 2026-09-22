#!/usr/bin/env bash
# Build the Linux image, check its real entrypoint, then export it for docker load.
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=${IMAGE:-sgl-model-gateway-cache-v1:latest}
platform=${PLATFORM:-linux/amd64}
output=${OUTPUT:-"$repo_dir/../artifacts/pre-kv-cache-final-v1-image/sgl-model-gateway-cache-v1-linux-amd64.tar.gz"}

command -v docker >/dev/null || { echo 'Docker is required to build this image.' >&2; exit 1; }
docker info >/dev/null
mkdir -p -- "$(dirname -- "$output")"
docker buildx build --platform "$platform" --load \
  --build-arg "BUILD_PROFILE=${BUILD_PROFILE:-ci}" \
  --build-arg "RUST_IMAGE=${RUST_IMAGE:-rust:1.93.1-bookworm}" \
  --build-arg "RUNTIME_IMAGE=${RUNTIME_IMAGE:-debian:bookworm-slim}" \
  --file "$repo_dir/docker/sgl-model-gateway.Dockerfile" \
  --tag "$image" "$repo_dir"

# These checks execute the Linux binary, detecting missing runtime libraries too.
docker run --rm --platform "$platform" "$image" --backend vllm --version-verbose
help=$(docker run --rm --platform "$platform" "$image" --help)
[[ "$help" == *pre_kv_cache* && "$help" == *--pre-kv-cache-config-json* && "$help" == *PRE_KV_CACHE_CONFIG_JSON* ]]
[[ "$help" == *--pre-kv-cache-profile* && "$help" == *--pre-kv-cache-token-source* ]]
[[ "$help" != *SMG_PRE_KV_CACHE_CONFIG_JSON* ]]
docker image inspect "$image" --format '{{.Os}}/{{.Architecture}} {{.Id}}'

# Replace the delivery archive only after the complete export succeeds.
docker save "$image" | gzip -1 > "$output.partial"
mv -- "$output.partial" "$output"
if command -v sha256sum >/dev/null; then
  sha256sum "$output"
else
  shasum -a 256 "$output"
fi
printf 'Image archive: %s\n' "$output"
