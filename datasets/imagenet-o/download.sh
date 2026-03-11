#!/usr/bin/env bash

set -euo pipefail

DATASET_URL="https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$(mktemp -d)"
ARCHIVE_PATH="${TMP_DIR}/imagenet-o.tar"

cleanup() {
  rm -rf "${TMP_DIR}"
}

trap cleanup EXIT

echo "Downloading ImageNet-O to ${ARCHIVE_PATH}"
curl -fL "${DATASET_URL}" -o "${ARCHIVE_PATH}"

echo "Extracting archive"
tar --strip-components=1 --exclude='README.md' -xf "${ARCHIVE_PATH}" -C "${SCRIPT_DIR}"

echo "ImageNet-O extracted under ${SCRIPT_DIR}"
