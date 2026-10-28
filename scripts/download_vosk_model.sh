#!/usr/bin/env bash
# Downloads the Vosk small English model used by the "Jarvis" voice control
# and places it in ./model (project root).
set -euo pipefail

MODEL_NAME="vosk-model-small-en-us-0.15"
URL="https://alphacephei.com/vosk/models/${MODEL_NAME}.zip"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/model"

if [ -d "${TARGET_DIR}" ]; then
  echo "Vosk model already present in ${TARGET_DIR}"
  exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "Downloading ${MODEL_NAME}..."
curl -L --fail -o "${TMP_DIR}/${MODEL_NAME}.zip" "${URL}"
unzip -q "${TMP_DIR}/${MODEL_NAME}.zip" -d "${TMP_DIR}"
mv "${TMP_DIR}/${MODEL_NAME}" "${TARGET_DIR}"
echo "Vosk model installed in ${TARGET_DIR}"
