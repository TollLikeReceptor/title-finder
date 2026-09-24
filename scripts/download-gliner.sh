#!/usr/bin/env sh
# Downloads a GLiNER model for job-title extraction into models/.
#
#   ./scripts/download-gliner.sh                   # gliner_large-v2.1, full precision (~1.8 GB)
#   ./scripts/download-gliner.sh large int8        # gliner_large-v2.1, quantized (~655 MB)
#   ./scripts/download-gliner.sh small             # gliner_small-v2.1, full precision (~610 MB)
#   ./scripts/download-gliner.sh small int8        # gliner_small-v2.1, quantized (~185 MB)
#   ./scripts/download-gliner.sh x-large           # gliner-x-large, quantized (~610 MB)
#
# large-v2.1 is the default: it scored real titles 0.88-0.97 and flagged nothing
# in title-free text, where small-v2.1 scored 0.70-0.87 and the quantized
# x-large ranked names and companies alongside titles. Point the server at
# another model with GLINER_MODEL_DIR (default: models/gliner_large-v2.1).
set -eu

MODEL="${1:-large}"
PRECISION="${2:-}"

case "${MODEL}" in
  x-large) REPO="knowledgator/gliner-x-large";      DIR="models/gliner-x-large" ;;
  small)   REPO="onnx-community/gliner_small-v2.1"; DIR="models/gliner_small-v2.1" ;;
  large)   REPO="onnx-community/gliner_large-v2.1"; DIR="models/gliner_large-v2.1" ;;
  *) echo "usage: $0 [small|large|x-large] [full|quantized|int8]" >&2; exit 1 ;;
esac

# gliner-x-large's full-precision export is one 2.4 GB protobuf with no
# external-data file, which ONNX Runtime cannot parse (protobuf's 2 GB limit),
# so only its quantized export is usable.
case "${MODEL}" in
  x-large) PRECISION="${PRECISION:-quantized}" ;;
  small|large) PRECISION="${PRECISION:-full}" ;;
esac

case "${MODEL}:${PRECISION}" in
  x-large:full)
    echo "gliner-x-large's full-precision ONNX (2.4 GB, single file) exceeds ONNX Runtime's 2 GB protobuf limit; use 'quantized'" >&2
    exit 1 ;;
  *:full)            ONNX="model.onnx" ;;
  x-large:quantized) ONNX="model_quantized.onnx" ;;
  small:int8|large:int8) ONNX="model_int8.onnx" ;;
  *) echo "unsupported precision '${PRECISION}' for ${MODEL}" >&2; exit 1 ;;
esac

BASE="https://huggingface.co/${REPO}/resolve/main"
mkdir -p "${DIR}/onnx"

echo "Downloading ${REPO} tokenizer..."
curl -fL --retry 3 -o "${DIR}/tokenizer.json" "${BASE}/tokenizer.json"

echo "Downloading ${REPO} onnx/${ONNX}..."
curl -fL --retry 3 -o "${DIR}/onnx/model.onnx" "${BASE}/onnx/${ONNX}"

echo "GLiNER model ready in ${DIR}"
