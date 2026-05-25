#!/usr/bin/env bash
set -eo pipefail
set -u

# 今いる作業ディレクトリ配下に全部出す
BASE_DIR="/root/AIMET_ENV/tools/ruri/convert"

# 量子化済みモデルのある場所
MODEL_DIR="/root/AIMET_ENV/tools/ruri/model/script_output"

# config
CONFIG_JSON="${BASE_DIR}/config.json"

# env
. /usr/venv/bin/activate

set +u
source "${QNN_SDK_ROOT}/bin/envsetup.sh"
set -u

TARGET="x86_64-linux-clang"
BACKEND_LIB="${QNN_SDK_ROOT}/lib/${TARGET}/libQnnHtp.so"

DATASETS=(abs art mix)
MASKS=(neg6 fp32min neg10)
METHODS=(tf_enhanced minmax)

mkdir -p "${BASE_DIR}/output"

cd "${BASE_DIR}"

for dataset in "${DATASETS[@]}"; do
  for mask in "${MASKS[@]}"; do
    for method in "${METHODS[@]}"; do

      MODEL_NAME="ruri_w8a16_${dataset}_mask_${mask}_${method}"

      echo "============================================================"
      echo "[START] ${MODEL_NAME}"

      ONNX_PATH="${MODEL_DIR}/${MODEL_NAME}.onnx"
      ENCODINGS_PATH="${MODEL_DIR}/${MODEL_NAME}.encodings"

      if [[ ! -f "${ONNX_PATH}" ]]; then
        echo "[SKIP] ONNX not found: ${ONNX_PATH}"
        continue
      fi

      if [[ ! -f "${ENCODINGS_PATH}" ]]; then
        echo "[SKIP] Encodings not found: ${ENCODINGS_PATH}"
        continue
      fi

      CPP_DIR="${BASE_DIR}/cpp_${MODEL_NAME}"
      LIB_OUT_DIR="${BASE_DIR}/lib_${MODEL_NAME}"
      CTX_OUT_DIR="${BASE_DIR}/output/script_output"

      mkdir -p "${CPP_DIR}" "${LIB_OUT_DIR}" "${CTX_OUT_DIR}"

      CPP_PATH="${CPP_DIR}/model.cpp"
      BIN_PATH="${CPP_DIR}/model.bin"

      echo "[1/3] qnn-onnx-converter"
      qnn-onnx-converter \
        --input_network "${ONNX_PATH}" \
        --quantization_overrides "${ENCODINGS_PATH}" \
        --keep_quant_nodes \
        --output_path "${CPP_PATH}" \
        --float_fallback

      echo "[2/3] qnn-model-lib-generator"
      qnn-model-lib-generator \
        -c "${CPP_PATH}" \
        -b "${BIN_PATH}" \
        -o "${LIB_OUT_DIR}" \
        -t "${TARGET}"

      MODEL_SO="${LIB_OUT_DIR}/${TARGET}/libmodel.so"
      if [[ ! -f "${MODEL_SO}" ]]; then
        echo "[ERROR] Generated libmodel.so not found: ${MODEL_SO}"
        continue
      fi

      echo "[3/3] qnn-context-binary-generator"
      qnn-context-binary-generator \
        --model "${MODEL_SO}" \
        --backend "${BACKEND_LIB}" \
        --binary_file "${MODEL_NAME}" \
        --config_file "${CONFIG_JSON}" \
        --output_dir "${CTX_OUT_DIR}"

      echo "[DONE] ${MODEL_NAME}"
      echo "  cpp/bin        : ${CPP_DIR}"
      echo "  lib            : ${LIB_OUT_DIR}"
      echo "  context binary : ${CTX_OUT_DIR}/${MODEL_NAME}.bin"

    done
  done
done

echo "============================================================"
echo "All conversions completed."