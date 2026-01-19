#!/bin/bash

# BigCodec 完整Pipeline
# 1. 產生推論檔案
# 2. 執行評估
# 3. 刪除推論檔案

set -e

echo "========================================="
echo "BigCodec Full Pipeline - Starting"
echo "========================================="

# 環境設定
EVAL_ENV="codec_eval_pip_py39"
EVAL_SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/docs/eval_sh"
INFERENCE_SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/docs/eval_sh/inference"
INFERENCE_DIR="/mnt/Internal/jieshiang/Inference_Result/BigCodec/80Hz"

# Step 1: 產生推論檔案
echo ""
echo "Step 1: Generating inference files..."
bash "${INFERENCE_SCRIPT_DIR}/inference_bigcodec.sh"

# Step 2: 執行評估
echo ""
echo "Step 2: Running evaluations..."
cd /home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison
conda run -n "$EVAL_ENV" bash "${EVAL_SCRIPT_DIR}/run_bigcodec_evaluations.sh"

# Step 3: 刪除推論檔案（保留資料夾結構）
echo ""
echo "Step 3: Cleaning up inference files..."

# 刪除 librispeech 推論檔案
if [ -d "${INFERENCE_DIR}/librispeech" ]; then
    echo "Removing ${INFERENCE_DIR}/librispeech/*.wav"
    find "${INFERENCE_DIR}/librispeech" -name "*_inference.wav" -type f -delete
fi

# 刪除 commonvoice 推論檔案
if [ -d "${INFERENCE_DIR}/commonvoice" ]; then
    echo "Removing ${INFERENCE_DIR}/commonvoice/*.wav"
    find "${INFERENCE_DIR}/commonvoice" -name "*_inference.wav" -type f -delete
fi

# 刪除 aishell 推論檔案
if [ -d "${INFERENCE_DIR}/aishell" ]; then
    echo "Removing ${INFERENCE_DIR}/aishell/*.wav"
    find "${INFERENCE_DIR}/aishell" -name "*_inference.wav" -type f -delete
fi

echo ""
echo "========================================="
echo "BigCodec Full Pipeline - Completed"
echo "========================================="
echo "Results saved to: /home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison"
