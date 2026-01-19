#!/bin/bash

# NanoCodec 完整Pipeline
# 1. 產生推論檔案
# 2. 執行評估
# 3. 刪除推論檔案

set -e

echo "========================================="
echo "NanoCodec Full Pipeline - Starting"
echo "========================================="

# 環境設定
EVAL_ENV="codec_eval_pip_py39"
EVAL_SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/docs/eval_sh"
INFERENCE_SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/docs/eval_sh/inference"
INFERENCE_BASE_DIR="/mnt/Internal/jieshiang/Inference_Result/NanoCodec"

# Step 1: 產生推論檔案
echo ""
echo "Step 1: Generating inference files..."
bash "${INFERENCE_SCRIPT_DIR}/inference_nanocodec.sh"

# Step 2: 執行評估
echo ""
echo "Step 2: Running evaluations..."
cd /home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison
conda run -n "$EVAL_ENV" bash "${EVAL_SCRIPT_DIR}/run_nanocodec_evaluations.sh"

# Step 3: 刪除推論檔案（保留資料夾結構）
echo ""
echo "Step 3: Cleaning up inference files..."

# 使用新的目錄結構: 12.5Hz_2k, 12.5Hz_4k, 21.5Hz_2k
MODELS=("12.5Hz_2k" "12.5Hz_4k" "21.5Hz_2k")
DATASETS=("librispeech" "commonvoice" "aishell")

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        INFERENCE_DIR="${INFERENCE_BASE_DIR}/${MODEL}/${DATASET}"
        if [ -d "$INFERENCE_DIR" ]; then
            echo "Removing ${INFERENCE_DIR}/*_inference.wav"
            find "$INFERENCE_DIR" -name "*_inference.wav" -type f -delete
        fi
    done
done

echo ""
echo "========================================="
echo "NanoCodec Full Pipeline - Completed"
echo "========================================="
echo "Results saved to: /home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison"
