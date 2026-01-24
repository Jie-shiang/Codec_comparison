#!/bin/bash

# EnCodec Inference Script - MDCC Only
# 生成 EnCodec 24khz 1.5kbps, 3.0kbps, 6.0kbps, 12.0kbps, 24.0kbps 的推論檔案 (MDCC 資料集)

set -e

echo "========================================="
echo "EnCodec Inference (MDCC) - Starting"
echo "========================================="

# 環境設定
CONDA_ENV="codec_eval"
BASE_AUDIO_DIR="/mnt/Internal/ASR"
OUTPUT_DIR="/mnt/Internal/jieshiang/Inference_Result"
ENCODEC_DIR="/home/jieshiang/Desktop/GitHub/EnCodec"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
GPU_ID=0

# CSV 檔案
MDCC_CSV="${CSV_DIR}/mdcc_filtered_clean.csv"

cd "$ENCODEC_DIR"

echo "Running EnCodec 24khz (1.5, 3.0, 6.0, 12.0, 24.0 kbps) inference on MDCC..."
echo "Environment: $CONDA_ENV"
echo "Output: $OUTPUT_DIR/EnCodec/75Hz_{2,4,8,16,32}k/mdcc"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n "$CONDA_ENV" python encodec_batch_inference.py \
    --base_audio_dir "$BASE_AUDIO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --mdcc_csv "$MDCC_CSV" \
    --model_type 24khz \
    --bandwidths 1.5 3.0 6.0 12.0 24.0 \
    --device cuda

echo "========================================="
echo "EnCodec Inference (MDCC) - Completed"
echo "========================================="
