#!/bin/bash

# LSCodec Inference Script - Hokkien Only
# 生成 LSCodec 25Hz 和 50Hz 的推論檔案 (Hokkien 資料集)

set -e

echo "========================================="
echo "LSCodec Inference (Hokkien) - Starting"
echo "========================================="

# 環境設定
CONDA_ENV="lscodec"
BASE_AUDIO_DIR="/mnt/Internal/ASR"
OUTPUT_DIR="/mnt/Internal/jieshiang/Inference_Result_2"
LSCODEC_DIR="/home/jieshiang/Desktop/GitHub/LSCodec-Inference"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
GPU_ID=1

# CSV 檔案
HOKKIEN_CSV="${CSV_DIR}/hokkien_filtered_clean.csv"

cd "$LSCODEC_DIR"

echo "Running LSCodec 25Hz and 50Hz inference on Hokkien..."
echo "Environment: $CONDA_ENV"
echo "Output: $OUTPUT_DIR/LSCodec/{25Hz,50Hz}/hokkien"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n "$CONDA_ENV" python lscodec_batch_inference.py \
    --base_audio_dir "$BASE_AUDIO_DIR" \
    --lscodec_dir "$LSCODEC_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --hokkien_csv "$HOKKIEN_CSV" \
    --frequencies 25Hz 50Hz

echo "========================================="
echo "LSCodec Inference (Hokkien) - Completed"
echo "========================================="
