#!/bin/bash

# LSCodec Inference Script - VIVOS Only
# 生成 LSCodec 25Hz 和 50Hz 的推論檔案 (VIVOS 資料集)

set -e

echo "========================================="
echo "LSCodec Inference (VIVOS) - Starting"
echo "========================================="

# 環境設定
CONDA_ENV="lscodec"
BASE_AUDIO_DIR="/mnt/Internal/ASR"
OUTPUT_DIR="/mnt/Internal/jieshiang/Inference_Result"
LSCODEC_DIR="/home/jieshiang/Desktop/GitHub/LSCodec-Inference"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
GPU_ID=1

# CSV 檔案
VIVOS_CSV="${CSV_DIR}/vivos_filtered_clean_test.csv"

cd "$LSCODEC_DIR"

echo "Running LSCodec 25Hz and 50Hz inference on VIVOS..."
echo "Environment: $CONDA_ENV"
echo "Output: $OUTPUT_DIR/LSCodec/{25Hz,50Hz}/vivos"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n "$CONDA_ENV" python lscodec_batch_inference.py \
    --base_audio_dir "$BASE_AUDIO_DIR" \
    --lscodec_dir "$LSCODEC_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --vivos_csv "$VIVOS_CSV" \
    --frequencies 25Hz 50Hz

echo "========================================="
echo "LSCodec Inference (VIVOS) - Completed"
echo "========================================="
