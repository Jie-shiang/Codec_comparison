#!/bin/bash

# LSCodec Inference Script - LibriSpeech Only
# 生成 LSCodec 25Hz 和 50Hz 的推論檔案 (LibriSpeech 資料集)

set -e

echo "========================================="
echo "LSCodec Inference (LibriSpeech) - Starting"
echo "========================================="

# 環境設定
CONDA_ENV="lscodec"
BASE_AUDIO_DIR="/mnt/Internal/ASR"
OUTPUT_DIR="/mnt/Internal/jieshiang/Inference_Result_2"
LSCODEC_DIR="/home/jieshiang/Desktop/GitHub/LSCodec-Inference"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
GPU_ID=1

# CSV 檔案
LIBRISPEECH_CSV="${CSV_DIR}/librispeech_filtered_clean.csv"

cd "$LSCODEC_DIR"

echo "Running LSCodec 25Hz and 50Hz inference on LibriSpeech..."
echo "Environment: $CONDA_ENV"
echo "Output: $OUTPUT_DIR/LSCodec/{25Hz,50Hz}/librispeech"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n "$CONDA_ENV" python lscodec_batch_inference.py \
    --base_audio_dir "$BASE_AUDIO_DIR" \
    --lscodec_dir "$LSCODEC_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --librispeech_csv "$LIBRISPEECH_CSV" \
    --frequencies 25Hz 50Hz

echo "========================================="
echo "LSCodec Inference (LibriSpeech) - Completed"
echo "========================================="
