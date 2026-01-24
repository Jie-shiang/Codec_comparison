#!/bin/bash

# EnCodec Inference Script - LibriSpeech Only
# 生成 EnCodec 24khz 1.5kbps, 3.0kbps, 6.0kbps, 12.0kbps, 24.0kbps 的推論檔案 (LibriSpeech 資料集)

set -e

echo "========================================="
echo "EnCodec Inference (LibriSpeech) - Starting"
echo "========================================="

# 環境設定
CONDA_ENV="codec_eval"
BASE_AUDIO_DIR="/mnt/Internal/ASR"
OUTPUT_DIR="/mnt/Internal/jieshiang/Inference_Result_2"
ENCODEC_DIR="/home/jieshiang/Desktop/GitHub/EnCodec"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
GPU_ID=1

# CSV 檔案
LIBRISPEECH_CSV="${CSV_DIR}/librispeech_filtered_clean.csv"

cd "$ENCODEC_DIR"

echo "Running EnCodec 24khz (1.5, 3.0, 6.0, 12.0, 24.0 kbps) inference on LibriSpeech..."
echo "Environment: $CONDA_ENV"
echo "Output: $OUTPUT_DIR/EnCodec/24khz_{1.5,3.0,6.0,12.0,24.0}kbps/librispeech"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n "$CONDA_ENV" python encodec_batch_inference.py \
    --base_audio_dir "$BASE_AUDIO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --librispeech_csv "$LIBRISPEECH_CSV" \
    --model_type 24khz \
    --bandwidths 1.5 3.0 6.0 12.0 24.0 \
    --device cuda

echo "========================================="
echo "EnCodec Inference (LibriSpeech) - Completed"
echo "========================================="
