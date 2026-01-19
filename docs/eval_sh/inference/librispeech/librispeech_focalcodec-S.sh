#!/bin/bash

# FocalCodec-S Inference Script - LibriSpeech Only
# 生成 FocalCodec-S 50Hz_2k, 50Hz_4k, 50Hz_65k 的推論檔案 (LibriSpeech 資料集)

set -e

echo "========================================="
echo "FocalCodec-S Inference (LibriSpeech) - Starting"
echo "========================================="

# 環境設定
CONDA_ENV="focalcodec"
BASE_AUDIO_DIR="/mnt/Internal/ASR"
OUTPUT_DIR="/mnt/Internal/jieshiang/Inference_Result_2"
FOCALCODEC_DIR="/home/jieshiang/Desktop/GitHub/FocalCodec"
MODEL_CACHE_DIR="/mnt/Internal/jieshiang/Model/FocalCodec"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
GPU_ID=1

# CSV 檔案
LIBRISPEECH_CSV="${CSV_DIR}/librispeech_filtered_clean.csv"

cd "$FOCALCODEC_DIR"

echo "Running FocalCodec-S (50Hz_2k, 50Hz_4k, 50Hz_65k) inference on LibriSpeech..."
echo "Environment: $CONDA_ENV"
echo "Output: $OUTPUT_DIR/FocalCodec/{50HZ_2K,50HZ_4K,50HZ_65K}/librispeech"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n "$CONDA_ENV" python focalcodec_batch_inference.py \
    --base_audio_dir "$BASE_AUDIO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_cache_dir "$MODEL_CACHE_DIR" \
    --librispeech_csv "$LIBRISPEECH_CSV" \
    --models 50hz_2k 50hz_4k 50hz_65k \
    --device cuda

echo "========================================="
echo "FocalCodec-S Inference (LibriSpeech) - Completed"
echo "========================================="
