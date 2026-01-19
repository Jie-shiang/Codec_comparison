#!/bin/bash

# FocalCodec-S Inference Script - Aishell Only
# 生成 FocalCodec-S 50Hz_2k, 50Hz_4k, 50Hz_65k 的推論檔案 (Aishell 資料集)

set -e

echo "========================================="
echo "FocalCodec-S Inference (Aishell) - Starting"
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
AISHELL_CSV="${CSV_DIR}/aishell_filtered_clean.csv"

cd "$FOCALCODEC_DIR"

echo "Running FocalCodec-S (50Hz_2k, 50Hz_4k, 50Hz_65k) inference on Aishell..."
echo "Environment: $CONDA_ENV"
echo "Output: $OUTPUT_DIR/FocalCodec/{50HZ_2K,50HZ_4K,50HZ_65K}/aishell"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n "$CONDA_ENV" python focalcodec_batch_inference.py \
    --base_audio_dir "$BASE_AUDIO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_cache_dir "$MODEL_CACHE_DIR" \
    --aishell_csv "$AISHELL_CSV" \
    --models 50hz_2k 50hz_4k 50hz_65k \
    --device cuda

echo "========================================="
echo "FocalCodec-S Inference (Aishell) - Completed"
echo "========================================="
