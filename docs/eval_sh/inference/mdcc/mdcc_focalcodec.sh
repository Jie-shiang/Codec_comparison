#!/bin/bash

# FocalCodec Inference Script - MDCC Only
# 生成 FocalCodec 12.5Hz, 25Hz, 50Hz 的推論檔案 (MDCC 資料集)

set -e

echo "========================================="
echo "FocalCodec Inference (MDCC) - Starting"
echo "========================================="

# 環境設定
CONDA_ENV="focalcodec"
BASE_AUDIO_DIR="/mnt/Internal/ASR"
OUTPUT_DIR="/mnt/Internal/jieshiang/Inference_Result"
FOCALCODEC_DIR="/home/jieshiang/Desktop/GitHub/FocalCodec"
MODEL_CACHE_DIR="/mnt/Internal/jieshiang/Model/FocalCodec"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
GPU_ID=1

# CSV 檔案
MDCC_CSV="${CSV_DIR}/mdcc_filtered_clean_test.csv"

cd "$FOCALCODEC_DIR"

echo "Running FocalCodec (12.5Hz, 25Hz, 50Hz) inference on MDCC..."
echo "Environment: $CONDA_ENV"
echo "Output: $OUTPUT_DIR/FocalCodec/{12.5HZ,25HZ,50HZ}/mdcc"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n "$CONDA_ENV" python focalcodec_batch_inference.py \
    --base_audio_dir "$BASE_AUDIO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_cache_dir "$MODEL_CACHE_DIR" \
    --mdcc_csv "$MDCC_CSV" \
    --models 12.5hz 25hz 50hz \
    --device cuda

echo "========================================="
echo "FocalCodec Inference (MDCC) - Completed"
echo "========================================="
