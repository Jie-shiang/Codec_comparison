#!/bin/bash

# DAC Inference Script - CommonVoice Only
# 生成 DAC 16kHz, 24kHz, 44kHz 的推論檔案 (CommonVoice 資料集)

set -e

echo "========================================="
echo "DAC Inference (CommonVoice) - Starting"
echo "========================================="

# 環境設定
CONDA_ENV="codec_eval"
BASE_AUDIO_DIR="/mnt/Internal/ASR"
OUTPUT_DIR="/mnt/Internal/jieshiang/Inference_Result_2"
DAC_DIR="/home/jieshiang/Desktop/GitHub/DAC"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
GPU_ID=1

# CSV 檔案
COMMONVOICE_CSV="${CSV_DIR}/commonvoice_filtered_clean.csv"

cd "$DAC_DIR"

echo "Running DAC (16kHz, 24kHz, 44kHz) inference on CommonVoice..."
echo "Environment: $CONDA_ENV"
echo "Output: $OUTPUT_DIR/DAC/{16khz,24khz,44khz}_8kbps/commonvoice"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n "$CONDA_ENV" python dac_batch_inference.py \
    --base_audio_dir "$BASE_AUDIO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --commonvoice_csv "$COMMONVOICE_CSV" \
    --model_types 16khz 24khz 44khz \
    --device cuda

echo "========================================="
echo "DAC Inference (CommonVoice) - Completed"
echo "========================================="
