#!/bin/bash

# SpeechTokenizer Inference Script - MDCC Only
# 生成 SpeechTokenizer 50Hz 的推論檔案 (MDCC 資料集)

set -e

echo "========================================="
echo "SpeechTokenizer Inference (MDCC) - Starting"
echo "========================================="

# 環境設定
CONDA_ENV="codec_eval"
BASE_AUDIO_DIR="/mnt/Internal/ASR"
OUTPUT_DIR="/mnt/Internal/jieshiang/Inference_Result"
SPEECHTOKENIZER_DIR="/home/jieshiang/Desktop/GitHub/SpeechTokenizer"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
GPU_ID=0

# CSV 檔案
MDCC_CSV="${CSV_DIR}/mdcc_filtered_clean.csv"

cd "$SPEECHTOKENIZER_DIR"

echo "Running SpeechTokenizer 50Hz inference on MDCC..."
echo "Environment: $CONDA_ENV"
echo "Output: $OUTPUT_DIR/SpeechTokenizer/50Hz/mdcc"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n "$CONDA_ENV" python speechtokenizer_batch_inference.py \
    --csv_file "$MDCC_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --dataset mdcc \
    --model_name speechtokenizer_hubert_avg \
    --device cuda \
    --audio_types complete \
    --base_audio_dir "$BASE_AUDIO_DIR"

echo "========================================="
echo "SpeechTokenizer Inference (MDCC) - Completed"
echo "========================================="
