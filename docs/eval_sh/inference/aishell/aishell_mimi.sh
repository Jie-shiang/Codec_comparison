#!/bin/bash

# Mimi Inference Script - Aishell Only
# 生成 Mimi 12.5Hz_8k 和 12.5Hz_16k 的推論檔案 (Aishell 資料集)

set -e

echo "========================================="
echo "Mimi Inference (Aishell) - Starting"
echo "========================================="

# 環境設定
CONDA_ENV="mimi_env"
BASE_AUDIO_DIR="/mnt/Internal/ASR"
OUTPUT_DIR="/mnt/Internal/jieshiang/Inference_Result_2"
MIMI_DIR="/home/jieshiang/Desktop/GitHub/Mimi"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"

# CSV 檔案
AISHELL_CSV="${CSV_DIR}/aishell_filtered_clean.csv"

cd "$MIMI_DIR"

echo "Running Mimi inference on Aishell..."
echo "Environment: $CONDA_ENV"

# Mimi 12.5Hz_8k (4 layers)
echo ""
echo "Running Mimi 12.5Hz_8k (4 layers) on Aishell..."
echo "Output: $OUTPUT_DIR/MimiCodec/12.5Hz_8k/aishell"

conda run -n "$CONDA_ENV" python mimi_batch_inference.py \
    --base_audio_dir "$BASE_AUDIO_DIR" \
    --csv_file "$AISHELL_CSV" \
    --output_dir "$OUTPUT_DIR/MimiCodec/12.5Hz_8k" \
    --dataset aishell \
    --hf_repo kyutai/moshika-pytorch-bf16 \
    --num_codebooks 4 \
    --device cuda \
    --audio_types complete

# Mimi 12.5Hz_16k (8 layers)
echo ""
echo "Running Mimi 12.5Hz_16k (8 layers) on Aishell..."
echo "Output: $OUTPUT_DIR/MimiCodec/12.5Hz_16k/aishell"

conda run -n "$CONDA_ENV" python mimi_batch_inference.py \
    --base_audio_dir "$BASE_AUDIO_DIR" \
    --csv_file "$AISHELL_CSV" \
    --output_dir "$OUTPUT_DIR/MimiCodec/12.5Hz_16k" \
    --dataset aishell \
    --hf_repo kyutai/moshika-pytorch-bf16 \
    --num_codebooks 8 \
    --device cuda \
    --audio_types complete

echo "========================================="
echo "Mimi Inference (Aishell) - Completed"
echo "========================================="
