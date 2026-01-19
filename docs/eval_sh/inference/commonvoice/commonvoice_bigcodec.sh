#!/bin/bash

# BigCodec Inference Script - CommonVoice Only
# 生成 BigCodec 80Hz 的推論檔案 (CommonVoice 資料集)

set -e

echo "========================================="
echo "BigCodec Inference (CommonVoice) - Starting"
echo "========================================="

# 環境設定
CONDA_ENV="codec_eval"
BASE_AUDIO_DIR="/mnt/Internal/ASR"
OUTPUT_DIR="/mnt/Internal/jieshiang/Inference_Result_2"
BIGCODEC_DIR="/home/jieshiang/Desktop/GitHub/BigCodec"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
GPU_ID=1

# CSV 檔案
COMMONVOICE_CSV="${CSV_DIR}/commonvoice_filtered_clean.csv"

# 模型參數
CODEC_NAME="BigCodec"
BITRATE="80Hz"
CHECKPOINT="${BIGCODEC_DIR}/bigcodec.pt"

cd "$BIGCODEC_DIR"

echo "Running BigCodec 80Hz inference on CommonVoice..."
echo "Environment: $CONDA_ENV"
echo "Output: $OUTPUT_DIR/$CODEC_NAME/$BITRATE/commonvoice"

CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n "$CONDA_ENV" python bigcodec_batch_inference.py \
    --base_audio_dir "$BASE_AUDIO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --commonvoice_csv "$COMMONVOICE_CSV" \
    --ckpt "$CHECKPOINT" \
    --codec_name "$CODEC_NAME" \
    --bitrate "$BITRATE" \
    --device cuda

echo "========================================="
echo "BigCodec Inference (CommonVoice) - Completed"
echo "========================================="
