#!/bin/bash

# NanoCodec Inference Script - VIVOS Only
# 生成 NanoCodec 12.5Hz_2k, 12.5Hz_4k, 21.5Hz_2k 的推論檔案 (VIVOS 資料集)

set -e

echo "========================================="
echo "NanoCodec Inference (VIVOS) - Starting"
echo "========================================="

# 環境設定
CONDA_ENV="nemo"
BASE_AUDIO_DIR="/mnt/Internal/ASR"
OUTPUT_DIR="/mnt/Internal/jieshiang/Inference_Result"
NANOCODEC_DIR="/home/jieshiang/Desktop/GitHub/NanoCodec"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"

cd "$NANOCODEC_DIR"

# 設定 HuggingFace 快取目錄到指定的模型路徑
export HF_HOME="/mnt/Internal/jieshiang/Model"
export TRANSFORMERS_CACHE="/mnt/Internal/jieshiang/Model/transformers_cache"
export HF_DATASETS_CACHE="/mnt/Internal/jieshiang/Model/datasets_cache"

# 定義模型和資料集
MODELS=("12.5Hz_2k" "12.5Hz_4k" "21.5Hz_2k")
DATASET="vivos"
CSV_PATH="${CSV_DIR}/vivos_filtered_clean.csv"

echo "Models to process: ${MODELS[@]}"
echo "Dataset: $DATASET"
echo "Environment: $CONDA_ENV"
echo "Model cache: $HF_HOME"
echo "Output: $OUTPUT_DIR/NanoCodec/"
echo ""

# 循環處理所有模型
for MODEL in "${MODELS[@]}"; do
    echo "========================================="
    echo "Processing: Model=$MODEL, Dataset=$DATASET"
    echo "========================================="

    if [ ! -f "$CSV_PATH" ]; then
        echo "Error: CSV file not found: $CSV_PATH"
        exit 1
    fi

    conda run -n "$CONDA_ENV" python nanocodec_batch_inference.py \
        --base_audio_dir "$BASE_AUDIO_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --model "$MODEL" \
        --csv_path "$CSV_PATH" \
        --dataset_name "$DATASET" \
        --device cuda:0 \
        --use_local

    echo "Completed: Model=$MODEL, Dataset=$DATASET"
    echo ""
done

echo "========================================="
echo "NanoCodec Inference (VIVOS) - All Completed"
echo "========================================="
