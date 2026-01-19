#!/bin/bash

# EnCodec Evaluation Script - Hokkien Dataset
# V3 Metrics with full metric suite

set -e

OUTPUT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison_pitch"
SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
EVAL_ENV="codec_eval_pip_py39"

echo "Starting EnCodec Evaluation - Hokkien Dataset"

# EnCodec 75Hz_4k - Hokkien (min)
echo "Running: EnCodec 75Hz_4k - Hokkien (min)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/EnCodec/75Hz_4k_finetune/stage1_full \
    --csv_file "${CSV_DIR}/hokkien_filtered_clean.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "EnCodec" \
    --frequency "75Hz_4k_stage_1" \
    --causality "Causal" \
    --bit_rate "3.0" \
    --quantizers "4" \
    --codebook_size "1024" \
    --n_params "16M" \
    --training_set "Multi-domain" \
    --testing_set "LibriSpeech test-clean" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe ter semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "hokkien" \
    --language min \
    --use_gpu \
    --gpu_id 0 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

echo "EnCodec evaluation on Hokkien completed."
