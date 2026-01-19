#!/bin/bash

# Mimi Evaluation Script - Commonvoice Dataset
# V3 Metrics with full metric suite

set -e

OUTPUT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison_test"
SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
EVAL_ENV="codec_eval_pip_py39"

echo "Starting Mimi Evaluation - Commonvoice Dataset"

# Mimi 12.5Hz_8k - Commonvoice (zh)
echo "Running: Mimi 12.5Hz_8k - Commonvoice (zh)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/MimiCodec/12.5Hz_8k/commonvoice \
    --csv_file "${CSV_DIR}/commonvoice_filtered_clean.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "Mimi" \
    --frequency "12.5Hz_8k" \
    --causality "Causal" \
    --bit_rate "0.55" \
    --quantizers "4" \
    --codebook_size "2048" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe ter semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "commonvoice" \
    --language zh \
    --use_gpu \
    --gpu_id 3 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# Mimi 12.5Hz_16k - Commonvoice (zh)
echo "Running: Mimi 12.5Hz_16k - Commonvoice (zh)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/MimiCodec/12.5Hz_16k/commonvoice \
    --csv_file "${CSV_DIR}/commonvoice_filtered_clean.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "Mimi" \
    --frequency "12.5Hz_16k" \
    --causality "Causal" \
    --bit_rate "1.1" \
    --quantizers "8" \
    --codebook_size "2048" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe ter semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "commonvoice" \
    --language zh \
    --use_gpu \
    --gpu_id 3 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

echo "Mimi evaluation on Commonvoice completed."
