#!/bin/bash

# DAC Evaluation Script - Hokkien Dataset
# V3 Metrics with full metric suite

set -e

OUTPUT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison"
SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
EVAL_ENV="codec_eval_pip_py39"

echo "Starting DAC Evaluation - Hokkien Dataset"

# DAC 50Hz_16kHz - Hokkien (min)
echo "Running: DAC 50Hz_16kHz - Hokkien (min)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/DAC/50Hz_16kHz/hokkien \
    --csv_file "${CSV_DIR}/hokkien_filtered_clean_test.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "DAC" \
    --frequency "50Hz_16kHz" \
    --causality "Non-Causal" \
    --bit_rate "8.0" \
    --quantizers "9" \
    --codebook_size "1024" \
    --n_params "73M" \
    --training_set "Multi-domain" \
    --testing_set "LibriSpeech + others" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe ter semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "hokkien" \
    --language min \
    --use_gpu \
    --gpu_id 1 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# DAC 50Hz_24kHz - Hokkien (min)
echo "Running: DAC 50Hz_24kHz - Hokkien (min)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/DAC/50Hz_24kHz/hokkien \
    --csv_file "${CSV_DIR}/hokkien_filtered_clean_test.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "DAC" \
    --frequency "50Hz_24kHz" \
    --causality "Non-Causal" \
    --bit_rate "8.0" \
    --quantizers "9" \
    --codebook_size "1024" \
    --n_params "73M" \
    --training_set "Multi-domain" \
    --testing_set "LibriSpeech + others" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe ter semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "hokkien" \
    --language min \
    --use_gpu \
    --gpu_id 1 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# DAC 50Hz_44kHz - Hokkien (min)
echo "Running: DAC 50Hz_44kHz - Hokkien (min)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/DAC/50Hz_44kHz/hokkien \
    --csv_file "${CSV_DIR}/hokkien_filtered_clean_test.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "DAC" \
    --frequency "50Hz_44kHz" \
    --causality "Non-Causal" \
    --bit_rate "8.0" \
    --quantizers "9" \
    --codebook_size "1024" \
    --n_params "73M" \
    --training_set "Multi-domain" \
    --testing_set "LibriSpeech + others" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe ter semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "hokkien" \
    --language min \
    --use_gpu \
    --gpu_id 1 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

echo "DAC evaluation on Hokkien completed."
