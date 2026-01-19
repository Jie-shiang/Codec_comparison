#!/bin/bash

# DAC Evaluation Script - Mdcc Dataset
# V3 Metrics with full metric suite

set -e

OUTPUT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison_test_2"
SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
EVAL_ENV="codec_eval_pip_py39"

echo "Starting DAC Evaluation - Mdcc Dataset"

# DAC 50Hz_16kHz - Mdcc (yue)
echo "Running: DAC 50Hz_16kHz - Mdcc (yue)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/DAC/50Hz_16kHz/mdcc \
    --csv_file "${CSV_DIR}/mdcc_filtered_clean.csv" \
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
    --dataset_name "mdcc" \
    --language yue \
    --use_gpu \
    --gpu_id 0 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# DAC 50Hz_24kHz - Mdcc (yue)
echo "Running: DAC 50Hz_24kHz - Mdcc (yue)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/DAC/50Hz_24kHz/mdcc \
    --csv_file "${CSV_DIR}/mdcc_filtered_clean.csv" \
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
    --dataset_name "mdcc" \
    --language yue \
    --use_gpu \
    --gpu_id 0 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# DAC 50Hz_44kHz - Mdcc (yue)
echo "Running: DAC 50Hz_44kHz - Mdcc (yue)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/DAC/50Hz_44kHz/mdcc \
    --csv_file "${CSV_DIR}/mdcc_filtered_clean.csv" \
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
    --dataset_name "mdcc" \
    --language yue \
    --use_gpu \
    --gpu_id 0 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

echo "DAC evaluation on Mdcc completed."
