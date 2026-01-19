#!/bin/bash

# DAC Evaluation Script - LibriSpeech Dataset
# V3 Metrics with full metric suite

set -e

OUTPUT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison_test"
SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
EVAL_ENV="codec_eval_pip_py39"

echo "Starting DAC Evaluation - LibriSpeech Dataset"

# DAC 50Hz_16kHz - LibriSpeech (English)
echo "Running: DAC 50Hz_16kHz - LibriSpeech (English)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/DAC/50Hz_16kHz/librispeech \
    --csv_file "${CSV_DIR}/librispeech_filtered_clean.csv" \
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
    --metrics dwer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "librispeech" \
    --language en \
    --use_gpu \
    --gpu_id 3 \
    --num_workers 64 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# DAC 50Hz_24kHz - LibriSpeech (English)
echo "Running: DAC 50Hz_24kHz - LibriSpeech (English)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/DAC/50Hz_24kHz/librispeech \
    --csv_file "${CSV_DIR}/librispeech_filtered_clean.csv" \
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
    --metrics dwer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "librispeech" \
    --language en \
    --use_gpu \
    --gpu_id 3 \
    --num_workers 64 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# DAC 50Hz_44kHz - LibriSpeech (English)
echo "Running: DAC 50Hz_44kHz - LibriSpeech (English)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/DAC/50Hz_44kHz/librispeech \
    --csv_file "${CSV_DIR}/librispeech_filtered_clean.csv" \
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
    --metrics dwer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "librispeech" \
    --language en \
    --use_gpu \
    --gpu_id 3 \
    --num_workers 64 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

echo "DAC evaluation on LibriSpeech completed."
