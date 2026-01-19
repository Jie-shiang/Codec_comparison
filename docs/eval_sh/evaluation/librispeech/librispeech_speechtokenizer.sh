#!/bin/bash

# SpeechTokenizer Evaluation Script - LibriSpeech Dataset
# V3 Metrics with full metric suite

set -e

OUTPUT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison_test"
SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
EVAL_ENV="codec_eval_pip_py39"

echo "Starting SpeechTokenizer Evaluation - LibriSpeech Dataset"

# SpeechTokenizer 50Hz - LibriSpeech (English)
echo "Running: SpeechTokenizer 50Hz - LibriSpeech (English)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/SpeechTokenizer/50Hz/librispeech \
    --csv_file "${CSV_DIR}/librispeech_filtered_clean.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "SpeechTokenizer" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.0" \
    --quantizers "8" \
    --codebook_size "1024" \
    --n_params "N/A" \
    --training_set "LibriSpeech" \
    --testing_set "LibriSpeech test-clean" \
    --metrics dwer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "librispeech" \
    --language en \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 64 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

echo "SpeechTokenizer evaluation on LibriSpeech completed."
