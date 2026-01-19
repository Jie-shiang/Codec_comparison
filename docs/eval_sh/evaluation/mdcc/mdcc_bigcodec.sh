#!/bin/bash

# BigCodec Evaluation Script - Mdcc Dataset
# V3 Metrics with full metric suite

set -e

OUTPUT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison"
SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
EVAL_ENV="codec_eval_pip_py39"

echo "Starting BigCodec Evaluation - Mdcc Dataset"

# BigCodec 80Hz - Mdcc (yue)
echo "Running: BigCodec 80Hz - Mdcc (yue)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/BigCodec/80Hz/mdcc \
    --csv_file "${CSV_DIR}/mdcc_filtered_clean_test.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "BigCodec" \
    --frequency "80Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.04" \
    --quantizers "1" \
    --codebook_size "8192" \
    --n_params "159M" \
    --training_set "LibriSpeech" \
    --testing_set "LibriSpeech + test-clean + MLS" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe ter semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "mdcc" \
    --language yue \
    --use_gpu \
    --gpu_id 1 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

echo "BigCodec evaluation on Mdcc completed."
