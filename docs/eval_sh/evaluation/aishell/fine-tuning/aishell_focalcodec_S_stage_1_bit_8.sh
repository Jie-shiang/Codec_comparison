#!/bin/bash

# FocalCodec-S Evaluation Script - Aishell Dataset
# V3 Metrics with full metric suite

set -e

OUTPUT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison_focal"
SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
EVAL_ENV="codec_eval_pip_py39"

echo "Starting FocalCodec-S Evaluation - Aishell Dataset"

# FocalCodec-S 50Hz_2k - Aishell (zh)
echo "Running: FocalCodec-S 50Hz_2k - Aishell (zh)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50Hz_2k_finetune/trained_stage_1_official_8bit \
    --csv_file "${CSV_DIR}/aishell_filtered_clean.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "FocalCodec-S" \
    --frequency "50Hz_2k_stage_1_8bit" \
    --causality "Causal" \
    --bit_rate "0.55" \
    --quantizers "1" \
    --codebook_size "2048" \
    --n_params "249M" \
    --training_set "LibriTTS + Libri-Light-medium" \
    --testing_set "LibriSpeech test-clean + Multilingual" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --dataset_name "aishell" \
    --language zh \
    --use_gpu \
    --gpu_id 0 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

echo "FocalCodec-S evaluation on Aishell completed."
