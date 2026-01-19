#!/bin/bash
set -e

OUTPUT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison_focal"
SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
EVAL_ENV="codec_eval_pip_py39"

echo "Starting FocalCodec-S Stage 2.1 (11-bit) Evaluation"

conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50Hz_2k_finetune/trained_stage_2.1_11bit \
    --csv_file "${CSV_DIR}/aishell_filtered_clean.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "FocalCodec-S" \
    --frequency "50Hz_2k_stage_2.1_11bit" \
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

echo "Evaluation completed."
