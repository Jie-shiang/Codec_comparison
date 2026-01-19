#!/bin/bash

# NanoCodec Evaluation Script - Aishell Dataset
# V3 Metrics with full metric suite

set -e

OUTPUT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison_test"
SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
EVAL_ENV="codec_eval_pip_py39"

echo "Starting NanoCodec Evaluation - Aishell Dataset"

# NanoCodec 12.5Hz_2k - Aishell (zh)
echo "Running: NanoCodec 12.5Hz_2k - Aishell (zh)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/NanoCodec/12.5Hz_2k/aishell \
    --csv_file "${CSV_DIR}/aishell_filtered_clean.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "NanoCodec" \
    --frequency "12.5Hz_2k" \
    --causality "Causal" \
    --bit_rate "1.78" \
    --quantizers "2" \
    --codebook_size "1024" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe ter semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "aishell" \
    --language zh \
    --use_gpu \
    --gpu_id 3 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# NanoCodec 12.5Hz_4k - Aishell (zh)
echo "Running: NanoCodec 12.5Hz_4k - Aishell (zh)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/NanoCodec/12.5Hz_4k/aishell \
    --csv_file "${CSV_DIR}/aishell_filtered_clean.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "NanoCodec" \
    --frequency "12.5Hz_4k" \
    --causality "Causal" \
    --bit_rate "0.6" \
    --quantizers "4" \
    --codebook_size "1024" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe ter semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "aishell" \
    --language zh \
    --use_gpu \
    --gpu_id 3 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# NanoCodec 21.5Hz_2k - Aishell (zh)
echo "Running: NanoCodec 21.5Hz_2k - Aishell (zh)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/NanoCodec/21.5Hz_2k/aishell \
    --csv_file "${CSV_DIR}/aishell_filtered_clean.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "NanoCodec" \
    --frequency "21.5Hz_2k" \
    --causality "Causal" \
    --bit_rate "1.89" \
    --quantizers "2" \
    --codebook_size "1024" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe ter semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "aishell" \
    --language zh \
    --use_gpu \
    --gpu_id 3 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

echo "NanoCodec evaluation on Aishell completed."
