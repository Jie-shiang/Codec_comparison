#!/bin/bash

# LSCodec Evaluation Script - Mdcc Dataset
# V3 Metrics with full metric suite

set -e

OUTPUT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison"
SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
EVAL_ENV="codec_eval_pip_py39"

echo "Starting LSCodec Evaluation - Mdcc Dataset"

# LSCodec 25Hz - Mdcc (yue)
echo "Running: LSCodec 25Hz - Mdcc (yue)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/25Hz/mdcc \
    --csv_file "${CSV_DIR}/mdcc_filtered_clean_test.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "LSCodec" \
    --frequency "25Hz" \
    --causality "Non-Causal" \
    --bit_rate "0.25" \
    --quantizers "1" \
    --codebook_size "1024" \
    --n_params "49M" \
    --training_set "LibriTTS All train splits" \
    --testing_set "LibriTTS test-clean + testset-B" \
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

# LSCodec 50Hz - Mdcc (yue)
echo "Running: LSCodec 50Hz - Mdcc (yue)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/mdcc \
    --csv_file "${CSV_DIR}/mdcc_filtered_clean_test.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "LSCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "0.45" \
    --quantizers "1" \
    --codebook_size "300" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
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

echo "LSCodec evaluation on Mdcc completed."
