#!/bin/bash

# FocalCodec Evaluation Script - Mdcc Dataset
# V3 Metrics with full metric suite

set -e

OUTPUT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison"
SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
EVAL_ENV="codec_eval_pip_py39"

echo "Starting FocalCodec Evaluation - Mdcc Dataset"

# FocalCodec 12.5Hz - Mdcc (yue)
echo "Running: FocalCodec 12.5Hz - Mdcc (yue)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/12.5HZ/mdcc \
    --csv_file "${CSV_DIR}/mdcc_filtered_clean_test.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "FocalCodec" \
    --frequency "12.5Hz" \
    --causality "Non-Causal" \
    --bit_rate "0.16" \
    --quantizers "1" \
    --codebook_size "8192" \
    --n_params "142M" \
    --training_set "LibriTTS train-clean-100" \
    --testing_set "LibriSpeech test-clean + Multilingual" \
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

# FocalCodec 25Hz - Mdcc (yue)
echo "Running: FocalCodec 25Hz - Mdcc (yue)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/25HZ/mdcc \
    --csv_file "${CSV_DIR}/mdcc_filtered_clean_test.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "FocalCodec" \
    --frequency "25Hz" \
    --causality "Non-Causal" \
    --bit_rate "0.33" \
    --quantizers "1" \
    --codebook_size "8192" \
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

# FocalCodec 50Hz - Mdcc (yue)
echo "Running: FocalCodec 50Hz - Mdcc (yue)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/50HZ/mdcc \
    --csv_file "${CSV_DIR}/mdcc_filtered_clean_test.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "FocalCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "0.65" \
    --quantizers "1" \
    --codebook_size "8192" \
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

echo "FocalCodec evaluation on Mdcc completed."
