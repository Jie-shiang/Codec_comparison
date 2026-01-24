#!/bin/bash

# EnCodec Evaluation Script - Commonvoice Dataset
# V3 Metrics with full metric suite

set -e

OUTPUT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison_test"
SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
EVAL_ENV="codec_eval_pip_py39"

echo "Starting EnCodec Evaluation - Commonvoice Dataset"


# EnCodec 75Hz_1.5k - COMMONVOICE (en)
echo "Running: EnCodec 75Hz_1.5k - COMMONVOICE (en)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/EnCodec/75Hz_1.5k/commonvoice \
    --csv_file "${CSV_DIR}/commonvoice_filtered_clean.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "EnCodec" \
    --frequency "75Hz_1.5k" \
    --causality "Causal" \
    --bit_rate "1.5" \
    --quantizers "2" \
    --codebook_size "1024" \
    --n_params "16M" \
    --training_set "Multi-domain" \
    --testing_set "LibriSpeech test-clean" \
    --metrics dwer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "commonvoice" \
    --language en \
    --use_gpu \
    --gpu_id 0 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# EnCodec 75Hz_4k - Commonvoice (zh)
echo "Running: EnCodec 75Hz_4k - Commonvoice (zh)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/EnCodec/75Hz_4k/commonvoice \
    --csv_file "${CSV_DIR}/commonvoice_filtered_clean.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "EnCodec" \
    --frequency "75Hz_4k" \
    --causality "Causal" \
    --bit_rate "3.0" \
    --quantizers "4" \
    --codebook_size "1024" \
    --n_params "16M" \
    --training_set "Multi-domain" \
    --testing_set "LibriSpeech test-clean" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe ter semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "commonvoice" \
    --language zh \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# EnCodec 75Hz_8k - Commonvoice (zh)
echo "Running: EnCodec 75Hz_8k - Commonvoice (zh)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/EnCodec/75Hz_8k/commonvoice \
    --csv_file "${CSV_DIR}/commonvoice_filtered_clean.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "EnCodec" \
    --frequency "75Hz_8k" \
    --causality "Causal" \
    --bit_rate "6.0" \
    --quantizers "8" \
    --codebook_size "1024" \
    --n_params "16M" \
    --training_set "Multi-domain" \
    --testing_set "LibriSpeech test-clean" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe ter semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "commonvoice" \
    --language zh \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# EnCodec 75Hz_16k - Commonvoice (zh)
echo "Running: EnCodec 75Hz_16k - Commonvoice (zh)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/EnCodec/75Hz_16k/commonvoice \
    --csv_file "${CSV_DIR}/commonvoice_filtered_clean.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "EnCodec" \
    --frequency "75Hz_16k" \
    --causality "Causal" \
    --bit_rate "12.0" \
    --quantizers "16" \
    --codebook_size "1024" \
    --n_params "16M" \
    --training_set "Multi-domain" \
    --testing_set "LibriSpeech test-clean" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe ter semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "commonvoice" \
    --language zh \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# EnCodec 75Hz_32k - Commonvoice (zh)
echo "Running: EnCodec 75Hz_32k - Commonvoice (zh)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/EnCodec/75Hz_32k/commonvoice \
    --csv_file "${CSV_DIR}/commonvoice_filtered_clean.csv" \
    --original_dir /mnt/Internal/ASR \
    --model_name "EnCodec" \
    --frequency "75Hz_32k" \
    --causality "Causal" \
    --bit_rate "24.0" \
    --quantizers "32" \
    --codebook_size "1024" \
    --n_params "16M" \
    --training_set "Multi-domain" \
    --testing_set "LibriSpeech test-clean" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity vde f0_rmse gpe ter semantic_similarity \
    --dataset_type "clean" \
    --dataset_name "commonvoice" \
    --language zh \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

echo "EnCodec evaluation on Commonvoice completed."
