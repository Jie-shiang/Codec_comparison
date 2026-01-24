#!/bin/bash

# EnCodec Evaluation Script - LibriSpeech Dataset
# V3 Metrics with full metric suite

set -e

OUTPUT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison_test"
SCRIPT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison"
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
EVAL_ENV="codec_eval_pip_py39"

echo "Starting EnCodec Evaluation - LibriSpeech Dataset"


# EnCodec 75Hz_1.5k - LIBRISPEECH (en)
echo "Running: EnCodec 75Hz_1.5k - LIBRISPEECH (en)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/EnCodec/75Hz_1.5k/librispeech \
    --csv_file "${CSV_DIR}/librispeech_test_clean.csv" \
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
    --dataset_name "librispeech" \
    --language en \
    --use_gpu \
    --gpu_id 0 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v3_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# EnCodec 75Hz_4k - LibriSpeech (English) - 3.0kbps
echo "Running: EnCodec 75Hz_4k - LibriSpeech (English)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/EnCodec/75Hz_4k/librispeech \
    --csv_file "${CSV_DIR}/librispeech_filtered_clean.csv" \
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

# EnCodec 75Hz_8k - LibriSpeech (English) - 6.0kbps
echo "Running: EnCodec 75Hz_8k - LibriSpeech (English)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/EnCodec/75Hz_8k/librispeech \
    --csv_file "${CSV_DIR}/librispeech_filtered_clean.csv" \
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

# EnCodec 75Hz_16k - LibriSpeech (English) - 12.0kbps
echo "Running: EnCodec 75Hz_16k - LibriSpeech (English)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/EnCodec/75Hz_16k/librispeech \
    --csv_file "${CSV_DIR}/librispeech_filtered_clean.csv" \
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

# EnCodec 75Hz_32k - LibriSpeech (English) - 24.0kbps
echo "Running: EnCodec 75Hz_32k - LibriSpeech (English)"
conda run -n "$EVAL_ENV" python "${SCRIPT_DIR}/fast_evaluation_pipeline.py" \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/EnCodec/75Hz_32k/librispeech \
    --csv_file "${CSV_DIR}/librispeech_filtered_clean.csv" \
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

echo "EnCodec evaluation on LibriSpeech completed."
