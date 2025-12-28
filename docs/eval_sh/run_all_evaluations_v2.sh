#!/bin/bash

set -e

echo "Starting all V2 Metrics Evaluation Pipeline commands on GPU 2..."
echo "Output directory: /home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Test_New_Metrics"
echo ""

# Activate conda environment
echo "Activating conda environment: codec_eval_pip_py39"
source /opt/conda/anaconda3/etc/profile.d/conda.sh
conda activate codec_eval_pip_py39

# Output base directory for V2 metrics
OUTPUT_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Test_New_Metrics"

################################################################################
# LSCodec Evaluations (Commands 1-4)
################################################################################

# 1. LSCodec 25Hz - LibriSpeech
echo "Running 1/18: LSCodec 25Hz - LibriSpeech"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/25Hz/librispeech_recon \
    --csv_file librispeech_test_clean_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --model_name "LSCodec" \
    --frequency "25Hz" \
    --causality "Non-Causal" \
    --bit_rate "0.25" \
    --quantizers "1" \
    --codebook_size "1024" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dwer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# 2. LSCodec 25Hz - Common Voice
echo "Running 2/18: LSCodec 25Hz - Common Voice"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/25Hz/common_voice_zh_CN_recon \
    --csv_file common_voice_zh_CN_train_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --model_name "LSCodec" \
    --frequency "25Hz" \
    --causality "Non-Causal" \
    --bit_rate "0.25" \
    --quantizers "1" \
    --codebook_size "1024" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# 3. LSCodec 50Hz - LibriSpeech
echo "Running 3/18: LSCodec 50Hz - LibriSpeech"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/librispeech_recon \
    --csv_file librispeech_test_clean_filtered.csv \
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
    --metrics dwer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# 4. LSCodec 50Hz - Common Voice
echo "Running 4/18: LSCodec 50Hz - Common Voice"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/25Hz/common_voice_zh_CN_recon \
    --csv_file common_voice_zh_CN_train_filtered.csv \
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
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

################################################################################
# FocalCodec Evaluations (Commands 5-10)
################################################################################

# 5. FocalCodec 12.5Hz - LibriSpeech
echo "Running 5/18: FocalCodec 12.5Hz - LibriSpeech"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/12.5HZ/librispeech \
    --csv_file librispeech_test_clean_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --model_name "FocalCodec" \
    --frequency "12.5Hz" \
    --causality "Non-Causal" \
    --bit_rate "0.16" \
    --quantizers "1" \
    --codebook_size "8192" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dwer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# 6. FocalCodec 12.5Hz - Common Voice
echo "Running 6/18: FocalCodec 12.5Hz - Common Voice"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/12.5HZ/commonvoice \
    --csv_file common_voice_zh_CN_train_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --model_name "FocalCodec" \
    --frequency "12.5Hz" \
    --causality "Non-Causal" \
    --bit_rate "0.16" \
    --quantizers "1" \
    --codebook_size "8192" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# 7. FocalCodec 25Hz - LibriSpeech
echo "Running 7/18: FocalCodec 25Hz - LibriSpeech"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/25HZ/librispeech \
    --csv_file librispeech_test_clean_filtered.csv \
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
    --metrics dwer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# 8. FocalCodec 25Hz - Common Voice
echo "Running 8/18: FocalCodec 25Hz - Common Voice"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/25HZ/commonvoice \
    --csv_file common_voice_zh_CN_train_filtered.csv \
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
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# 9. FocalCodec 50Hz - LibriSpeech
echo "Running 9/18: FocalCodec 50Hz - LibriSpeech"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/50HZ/librispeech \
    --csv_file librispeech_test_clean_filtered.csv \
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
    --metrics dwer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# 10. FocalCodec 50Hz - Common Voice
echo "Running 10/18: FocalCodec 50Hz - Common Voice"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/50HZ/commonvoice \
    --csv_file common_voice_zh_CN_train_filtered.csv \
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
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

################################################################################
# FocalCodec-S Evaluations (Commands 11-16)
################################################################################

# 11. FocalCodec-S 50Hz_2k - LibriSpeech
echo "Running 11/18: FocalCodec-S 50Hz_2k - LibriSpeech"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50Hz_2k/librispeech \
    --csv_file librispeech_test_clean_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --model_name "FocalCodec-S" \
    --frequency "50Hz_2k" \
    --causality "Causal" \
    --bit_rate "0.55" \
    --quantizers "1" \
    --codebook_size "2048" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dwer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# 12. FocalCodec-S 50Hz_2k - Common Voice
echo "Running 12/18: FocalCodec-S 50Hz_2k - Common Voice"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50Hz_2k/commonvoice \
    --csv_file common_voice_zh_CN_train_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --model_name "FocalCodec-S" \
    --frequency "50Hz_2k" \
    --causality "Causal" \
    --bit_rate "0.55" \
    --quantizers "1" \
    --codebook_size "2048" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# 13. FocalCodec-S 50Hz_4k - LibriSpeech
echo "Running 13/18: FocalCodec-S 50Hz_4k - LibriSpeech"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50Hz_4k/librispeech \
    --csv_file librispeech_test_clean_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --model_name "FocalCodec-S" \
    --frequency "50Hz_4k" \
    --causality "Causal" \
    --bit_rate "0.60" \
    --quantizers "1" \
    --codebook_size "4096" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dwer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# 14. FocalCodec-S 50Hz_4k - Common Voice
echo "Running 14/18: FocalCodec-S 50Hz_4k - Common Voice"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50Hz_4k/commonvoice \
    --csv_file common_voice_zh_CN_train_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --model_name "FocalCodec-S" \
    --frequency "50Hz_4k" \
    --causality "Causal" \
    --bit_rate "0.60" \
    --quantizers "1" \
    --codebook_size "4096" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# 15. FocalCodec-S 50Hz_65k - LibriSpeech
echo "Running 15/18: FocalCodec-S 50Hz_65k - LibriSpeech"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50Hz_65k/librispeech \
    --csv_file librispeech_test_clean_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --model_name "FocalCodec-S" \
    --frequency "50Hz_65k" \
    --causality "Causal" \
    --bit_rate "0.80" \
    --quantizers "1" \
    --codebook_size "65536" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dwer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# 16. FocalCodec-S 50Hz_65k - Common Voice
echo "Running 16/18: FocalCodec-S 50Hz_65k - Common Voice"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50Hz_65k/commonvoice \
    --csv_file common_voice_zh_CN_train_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --model_name "FocalCodec-S" \
    --frequency "50Hz_65k" \
    --causality "Causal" \
    --bit_rate "0.80" \
    --quantizers "1" \
    --codebook_size "65536" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging
    
################################################################################
# BigCodec Evaluations (Commands 17-18)
################################################################################

# 17. BigCodec 80Hz - LibriSpeech
echo "Running 17/18: BigCodec 80Hz - LibriSpeech"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/BigCodec/80Hz/librispeech \
    --csv_file librispeech_test_clean_filtered.csv \
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
    --metrics dwer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

# 18. BigCodec 80Hz - Common Voice
echo "Running 18/18: BigCodec 80Hz - Common Voice"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/BigCodec/80Hz/commonvoice_test \
    --csv_file common_voice_zh_CN_train_filtered.csv \
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
    --metrics dcer MOS_Quality MOS_Naturalness pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64 \
    --use_v2_metrics \
    --output_base_dir "$OUTPUT_DIR" \
    --enable_logging

echo "All 18 V2 metrics evaluation commands completed."
echo "Results saved to: $OUTPUT_DIR"
