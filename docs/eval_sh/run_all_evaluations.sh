#!/bin/bash

# 確保任何命令失敗時立即退出 

# chmod +x run_all_evaluations_*.sh

set -e

echo "Starting all 16 Enhanced Evaluation Pipeline commands on GPU 2..."

################################################################################
# LSCodec Evaluations (Commands 1-4)
################################################################################

# 1. LSCodec 25Hz - LibriSpeech
echo "Running 1/16: LSCodec 25Hz - LibriSpeech"
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
    --metrics dwer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

# 2. LSCodec 25Hz - Common Voice
echo "Running 2/16: LSCodec 25Hz - Common Voice"
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
    --metrics dcer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

# 3. LSCodec 50Hz - LibriSpeech
echo "Running 3/16: LSCodec 50Hz - LibriSpeech"
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
    --metrics dwer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

# 4. LSCodec 50Hz - Common Voice
echo "Running 4/16: LSCodec 50Hz - Common Voice"
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
    --metrics dcer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

################################################################################
# FocalCodec Evaluations (Commands 5-10)
################################################################################

# 5. FocalCodec 12.5Hz - LibriSpeech
echo "Running 5/16: FocalCodec 12.5Hz - LibriSpeech"
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
    --metrics dwer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

# 6. FocalCodec 12.5Hz - Common Voice
echo "Running 6/16: FocalCodec 12.5Hz - Common Voice"
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
    --metrics dcer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

# 7. FocalCodec 25Hz - LibriSpeech
echo "Running 7/16: FocalCodec 25Hz - LibriSpeech"
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
    --metrics dwer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

# 8. FocalCodec 25Hz - Common Voice
echo "Running 8/16: FocalCodec 25Hz - Common Voice"
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
    --metrics dcer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

# 9. FocalCodec 50Hz - LibriSpeech
echo "Running 9/16: FocalCodec 50Hz - LibriSpeech"
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
    --metrics dwer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

# 10. FocalCodec 50Hz - Common Voice
echo "Running 10/16: FocalCodec 50Hz - Common Voice"
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
    --metrics dcer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

################################################################################
# FocalCodec-S Evaluations (Commands 11-16)
################################################################################

# 11. FocalCodec-S 50Hz_2k - LibriSpeech
echo "Running 11/16: FocalCodec-S 50Hz_2k - LibriSpeech"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/2K/librispeech \
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
    --metrics dwer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

# 12. FocalCodec-S 50Hz_2k - Common Voice
echo "Running 12/16: FocalCodec-S 50Hz_2k - Common Voice"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/2K/commonvoice \
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
    --metrics dcer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

# 13. FocalCodec-S 50Hz_4k - LibriSpeech
echo "Running 13/16: FocalCodec-S 50Hz_4k - LibriSpeech"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/4K/librispeech \
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
    --metrics dwer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

# 14. FocalCodec-S 50Hz_4k - Common Voice
echo "Running 14/16: FocalCodec-S 50Hz_4k - Common Voice"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/4K/commonvoice \
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
    --metrics dcer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

# 15. FocalCodec-S 50Hz_65k - LibriSpeech
echo "Running 15/16: FocalCodec-S 50Hz_65k - LibriSpeech"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/65K/librispeech \
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
    --metrics dwer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

# 16. FocalCodec-S 50Hz_65k - Common Voice
echo "Running 16/16: FocalCodec-S 50Hz_65k - Common Voice"
python fast_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/65K/commonvoice \
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
    --metrics dcer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

echo "All 16 evaluation commands completed."