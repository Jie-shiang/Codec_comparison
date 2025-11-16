#!/bin/bash

# 確保任何命令失敗時立即退出
set -e

echo "Starting all 18 Segmented Evaluation Pipeline (0.5s) commands on GPU 2..."

################################################################################
# LSCodec Evaluations (Commands 1-4)
################################################################################

# 1. LSCodec 25Hz - LibriSpeech (0.5s)
echo "Running 1/16: LSCodec 25Hz - LibriSpeech"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/25Hz/librispeech_recon/0.5s \
    --segment_csv_file librispeech_test_clean_filtered_0.5s.csv \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

# 2. LSCodec 25Hz - Common Voice (0.5s)
echo "Running 2/16: LSCodec 25Hz - Common Voice"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/25Hz/common_voice_zh_CN_recon/0.5s \
    --segment_csv_file common_voice_zh_CN_train_filtered_0.5s.csv \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

# 3. LSCodec 50Hz - LibriSpeech (0.5s)
echo "Running 3/16: LSCodec 50Hz - LibriSpeech"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/librispeech_recon/0.5s \
    --segment_csv_file librispeech_test_clean_filtered_0.5s.csv \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

# 4. LSCodec 50Hz - Common Voice (0.5s)
echo "Running 4/16: LSCodec 50Hz - Common Voice"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/common_voice_zh_CN_recon/0.5s \
    --segment_csv_file common_voice_zh_CN_train_filtered_0.5s.csv \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result
    
################################################################################
# FocalCodec Evaluations (Commands 5-10)
################################################################################

# 5. FocalCodec 12.5Hz - LibriSpeech (0.5s)
echo "Running 5/16: FocalCodec 12.5Hz - LibriSpeech (0.5s segments)"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/12.5HZ/librispeech/0.5s \
    --segment_csv_file librispeech_test_clean_filtered_0.5s.csv  \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

# 6. FocalCodec 12.5Hz - Common Voice (0.5s)
echo "Running 6/16: FocalCodec 12.5Hz - Common Voice (0.5s segments)"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/12.5HZ/commonvoice/0.5s \
    --segment_csv_file common_voice_zh_CN_train_filtered_0.5s.csv \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

# 7. FocalCodec 25Hz - LibriSpeech (0.5s)
echo "Running 7/16: FocalCodec 25Hz - LibriSpeech (0.5s segments)"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/25HZ/librispeech/0.5s \
    --segment_csv_file librispeech_test_clean_filtered_0.5s.csv \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

# 8. FocalCodec 25Hz - Common Voice (0.5s)
echo "Running 8/16: FocalCodec 25Hz - Common Voice (0.5s segments)"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/25HZ/commonvoice/0.5s \
    --segment_csv_file common_voice_zh_CN_train_filtered_0.5s.csv \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

# 9. FocalCodec 50Hz - LibriSpeech (0.5s)
echo "Running 9/16: FocalCodec 50Hz - LibriSpeech (0.5s segments)"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/50HZ/librispeech/0.5s \
    --segment_csv_file librispeech_test_clean_filtered_0.5s.csv \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

# 10. FocalCodec 50Hz - Common Voice (0.5s)
echo "Running 10/16: FocalCodec 50Hz - Common Voice (0.5s segments)"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/50HZ/commonvoice/0.5s \
    --segment_csv_file common_voice_zh_CN_train_filtered_0.5s.csv \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

################################################################################
# FocalCodec-S Evaluations (Commands 11-16)
################################################################################

# 11. FocalCodec-S 50Hz_2k - LibriSpeech (0.5s)
echo "Running 11/16: FocalCodec-S 50Hz_2k - LibriSpeech (0.5s segments)"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/2K/librispeech/0.5s \
    --segment_csv_file librispeech_test_clean_filtered_0.5s.csv \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

# 12. FocalCodec-S 50Hz_2k - Common Voice (0.5s)
echo "Running 12/16: FocalCodec-S 50Hz_2k - Common Voice (0.5s segments)"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/2K/commonvoice/0.5s \
    --segment_csv_file common_voice_zh_CN_train_filtered_0.5s.csv \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

# 13. FocalCodec-S 50Hz_4k - LibriSpeech (0.5s)
echo "Running 13/16: FocalCodec-S 50Hz_4k - LibriSpeech (0.5s segments)"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/4K/librispeech/0.5s \
    --segment_csv_file librispeech_test_clean_filtered_0.5s.csv \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

# 14. FocalCodec-S 50Hz_4k - Common Voice (0.5s)
echo "Running 14/16: FocalCodec-S 50Hz_4k - Common Voice (0.5s segments)"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/4K/commonvoice/0.5s \
    --segment_csv_file common_voice_zh_CN_train_filtered_0.5s.csv \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

# 15. FocalCodec-S 50Hz_65k - LibriSpeech (0.5s)
echo "Running 15/16: FocalCodec-S 50Hz_65k - LibriSpeech (0.5s segments)"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/65K/librispeech/0.5s \
    --segment_csv_file librispeech_test_clean_filtered_0.5s.csv \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

# 16. FocalCodec-S 50Hz_65k - Common Voice (0.5s)
echo "Running 16/16: FocalCodec-S 50Hz_65k - Common Voice (0.5s segments)"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/65K/commonvoice/0.5s \
    --segment_csv_file common_voice_zh_CN_train_filtered_0.5s.csv \
    --segment_length 0.5 \
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
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

################################################################################
# BigCodec Evaluations (Commands 17-18)
################################################################################

# 17. BigCodec 80Hz - LibriSpeech
echo "Running 17/18: BigCodec 80Hz - LibriSpeech (0.5s segments)"
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/BigCodec/80Hz/librispeech/0.5s \
    --segment_csv_file librispeech_test_clean_filtered_0.5s.csv  \
    --segment_length 0.5 \
    --model_name "BigCodec" \
    --frequency "80Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.04" \
    --quantizers "1" \
    --codebook_size "8192" \
    --n_params "159M" \
    --training_set "LibriSpeech" \
    --testing_set "LibriSpeech + test-clean + MLS" \
    --metrics dwer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

# 18. BigCodec 80Hz - Common Voice
echo "Running 18/18: BigCodec 80Hz - Common Voice (0.5s segments)"
python segmented_evaluation_pipeline.py\
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/BigCodec/80Hz/commonvoice/0.5s \
    --segment_csv_file common_voice_zh_CN_train_filtered_0.5s.csv \
    --segment_length 0.5 \
    --model_name "BigCodec" \
    --frequency "80Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.04" \
    --quantizers "1" \
    --codebook_size "8192" \
    --n_params "159M" \
    --training_set "LibriSpeech" \
    --testing_set "LibriSpeech + test-clean + MLS" \
    --metrics dcer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64 \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result

echo "All 18 Segmented evaluation (0.5s) commands completed."
