#!/bin/bash

# 確保任何命令失敗時立即退出
set -e

echo "Starting all Noise Evaluation Pipeline commands on GPU 2..."

################################################################################
# LSCodec Evaluations (Commands 1-4)
################################################################################

# 1. LSCodec 25Hz - LibriSpeech (Noise)
echo "Running 1/28: LSCodec 25Hz - LibriSpeech (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/25Hz/librispeech_recon/noise \
    --csv_file noise/librispeech_test_clean_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
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
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 2. LSCodec 25Hz - Common Voice (Noise)
echo "Running 2/28: LSCodec 25Hz - Common Voice (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/25Hz/common_voice_zh_CN_recon/noise \
    --csv_file noise/common_voice_zh_CN_train_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
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
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 3. LSCodec 50Hz - LibriSpeech (Noise)
echo "Running 3/28: LSCodec 50Hz - LibriSpeech (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/librispeech_recon/noise \
    --csv_file noise/librispeech_test_clean_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
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
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 4. LSCodec 50Hz - Common Voice (Noise)
echo "Running 4/28: LSCodec 50Hz - Common Voice (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/common_voice_zh_CN_recon/noise \
    --csv_file noise/common_voice_zh_CN_train_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
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
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

################################################################################
# FocalCodec Evaluations (Commands 5-16)
################################################################################

# 5. FocalCodec 12.5Hz - LibriSpeech (Noise)
echo "Running 5/28: FocalCodec 12.5Hz - LibriSpeech (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/12.5HZ/librispeech/noise \
    --csv_file noise/librispeech_test_clean_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
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
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 6. FocalCodec 12.5Hz - Common Voice (Noise)
echo "Running 6/28: FocalCodec 12.5Hz - Common Voice (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/12.5HZ/commonvoice/noise \
    --csv_file noise/common_voice_zh_CN_train_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
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
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 7. FocalCodec 25Hz - LibriSpeech (Noise)
echo "Running 7/28: FocalCodec 25Hz - LibriSpeech (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/25HZ/librispeech/noise \
    --csv_file noise/librispeech_test_clean_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
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
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 8. FocalCodec 25Hz - Common Voice (Noise)
echo "Running 8/28: FocalCodec 25Hz - Common Voice (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/25HZ/commonvoice/noise \
    --csv_file noise/common_voice_zh_CN_train_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
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
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 9. FocalCodec 50Hz - LibriSpeech (Noise)
echo "Running 9/28: FocalCodec 50Hz - LibriSpeech (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/50HZ/librispeech/noise \
    --csv_file noise/librispeech_test_clean_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
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
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 10. FocalCodec 50Hz - Common Voice (Noise)
echo "Running 10/28: FocalCodec 50Hz - Common Voice (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec/50HZ/commonvoice/noise \
    --csv_file noise/common_voice_zh_CN_train_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
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
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

################################################################################
# FocalCodec-S Evaluations (Commands 11-16)
################################################################################

# 11. FocalCodec-S 50Hz_2k - LibriSpeech (Noise)
echo "Running 11/28: FocalCodec-S 50Hz_2k - LibriSpeech (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/2K/librispeech/noise \
    --csv_file noise/librispeech_test_clean_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
    --model_name "FocalCodec-S" \
    --frequency "50Hz_2k" \
    --causality "Causal" \
    --bit_rate "0.26" \
    --quantizers "1" \
    --codebook_size "2048" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dwer utmos pesq stoi speaker_similarity \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 12. FocalCodec-S 50Hz_2k - Common Voice (Noise)
echo "Running 12/28: FocalCodec-S 50Hz_2k - Common Voice (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/2K/commonvoice/noise \
    --csv_file noise/common_voice_zh_CN_train_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
    --model_name "FocalCodec-S" \
    --frequency "50Hz_2k" \
    --causality "Causal" \
    --bit_rate "0.26" \
    --quantizers "1" \
    --codebook_size "2048" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dcer utmos pesq stoi speaker_similarity \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 13. FocalCodec-S 50Hz_4k - LibriSpeech (Noise)
echo "Running 13/28: FocalCodec-S 50Hz_4k - LibriSpeech (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/4K/librispeech/noise \
    --csv_file noise/librispeech_test_clean_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
    --model_name "FocalCodec-S" \
    --frequency "50Hz_4k" \
    --causality "Causal" \
    --bit_rate "0.40" \
    --quantizers "1" \
    --codebook_size "4096" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dwer utmos pesq stoi speaker_similarity \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 14. FocalCodec-S 50Hz_4k - Common Voice (Noise)
echo "Running 14/28: FocalCodec-S 50Hz_4k - Common Voice (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/4K/commonvoice/noise \
    --csv_file noise/common_voice_zh_CN_train_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
    --model_name "FocalCodec-S" \
    --frequency "50Hz_4k" \
    --causality "Causal" \
    --bit_rate "0.40" \
    --quantizers "1" \
    --codebook_size "4096" \
    --n_params "N/A" \
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dcer utmos pesq stoi speaker_similarity \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 15. FocalCodec-S 50Hz_65k - LibriSpeech (Noise)
echo "Running 15/28: FocalCodec-S 50Hz_65k - LibriSpeech (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/65K/librispeech/noise \
    --csv_file noise/librispeech_test_clean_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
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
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 16. FocalCodec-S 50Hz_65k - Common Voice (Noise)
echo "Running 16/28: FocalCodec-S 50Hz_65k - Common Voice (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/FocalCodec-S/50HZ/65K/commonvoice/noise \
    --csv_file noise/common_voice_zh_CN_train_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
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
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

################################################################################
# BigCodec Evaluations (Commands 17-18)
################################################################################

# 17. BigCodec 80Hz - LibriSpeech (Noise)
echo "Running 17/28: BigCodec 80Hz - LibriSpeech (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/BigCodec/80Hz/librispeech/noise \
    --csv_file noise/librispeech_test_clean_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
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
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 18. BigCodec 80Hz - Common Voice (Noise)
echo "Running 18/28: BigCodec 80Hz - Common Voice (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/BigCodec/80Hz/commonvoice/noise \
    --csv_file noise/common_voice_zh_CN_train_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
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
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

################################################################################
# NanoCodec Evaluations (Commands 19-24)
################################################################################

# 19. NanoCodec 12.5Hz_2k - LibriSpeech (Noise)
echo "Running 19/28: NanoCodec 12.5Hz_2k - LibriSpeech (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/NanoCodec/12.5Hz_2k/librispeech/noise \
    --csv_file noise/librispeech_test_clean_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
    --model_name "NanoCodec" \
    --frequency "12.5Hz_2k" \
    --causality "Non-Causal" \
    --bit_rate "0.60" \
    --quantizers "4" \
    --codebook_size "4032" \
    --n_params "N/A" \
    --training_set "Common Voice 3200 hrs + MLS English 25500 hrs" \
    --testing_set "MLS test set + DAPS clean dataset" \
    --metrics dwer utmos pesq stoi speaker_similarity \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 20. NanoCodec 12.5Hz_2k - Common Voice (Noise)
echo "Running 20/28: NanoCodec 12.5Hz_2k - Common Voice (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/NanoCodec/12.5Hz_2k/commonvoice/noise \
    --csv_file noise/common_voice_zh_CN_train_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
    --model_name "NanoCodec" \
    --frequency "12.5Hz_2k" \
    --causality "Non-Causal" \
    --bit_rate "0.60" \
    --quantizers "4" \
    --codebook_size "4032" \
    --n_params "N/A" \
    --training_set "Common Voice 3200 hrs + MLS English 25500 hrs" \
    --testing_set "MLS test set + DAPS clean dataset" \
    --metrics dcer utmos pesq stoi speaker_similarity \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 21. NanoCodec 12.5Hz_4k - LibriSpeech (Noise)
echo "Running 21/28: NanoCodec 12.5Hz_4k - LibriSpeech (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/NanoCodec/12.5Hz_4k/librispeech/noise \
    --csv_file noise/librispeech_test_clean_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
    --model_name "NanoCodec" \
    --frequency "12.5Hz_4k" \
    --causality "Non-Causal" \
    --bit_rate "1.78" \
    --quantizers "13" \
    --codebook_size "2016" \
    --n_params "N/A" \
    --training_set "Common Voice 3200 hrs + MLS English 25500 hrs" \
    --testing_set "MLS test set + DAPS clean dataset" \
    --metrics dwer utmos pesq stoi speaker_similarity \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 22. NanoCodec 12.5Hz_4k - Common Voice (Noise)
echo "Running 22/28: NanoCodec 12.5Hz_4k - Common Voice (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/NanoCodec/12.5Hz_4k/commonvoice/noise \
    --csv_file noise/common_voice_zh_CN_train_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
    --model_name "NanoCodec" \
    --frequency "12.5Hz_4k" \
    --causality "Non-Causal" \
    --bit_rate "1.78" \
    --quantizers "13" \
    --codebook_size "2016" \
    --n_params "N/A" \
    --training_set "Common Voice 3200 hrs + MLS English 25500 hrs" \
    --testing_set "MLS test set + DAPS clean dataset" \
    --metrics dcer utmos pesq stoi speaker_similarity \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 23. NanoCodec 21.5Hz_2k - LibriSpeech (Noise)
echo "Running 23/28: NanoCodec 21.5Hz_2k - LibriSpeech (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/NanoCodec/21.5Hz_2k/librispeech/noise \
    --csv_file noise/librispeech_test_clean_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
    --model_name "NanoCodec" \
    --frequency "21.5Hz_2k" \
    --causality "Non-Causal" \
    --bit_rate "1.89" \
    --quantizers "8" \
    --codebook_size "2016" \
    --n_params "N/A" \
    --training_set "Common Voice 3200 hrs + MLS English 25500 hrs" \
    --testing_set "MLS test set + DAPS clean dataset" \
    --metrics dwer utmos pesq stoi speaker_similarity \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

# 24. NanoCodec 21.5Hz_2k - Common Voice (Noise)
echo "Running 24/28: NanoCodec 21.5Hz_2k - Common Voice (Noise)"
python noise_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/NanoCodec/21.5Hz_2k/commonvoice/noise \
    --csv_file noise/common_voice_zh_CN_train_noise.csv \
    --original_dir /mnt/Internal/jieshiang/Noise_Result \
    --model_name "NanoCodec" \
    --frequency "21.5Hz_2k" \
    --causality "Non-Causal" \
    --bit_rate "1.89" \
    --quantizers "8" \
    --codebook_size "2016" \
    --n_params "N/A" \
    --training_set "Common Voice 3200 hrs + MLS English 25500 hrs" \
    --testing_set "MLS test set + DAPS clean dataset" \
    --metrics dcer utmos pesq stoi speaker_similarity \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32 \
    --asr_batch_size 64

echo "All 24 Noise evaluation commands completed."