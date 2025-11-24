# FocalCodec-S 50Hz_2k - LibriSpeech
echo "Running 1/2: FocalCodec-S 50Hz_2k - LibriSpeech"
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
    --metrics dwer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64

# FocalCodec-S 50Hz_2k - Common Voice
echo "Running 2/2: FocalCodec-S 50Hz_2k - Common Voice"
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
    --metrics dcer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 2 \
    --num_workers 32  \
    --asr_batch_size 64