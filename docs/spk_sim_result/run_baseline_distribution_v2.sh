#!/bin/bash

set -e

echo "Starting Speaker Similarity Baseline Distribution Calculation V2"
echo "================================================================"
echo ""

# Activate conda environment if needed
# Uncomment and modify if you need to activate a specific environment
# source /opt/conda/anaconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# Base paths
CSV_DIR="/home/jieshiang/Desktop/GitHub/Codec_comparison/csv"
AUDIO_BASE="/mnt/Internal/ASR"
OUTPUT_BASE="/home/jieshiang/Desktop/GitHub/Codec_comparison"
SCRIPT_PATH="/home/jieshiang/Desktop/GitHub/Codec_comparison/utils/compute_baseline_distribution_v2.py"

################################################################################
# Common Voice (Chinese)
################################################################################
echo "1/3: Processing Common Voice (Chinese)..."
echo "Using CAM++ model for Chinese speaker similarity"
echo ""

python "${SCRIPT_PATH}" \
    --dataset commonvoice \
    --language zh \
    --csv_path "${CSV_DIR}/common_voice_zh_CN_train_filtered.csv" \
    --audio_base_path "${AUDIO_BASE}" \
    --output_dir "${OUTPUT_BASE}/common_voice_spk_sim_result" \
    --gpu_id 3 \
    --num_negative_speakers 5 \
    --max_positive_pairs 20

echo ""
echo "Common Voice baseline calculation completed!"
echo ""

################################################################################
# AISHELL (Chinese)
################################################################################
echo "2/3: Processing AISHELL (Chinese)..."
echo "Using CAM++ model for Chinese speaker similarity"
echo ""

python "${SCRIPT_PATH}" \
    --dataset aishell \
    --language zh \
    --csv_path "${CSV_DIR}/aishell_filtered_clean.csv" \
    --audio_base_path "${AUDIO_BASE}" \
    --output_dir "${OUTPUT_BASE}/aishell_spk_sim_result" \
    --gpu_id 3 \
    --num_negative_speakers 5 \
    --max_positive_pairs 20

echo ""
echo "AISHELL baseline calculation completed!"
echo ""

################################################################################
# LibriSpeech (English)
################################################################################
echo "3/3: Processing LibriSpeech (English)..."
echo "Using ResNet3 (WeSpeaker) model for English speaker similarity"
echo ""

python "${SCRIPT_PATH}" \
    --dataset librispeech \
    --language en \
    --csv_path "${CSV_DIR}/librispeech_test_clean_filtered.csv" \
    --audio_base_path "${AUDIO_BASE}" \
    --output_dir "${OUTPUT_BASE}/librispeech_spk_sim_result" \
    --gpu_id 3 \
    --num_negative_speakers 5 \
    --max_positive_pairs 20

echo ""
echo "LibriSpeech baseline calculation completed!"
echo ""

################################################################################
# Summary
################################################################################
echo "================================================================"
echo "All baseline distributions calculated successfully!"
echo "================================================================"
echo ""
echo "Results saved to:"
echo "  - ${OUTPUT_BASE}/common_voice_spk_sim_result/"
echo "  - ${OUTPUT_BASE}/aishell_spk_sim_result/"
echo "  - ${OUTPUT_BASE}/librispeech_spk_sim_result/"
echo ""
echo "Each directory contains:"
echo "  - baseline_statistics_*.json (statistical summary)"
echo "  - positive_samples_*.csv (same-speaker pairs)"
echo "  - negative_samples_*.csv (different-speaker pairs)"
echo "  - distribution_*.png (visualization)"
echo "  - baseline_report_*.txt (human-readable report)"
echo ""
