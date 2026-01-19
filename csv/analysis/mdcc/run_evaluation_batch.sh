#!/bin/bash
#
# Cantonese Audio TRUE BATCH Evaluation Runner (Optimized)
# Usage: bash run_evaluation_batch.sh [gpu_id] [batch_size]
#
# This is the OPTIMIZED version with TRUE BATCH PROCESSING
#

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default parameters
GPU_ID=${1:-0}
BATCH_SIZE=${2:-128}  # Default 32 for TRUE batch processing

# Input and output files (same file - will add evaluation columns)
INPUT_CSV="mdcc_filtered_full.csv"
OUTPUT_CSV="mdcc_filtered_full.csv"

# Base path for audio files
BASE_PATH="/mnt/Internal/ASR/mdcc/dataset"

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="evaluation_cantonese_${TIMESTAMP}.log"

echo "================================================================================"
echo "Cantonese Audio TRUE BATCH Evaluation (OPTIMIZED)"
echo "================================================================================"
echo "Working Dir:  $SCRIPT_DIR"
echo "Input CSV:    $INPUT_CSV"
echo "Output CSV:   $OUTPUT_CSV"
echo "GPU ID:       $GPU_ID"
echo "Batch Size:   $BATCH_SIZE (TRUE batch processing)"
echo "Base Path:    $BASE_PATH"
echo "Log File:     $LOG_FILE"
echo "================================================================================"
echo ""
echo "ASR Model:    Whisper-large-v3 (Cantonese)"
echo "CER:          fast_cer() from metrics_evaluator_v3.py"
echo "TER:          calculate_ter() from metrics_evaluator_v3.py (fixed pycantonese)"
echo "MOS:          NISQA v2 + UTMOS"
echo ""
echo "Progress will be logged to: $LOG_FILE"
echo "CSV is saved after each batch for resume capability"
echo ""

# Activate conda environment and run BATCH version
conda run -n codec_eval_pip_py39 python evaluate_cantonese_batch.py \
    "$INPUT_CSV" \
    "$OUTPUT_CSV" \
    --base-path "$BASE_PATH" \
    --gpu "$GPU_ID" \
    --batch-size "$BATCH_SIZE" \
    --log-file "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
else
    echo "Evaluation exited with code $EXIT_CODE"
fi
echo "Results saved to: $OUTPUT_CSV"
echo "Log saved to: $LOG_FILE"
echo "================================================================================"

exit $EXIT_CODE
