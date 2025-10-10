# Segmented Audio Evaluation

## Overview

This feature enables evaluation of audio codecs on segmented audio files. This is particularly useful for:
- Testing codec performance on different audio lengths
- Analyzing quality degradation over time
- Optimizing codec parameters for specific segment lengths

## Complete Workflow

### **Step 1: Split Audio Files**

Split your original audio files into segments of specified length:

```bash
python audio_splitter.py \
    --csv_file librispeech_test_clean_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --segment_length 1.0 \
    --output_format wav

# Split Common Voice files (2.0s segments)
python audio_splitter.py \
    --csv_file common_voice_zh_CN_train_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --segment_length 2.0 \
    --output_format wav
```

**What it does**:
- Splits each audio file into segments of specified length
- Last segment preserves remaining audio (even if shorter)
- Converts all segments to specified format (default: wav)
- Names segments: `original_001.wav`, `original_002.wav`, etc.

**Output Structure**:
```
Original files:
/mnt/Internal/ASR/librispeech/LibriSpeech/test-clean/
└── 61/70968/
    └── 61-70968-0000.flac  (3.5s)

Split files (1.0s):
/mnt/Internal/ASR/librispeech/LibriSpeech/test-clean/1.0s/
└── 61/70968/
    ├── 61-70968-0000_001.wav  (1.0s)
    ├── 61-70968-0000_002.wav  (1.0s)
    ├── 61-70968-0000_003.wav  (1.0s)
    └── 61-70968-0000_004.wav  (0.5s)

Segment CSV:
csv/librispeech_test_clean_filtered_1.0s.csv
```

**Generated CSV Format**:
The segment CSV contains metadata for all segments:
- `segment_file_name`: Name of segment file (e.g., `61-70968-0000_001.wav`)
- `segment_file_path`: Relative path to segment
- `original_file_name`: Original file name
- `original_file_path`: Original file path
- `segment_index`: Segment number (001, 002, 003, etc.)
- `segment_duration`: Actual duration of segment
- `original_duration`: Duration of original file
- `transcription`: Full transcription (same for all segments)
- `speaker_id`: Speaker ID (for Common Voice datasets)

---

### **Step 2: Process Segments with Your Codec**

Process the split segments with your codec. Your codec should:
1. Read from: `/mnt/Internal/ASR/.../test-clean/1.0s/`
2. Process each segment independently
3. Save to: `/mnt/Internal/jieshiang/Inference_Result/YourCodec/YourFreq/dataset_name/1.0s/`
4. Name inference files: `original_001_inference.wav`, `original_002_inference.wav`, etc.

**Expected Inference Structure**:
```
/mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/librispeech_recon/
└── 1.0s/
    ├── 61-70968-0000_001_inference.wav
    ├── 61-70968-0000_002_inference.wav
    ├── 61-70968-0000_003_inference.wav
    ├── 61-70968-0000_004_inference.wav
    └── ...
```

---

### **Step 3: Evaluate Segmented Files**

The pipeline automatically:
1. Finds all segments for each original file
2. Merges segments back to full length
3. Computes metrics on merged audio
4. Generates detailed and summary reports

```bash
# Evaluate 1.0s segmented LibriSpeech files
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/librispeech_recon \
    --segment_csv_file librispeech_test_clean_filtered_1.0s.csv \
    --segment_length 1.0 \
    --original_dir /mnt/Internal/ASR \
    --model_name "LSCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "0.45" \
    --metrics dwer utmos pesq stoi \
    --keep_merged_files \
    --use_gpu --gpu_id 0

# Evaluate 2.0s segmented Common Voice files
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/common_voice_recon \
    --segment_csv_file common_voice_zh_CN_train_filtered_2.0s.csv \
    --segment_length 2.0 \
    --original_dir /mnt/Internal/ASR \
    --model_name "LSCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "0.45" \
    --metrics dcer utmos pesq stoi \
    --keep_merged_files \
    --use_gpu --gpu_id 1
```

**What happens during evaluation**:
1. Loads segment CSV and groups by original file
2. For each original file:
   - Finds all inference segments (`_001_inference.wav`, `_002_inference.wav`, etc.)
   - Finds all original segments (`_001.wav`, `_002.wav`, etc.)
   - Validates segment integrity (same count for both)
   - Merges inference segments → `merged_inference.wav`
   - Merges original segments → `merged_original.wav`
   - Computes all requested metrics on merged files
3. Saves detailed and summary results

**Output Structure**:
```
Results:
result/
├── detailed_results_LSCodec_1.0s_clean_librispeech.csv
└── summary_results_LSCodec_1.0s_clean_librispeech.csv

Merged files (if --keep_merged_files):
/mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/librispeech_recon/merged/
├── original/
│   ├── 61-70968-0000_merged.wav
│   └── ...
└── inference/
    ├── 61-70968-0000_merged_inference.wav
    └── ...
```

---

## Parameters Reference

### audio_splitter.py
- `--csv_file`: Input CSV file (required)
- `--original_dir`: Root directory for original audio files (required)
- `--segment_length`: Segment length in seconds, e.g., 1.0, 2.0, 3.0 (required)
- `--output_format`: Output format (wav/flac), default: wav
- `--sample_rate`: Sample rate for output files, default: 16000
- `--project_dir`: Project root directory

### segmented_evaluation_pipeline.py
- `--segment_csv_file`: CSV file generated by audio_splitter.py (required)
- `--segment_length`: Must match the split length (required)
- `--inference_dir`: Directory containing inference segments (required)
- `--original_dir`: Root directory for original audio files
- `--keep_merged_files`: Keep merged audio for debugging (default: enabled)
- `--no_keep_merged_files`: Delete merged files after evaluation
- All standard evaluation parameters (model_name, frequency, metrics, etc.)

---

## Complete Example Workflow

```bash
# 1. Split original files into 1.0s segments
python audio_splitter.py \
    --csv_file librispeech_test_clean_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --segment_length 1.0

# Output: 
# - Split files in: /mnt/Internal/ASR/librispeech/LibriSpeech/test-clean/1.0s/
# - Segment CSV: csv/librispeech_test_clean_filtered_1.0s.csv

# 2. Process segments with your codec (user responsibility)
# Your codec reads from: /mnt/Internal/ASR/.../test-clean/1.0s/
# Your codec outputs to: /mnt/Internal/.../Inference_Result/.../librispeech_recon/1.0s/

# 3. Evaluate segmented files
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/librispeech_recon \
    --segment_csv_file librispeech_test_clean_filtered_1.0s.csv \
    --segment_length 1.0 \
    --original_dir /mnt/Internal/ASR \
    --model_name "LSCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "0.45" \
    --metrics dwer utmos pesq stoi \
    --keep_merged_files \
    --use_gpu --gpu_id 0

# Output:
# - Detailed CSV: result/detailed_results_LSCodec_1.0s_clean_librispeech.csv
# - Summary CSV: result/summary_results_LSCodec_1.0s_clean_librispeech.csv
# - Merged files: /mnt/Internal/.../Inference_Result/.../librispeech_recon/merged/
```