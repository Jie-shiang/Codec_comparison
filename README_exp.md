# General Segmented Evaluation Workflow

This guide applies to the workflow for evaluating segmented audio files using **any Codec**.

---

## 📋 Complete Workflow Overview

```
1. Prepare split files
   ↓
2. Process split files with your Codec
   ↓
3. Verify inference file naming and location
   ↓
4. Execute evaluation
```

---

## Step 1: Prepare Split Files

### 1.1 Execute Audio Splitting

```bash
cd /home/jieshiang/Desktop/GitHub/Codec_comparison

# LibriSpeech 1.0s splitting
python audio_splitter.py \
    --csv_file librispeech_test_clean_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result \
    --segment_length 1.0

# Common Voice zh-CN 1.0s splitting
python audio_splitter.py \
    --csv_file common_voice_zh_CN_train_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result \
    --segment_length 1.0

# Common Voice zh-TW 1.0s splitting
python audio_splitter.py \
    --csv_file common_voice_zh_TW_train_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result \
    --segment_length 1.0
```

### 1.2 Split File Locations

**LibriSpeech:**
```
/mnt/Internal/jieshiang/Split_Result/
└── librispeech/LibriSpeech/test-clean/1.0s/
    └── 8463/287645/
        ├── 8463-287645-0001_001.wav
        ├── 8463-287645-0001_002.wav
        ├── 8463-287645-0001_003.wav
        └── 8463-287645-0001_004.wav
```

**Common Voice:**
```
/mnt/Internal/jieshiang/Split_Result/
└── common_voice/cv-corpus-22.0-2025-06-20/
    ├── zh-CN/clips/1.0s/
    │   ├── common_voice_zh-CN_19485265_001.wav
    │   ├── common_voice_zh-CN_19485265_002.wav
    │   └── ...
    └── zh-TW/clips/1.0s/
        └── ...
```

### 1.3 Generated CSV and Metadata

**Location:**
```
/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/
├── librispeech_test_clean_filtered_1.0s.csv
├── librispeech_test_clean_filtered_1.0s.metadata.txt
├── common_voice_zh_CN_train_filtered_1.0s.csv
├── common_voice_zh_CN_train_filtered_1.0s.metadata.txt
├── common_voice_zh_TW_train_filtered_1.0s.csv
└── common_voice_zh_TW_train_filtered_1.0s.metadata.txt
```

**Metadata Content Example:**
```
split_output_dir=/mnt/Internal/jieshiang/Split_Result
segment_length=1.0
output_format=wav
sample_rate=16000
```

---

## Step 2: Process Split Files with Your Codec

### 2.1 Input Files

Your Codec should read from:
```
/mnt/Internal/jieshiang/Split_Result/librispeech/LibriSpeech/test-clean/1.0s/
```

### 2.2 Process All Split Segments

**Important:** Process **all** split segments, including:
- `*_001.wav`
- `*_002.wav`
- `*_003.wav`
- `*_004.wav` (last segment, may have overlap)
- ... etc.

### 2.3 Output File Naming Convention (Important!)

**Naming Format:** `{original_filename}_{segment_number}_inference.wav`

**Examples:**
- Input: `8463-287645-0001_001.wav`
- Output: `8463-287645-0001_001_inference.wav` ✅

- Input: `8463-287645-0001_002.wav`
- Output: `8463-287645-0001_002_inference.wav` ✅

**Incorrect Examples:**
- ❌ `8463-287645-0001_001.wav` (missing `_inference` suffix)
- ❌ `8463-287645-0001_inference.wav` (missing segment number)
- ❌ `inference_8463-287645-0001_001.wav` (incorrect `_inference` position)

### 2.4 Recommended Output Path Structure

**Format:**
```
/mnt/Internal/jieshiang/Inference_Result/{CodecName}/{Frequency}/{Dataset}_recon/{SegmentLength}/
```

**LibriSpeech Example:**
```
/mnt/Internal/jieshiang/Inference_Result/MyCodec/50Hz/librispeech_recon/1.0s/
├── 8463-287645-0001_001_inference.wav
├── 8463-287645-0001_002_inference.wav
├── 8463-287645-0001_003_inference.wav
├── 8463-287645-0001_004_inference.wav
├── 8463-287645-0001_001_inference.wav
└── ...
```

**Common Voice Example:**
```
/mnt/Internal/jieshiang/Inference_Result/MyCodec/50Hz/common_voice_zh_CN_recon/1.0s/
├── common_voice_zh-CN_19485265_001_inference.wav
├── common_voice_zh-CN_19485265_002_inference.wav
└── ...
```

**Note:** The path structure can be adjusted according to your needs, as long as you ensure:
1. All inference files are in the same directory
2. File naming is correct

---

## Step 3: Execute Evaluation

### 3.1 LibriSpeech Evaluation

```bash
cd /home/jieshiang/Desktop/GitHub/Codec_comparison

python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/MyCodec/50Hz/librispeech_recon \
    --segment_csv_file librispeech_test_clean_filtered_1.0s.csv \
    --segment_length 1.0 \
    --model_name "MyCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.0" \
    --metrics dwer utmos pesq stoi \
    --use_gpu --gpu_id 0
```

### 3.2 Common Voice zh-CN Evaluation

```bash
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/MyCodec/50Hz/common_voice_zh_CN_recon \
    --segment_csv_file common_voice_zh_CN_train_filtered_1.0s.csv \
    --segment_length 1.0 \
    --model_name "MyCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.0" \
    --metrics dcer utmos pesq stoi \
    --use_gpu --gpu_id 1
```

### 3.3 Common Voice zh-TW Evaluation

```bash
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/MyCodec/50Hz/common_voice_zh_TW_recon \
    --segment_csv_file common_voice_zh_TW_train_filtered_1.0s.csv \
    --segment_length 1.0 \
    --model_name "MyCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.0" \
    --metrics dcer utmos pesq stoi \
    --use_gpu --gpu_id 1
```

### 3.4 Parameter Descriptions

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--inference_dir` | Directory containing inference files (without segment_length subdirectory) | `/mnt/Internal/.../librispeech_recon` |
| `--segment_csv_file` | CSV file generated after splitting | `librispeech_test_clean_filtered_1.0s.csv` |
| `--segment_length` | Segment length in seconds (without 's') | `1.0` |
| `--model_name` | Codec name | `"MyCodec"` |
| `--frequency` | Frequency parameter | `"50Hz"` |
| `--causality` | Causality type | `"Causal"` or `"Non-Causal"` |
| `--bit_rate` | Compression bit rate | `"1.0"` |
| `--metrics` | Metrics to calculate | `dwer utmos pesq stoi` (English) or `dcer utmos pesq stoi` (Chinese) |
| `--use_gpu` | Enable GPU | (flag) |
| `--gpu_id` | GPU number | `0` or `1` |

---

## 📊 Evaluation Results Output

### Output File Locations

```
/home/jieshiang/Desktop/GitHub/Codec_comparison/
├── result/
│   ├── detailed_results_MyCodec_1.0s_clean_librispeech.csv
│   └── summary_results_MyCodec_1.0s_clean_librispeech.csv
│
├── audio_1.0s/                    # ← Audio file copy location (includes segment length)
│   ├── LibriSpeech/
│   │   └── original/ and inference/
│   └── CommonVoice/
│       └── original/ and inference/
│
└── configs_1.0s/                  # ← JSON configs (includes segment length)
    ├── MyCodec_1.0s_clean.json
    └── ...
```

### Merged File Locations (if retained)

```
/mnt/Internal/jieshiang/Inference_Result/MyCodec/50Hz/librispeech_recon/
└── merged/
    ├── original/
    │   └── 8463-287645-0001_merged.wav  (reconstructed from split segments)
    └── inference/
        └── 8463-287645-0001_merged_inference.wav  (reconstructed from split segments)
```

### Maintaining Clean Directory Organization

```
Recommended directory structure:
/mnt/Internal/jieshiang/Inference_Result/
└── {CodecName}/
    ├── 50Hz/
    │   ├── librispeech_recon/
    │   │   ├── 0.5s/
    │   │   ├── 1.0s/
    │   │   ├── 2.0s/
    │   │   └── 3.0s/
    │   └── common_voice_zh_CN_recon/
    │       └── 1.0s/
    └── 25Hz/
        └── ...
```

---

## 📋 Quick Reference

### Complete Workflow (Single Configuration)

```bash
# 1. Split
python audio_splitter.py \
    --csv_file librispeech_test_clean_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result \
    --segment_length 1.0

# 2. Process (using your codec)
# your_codec_process.sh /mnt/Internal/jieshiang/Split_Result/... \
#     --output /mnt/Internal/jieshiang/Inference_Result/MyCodec/50Hz/librispeech_recon/1.0s/

# 3. Evaluate
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/MyCodec/50Hz/librispeech_recon \
    --segment_csv_file librispeech_test_clean_filtered_1.0s.csv \
    --segment_length 1.0 \
    --model_name "MyCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.0" \
    --metrics dwer utmos pesq stoi \
    --use_gpu --gpu_id 0
```

### Key File Naming

| Stage | File Example |
|-------|--------------|
| Original file | `8463-287645-0001.flac` |
| Split file | `8463-287645-0001_001.wav` |
| Inference file | `8463-287645-0001_001_inference.wav` ✅ |
| Merged original | `8463-287645-0001_merged.wav` |
| Merged inference | `8463-287645-0001_merged_inference.wav` |