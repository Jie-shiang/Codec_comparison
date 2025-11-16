# Neural Audio Codec Evaluation Pipeline

**🌐 [View Live Demo](https://jie-shiang.github.io/Codec_comparison/)**

A comprehensive evaluation pipeline for neural audio codecs with GPU acceleration, batch processing, and support for complete, segmented, and noise-corrupted audio evaluation.

---

## 📌 Project Overview

### **Purpose**

This project provides a systematic framework for evaluating the performance of multiple neural audio codecs across various conditions and metrics.

**Supported Codecs:**
- LSCodec (25Hz, 50Hz)
- FocalCodec (12.5Hz, 25Hz, 50Hz)
- FocalCodec-S (50Hz_2k, 50Hz_4k, 50Hz_65k)
- BigCodec (80Hz)
- NanoCodec (12.5Hz_2k, 12.5Hz_4k, 21.5Hz_2k)

**Evaluation Dimensions:**
- **Speech Quality:** UTMOS, PESQ, STOI, Speaker Similarity
- **Transcription Accuracy:** dWER (English), dCER (Chinese)

**Experiment Types:**
1. **Original Audio Evaluation** - Complete audio files
2. **Segmented Audio Evaluation** - Audio split into 0.5s, 1.0s, 2.0s segments
3. **Noise Robustness Evaluation** - Audio with added noise (SNR 5-15dB)

---

## 📂 Project Structure

```
/home/jieshiang/Desktop/GitHub/Codec_comparison/
├── docs/                                # Documentation and configurations
│   ├── codecs_config.json              # Original experiment codec config
│   ├── codecs_config_segmented.json    # Segmented experiment codec config
│   ├── codecs_config_noise.json        # Noise experiment codec config
│   ├── codec_loader.js                 # Shared JavaScript module
│   └── eval_sh/                        # Batch evaluation scripts
│       ├── run_all_evaluations.sh
│       ├── run_all_evaluations_segmented_0.5s.sh
│       ├── run_all_evaluations_segmented_1.0s.sh
│       ├── run_all_evaluations_segmented_2.0s.sh
│       └── run_all_noise_evaluations.sh
│
├── csv/                                # Dataset CSV files
│   ├── librispeech_test_clean_filtered.csv
│   ├── common_voice_zh_CN_train_filtered.csv
│   ├── noise/                          # Noise experiment CSVs
│   └── split/                          # Segmented audio CSVs
│
├── audio/                              # [AUTO-GENERATED] Original experiment audio
│   ├── LibriSpeech/
│   │   ├── original/                   # Original audio files (.flac)
│   │   ├── LSCodec/{25Hz,50Hz}/       # Reconstructed audio (.wav)
│   │   ├── FocalCodec/{12.5Hz,25Hz,50Hz}/
│   │   ├── FocalCodec-S/{50Hz_2k,50Hz_4k,50Hz_65k}/
│   │   ├── BigCodec/80Hz/
│   │   └── NanoCodec/{12.5Hz_2k,12.5Hz_4k,21.5Hz_2k}/
│   └── CommonVoice/                    # Same structure as LibriSpeech
│
├── audio_0.5s/                         # [AUTO-GENERATED] 0.5s segmented audio
├── audio_1.0s/                         # [AUTO-GENERATED] 1.0s segmented audio
├── audio_2.0s/                         # [AUTO-GENERATED] 2.0s segmented audio
├── audio_noise/                        # [AUTO-GENERATED] Noise experiment audio
│   ├── LibriSpeech/Noise/
│   └── CommonVoice/Noise/
│
├── configs/                            # [AUTO-GENERATED] Original experiment configs
│   ├── LSCodec_25Hz_config.json
│   ├── LSCodec_50Hz_config.json
│   └── ... (other codec configs)
│
├── configs_0.5s/                       # [AUTO-GENERATED] 0.5s segment configs
├── configs_1.0s/                       # [AUTO-GENERATED] 1.0s segment configs
├── configs_2.0s/                       # [AUTO-GENERATED] 2.0s segment configs
├── configs_noise/                      # [AUTO-GENERATED] Noise experiment configs
│
├── result/                             # [AUTO-GENERATED] Original experiment results
│   └── {CodecName}/{Frequency}/
│       ├── detailed_results_*.csv      # Per-file metrics
│       └── summary_results_*.csv       # Statistical summary
│
├── result_0.5s/                        # [AUTO-GENERATED] 0.5s segment results
├── result_1.0s/                        # [AUTO-GENERATED] 1.0s segment results
├── result_2.0s/                        # [AUTO-GENERATED] 2.0s segment results
├── result_noise/                       # [AUTO-GENERATED] Noise experiment results
│
├── enhanced_evaluation_pipeline.py     # Complete audio evaluation
├── segmented_evaluation_pipeline.py    # Segmented audio evaluation
├── metrics_evaluator.py                # Optimized metrics module
├── audio_splitter.py                   # Audio segmentation tool
├── segment_utils.py                    # Segmentation utilities
├── test_and_validation.py             # Validation script
│
├── index.html                          # Original experiments web interface
├── index_segmented.html                # Segmented experiments web interface
├── index_noise.html                    # Noise experiments web interface
│
└── README.md                           # This file
```

---

## 📁 Directory Explanations

### **`audio/` and `audio_*/` Directories**
**Purpose:** Store audio files for web playback and verification

- **`audio/`**: Original experiment audio (complete files)
- **`audio_0.5s/`, `audio_1.0s/`, `audio_2.0s/`**: Segmented audio files
- **`audio_noise/`**: Noise-corrupted audio files

**Structure:**
```
audio/
├── {Dataset}/                  # LibriSpeech or CommonVoice
    ├── original/               # Ground truth audio
    └── {CodecName}/{Config}/   # Reconstructed audio
```

**Contents:**
- Sample files for web interface playback
- Typically includes: Total stats + 5 samples + 1 error example per codec/dataset
- Automatically copied during evaluation pipeline execution

### **`configs/` and `configs_*/` Directories**
**Purpose:** Store evaluation results in JSON format for web visualization

**Generated by:** Evaluation pipeline (`enhanced_evaluation_pipeline.py`, `segmented_evaluation_pipeline.py`)

**Structure:**
```json
{
  "model_info": {
    "modelName": "LSCodec",
    "causality": "Non-Causal",
    "bitRate": "0.45",
    "parameters": {
      "frameRate": "50",
      "quantizers": "1",
      "codebookSize": "300",
      "nParams": "N/A"
    }
  },
  "LibriSpeech": {
    "Total": {
      "UTMOS": "4.2",
      "PESQ": "3.8",
      "STOI": "0.95",
      "Speaker_Sim": "0.92",
      "dWER": "0.05"
    },
    "Sample_1": {
      "File_name": "8463-287645-0001",
      "Transcription": "...",
      "Origin": "...",
      "Inference": "...",
      "UTMOS": "4.1",
      "dWER": "0.03",
      "PESQ": "3.7",
      "STOI": "0.94",
      "Speaker_Sim": "0.91"
    },
    ...
  },
  "CommonVoice": { ... }
}
```

**Usage:** Web pages (`index.html`, etc.) fetch these JSON files to display results

### **`result/` and `result_*/` Directories**
**Purpose:** Store detailed CSV evaluation results for analysis

**Generated by:** Evaluation pipeline

**Contents:**
- **`detailed_results_*.csv`**: Per-file metrics for all evaluated samples
- **`summary_results_*.csv`**: Statistical summary (mean, std, min, max, quartiles)

**Example:**
```csv
File_name,Original_Path,Inference_Path,UTMOS,PESQ,STOI,dWER,Speaker_Sim
8463-287645-0001,/path/to/original.flac,/path/to/inference.wav,4.1,3.7,0.94,0.03,0.91
...
```

---

## 🌐 Web Interface Implementation

### **Architecture Overview**

The web interface uses a **configuration-driven** architecture for easy maintenance and scalability.

### **Key Components**

1. **Configuration Files** (`docs/codecs_config*.json`)
   - Define available codecs and their configurations
   - Web pages dynamically generate UI based on these files

2. **Shared JavaScript Module** (`docs/codec_loader.js`)
   - `CodecLoader` class handles all data loading
   - Shared across all three web pages
   - Supports dynamic path switching (original/segmented/noise)

3. **Web Pages**
   - `index.html`: Original experiments
   - `index_segmented.html`: Segmented experiments (0.5s/1.0s/2.0s)
   - `index_noise.html`: Noise experiments

### **How It Works**

```
1. Page loads → Fetch codec config (codecs_config.json)
2. Generate UI controls dynamically based on config
3. Load codec data (JSON files from configs/)
4. User selects codec/frequency → Update display
5. Audio playback uses files from audio/ directories
```

### **Adding a New Codec**

1. Run evaluation pipeline to generate configs and copy audio
2. Config JSON files are automatically created in `configs/`
3. Web interface automatically detects and displays the new codec
4. **No HTML editing required!**

Example: To add "NewCodec" with "60Hz" configuration:
```bash
python enhanced_evaluation_pipeline.py \
    --model_name "NewCodec" \
    --frequency "60Hz" \
    ...
```

The web interface will automatically include NewCodec in the dropdown menu.

---

## 🚀 Running Evaluations

### **Path Configuration**

| Component | Path |
|-----------|------|
| Project Directory | `/home/jieshiang/Desktop/GitHub/Codec_comparison` |
| Original Audio Root | `/mnt/Internal/ASR` |
| Inference Output Root | `/mnt/Internal/jieshiang/Inference_Result` |
| Split Audio Root | `/mnt/Internal/jieshiang/Split_Result` |

### **Method 1: Batch Evaluation (Recommended)**

Use the provided shell scripts in `docs/eval_sh/`:

#### **Original Audio Evaluation**
```bash
bash docs/eval_sh/run_all_evaluations.sh
```

This script evaluates all codecs on complete audio files.

#### **Segmented Audio Evaluation**
```bash
# 0.5-second segments
bash docs/eval_sh/run_all_evaluations_segmented_0.5s.sh

# 1.0-second segments
bash docs/eval_sh/run_all_evaluations_segmented_1.0s.sh

# 2.0-second segments
bash docs/eval_sh/run_all_evaluations_segmented_2.0s.sh
```

#### **Noise Robustness Evaluation**
```bash
bash docs/eval_sh/run_all_noise_evaluations.sh
```

**What these scripts do:**
1. Iterate through all codec/frequency combinations
2. Run evaluation on both LibriSpeech and Common Voice datasets
3. Generate detailed CSVs, summary CSVs, and JSON configs
4. Copy sample audio files to web interface directories

### **Method 2: Manual Single Codec Evaluation**

#### **Original Audio**

**LibriSpeech Example:**
```bash
python enhanced_evaluation_pipeline.py \
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
    --training_set "LibriSpeech" \
    --testing_set "LibriSpeech test-clean" \
    --metrics dwer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 0
```

**Common Voice Example:**
```bash
python enhanced_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/commonvoice \
    --csv_file common_voice_zh_CN_train_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --model_name "LSCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "0.45" \
    --quantizers "1" \
    --codebook_size "300" \
    --n_params "N/A" \
    --training_set "LibriSpeech" \
    --testing_set "Common Voice zh-CN" \
    --metrics dcer utmos pesq stoi speaker_similarity \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 0
```

#### **Segmented Audio**

**Step 1: Split Audio (if not already done)**
```bash
python audio_splitter.py \
    --csv_file librispeech_test_clean_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --split_output_dir /mnt/Internal/jieshiang/Split_Result \
    --segment_length 1.0
```

**Step 2: Process Segments with Your Codec**

Your codec should process files from:
```
/mnt/Internal/jieshiang/Split_Result/librispeech/LibriSpeech/test-clean/1.0s/
```

Output naming: `{original_filename}_{segment_num}_inference.wav`

Example:
- Input: `8463-287645-0001_001.wav`
- Output: `8463-287645-0001_001_inference.wav`

**Step 3: Evaluate Segments**
```bash
python segmented_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/librispeech_recon \
    --segment_csv_file librispeech_test_clean_filtered_1.0s.csv \
    --segment_length 1.0 \
    --model_name "LSCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "0.45" \
    --quantizers "1" \
    --codebook_size "300" \
    --n_params "N/A" \
    --training_set "LibriSpeech" \
    --testing_set "LibriSpeech test-clean" \
    --metrics dwer utmos pesq stoi speaker_similarity \
    --use_gpu \
    --gpu_id 0
```

#### **Noise Robustness**

Similar to original audio evaluation, but use noise-corrupted inputs and specify `--dataset_type "noise"`.

---

## 📊 Evaluation Metrics

### **ASR Metrics**
- **dWER** (English): Degradation in Word Error Rate
- **dCER** (Chinese): Degradation in Character Error Rate
- **ASR Model**: OpenAI Whisper-large-v3

Formula: `dWER = WER(reconstructed) - WER(original)`

### **Audio Quality Metrics**
- **UTMOS**: Speech quality score (1-5, higher is better) - GPU accelerated
- **PESQ**: Perceptual Evaluation of Speech Quality (0.5-4.5) - Multi-process
- **STOI**: Short-Time Objective Intelligibility (0-1) - Multi-process
- **Speaker Similarity**: Identity preservation (0-1, cosine similarity) - GPU accelerated

---

## 📈 Output Files

### **CSV Results**
- **Detailed CSV**: Per-file metrics for all evaluated samples
- **Summary CSV**: Statistical summary (count, mean, std, min, max, Q25, median, Q75)

### **JSON Configs**
- Structured data for web visualization
- Includes model parameters, dataset results, and sample files

### **Audio Samples**
- Automatically copied to `audio/` directories
- Used for web playback and verification

---

## 🎯 Supported Datasets

| Dataset | Language | CSV File | Metric | Samples |
|---------|----------|----------|--------|---------|
| LibriSpeech test-clean | English | `librispeech_test_clean_filtered.csv` | dWER | 2,229 |
| Common Voice zh-CN | Chinese (Simplified) | `common_voice_zh_CN_train_filtered.csv` | dCER | 2,132 |
| Common Voice zh-TW | Chinese (Traditional) | `common_voice_zh_TW_train_filtered.csv` | dCER | 2,132 |

---

## 🔧 File Naming Requirements

### **Complete Audio**
- ✅ `{original_filename}_inference.wav`
- ✅ `{original_filename}_inference.flac`
- ⚠️ `{original_filename}.wav` (alternative, less preferred)

### **Segmented Audio**
- ✅ `{original_filename}_{segment_num}_inference.wav`
- ❌ `{original_filename}_{segment_num}.wav` (missing `_inference`)
- ❌ `{original_filename}_inference.wav` (missing segment number)

Example for file `8463-287645-0001.flac` split into 1.0s segments:
```
8463-287645-0001_001_inference.wav  ✅
8463-287645-0001_002_inference.wav  ✅
8463-287645-0001_003_inference.wav  ✅
```