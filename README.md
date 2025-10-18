# Neural Audio Codec Evaluation Pipeline

**🌐 [View Live Demo](https://jie-shiang.github.io/Codec_comparison/)**

A comprehensive evaluation pipeline for neural audio codecs with automated metrics assessment, selective metric calculation, multi-language support, and segmented audio evaluation.

## Project Structure

```
/home/jieshiang/Desktop/GitHub/Codec_comparison/
├── enhanced_evaluation_pipeline.py   # Enhanced evaluation with selective metrics
├── segmented_evaluation_pipeline.py  # Segmented audio evaluation pipeline
├── test_and_validation.py            # Test and validation tools
├── metrics_evaluator.py              # Audio metrics evaluation module
├── cleanup_test_files.py             # Clean testing files
├── audio_splitter.py                 # Audio segmentation tool
├── segment_utils.py                  # Segmentation utilities
├── requirements.txt                  # Python dependencies
├── index.html                        # Web interface
├── csv/                              # Dataset files
│   ├── librispeech_test_clean_filtered.csv
│   ├── common_voice_zh_TW_train_filtered.csv
│   ├── common_voice_zh_CN_train_filtered.csv
│   ├── librispeech_test_clean_filtered_1.0s.csv  # Segmented CSV
│   └── common_voice_zh_CN_train_filtered_1.0s.csv
├── result/                           # Evaluation reports (auto-generated)
│   ├── detailed_results_ModelName_clean_librispeech.csv
│   ├── summary_results_ModelName_clean_librispeech.csv
│   ├── detailed_results_ModelName_1.0s_clean_librispeech.csv  # Segmented results
│   └── test_results/                # Test mode results
├── audio/                           # Audio files (auto-generated)
│   ├── LibriSpeech/
│   ├── CommonVoice/
│   └── merged/                      # Merged segment files
├── configs/                         # JSON configurations (auto-generated)
└── README.md                        # This documentation
```

## Quick Start - Recommended Workflow

### Step 1: Environment Setup

1. Create and activate Python environment:
```bash
# Using conda
conda create -n codec_eval python=3.9
conda activate codec_eval

# Or using venv
python -m venv codec_eval
source codec_eval/bin/activate  # Linux/Mac
# codec_eval\Scripts\activate   # Windows
```

2. Navigate to project directory:
```bash
cd /home/jieshiang/Desktop/GitHub/Codec_comparison
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: File Preparation and Validation

Before running evaluation, validate your inference files:

⚠️ **Notice**: The CSV files provide relative paths instead of absolute paths.
Please check your paths carefully.

**Example**:
- CSV path: `./librispeech/LibriSpeech/test-clean`
- Original file at: `/mnt/Internal/ASR/librispeech/LibriSpeech/test-clean`
- `--original_dir` should be: `/mnt/Internal/ASR`

```bash
# Validate file naming and check for issues
python test_and_validation.py \
    --inference_dir /path/to/inference \
    --csv_file librispeech_test_clean_filtered.csv \
    --mode validate

# Fix naming issues if needed (dry run first, then actual fix)
python test_and_validation.py \
    --inference_dir /path/to/inference \
    --csv_file librispeech_test_clean_filtered.csv \
    --mode validate \
    --fix_naming

python test_and_validation.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/librispeech_recon/1.0s \
    --csv_file librispeech_test_clean_filtered_1.0s.csv \
    --mode validate \
    --fix_naming
```

### Step 3: Test Mode (Quick Validation)

Run a quick test on 20 samples to ensure everything works:

```bash
# Test evaluation with first 20 samples
python test_and_validation.py \
    --original_dir /mnt/Internal/ASR \
    --inference_dir /path/to/your/inference/files \
    --csv_file librispeech_test_clean_filtered.csv \
    --model_name "TestCodec" \
    --mode test \
    --num_samples 20 \
    --use_gpu \
    --gpu_id 0

# For Example - LibriSpeech
python test_and_validation.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/librispeech_recon \
    --csv_file librispeech_test_clean_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --model_name "TestCodec" \
    --mode test \
    --num_samples 20 \
    --use_gpu \
    --gpu_id 0

# For Example - Common Voice
python test_and_validation.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/common_voice_recon \
    --csv_file common_voice_zh_CN_train_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --model_name "TestCodec" \
    --mode test \
    --num_samples 20 \
    --use_gpu \
    --gpu_id 0
```

### Step 4: Production Evaluation

Once testing is successful, you can run the cleanup script then run the full evaluation:

⚠️ **Make sure your result folder is clean.**

```bash
# Clean Test files
python cleanup_test_files.py --dry_run
python cleanup_test_files.py

# For Example
python cleanup_test_files.py \
    --project_dir /home/jieshiang/Desktop/GitHub/Codec_comparison \
    --yes

# Option A: Compute all metrics at once - LibriSpeech (English)
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
    --training_set "N/A" \
    --testing_set "N/A" \
    --metrics dwer utmos pesq stoi \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 0

# Option A: Compute all metrics at once - Common Voice (Chinese)
python enhanced_evaluation_pipeline.py \
    --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/common_voice_recon \
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
    --metrics dcer utmos pesq stoi \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 1

# Option B: Compute metrics incrementally (recommended for large datasets)
# Step 1: Compute dWER first (fastest)
python enhanced_evaluation_pipeline.py \
    --inference_dir /path/to/your/inference/files \
    --csv_file librispeech_test_clean_filtered.csv \
    --model_name "MyCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.5" \
    --metrics dwer \
    --use_gpu \
    --gpu_id 0

# Step 2: Add UTMOS
python enhanced_evaluation_pipeline.py \
    --inference_dir /path/to/your/inference/files \
    --csv_file librispeech_test_clean_filtered.csv \
    --model_name "MyCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.5" \
    --metrics utmos \
    --use_gpu \
    --gpu_id 0

# Step 3: Add PESQ and STOI
python enhanced_evaluation_pipeline.py \
    --inference_dir /path/to/your/inference/files \
    --csv_file librispeech_test_clean_filtered.csv \
    --model_name "MyCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.5" \
    --metrics pesq stoi \
    --cpu_only
```

## File Naming Requirements

### Supported Inference File Patterns

The pipeline automatically detects inference files using these patterns:
- `{original_filename}_inference.wav` ✅ **Recommended**
- `{original_filename}_inference.flac` ✅ **Recommended**
- `{original_filename}.wav` ⚠️ **Needs processing**
- `{original_filename}.flac` ⚠️ **Needs processing**

## Output Files

### Enhanced Pipeline Outputs

1. **Detailed Results**: `detailed_results_{model_name}_{dataset_type}.csv`
   - Individual file evaluation data
   - Transcriptions, metrics, file paths
   - Supports incremental updates

2. **Summary Statistics**: `summary_results_{model_name}_{dataset_type}.csv`
   - Comprehensive statistics (Mean, Std, Min, Max, Median, Q25, Q75)
   - Model comparison data
   - Automatic metric aggregation

## Evaluation Metrics

### ASR Accuracy Metrics
- **dWER (English)**: Word Error Rate difference between reconstruction and original
- **dCER (Chinese)**: Character Error Rate difference between reconstruction and original
- **ASR Model**: OpenAI Whisper-large-v3 for both languages

### Audio Quality Metrics
- **UTMOS**: Predicted Mean Opinion Score for overall speech quality (1-5 scale)
- **PESQ**: Perceptual Evaluation of Speech Quality (0.5-4.5 scale)
- **STOI**: Short-Time Objective Intelligibility (0-1 scale)

## Supported Datasets

### LibriSpeech (English)
- **File**: `csv/librispeech_test_clean_filtered.csv`
- **Language**: English
- **Primary Metric**: dWER
- **Sample Count**: 2,229 files
- **Duration**: 3.0+ seconds per file

### Common Voice zh-TW (Chinese Traditional)
- **File**: `csv/common_voice_zh_TW_train_filtered.csv`
- **Language**: Chinese (Traditional)
- **Primary Metric**: dCER
- **Sample Count**: 2,132 files
- **Duration**: 3.0+ seconds per file
- **Max samples per speaker**: 20

### Common Voice zh-CN (Chinese Simplified)
- **File**: `csv/common_voice_zh_CN_train_filtered.csv`
- **Language**: Chinese (Simplified)
- **Primary Metric**: dCER
- **Sample Count**: 2,132 files
- **Duration**: 3.0+ seconds per file
- **Max samples per speaker**: 20

## Advanced Usage

### Custom Parameters

```bash
# Custom model parameters
python enhanced_evaluation_pipeline.py \
    --inference_dir /path/to/inference \
    --csv_file your_dataset.csv \
    --model_name "CustomCodec" \
    --frequency "25Hz" \
    --causality "Causal" \
    --bit_rate "0.75" \
    --quantizers "8" \
    --codebook_size "512" \
    --n_params "12M" \
    --training_set "Custom Training Set" \
    --testing_set "Custom Test Set" \
    --metrics dwer utmos pesq stoi \
    --dataset_type "clean" \
    --use_gpu \
    --gpu_id 0
```