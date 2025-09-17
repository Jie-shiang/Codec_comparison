# Neural Audio Codec Evaluation Pipeline

A comprehensive evaluation pipeline for neural audio codecs with automated metrics assessment, web interface generation, and multi-language support.

## Features

- **Multi-language ASR Evaluation**: Automatic dWER (English) and dCER (Chinese) calculation using Whisper-large-v3
- **Comprehensive Quality Metrics**: UTMOS, PESQ, and STOI assessment
- **Dataset Type Support**: Clean, noise, and blank conditions for thorough evaluation
- **Modular Architecture**: Separate metrics evaluator and pipeline modules
- **Automated Web Interface**: Generates complete files for interactive comparison
- **Performance Monitoring**: Detailed timing and success rate tracking
- **Real Filename Usage**: Uses actual audio filenames instead of generic placeholders

## Quick Start

### Installation

1. Navigate to project directory:
```bash
cd ./Codec_comparison
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
/home/jieshiang/Desktop/GitHub/Codec_comparison/
├── evaluation_pipeline.py           # Main evaluation pipeline
├── metrics_evaluator.py            # Audio metrics evaluation module
├── requirements.txt                 # Python dependencies
├── index.html                      # Web interface
├── csv/                           # Dataset files
│   ├── librispeech_test_clean_filtered.csv
│   └── common_voice_zh_TW_train_filtered.csv
├── result/                        # Evaluation reports (auto-generated)
├── audio/                         # Audio files (auto-generated)
├── configs/                       # JSON configurations (auto-generated)
└── README.md                      # This documentation
```

## Usage

### Basic Command Structure

```bash
python evaluation_pipeline.py \
    --inference_dir /path/to/inference/results \
    --csv_file DATASET_FILE \
    --model_name "ModelName" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.5" \
    --dataset_type "clean"
```

### English Dataset (LibriSpeech) Examples

```bash
# Clean samples
python evaluation_pipeline.py \
    --inference_dir /path/to/inference/results \
    --csv_file librispeech_test_clean_filtered.csv \
    --model_name "MyCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.5" \
    --dataset_type "clean"

# Noisy samples
python evaluation_pipeline.py \
    --inference_dir /path/to/inference/results \
    --csv_file librispeech_test_clean_filtered.csv \
    --model_name "MyCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.5" \
    --dataset_type "noise"

# Blank/silence samples
python evaluation_pipeline.py \
    --inference_dir /path/to/inference/results \
    --csv_file librispeech_test_clean_filtered.csv \
    --model_name "MyCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.5" \
    --dataset_type "blank"
```

### Chinese Dataset (Common Voice) Example

```bash
python evaluation_pipeline.py \
    --inference_dir /path/to/inference/results \
    --csv_file common_voice_zh_TW_train_filtered.csv \
    --model_name "MyCodec" \
    --frequency "50Hz" \
    --causality "Non-Causal" \
    --bit_rate "1.5" \
    --dataset_type "clean"
```

## Inference Audio Requirements

### Supported File Naming Conventions

The pipeline automatically detects inference files using these patterns:
- `{original_filename}_inference.wav`
- `{original_filename}_inference.flac`
- `{original_filename}.wav`
- `{original_filename}.flac`

## Pipeline Output

### Generated Directory Structure

```
result/
├── detailed_results_MyCodec_20241217_143022.csv    # Complete per-file results
└── summary_results_MyCodec_20241217_143022.csv     # Statistical summary

configs/
└── MyCodec_50Hz_clean_config.json                  # Web interface config

audio/
├── LibriSpeech/                                     # Clean samples
│   ├── original/
│   │   ├── 61-70968-0013.flac                      # Actual filenames
│   │   ├── 121-127105-0001.flac
│   │   └── 237-134493-0000.flac
│   └── MyCodec/
│       └── 50Hz/
│           ├── 61-70968-0013.wav
│           ├── 121-127105-0001.wav
│           └── 237-134493-0000.wav
├── LibriSpeech/Noise/                              # Noisy samples
└── LibriSpeech/Blank/                              # Blank samples
```

### Output Files Description

1. **Detailed Results CSV**: Complete evaluation data for all processed files
2. **Summary Results CSV**: Statistical overview and model performance metrics  
3. **JSON Configuration**: Web interface configuration with actual sample data
4. **Audio Files**: Selected representative samples organized for web interface
5. **Performance Log**: Execution timing and success rate information

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
- **Sample Count**: ~2200 files
- **Duration**: 3.0+ seconds per file
- **Format**: speaker-chapter-utterance ID structure

### Common Voice zh-TW (Chinese Traditional)
- **File**: `csv/common_voice_zh_TW_train_filtered.csv`
- **Language**: Chinese (Traditional)
- **Primary Metric**: dCER  
- **Sample Count**: ~4100 files
- **Duration**: 3.0+ seconds per file
- **Format**: Mozilla Common Voice structure

## Sample Selection Logic

The pipeline intelligently selects representative samples for web interface:

### For Total Statistics
- Overall mean scores across all evaluated files

### For Sample_1 to Sample_5
- **LibriSpeech**: First utterance from top 5 different speakers (sorted by speaker ID)
- **Common Voice**: First 5 samples from the dataset

### For Error_Sample_1
- Sample with highest dWER/dCER score (automatically highlighted in red)