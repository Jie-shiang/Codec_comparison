# Neural Audio Codec Comparison - Project Structure

## Overview

This project provides a web-based comparison tool for Neural Audio Codecs (NACs), specifically LSCodec and FocalCodec, across different datasets and configurations.

## Project Directory Structure

```
neural-codec-comparison/
├── index.html                          # Main HTML interface
├── configs/                           # JSON configuration files
│   ├── LSCodec_50Hz_config.json       # LSCodec 50Hz configuration
│   ├── FocalCodec_12.5Hz_config.json  # FocalCodec 12.5Hz configuration
│   ├── FocalCodec_25Hz_config.json    # FocalCodec 25Hz configuration
│   └── FocalCodec_50Hz_config.json    # FocalCodec 50Hz configuration
├── audio/                             # Audio files organized by dataset
│   ├── LibriSpeech/                   # Clean LibriSpeech dataset
│   │   ├── original/
│   │   │   ├── Sample_1.flac
│   │   │   ├── Sample_2.flac
│   │   │   ├── Sample_3.flac
│   │   │   ├── Sample_4.flac
│   │   │   ├── Sample_5.flac
│   │   │   └── Error_Sample_1.flac
│   │   ├── LSCodec/
│   │   │   └── 50Hz/
│   │   │       ├── Sample_1.wav
│   │   │       ├── Sample_2.wav
│   │   │       ├── Sample_3.wav
│   │   │       ├── Sample_4.wav
│   │   │       ├── Sample_5.wav
│   │   │       └── Error_Sample_1.wav
│   │   └── FocalCodec/
│   │       ├── 12.5Hz/
│   │       │   ├── Sample_1.wav
│   │       │   └── ... (all 6 samples)
│   │       ├── 25Hz/
│   │       │   ├── Sample_1.wav
│   │       │   └── ... (all 6 samples)
│   │       └── 50Hz/
│   │           ├── Sample_1.wav
│   │           └── ... (all 6 samples)
│   ├── LibriSpeech/Noise/             # Noisy LibriSpeech dataset
│   │   ├── original/                  # Original noisy samples (.flac)
│   │   ├── LSCodec/
│   │   │   └── 50Hz/                  # Processed samples (.wav)
│   │   └── FocalCodec/
│   │       ├── 12.5Hz/
│   │       ├── 25Hz/
│   │       └── 50Hz/
│   ├── LibriSpeech/Blank/             # Blank/silence LibriSpeech samples
│   │   ├── original/                  # Original blank samples (.flac)
│   │   ├── LSCodec/
│   │   │   └── 50Hz/                  # Processed samples (.wav)
│   │   └── FocalCodec/
│   │       ├── 12.5Hz/
│   │       ├── 25Hz/
│   │       └── 50Hz/
│   ├── CommonVoice/                   # Clean Common Voice dataset
│   │   ├── original/                  # Original samples (.flac)
│   │   ├── LSCodec/
│   │   │   └── 50Hz/                  # Processed samples (.wav)
│   │   └── FocalCodec/
│   │       ├── 12.5Hz/
│   │       ├── 25Hz/
│   │       └── 50Hz/
│   ├── CommonVoice/Noise/             # Noisy Common Voice dataset
│   │   ├── original/
│   │   ├── LSCodec/
│   │   │   └── 50Hz/
│   │   └── FocalCodec/
│   │       ├── 12.5Hz/
│   │       ├── 25Hz/
│   │       └── 50Hz/
│   └── CommonVoice/Blank/             # Blank/silence Common Voice samples
│       ├── original/
│       ├── LSCodec/
│       │   └── 50Hz/
│       └── FocalCodec/
│           ├── 12.5Hz/
│           ├── 25Hz/
│           └── 50Hz/
└── README.md                          # This documentation file
```

## Audio File Organization

### File Naming Convention
Each dataset contains exactly 6 audio files:
- **Sample_1** to **Sample_5**: Normal reconstruction examples
- **Error_Sample_1**: Example with reconstruction artifacts (highlighted in red in UI)


### Dataset Path Mapping
The web interface maps dataset selections to file paths as follows:
- `librispeech` → `LibriSpeech/`
- `librispeech-noise` → `LibriSpeech/Noise/`
- `librispeech-blank` → `LibriSpeech/Blank/`
- `commonvoice` → `CommonVoice/`
- `commonvoice-noise` → `CommonVoice/Noise/`
- `commonvoice-blank` → `CommonVoice/Blank/`

## JSON Configuration File Format

Configuration files must follow the naming pattern: `{CodecName}_{Frequency}_config.json`

### Required Structure

```json
{
  "model_info": {
    "modelName": "FocalCodec",
    "causality": "Non-Causal",
    "trainingSet": "LibriTTS + LibriTTS train-clean-100",
    "testingSet": "LibriSpeech + test-clean + Multilingual Lspeech + VoiceBank + LibrilMix + VCTK",
    "bitRate": "0.33",
    "parameters": {
      "frameRate": "25",
      "quantizers": "1",
      "codebookSize": "8192",
      "nParams": "N/A"
    }
  },
  "LibriSpeech": {
    "Total": {
      "Transcription": "-",
      "dWER": "N/A",
      "UTMOS": "N/A",
      "PESQ": "N/A",
      "STOI": "N/A"
    },
    "Sample_1": {
      "Transcription": "HE HOPED THERE WOULD BE STEW FOR DINNER",
      "dWER": "0.0",
      "UTMOS": "4.4",
      "PESQ": "3.2",
      "STOI": "0.97"
    }
    // ... Additional samples (Sample_2 through Sample_5, Error_Sample_1)
  },
  "LibriSpeech_Noise": {
    // Same structure as LibriSpeech
  },
  "LibriSpeech_Blank": {
    // Same structure as LibriSpeech
  },
  "CommonVoice": {
    // Same structure as LibriSpeech
  },
  "CommonVoice_Noise": {
    // Same structure as LibriSpeech
  },
  "CommonVoice_Blank": {
    // Same structure as LibriSpeech
  }
}
```

### Configuration Details

#### Model Info Section
- **modelName**: Display name for the codec (e.g., "LSCodec", "FocalCodec")
- **causality**: "Causal" or "Non-Causal"
- **trainingSet**: Description of training data
- **testingSet**: Description of test data
- **bitRate**: Compression rate in kbps as string
- **parameters**: Object containing:
  - **frameRate**: Frame rate in Hz as string
  - **quantizers**: Number of quantizers as string
  - **codebookSize**: Size of codebook as string
  - **nParams**: Number of parameters (or "N/A")

#### Dataset Sections
Each dataset section (LibriSpeech, LibriSpeech_Noise, etc.) contains:

- **Total**: Aggregate statistics row with Transcription set to "-"
- **Sample_1** through **Sample_5**: Individual sample data
- **Error_Sample_1**: Error case example

#### Sample Data Fields
- **Transcription**: Text transcription of the audio
- **dWER**: Word Error Rate (string format)
- **UTMOS**: Speech quality score (string format)  
- **PESQ**: Perceptual Evaluation of Speech Quality (string format)
- **STOI**: Short-Time Objective Intelligibility (string format)

### Placeholder Values
Use these values when actual data is not available:
- **Transcription**: "N/A" (or "-" for Total rows)
- **Metrics**: "N/A"