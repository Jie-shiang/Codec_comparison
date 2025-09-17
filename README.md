# Neural Audio Codec Comparison - Project Structure

## Recommended Directory Structure

```
neural-codec-comparison/
├── index.html                          # Main HTML file
├── configs/                           # JSON configuration files
│   ├── LSCodec_50Hz_config.json      # LSCodec 50Hz configuration
│   ├── FocalCodec_12.5Hz_config.json # FocalCodec 12.5Hz configuration
│   ├── FocalCodec_25Hz_config.json   # FocalCodec 25Hz configuration
│   └── FocalCodec_50Hz_config.json   # FocalCodec 50Hz configuration
├── audio/                            # Audio files directory
│   ├── LibriSpeech/                  # LibriSpeech dataset
│   │   ├── original/                 # Original audio files
│   │   │   ├── Sample_1.flac
│   │   │   ├── Sample_2.flac
│   │   │   ├── Sample_3.flac
│   │   │   ├── Sample_4.flac
│   │   │   ├── Sample_5.flac
│   │   │   └── Error_Sample_1.flac
│   │   ├── LSCodec/                  # LSCodec processed files
│   │   │   └── 50Hz/
│   │   │       ├── Sample_1.wav
│   │   │       ├── Sample_2.wav
│   │   │       ├── Sample_3.wav
│   │   │       ├── Sample_4.wav
│   │   │       ├── Sample_5.wav
│   │   │       └── Error_Sample_1.wav
│   │   └── FocalCodec/              # FocalCodec processed files
│   │       ├── 12.5Hz/
│   │       │   ├── Sample_1.wav
│   │       │   └── ... (all 6 files)
│   │       ├── 25Hz/
│   │       │   ├── Sample_1.wav
│   │       │   └── ... (all 6 files)
│   │       └── 50Hz/
│   │           ├── Sample_1.wav
│   │           └── ... (all 6 files)
│   ├── LibriSpeech/Noise/           # Noisy LibriSpeech dataset
│   │   ├── original/
│   │   │   ├── Sample_1.flac
│   │   │   ├── Sample_2.flac
│   │   │   ├── Sample_3.flac
│   │   │   ├── Sample_4.flac
│   │   │   ├── Sample_5.flac
│   │   │   └── Error_Sample_1.flac
│   │   ├── LSCodec/
│   │   │   └── 50Hz/ (6 files)
│   │   └── FocalCodec/
│   │       ├── 12.5Hz/ (6 files)
│   │       ├── 25Hz/ (6 files)
│   │       └── 50Hz/ (6 files)
│   └── LibriSpeech/Blank/           # Blank/silence samples
│       ├── original/ (6 files)
│       ├── LSCodec/
│       │   └── 50Hz/ (6 files)
│       └── FocalCodec/
│           ├── 12.5Hz/ (6 files)
│           ├── 25Hz/ (6 files)
│           └── 50Hz/ (6 files)
└── README.md                        # Project documentation
```

## Sample Files Per Dataset

Each dataset contains exactly 6 audio samples:
- **Sample_1** to **Sample_5**: Normal reconstruction examples
- **Error_Sample_1**: Example with reconstruction errors (highlighted in red)

## JSON Configuration File Format

Each model configuration file should follow this structure:

### Key Components:

1. **model_info**: Basic model metadata
   - `modelName`: Display name (LSCodec/FocalCodec)
   - `causality`: Causal or Non-Causal
   - `trainingSet`: Training dataset description
   - `testingSet`: Testing dataset description  
   - `bitRate`: Compression rate in kbps
   - `parameters`: Technical parameters object

2. **Dataset Sections**: One section per dataset
   - `LibriSpeech`: Clean speech samples
   - `LibriSpeech_Noise`: Noisy conditions
   - `LibriSpeech_Blank`: Silent/blank samples

3. **Sample Data**: For each audio file ID
   - `Transcription`: Text transcription ("X" for placeholder)
   - `dWER`: Word Error Rate ("N/A" for placeholder)
   - `UTMOS`: Speech quality score ("X" for placeholder)
   - `PESQ`: Perceptual quality metric ("X" for placeholder)
   - `STOI`: Speech intelligibility metric ("X" for placeholder)

