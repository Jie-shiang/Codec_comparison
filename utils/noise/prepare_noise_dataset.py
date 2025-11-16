#!/usr/bin/env python3
"""
Prepare Noise Dataset
- Select 10 utterances per speaker from clean CSV
- Add MUSAN noise with random SNR (5-15 dB)
- Save to /mnt/Internal/jieshiang/Noise_Result/

1. 準備 LibriSpeech 噪音資料集
python prepare_noise_dataset.py \
    --csv_file /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/librispeech_test_clean_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --output_dir /mnt/Internal/jieshiang/Noise_Result \
    --output_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/noise/librispeech_test_clean_noise.csv \
    --n_per_speaker 22 \
    --snr_min 5.0 \
    --snr_max 15.0
    
2. 準備 Common Voice 噪音資料集
python prepare_noise_dataset.py \
    --csv_file /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/common_voice_zh_CN_train_filtered.csv \
    --original_dir /mnt/Internal/ASR \
    --output_dir /mnt/Internal/jieshiang/Noise_Result \
    --output_csv /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/noise/common_voice_zh_CN_train_noise.csv \
    --n_per_speaker 7 \
    --snr_min 5.0 \
    --snr_max 15.0
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys

# Add utils/noise to path
sys.path.insert(0, '/home/jieshiang/Desktop/GitHub/Codec_comparison/utils/noise')
from noise_generator import NoiseGenerator


def select_per_speaker(df: pd.DataFrame, n_per_speaker: int = 10, seed: int = 42):
    """Select n utterances per speaker"""
    np.random.seed(seed)
    
    # Auto-detect speaker column
    speaker_col = None
    for col in ['speaker_id', 'client_id', 'speaker']:
        if col in df.columns:
            speaker_col = col
            break
    
    if speaker_col is None:
        print("Warning: No speaker column found")
        # For LibriSpeech, extract from file_name
        if 'file_name' in df.columns:
            df['speaker_id'] = df['file_name'].apply(lambda x: x.split('-')[0])
            speaker_col = 'speaker_id'
        else:
            # Random sampling
            return df.sample(n=min(len(df), n_per_speaker * 200), random_state=seed)
    
    print(f"Using speaker column: {speaker_col}")
    
    # Group and sample
    selected = []
    for speaker_id, group in df.groupby(speaker_col):
        n_samples = min(n_per_speaker, len(group))
        sampled = group.sample(n=n_samples, random_state=seed)
        selected.append(sampled)
    
    result = pd.concat(selected, ignore_index=True)
    n_speakers = df[speaker_col].nunique()
    print(f"Selected {len(result)} utterances from {n_speakers} speakers")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare noise-corrupted dataset")
    parser.add_argument("--csv_file", required=True, 
                       help="Clean CSV file (e.g., csv/librispeech_test_clean_filtered.csv)")
    parser.add_argument("--original_dir", required=True,
                       help="Root dir of clean audio (e.g., /mnt/Internal/ASR)")
    parser.add_argument("--output_dir", required=True,
                       help="Output dir for noisy audio (e.g., /mnt/Internal/jieshiang/Noise_Result/librispeech)")
    parser.add_argument("--output_csv", required=True,
                       help="Output CSV file (e.g., csv/librispeech_test_clean_noise.csv)")
    parser.add_argument("--musan_dir", default="/mnt/External/ASR/musan/noise",
                       help="MUSAN noise directory")
    parser.add_argument("--n_per_speaker", type=int, default=10,
                       help="Number of utterances per speaker")
    parser.add_argument("--snr_min", type=float, default=5.0,
                       help="Minimum SNR in dB")
    parser.add_argument("--snr_max", type=float, default=15.0,
                       help="Maximum SNR in dB")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Preparing Noise Dataset")
    print("="*60)
    print(f"Input CSV: {args.csv_file}")
    print(f"Original dir: {args.original_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"SNR range: {args.snr_min}-{args.snr_max} dB")
    print(f"Per speaker: {args.n_per_speaker} utterances")
    print("="*60)
    
    # Load CSV
    df = pd.read_csv(args.csv_file, encoding='utf-8')
    print(f"\nLoaded {len(df)} utterances from CSV")
    
    # Select samples
    selected_df = select_per_speaker(df, args.n_per_speaker, args.seed)
    
    # Initialize noise generator
    noise_gen = NoiseGenerator(musan_noise_dir=args.musan_dir, seed=args.seed)
    
    # Process files
    results = []
    output_dir = Path(args.output_dir)
    
    print(f"\nAdding noise to {len(selected_df)} files...")
    for _, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Processing"):
        # Clean audio path
        file_path = row['file_path'].lstrip('./')
        input_path = Path(args.original_dir) / file_path
        
        if not input_path.exists():
            print(f"Warning: File not found: {input_path}")
            continue
        
        # Output path (keep same directory structure)
        output_path = output_dir / file_path
        
        # Random SNR in range
        snr_db = np.random.uniform(args.snr_min, args.snr_max)
        
        # Process
        success, noise_type = noise_gen.process_file(
            str(input_path), str(output_path), snr_db
        )
        
        if success:
            result_row = row.copy()
            result_row['snr_db'] = snr_db
            result_row['noise_type'] = noise_type
            results.append(result_row)
    
    # Save CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output_csv, index=False, encoding='utf-8')
    
    # Save metadata
    metadata = {
        'source_csv': args.csv_file,
        'original_dir': args.original_dir,
        'output_dir': str(output_dir),
        'musan_dir': args.musan_dir,
        'n_per_speaker': args.n_per_speaker,
        'snr_range': [args.snr_min, args.snr_max],
        'seed': args.seed,
        'total_samples': len(result_df)
    }
    metadata_path = Path(args.output_csv).with_suffix('.metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("✓ Noise dataset preparation completed!")
    print(f"{'='*60}")
    print(f"Total files: {len(result_df)}")
    print(f"Output dir: {output_dir}")
    print(f"CSV: {args.output_csv}")
    print(f"Metadata: {metadata_path}")
    print(f"\nSNR distribution:")
    print(result_df['snr_db'].describe())
    print(f"\nNoise type distribution:")
    print(result_df['noise_type'].value_counts())


if __name__ == "__main__":
    main()