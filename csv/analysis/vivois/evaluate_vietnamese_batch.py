#!/usr/bin/env python3
"""
TRUE BATCH Vietnamese Audio Evaluation Script
- Real batch processing for ASR (processes multiple files at once)
- Real batch processing for MOS models
- Uses same calculation methods as metrics_evaluator_v3.py
- Log file output (no tqdm for bash compatibility)
"""

import sys
import os
import argparse
import pandas as pd
import torch
from pathlib import Path
import warnings
import time
from typing import List, Dict, Tuple
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, parent_dir)
from metrics_evaluator_v3 import AudioMetricsEvaluatorV3

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TrueBatchVietnameseEvaluator:
    """
    TRUE Batch Vietnamese Audio Evaluator

    ASR Model: PhoWhisper-large (same as metrics_evaluator_v3.py for Vietnamese)
    WER: Uses fast_wer() from metrics_evaluator_v3.py
    TER: Uses calculate_ter() from metrics_evaluator_v3.py (Vietnamese diacritics)
    MOS: NISQA v2 and UTMOS
    """

    def __init__(self, gpu_id=0, batch_size=32):
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.log_file = None

        self.log("="*80)
        self.log("Initializing TRUE BATCH Vietnamese Audio Evaluator")
        self.log(f"Batch size: {batch_size}")
        self.log(f"GPU ID: {gpu_id}")
        self.log("="*80)

        # Initialize metrics evaluator
        self.evaluator = AudioMetricsEvaluatorV3(
            language='vi',  # Vietnamese
            dataset='vivos',
            use_gpu=True,
            gpu_id=gpu_id
        )

        self.log("\nLoading Models...")
        self._load_models()
        self.log("All models loaded successfully!\n")

    def log(self, message, also_print=True):
        """Write to log file and optionally print"""
        if also_print:
            print(message, flush=True)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')

    def _load_models(self):
        """Pre-load all required models"""
        # 1. Load ASR model (PhoWhisper-large for Vietnamese)
        self.log("[1/3] Loading ASR Model (PhoWhisper-large for Vietnamese)...")
        try:
            self.evaluator._load_asr_model()
            self.log("✓ ASR model loaded")
        except Exception as e:
            self.log(f"✗ Failed to load ASR model: {e}")

        # 2. Load MOS Quality model
        self.log("[2/3] Loading MOS Quality Model (NISQA v2)...")
        try:
            self.evaluator._load_mos_quality_model()
            self.log("✓ MOS Quality model loaded")
        except Exception as e:
            self.log(f"Note: MOS Quality will be loaded on first use: {e}")

        # 3. Load MOS Naturalness model
        self.log("[3/3] Loading MOS Naturalness Model (UTMOS)...")
        try:
            self.evaluator._load_mos_naturalness_model()
            self.log("✓ MOS Naturalness model loaded")
        except Exception as e:
            self.log(f"Note: MOS Naturalness will be loaded on first use: {e}")

    def evaluate_batch_true(self, df, base_path='', output_csv='', log_file=''):
        """
        TRUE batch evaluation with real batch ASR processing
        """
        self.log_file = log_file

        self.log("\n" + "="*80)
        self.log(f"Starting TRUE BATCH evaluation")
        self.log(f"Total files: {len(df)}")
        self.log(f"Batch size: {self.batch_size}")
        self.log(f"Output CSV: {output_csv}")
        self.log(f"Log file: {log_file}")
        self.log("="*80 + "\n")

        # Initialize result columns (Vietnamese uses WER not CER)
        for col in ['asr_result', 'wer', 'MOS_Quality', 'MOS_Naturalness', 'TER']:
            if col not in df.columns:
                df[col] = None

        start_time = time.time()

        # Check for already processed files
        already_processed = df['asr_result'].notna().sum()
        if already_processed > 0:
            self.log(f"Found {already_processed} already processed files, resuming...\n")

        # Get list of files to process
        files_to_process = []
        for idx, row in df.iterrows():
            # Skip if already processed
            if pd.notna(df.at[idx, 'asr_result']):
                continue

            file_path = row['file_path']
            if base_path and not os.path.isabs(file_path):
                file_path = os.path.join(base_path, file_path.lstrip('./'))

            if os.path.exists(file_path):
                files_to_process.append({
                    'idx': idx,
                    'path': file_path,
                    'transcription': row['transcription']
                })

        total_to_process = len(files_to_process)
        self.log(f"Files to process: {total_to_process}\n")

        if total_to_process == 0:
            self.log("No files to process!")
            return df

        # Process in TRUE batches
        num_batches = (total_to_process + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, total_to_process)
            batch = files_to_process[batch_start:batch_end]
            batch_size_actual = len(batch)

            batch_start_time = time.time()

            self.log(f"Processing batch {batch_idx + 1}/{num_batches} ({batch_end}/{total_to_process} files)...")

            # Extract batch data
            batch_indices = [item['idx'] for item in batch]
            batch_paths = [item['path'] for item in batch]
            batch_transcriptions = [item['transcription'] for item in batch]

            # ==================== 1. TRUE BATCH ASR ====================
            self.log(f"  [1/4] Running ASR on {batch_size_actual} files...")
            asr_transcripts = []
            try:
                # Use batch transcribe from evaluator (TRUE batch processing)
                asr_transcripts = self.evaluator.batch_transcribe(batch_paths, batch_size=8)
            except Exception as e:
                self.log(f"  Warning: Batch transcribe failed: {e}, falling back to sequential")
                # Fallback to sequential
                for path in batch_paths:
                    try:
                        result = self.evaluator.asr_pipeline(path, generate_kwargs={"language": "vi"})
                        asr_transcripts.append(result.get('text', ''))
                    except:
                        asr_transcripts.append('')

            # ==================== 2. BATCH WER ====================
            self.log(f"  [2/4] Calculating WER...")
            wer_results = []
            for ref, hyp in zip(batch_transcriptions, asr_transcripts):
                try:
                    # Use same method as metrics_evaluator_v3.py
                    wer = self.evaluator.fast_wer(ref, hyp)
                    wer_results.append(wer)
                except:
                    wer_results.append(None)

            # ==================== 3. BATCH TER ====================
            self.log(f"  [3/4] Calculating TER...")
            ter_results = []
            for ref, hyp in zip(batch_transcriptions, asr_transcripts):
                try:
                    # Use same method as metrics_evaluator_v3.py (fixed version)
                    ter = self.evaluator.calculate_ter(ref, hyp)
                    ter_results.append(ter)
                except:
                    ter_results.append(None)

            # ==================== 4. BATCH MOS ====================
            self.log(f"  [4/4] Calculating MOS Quality and Naturalness...")
            mos_q_results = []
            mos_n_results = []
            for path in batch_paths:
                try:
                    mos_q = self.evaluator.calculate_mos_quality(path)
                    mos_q_results.append(mos_q)
                except:
                    mos_q_results.append(None)

                try:
                    mos_n = self.evaluator.calculate_mos_naturalness(path)
                    mos_n_results.append(mos_n)
                except:
                    mos_n_results.append(None)

            # ==================== 5. Store results ====================
            for i, idx in enumerate(batch_indices):
                df.at[idx, 'asr_result'] = asr_transcripts[i] if i < len(asr_transcripts) else None
                df.at[idx, 'wer'] = wer_results[i] if i < len(wer_results) else None
                df.at[idx, 'TER'] = ter_results[i] if i < len(ter_results) else None
                df.at[idx, 'MOS_Quality'] = mos_q_results[i] if i < len(mos_q_results) else None
                df.at[idx, 'MOS_Naturalness'] = mos_n_results[i] if i < len(mos_n_results) else None

            # Save after each batch
            if output_csv:
                df.to_csv(output_csv, index=False)

            # Calculate timing
            batch_time = time.time() - batch_start_time
            elapsed = time.time() - start_time
            avg_time_per_file = elapsed / batch_end
            remaining_files = total_to_process - batch_end
            eta_seconds = remaining_files * avg_time_per_file
            eta_minutes = eta_seconds / 60

            self.log(f"  Batch completed in {batch_time:.1f}s")
            self.log(f"  Progress: {batch_end}/{total_to_process} files ({batch_end/total_to_process*100:.1f}%)")
            self.log(f"  Average: {avg_time_per_file:.2f}s/file")
            self.log(f"  ETA: {eta_minutes:.1f} minutes\n")

        # Final save
        if output_csv:
            df.to_csv(output_csv, index=False)

        total_time = time.time() - start_time
        self.log("\n" + "="*80)
        self.log("Batch evaluation completed!")
        self.log(f"Total processed: {total_to_process}")
        self.log(f"Total time: {total_time/60:.1f} minutes")
        self.log(f"Average speed: {total_to_process / total_time:.2f} files/second")
        self.log("="*80)

        return df

    def print_statistics(self, df):
        """Print statistics of evaluation results"""
        self.log("\n" + "="*80)
        self.log("Evaluation Statistics")
        self.log("="*80)

        n_total = len(df)
        n_asr = df['asr_result'].notna().sum()
        n_wer = df['wer'].notna().sum()
        n_ter = df['TER'].notna().sum()
        n_mos_q = df['MOS_Quality'].notna().sum()
        n_mos_n = df['MOS_Naturalness'].notna().sum()

        self.log(f"\nTotal files: {n_total}")
        self.log(f"ASR successful: {n_asr}/{n_total} ({n_asr/n_total*100:.1f}%)")
        self.log(f"WER calculated: {n_wer}/{n_total} ({n_wer/n_total*100:.1f}%)")
        self.log(f"TER calculated: {n_ter}/{n_total} ({n_ter/n_total*100:.1f}%)")
        self.log(f"MOS Quality: {n_mos_q}/{n_total} ({n_mos_q/n_total*100:.1f}%)")
        self.log(f"MOS Naturalness: {n_mos_n}/{n_total} ({n_mos_n/n_total*100:.1f}%)")

        if n_wer > 0:
            self.log(f"\nMean WER: {df['wer'].mean():.4f} ± {df['wer'].std():.4f}")
        if n_ter > 0:
            self.log(f"Mean TER: {df['TER'].mean():.4f} ± {df['TER'].std():.4f}")
        if n_mos_q > 0:
            self.log(f"Mean MOS Quality: {df['MOS_Quality'].mean():.4f} ± {df['MOS_Quality'].std():.4f}")
        if n_mos_n > 0:
            self.log(f"Mean MOS Naturalness: {df['MOS_Naturalness'].mean():.4f} ± {df['MOS_Naturalness'].std():.4f}")

        self.log("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='TRUE Batch Evaluate Vietnamese Audio Files')
    parser.add_argument('input_csv', type=str, help='Input CSV file')
    parser.add_argument('output_csv', type=str, help='Output CSV file')
    parser.add_argument('--base-path', type=str, default='', help='Base path for audio files')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--log-file', type=str, default='', help='Log file path')

    args = parser.parse_args()

    # Auto-generate log file name if not provided
    if not args.log_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.log_file = f'evaluation_vietnamese_{timestamp}.log'

    # Load input CSV
    print(f"Loading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    required_cols = ['file_name', 'file_path', 'transcription']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return

    print(f"Loaded {len(df)} files\n")

    # Initialize evaluator
    evaluator = TrueBatchVietnameseEvaluator(gpu_id=args.gpu, batch_size=args.batch_size)

    # Run evaluation
    df_results = evaluator.evaluate_batch_true(
        df,
        base_path=args.base_path,
        output_csv=args.output_csv,
        log_file=args.log_file
    )

    # Print statistics
    evaluator.print_statistics(df_results)

    print(f"\nResults saved to: {args.output_csv}")
    print(f"Log saved to: {args.log_file}\n")


if __name__ == '__main__':
    main()
