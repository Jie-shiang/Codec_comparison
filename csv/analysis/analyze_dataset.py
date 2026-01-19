#!/usr/bin/env python3
"""
Unified dataset analyzer for AISHELL, Common Voice, and LibriSpeech.

This script:
1. Reads dataset transcriptions (ground truth)
2. Runs ASR on audio files to get transcription results
3. Calculates WER/CER using jiwer between ASR results and ground truth
4. Evaluates audio quality metrics (MOS_Quality, MOS_Naturalness)
5. Generates CSV files with detailed and summary statistics
6. Creates distribution plots for each metric

# 測試 AISHELL (10 個檔案)
python analyze_dataset.py --dataset aishell --limit 10

# 測試 Common Voice (10 個檔案)
python analyze_dataset.py --dataset commonvoice --limit 10

# 測試 LibriSpeech (10 個檔案)
python analyze_dataset.py --dataset librispeech --limit 10

"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
import argparse

# Import the metrics evaluator
sys.path.insert(0, '/home/jieshiang/Desktop/GitHub/Codec_comparison')
from metrics_evaluator_v2 import AudioMetricsEvaluatorV2

warnings.filterwarnings('ignore')

# Matplotlib settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DatasetAnalyzer:
    """Unified analyzer for different datasets"""

    def __init__(self, dataset_name, gpu_id=0, use_cpu=False, batch_size=32, limit=None):
        self.dataset_name = dataset_name.lower()
        self.gpu_id = gpu_id
        self.use_cpu = use_cpu
        self.batch_size = batch_size
        self.limit = limit

        # Dataset configurations
        self.configs = {
            'aishell': {
                'language': 'zh',
                'metric_name': 'cer',
                'base_path': '/mnt/Internal/ASR',
                'transcript_file': '/mnt/Internal/ASR/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt',
                'audio_dir': '/mnt/Internal/ASR/aishell/data_aishell/wav/test',
                'output_dir': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/aishell',
            },
            'commonvoice': {
                'language': 'zh',
                'metric_name': 'cer',
                'base_path': '/mnt/Internal/ASR',
                'transcript_file': '/mnt/Internal/ASR/common_voice/cv-corpus-22.0-2025-06-20/zh-TW/test.tsv',
                'audio_dir': '/mnt/Internal/ASR/common_voice/cv-corpus-22.0-2025-06-20/zh-TW/clips',
                'output_dir': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/commonvoice',
            },
            'librispeech': {
                'language': 'en',
                'metric_name': 'wer',
                'base_path': '/mnt/Internal/ASR',
                'audio_dir': '/mnt/Internal/ASR/librispeech/LibriSpeech/test-clean',
                'output_dir': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/librispeech',
            }
        }

        if self.dataset_name not in self.configs:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(self.configs.keys())}")

        self.config = self.configs[self.dataset_name]
        self.output_dir = Path(self.config['output_dir'])
        self.plot_dir = self.output_dir / 'plot'

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def load_aishell_transcripts(self, transcript_file):
        """Load AISHELL transcripts"""
        transcripts = {}
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    file_id, transcript = parts
                    transcript = transcript.replace(' ', '')
                    transcripts[file_id] = transcript
        return transcripts

    def load_commonvoice_transcripts(self, tsv_file):
        """Load Common Voice test.tsv"""
        df = pd.read_csv(tsv_file, sep='\t')
        transcripts = {}
        speakers = {}

        for _, row in df.iterrows():
            file_name = row['path'].replace('.mp3', '')
            transcripts[file_name] = row['sentence']
            speakers[file_name] = row['client_id']

        return transcripts, speakers

    def load_librispeech_transcripts(self, audio_dir):
        """Load LibriSpeech transcripts from trans.txt files"""
        transcripts = {}

        for speaker_dir in sorted(os.listdir(audio_dir)):
            speaker_path = os.path.join(audio_dir, speaker_dir)
            if not os.path.isdir(speaker_path):
                continue

            for chapter_dir in sorted(os.listdir(speaker_path)):
                chapter_path = os.path.join(speaker_path, chapter_dir)
                if not os.path.isdir(chapter_path):
                    continue

                # Find trans.txt file
                trans_file = os.path.join(chapter_path, f"{speaker_dir}-{chapter_dir}.trans.txt")
                if os.path.exists(trans_file):
                    with open(trans_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split(maxsplit=1)
                            if len(parts) == 2:
                                file_id, transcript = parts
                                transcripts[file_id] = transcript

        return transcripts

    def collect_audio_files(self):
        """Collect audio files based on dataset type"""
        audio_data = []

        if self.dataset_name == 'aishell':
            transcripts = self.load_aishell_transcripts(self.config['transcript_file'])
            audio_dir = self.config['audio_dir']

            for speaker_dir in sorted(os.listdir(audio_dir)):
                speaker_path = os.path.join(audio_dir, speaker_dir)
                if not os.path.isdir(speaker_path):
                    continue

                for audio_file in sorted(os.listdir(speaker_path)):
                    if not audio_file.endswith('.wav'):
                        continue

                    file_id = audio_file.replace('.wav', '')
                    if file_id not in transcripts:
                        continue

                    relative_path = f"./aishell/data_aishell/wav/test/{speaker_dir}/{audio_file}"
                    full_path = os.path.join(self.config['base_path'], relative_path.replace('./', ''))

                    audio_data.append({
                        'file_id': file_id,
                        'relative_path': relative_path,
                        'full_path': full_path,
                        'speaker_id': speaker_dir,
                        'transcription': transcripts[file_id]
                    })

        elif self.dataset_name == 'commonvoice':
            transcripts, speakers = self.load_commonvoice_transcripts(self.config['transcript_file'])
            audio_dir = self.config['audio_dir']

            for file_name in transcripts.keys():
                audio_file = f"{file_name}.mp3"
                full_path = os.path.join(audio_dir, audio_file)

                if not os.path.exists(full_path):
                    continue

                relative_path = f"./common_voice/cv-corpus-22.0-2025-06-20/zh-TW/clips/{audio_file}"

                audio_data.append({
                    'file_id': file_name,
                    'relative_path': relative_path,
                    'full_path': full_path,
                    'speaker_id': speakers[file_name],
                    'transcription': transcripts[file_name]
                })

        elif self.dataset_name == 'librispeech':
            transcripts = self.load_librispeech_transcripts(self.config['audio_dir'])
            audio_dir = self.config['audio_dir']

            for speaker_dir in sorted(os.listdir(audio_dir)):
                speaker_path = os.path.join(audio_dir, speaker_dir)
                if not os.path.isdir(speaker_path):
                    continue

                for chapter_dir in sorted(os.listdir(speaker_path)):
                    chapter_path = os.path.join(speaker_path, chapter_dir)
                    if not os.path.isdir(chapter_path):
                        continue

                    for audio_file in sorted(os.listdir(chapter_path)):
                        if not audio_file.endswith('.flac'):
                            continue

                        file_id = audio_file.replace('.flac', '')
                        if file_id not in transcripts:
                            continue

                        relative_path = f"./librispeech/LibriSpeech/test-clean/{speaker_dir}/{chapter_dir}/{audio_file}"
                        full_path = os.path.join(chapter_path, audio_file)

                        audio_data.append({
                            'file_id': file_id,
                            'relative_path': relative_path,
                            'full_path': full_path,
                            'speaker_id': speaker_dir,
                            'transcription': transcripts[file_id]
                        })

        return audio_data

    def get_audio_duration(self, audio_path):
        """Get audio duration"""
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.duration
        except:
            try:
                import torchaudio
                waveform, sample_rate = torchaudio.load(audio_path)
                return waveform.shape[1] / sample_rate
            except:
                return None

    def calculate_error_rate(self, evaluator, asr_transcript, ground_truth):
        """Calculate WER or CER based on language"""
        try:
            if not asr_transcript or not ground_truth:
                return None

            # Convert and normalize
            if self.config['language'] == 'zh':
                asr_simplified = evaluator.convert_traditional_to_simplified(asr_transcript)
                gt_simplified = evaluator.convert_traditional_to_simplified(ground_truth)
                asr_norm = evaluator.normalize_text(asr_simplified)
                gt_norm = evaluator.normalize_text(gt_simplified)
                return evaluator.fast_cer(gt_norm, asr_norm)
            else:
                asr_norm = evaluator.normalize_text(asr_transcript)
                gt_norm = evaluator.normalize_text(ground_truth)
                return evaluator.fast_wer(gt_norm, asr_norm)

        except Exception as e:
            print(f"Error calculating error rate: {e}")
            return None

    def calculate_metrics_batch(self, evaluator, audio_data):
        """Calculate metrics for all audio files"""
        results = []

        audio_paths = [item['full_path'] for item in audio_data]
        transcriptions = [item['transcription'] for item in audio_data]

        print(f"\nTotal files to process: {len(audio_paths)}")

        # === 1. ASR & Error Rate ===
        metric_name = self.config['metric_name'].upper()
        print(f"\n=== [1/3] Running ASR and Calculating {metric_name} ===")
        asr_transcripts = evaluator.batch_transcribe(audio_paths)

        error_scores = []
        for gt_transcript, asr_transcript in zip(transcriptions, asr_transcripts):
            error = self.calculate_error_rate(evaluator, asr_transcript, gt_transcript)
            error_scores.append(error)

        valid_errors = [x for x in error_scores if x is not None]
        if valid_errors:
            print(f"Completed {metric_name}: {len(valid_errors)}/{len(error_scores)} valid scores, Mean: {np.mean(valid_errors):.4f}")
            print(f"Note: {metric_name} calculated using jiwer")

        # === 2. MOS Quality ===
        print("\n=== [2/3] Calculating MOS Quality (NISQA v2) ===")
        mos_quality_dict = evaluator.calculate_mos_quality_batch(audio_paths)
        mos_quality_scores = [mos_quality_dict.get(path) for path in audio_paths]

        valid_mos_q = [x for x in mos_quality_scores if x is not None]
        if valid_mos_q:
            print(f"Completed MOS Quality: {len(valid_mos_q)}/{len(mos_quality_scores)} valid scores, Mean: {np.mean(valid_mos_q):.4f}")

        # === 3. MOS Naturalness ===
        print("\n=== [3/3] Calculating MOS Naturalness (UTMOS) ===")
        mos_naturalness_dict = evaluator.calculate_mos_naturalness_batch(audio_paths, batch_size=self.batch_size)
        mos_naturalness_scores = [mos_naturalness_dict.get(path) for path in audio_paths]

        valid_mos_n = [x for x in mos_naturalness_scores if x is not None]
        if valid_mos_n:
            print(f"Completed MOS Naturalness: {len(valid_mos_n)}/{len(mos_naturalness_scores)} valid scores, Mean: {np.mean(valid_mos_n):.4f}")

        # Combine results
        print("\n=== Combining results ===")
        for i, item in enumerate(audio_data):
            results.append({
                'file_id': item['file_id'],
                'relative_path': item['relative_path'],
                'speaker_id': item['speaker_id'],
                'transcription': item['transcription'],
                'duration': self.get_audio_duration(item['full_path']),
                'asr_result': asr_transcripts[i] if i < len(asr_transcripts) else None,
                self.config['metric_name']: error_scores[i],
                'MOS_Quality': mos_quality_scores[i],
                'MOS_Naturalness': mos_naturalness_scores[i]
            })

        return results

    def create_distribution_plot(self, data, metric_name, output_path, xlabel=None, title=None):
        """Create distribution histogram with mean line"""
        data = [x for x in data if x is not None and not np.isnan(x)]

        if len(data) == 0:
            print(f"No valid data for {metric_name}, skipping plot")
            return

        mean_val = np.mean(data)

        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')

        plt.xlabel(xlabel or metric_name, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title or f'{metric_name} Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved plot: {output_path}")

    def run(self):
        """Run the analysis"""
        print("="*80)
        print(f"Dataset Analysis: {self.dataset_name.upper()}")
        print("="*80)
        print(f"Language: {self.config['language']}")
        print(f"Metric: {self.config['metric_name'].upper()}")
        print(f"GPU ID: {self.gpu_id}")
        print(f"Use CPU: {self.use_cpu}")
        print(f"Batch size: {self.batch_size}")
        if self.limit:
            print(f"File limit: {self.limit} (testing mode)")
        print(f"Output directory: {self.output_dir}")
        print("="*80)

        # Collect audio files
        print("\n[1/4] Collecting audio files and transcripts...")
        audio_data = self.collect_audio_files()
        print(f"Found {len(audio_data)} audio files with transcripts")

        if self.limit:
            audio_data = audio_data[:self.limit]
            print(f"Limited to first {self.limit} files for testing")

        # Initialize evaluator
        print("\n[2/4] Loading models...")
        evaluator = AudioMetricsEvaluatorV2(
            language=self.config['language'],
            device=None,
            use_gpu=not self.use_cpu,
            gpu_id=self.gpu_id
        )
        evaluator.load_models()

        # Calculate metrics
        print("\n[3/4] Calculating metrics...")
        results = self.calculate_metrics_batch(evaluator, audio_data)

        # Save results
        print("\n[4/4] Saving results...")
        df_detailed = pd.DataFrame(results)
        df_detailed = df_detailed.rename(columns={
            'file_id': 'file_name',
            'relative_path': 'file_path'
        })

        # Reorder columns
        column_order = [
            'file_name', 'file_path', 'duration', 'transcription', 'speaker_id',
            'asr_result', self.config['metric_name'], 'MOS_Quality', 'MOS_Naturalness'
        ]
        df_detailed = df_detailed[column_order]

        # Save detailed CSV
        detailed_csv_path = self.output_dir / f"{self.dataset_name}.csv"
        df_detailed.to_csv(detailed_csv_path, index=False, encoding='utf-8')
        print(f"Saved detailed results: {detailed_csv_path}")

        # Calculate summary statistics
        print("\n=== Calculating summary statistics ===")
        spk_num = df_detailed['speaker_id'].nunique()
        avg_utts = len(df_detailed) / spk_num

        metric_means = {}
        for metric in [self.config['metric_name'], 'MOS_Quality', 'MOS_Naturalness']:
            values = df_detailed[metric].dropna()
            if len(values) > 0:
                metric_means[f'{metric}_mean'] = values.mean()
            else:
                metric_means[f'{metric}_mean'] = None

        summary_data = {
            'base_path': [self.config['base_path']],
            'spk_num': [spk_num],
            'avg_utts': [avg_utts],
            **{k: [v] for k, v in metric_means.items()}
        }

        df_summary = pd.DataFrame(summary_data)
        summary_csv_path = self.output_dir / f"{self.dataset_name}_summary.csv"
        df_summary.to_csv(summary_csv_path, index=False, encoding='utf-8')
        print(f"Saved summary results: {summary_csv_path}")

        # Print summary
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Dataset: {self.dataset_name.upper()}")
        print(f"Base Path: {self.config['base_path']}")
        print(f"Total Speakers: {spk_num}")
        print(f"Total Utterances: {len(df_detailed)}")
        print(f"Average Utterances per Speaker: {avg_utts:.2f}")
        print(f"Total Duration: {df_detailed['duration'].sum()/3600:.2f} hours")
        print(f"\nMetric Averages:")
        for metric, value in metric_means.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: N/A")

        # Create distribution plots
        print("\n=== Creating distribution plots ===")
        plot_configs = [
            (self.config['metric_name'],
             f"{'Word Error Rate (WER)' if self.config['metric_name'] == 'wer' else 'Character Error Rate (CER)'}",
             f"{self.config['metric_name'].upper()} Distribution - {self.dataset_name.upper()}"),
            ('MOS_Quality', 'MOS Quality Score', f'MOS Quality Distribution - {self.dataset_name.upper()}'),
            ('MOS_Naturalness', 'MOS Naturalness Score', f'MOS Naturalness Distribution - {self.dataset_name.upper()}'),
        ]

        for metric, xlabel, title in plot_configs:
            output_path = self.plot_dir / f"{metric}_distribution.png"
            self.create_distribution_plot(
                df_detailed[metric].values,
                metric,
                output_path,
                xlabel=xlabel,
                title=title
            )

        print("\n" + "="*80)
        print("Analysis complete!")
        print("="*80)
        print(f"\nOutput files:")
        print(f"  - Detailed CSV: {detailed_csv_path}")
        print(f"  - Summary CSV: {summary_csv_path}")
        print(f"  - Plots: {self.plot_dir}")


def main():
    parser = argparse.ArgumentParser(description='Unified dataset analyzer for speech datasets')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['aishell', 'commonvoice', 'librispeech'],
                       help='Dataset to analyze')
    parser.add_argument('--gpu_id', type=int, default=1,
                       help='GPU ID to use (default: 1)')
    parser.add_argument('--use_cpu', action='store_true',
                       help='Use CPU instead of GPU')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for MOS Naturalness (default: 128)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of files to process (for testing)')

    args = parser.parse_args()

    analyzer = DatasetAnalyzer(
        dataset_name=args.dataset,
        gpu_id=args.gpu_id,
        use_cpu=args.use_cpu,
        batch_size=args.batch_size,
        limit=args.limit
    )

    analyzer.run()


if __name__ == "__main__":
    main()
