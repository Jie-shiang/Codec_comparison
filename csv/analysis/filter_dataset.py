#!/usr/bin/env python3
"""
Filter dataset results to select samples with lower CER/WER.

This script:
1. Reads analysis results (aishell.csv, commonvoice.csv, librispeech.csv)
2. Filters samples based on CER/WER to keep high-quality samples
3. Ensures each speaker is represented
4. Limits maximum samples per speaker
5. Outputs filtered CSV files

基本篩選（完整資料）
python /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/filter_dataset.py --dataset aishell

篩選 + 產生乾淨 CSV
python /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/filter_dataset.py --dataset aishell --clean

"""

import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Matplotlib settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DatasetFilter:
    """Filter dataset based on CER/WER with speaker balancing"""

    def __init__(self, dataset_name, target_size=2000, max_per_speaker=100, save_clean=False):
        self.dataset_name = dataset_name.lower()
        self.target_size = target_size
        self.max_per_speaker = max_per_speaker
        self.save_clean = save_clean

        # Dataset configurations
        self.configs = {
            'aishell': {
                'input_csv': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/aishell/aishell.csv',
                'output_csv': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/aishell/aishell_filtered.csv',
                'output_dir': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/aishell',
                'metric': 'cer',
                'metric_label': 'Character Error Rate (CER)',
            },
            'commonvoice': {
                'input_csv': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/commonvoice/commonvoice.csv',
                'output_csv': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/commonvoice/commonvoice_filtered.csv',
                'output_dir': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/commonvoice',
                'metric': 'cer',
                'metric_label': 'Character Error Rate (CER)',
            },
            'librispeech': {
                'input_csv': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/librispeech/librispeech.csv',
                'output_csv': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/librispeech/librispeech_filtered.csv',
                'output_dir': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/librispeech',
                'metric': 'wer',
                'metric_label': 'Word Error Rate (WER)',
            }
        }

        if self.dataset_name not in self.configs:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(self.configs.keys())}")

        self.config = self.configs[self.dataset_name]

    def load_data(self):
        """Load CSV data"""
        input_path = self.config['input_csv']
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} samples from {input_path}")
        return df

    def filter_by_speaker_balanced(self, df):
        """
        Filter data with speaker balancing strategy:
        1. Ensure every speaker has at least one sample
        2. Sort samples by CER/WER (lower is better)
        3. Select samples round-robin across speakers
        4. Limit max samples per speaker
        """
        metric = self.config['metric']

        # Remove samples with missing metric values
        df_valid = df.dropna(subset=[metric]).copy()
        print(f"Valid samples (non-null {metric.upper()}): {len(df_valid)}")

        # Sort by metric (ascending - lower is better)
        df_valid = df_valid.sort_values(metric)

        # Get unique speakers
        speakers = df_valid['speaker_id'].unique()
        num_speakers = len(speakers)
        print(f"Total speakers: {num_speakers}")

        # Calculate samples per speaker
        min_per_speaker = 1  # At least one sample per speaker
        samples_per_speaker = min(
            self.max_per_speaker,
            max(min_per_speaker, self.target_size // num_speakers)
        )

        print(f"Target: {self.target_size} samples")
        print(f"Min per speaker: {min_per_speaker}")
        print(f"Max per speaker: {self.max_per_speaker}")
        print(f"Initial allocation per speaker: {samples_per_speaker}")

        # Phase 1: Ensure every speaker has at least min_per_speaker samples
        selected_samples = []
        speaker_counts = {spk: 0 for spk in speakers}

        # Group by speaker
        grouped = df_valid.groupby('speaker_id')

        # First pass: get min_per_speaker samples from each speaker
        for speaker, group in grouped:
            # Sort group by metric
            group_sorted = group.sort_values(metric)
            # Take up to min_per_speaker samples
            n_samples = min(min_per_speaker, len(group_sorted))
            selected = group_sorted.head(n_samples)
            selected_samples.append(selected)
            speaker_counts[speaker] = n_samples

        # Combine first pass
        df_selected = pd.concat(selected_samples, ignore_index=True)
        print(f"After ensuring minimum coverage: {len(df_selected)} samples")

        # Phase 2: Fill remaining quota with best samples
        remaining_quota = self.target_size - len(df_selected)

        if remaining_quota > 0:
            # Get samples not yet selected
            selected_indices = set(df_selected.index)
            df_remaining = df_valid[~df_valid.index.isin(selected_indices)].copy()

            # Sort remaining samples by metric
            df_remaining = df_remaining.sort_values(metric)

            # Add samples round-robin, respecting max_per_speaker
            additional_samples = []
            for _, row in df_remaining.iterrows():
                if len(additional_samples) >= remaining_quota:
                    break

                speaker = row['speaker_id']
                if speaker_counts[speaker] < self.max_per_speaker:
                    additional_samples.append(row)
                    speaker_counts[speaker] += 1

            if additional_samples:
                df_additional = pd.DataFrame(additional_samples)
                df_selected = pd.concat([df_selected, df_additional], ignore_index=True)

        # Final sort by metric
        df_selected = df_selected.sort_values(metric).reset_index(drop=True)

        print(f"\n=== Filtering Summary ===")
        print(f"Total selected: {len(df_selected)} samples")
        print(f"Speakers represented: {df_selected['speaker_id'].nunique()}")
        print(f"Average per speaker: {len(df_selected) / df_selected['speaker_id'].nunique():.1f}")
        print(f"Min samples per speaker: {df_selected.groupby('speaker_id').size().min()}")
        print(f"Max samples per speaker: {df_selected.groupby('speaker_id').size().max()}")
        print(f"{metric.upper()} range: [{df_selected[metric].min():.4f}, {df_selected[metric].max():.4f}]")
        print(f"Mean {metric.upper()}: {df_selected[metric].mean():.4f}")

        return df_selected

    def print_speaker_distribution(self, df):
        """Print detailed speaker distribution"""
        speaker_counts = df.groupby('speaker_id').size().sort_values(ascending=False)

        print(f"\n=== Speaker Distribution (Top 20) ===")
        for i, (speaker, count) in enumerate(speaker_counts.head(20).items(), 1):
            print(f"{i:2d}. Speaker {speaker}: {count} samples")

        if len(speaker_counts) > 20:
            print(f"... and {len(speaker_counts) - 20} more speakers")

    def create_distribution_plot(self, data, metric_name, output_path, xlabel=None, title=None):
        """Create distribution histogram with mean line"""
        # Filter out None values
        data = [x for x in data if x is not None and not np.isnan(x)]

        if len(data) == 0:
            print(f"No valid data for {metric_name}, skipping plot")
            return

        # Calculate statistics
        mean_val = np.mean(data)

        # Create plot
        plt.figure(figsize=(10, 6))

        # Create histogram
        n, bins, patches = plt.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')

        # Add mean line
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_val:.4f}')

        # Labels and title
        plt.xlabel(xlabel or metric_name, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title or f'{metric_name} Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)

        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved plot: {output_path}")

    def create_plots(self, df):
        """Create distribution plots for filtered data"""
        print("\n=== Creating Distribution Plots ===")

        # Create plot directory
        plot_dir = os.path.join(self.config['output_dir'], 'plot')
        os.makedirs(plot_dir, exist_ok=True)

        metric = self.config['metric']
        metric_label = self.config['metric_label']

        # Plot configurations
        plot_configs = [
            (metric, metric_label, f"{metric.upper()} Distribution - {self.dataset_name.upper()} (Filtered)"),
            ('MOS_Quality', 'MOS Quality Score', f'MOS Quality Distribution - {self.dataset_name.upper()} (Filtered)'),
            ('MOS_Naturalness', 'MOS Naturalness Score', f'MOS Naturalness Distribution - {self.dataset_name.upper()} (Filtered)'),
        ]

        for metric_name, xlabel, title in plot_configs:
            output_path = os.path.join(plot_dir, f"{metric_name}_distribution_filtered.png")
            self.create_distribution_plot(
                df[metric_name].values,
                metric_name,
                output_path,
                xlabel=xlabel,
                title=title
            )

        print(f"✓ All plots saved to: {plot_dir}")

    def save_filtered_data(self, df):
        """Save filtered data to CSV"""
        output_path = self.config['output_csv']
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✓ Saved filtered data to: {output_path}")

        # Save clean CSV if requested
        if self.save_clean:
            clean_output_path = output_path.replace('_filtered.csv', '_filtered_clean.csv')
            clean_columns = ['file_name', 'file_path', 'duration', 'transcription', 'speaker_id']
            df_clean = df[clean_columns].copy()
            df_clean.to_csv(clean_output_path, index=False, encoding='utf-8')
            print(f"✓ Saved clean filtered data to: {clean_output_path}")

        # Also save a summary
        summary_path = output_path.replace('.csv', '_summary.txt')
        metric = self.config['metric']

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Dataset: {self.dataset_name.upper()}\n")
            f.write(f"Filtered from: {self.config['input_csv']}\n")
            f.write(f"\n=== Statistics ===\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Total speakers: {df['speaker_id'].nunique()}\n")
            f.write(f"Average per speaker: {len(df) / df['speaker_id'].nunique():.2f}\n")
            f.write(f"Min per speaker: {df.groupby('speaker_id').size().min()}\n")
            f.write(f"Max per speaker: {df.groupby('speaker_id').size().max()}\n")
            f.write(f"\n{metric.upper()} Statistics:\n")
            f.write(f"  Mean: {df[metric].mean():.4f}\n")
            f.write(f"  Std: {df[metric].std():.4f}\n")
            f.write(f"  Min: {df[metric].min():.4f}\n")
            f.write(f"  Max: {df[metric].max():.4f}\n")
            f.write(f"  Median: {df[metric].median():.4f}\n")
            f.write(f"\nMOS Quality Statistics:\n")
            f.write(f"  Mean: {df['MOS_Quality'].mean():.4f}\n")
            f.write(f"  Std: {df['MOS_Quality'].std():.4f}\n")
            f.write(f"\nMOS Naturalness Statistics:\n")
            f.write(f"  Mean: {df['MOS_Naturalness'].mean():.4f}\n")
            f.write(f"  Std: {df['MOS_Naturalness'].std():.4f}\n")

            # Speaker distribution
            f.write(f"\n=== Speaker Distribution ===\n")
            speaker_counts = df.groupby('speaker_id').size().sort_values(ascending=False)
            for speaker, count in speaker_counts.items():
                f.write(f"Speaker {speaker}: {count} samples\n")

        print(f"✓ Saved summary to: {summary_path}")

    def run(self):
        """Run the filtering process"""
        print("="*80)
        print(f"Dataset Filtering: {self.dataset_name.upper()}")
        print("="*80)
        print(f"Target size: {self.target_size} samples")
        print(f"Max per speaker: {self.max_per_speaker}")
        print(f"Metric: {self.config['metric'].upper()}")
        print("="*80)

        # Load data
        print("\n[1/3] Loading data...")
        df = self.load_data()

        # Print original statistics
        metric = self.config['metric']
        print(f"\n=== Original Data Statistics ===")
        print(f"Total samples: {len(df)}")
        print(f"Total speakers: {df['speaker_id'].nunique()}")
        print(f"{metric.upper()} mean: {df[metric].mean():.4f}")

        # Filter data
        print(f"\n[2/3] Filtering data...")
        df_filtered = self.filter_by_speaker_balanced(df)

        # Print speaker distribution
        self.print_speaker_distribution(df_filtered)

        # Save results
        print(f"\n[3/4] Saving results...")
        self.save_filtered_data(df_filtered)

        # Create plots
        print(f"\n[4/4] Creating plots...")
        self.create_plots(df_filtered)

        print("\n" + "="*80)
        print("Filtering complete!")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Filter dataset results based on CER/WER')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['aishell', 'commonvoice', 'librispeech'],
                       help='Dataset to filter')
    parser.add_argument('--target_size', type=int, default=2000,
                       help='Target number of samples to keep (default: 2000)')
    parser.add_argument('--max_per_speaker', type=int, default=100,
                       help='Maximum samples per speaker (default: 100)')
    parser.add_argument('--clean', action='store_true',
                       help='Also save a clean CSV with only essential columns (file_name, file_path, duration, transcription, speaker_id)')

    args = parser.parse_args()

    filter_obj = DatasetFilter(
        dataset_name=args.dataset,
        target_size=args.target_size,
        max_per_speaker=args.max_per_speaker,
        save_clean=args.clean
    )

    filter_obj.run()


if __name__ == "__main__":
    main()
