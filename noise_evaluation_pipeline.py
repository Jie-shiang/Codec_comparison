#!/usr/bin/env python3
"""
Noise Evaluation Pipeline
- Inherits from FastCodecEvaluationPipeline
- Saves to audio_noise/, configs_noise/, result/
- Original files copied from /mnt/Internal/jieshiang/Noise_Result/ (noisy audio)
"""

import argparse
import pandas as pd
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, '/home/jieshiang/Desktop/GitHub/Codec_comparison')
from fast_evaluation_pipeline import FastCodecEvaluationPipeline


class NoiseEvaluationPipeline(FastCodecEvaluationPipeline):
    """Noise evaluation pipeline with correct file organization"""
    
    def __init__(self, **kwargs):
        # Force dataset_type to "noise"
        kwargs['dataset_type'] = "noise"
        
        super().__init__(**kwargs)
        
        # Override directories for noise evaluation
        self.audio_dir = self.project_dir / "audio_noise"
        self.config_dir = self.project_dir / "configs_noise"
        # result_dir stays as "result" (not result_noise)
        
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Noise evaluation initialized:")
        print(f"  Audio samples dir: {self.audio_dir}")
        print(f"  Config dir: {self.config_dir}")
        print(f"  Result dir: {self.result_dir}")
    
    def copy_sample_files(self, selected_samples: list):
        """
        Copy sample audio files to audio_noise/ directory
        
        Directory structure: audio_noise/LibriSpeech/BigCodec/80Hz/xxx.wav
                            audio_noise/CommonVoice/BigCodec/80Hz/xxx.wav
        
        Files to copy:
        - First utterance from first 5 speakers (original + inference)
        - Highest WER/CER sample (original + inference)
        
        Original files are from /mnt/Internal/jieshiang/Noise_Result/ (NOISY audio)
        """
        # Determine dataset subdir
        if self.base_dataset_name == 'LibriSpeech':
            dataset_subdir = 'LibriSpeech'
        elif self.base_dataset_name == 'CommonVoice':
            dataset_subdir = 'CommonVoice'
        else:
            dataset_subdir = 'Other'
        
        # Create output directory structure
        output_base = self.audio_dir / dataset_subdir / self.model_name / self.frequency
        output_base.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying sample files to: {output_base}")
        
        copied_count = 0
        for sample_name, row in selected_samples:
            file_name = row['file_name']
            base_name = file_name
            
            # Find original (noisy) audio
            # Original path should be from self.original_dir which is /mnt/Internal/jieshiang/Noise_Result
            file_path = row.get('file_path', '').lstrip('./')
            if self.original_dir:
                original_path = self.original_dir / file_path
            else:
                print(f"Warning: original_dir not set, cannot copy original files")
                continue
            
            # Find inference audio
            inference_path = self.find_inference_audio(file_name)
            
            if original_path.exists() and inference_path and inference_path.exists():
                try:
                    # Copy both files
                    original_dest = output_base / f"{base_name}.wav"
                    inference_dest = output_base / f"{base_name}_inference.wav"
                    
                    shutil.copy2(original_path, original_dest)
                    shutil.copy2(inference_path, inference_dest)
                    
                    print(f"  Copied {sample_name}: {base_name}")
                    copied_count += 1
                    
                except Exception as e:
                    print(f"  Warning: Failed to copy {base_name}: {e}")
            else:
                if not original_path.exists():
                    print(f"  Warning: Original not found: {original_path}")
                if not inference_path or not inference_path.exists():
                    print(f"  Warning: Inference not found for: {base_name}")
        
        print(f"✓ Copied {copied_count} sample pairs")
    
    def save_results(self, results_df: pd.DataFrame):
        """Save results and add noise-specific summaries"""
        
        # Add SNR and noise_type info from CSV
        csv_df = pd.read_csv(self.csv_file, encoding='utf-8')
        
        if 'snr_db' in csv_df.columns:
            snr_map = dict(zip(csv_df['file_name'], csv_df['snr_db']))
            results_df['snr_db'] = results_df['file_name'].map(snr_map)
        
        if 'noise_type' in csv_df.columns:
            noise_map = dict(zip(csv_df['file_name'], csv_df['noise_type']))
            results_df['noise_type'] = results_df['file_name'].map(noise_map)
        
        # Call parent save (saves detailed and summary CSVs)
        super().save_results(results_df)
        
        # Create noise-specific summary
        if 'snr_db' in results_df.columns:
            self._save_noise_summary(results_df)
    
    def _save_noise_summary(self, results_df: pd.DataFrame):
        """
        Generate summary grouped by SNR bins and noise types
        
        Groups:
        - Overall
        - SNR 5-8 dB
        - SNR 8-11 dB  
        - SNR 11-15 dB
        - Noise: free-sound
        - Noise: sound-bible
        """
        summary_rows = []
        
        # Overall statistics
        summary_rows.append(self._calc_stats(results_df, "Overall"))
        
        # SNR bins
        snr_bins = [(5, 8), (8, 11), (11, 15)]
        for min_snr, max_snr in snr_bins:
            bin_df = results_df[
                (results_df['snr_db'] >= min_snr) & 
                (results_df['snr_db'] < max_snr)
            ]
            if len(bin_df) > 0:
                bin_name = f"SNR {min_snr}-{max_snr} dB"
                summary_rows.append(self._calc_stats(bin_df, bin_name))
        
        # Noise types
        if 'noise_type' in results_df.columns:
            for noise_type in results_df['noise_type'].unique():
                type_df = results_df[results_df['noise_type'] == noise_type]
                if len(type_df) > 0:
                    type_name = f"Noise: {noise_type}"
                    summary_rows.append(self._calc_stats(type_df, type_name))
        
        # Save
        summary_df = pd.DataFrame(summary_rows)
        summary_path = self.result_dir / f"noise_summary_{self.model_name}_{self.frequency}_noise_{self.dataset_name}.csv"
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        
        print(f"\n✓ Noise summary: {summary_path}")
        print("\nNoise Summary:")
        print(summary_df.to_string(index=False))
    
    def _calc_stats(self, df: pd.DataFrame, group_name: str) -> dict:
        """Calculate mean and std for each metric"""
        stats = {'Group': group_name, 'Count': len(df)}
        
        for metric in self.metrics_to_compute:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    stats[f'{metric}_mean'] = round(values.mean(), 4)
                    stats[f'{metric}_std'] = round(values.std(), 4)
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Noise Evaluation Pipeline")
    
    # Required arguments
    parser.add_argument("--inference_dir", required=True,
                       help="Inference audio directory (e.g., /mnt/Internal/jieshiang/Inference_Result/BigCodec/80Hz/librispeech_noise_recon)")
    parser.add_argument("--csv_file", required=True,
                       help="Noise CSV file in csv/ (e.g., librispeech_test_clean_noise.csv)")
    parser.add_argument("--model_name", required=True, help="Model name (e.g., BigCodec)")
    parser.add_argument("--frequency", required=True, help="Frequency (e.g., 80Hz)")
    parser.add_argument("--causality", required=True, choices=["Causal", "Non-Causal"])
    parser.add_argument("--bit_rate", required=True, help="Bit rate")
    
    # Important: original_dir should point to noisy audio location
    parser.add_argument("--original_dir", required=True,
                       help="Noisy audio root dir (e.g., /mnt/Internal/jieshiang/Noise_Result)")
    
    # Optional arguments
    parser.add_argument("--project_dir", default="/home/jieshiang/Desktop/GitHub/Codec_comparison")
    parser.add_argument("--quantizers", default="4")
    parser.add_argument("--codebook_size", default="1024")
    parser.add_argument("--n_params", default="45M")
    parser.add_argument("--training_set", default="Custom Dataset")
    parser.add_argument("--testing_set", default="Custom Test Set")
    parser.add_argument("--metrics", nargs='+',
                       choices=["dwer", "dcer", "utmos", "pesq", "stoi", "speaker_similarity"],
                       default=["dwer", "dcer", "utmos", "pesq", "stoi", "speaker_similarity"])
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--language", choices=["en", "zh"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--asr_batch_size", type=int, default=16)
    
    args = parser.parse_args()
    
    use_gpu = args.use_gpu and not args.cpu_only
    
    # Create pipeline
    pipeline = NoiseEvaluationPipeline(
        inference_dir=args.inference_dir,
        csv_file=args.csv_file,
        model_name=args.model_name,
        frequency=args.frequency,
        causality=args.causality,
        bit_rate=args.bit_rate,
        project_dir=args.project_dir,
        quantizers=args.quantizers,
        codebook_size=args.codebook_size,
        n_params=args.n_params,
        training_set=args.training_set,
        testing_set=args.testing_set,
        metrics_to_compute=args.metrics,
        use_gpu=use_gpu,
        gpu_id=args.gpu_id,
        original_dir=args.original_dir,
        language=args.language,
        num_workers=args.num_workers,
        asr_batch_size=args.asr_batch_size
    )
    
    try:
        pipeline.run_evaluation()
        print("\n✓ Noise evaluation completed successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()