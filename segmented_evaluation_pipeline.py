#!/usr/bin/env python3
"""
Segmented Audio Evaluation Pipeline

Evaluate neural audio codecs on segmented audio files by merging segments
before computing metrics.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import shutil

from enhanced_evaluation_pipeline import EnhancedCodecEvaluationPipeline
from segment_utils import (
    find_segment_files,
    merge_audio_segments,
    validate_segment_integrity
)


class SegmentedEvaluationPipeline(EnhancedCodecEvaluationPipeline):
    """Evaluation pipeline for segmented audio files"""
    
    def __init__(self,
                 segment_csv_file: str,
                 segment_length: float,
                 keep_merged_files: bool = True,
                 **kwargs):
        """
        Initialize segmented evaluation pipeline.
        
        Args:
            segment_csv_file: CSV file with segment metadata (e.g., xxx_1.0s.csv)
            segment_length: Segment length in seconds
            keep_merged_files: Whether to keep merged files for debugging
            **kwargs: All other arguments for EnhancedCodecEvaluationPipeline
        """
        
        # Initialize parent class with segment CSV
        super().__init__(csv_file=segment_csv_file, **kwargs)
        
        self.segment_length = segment_length
        self.segment_str = f"{segment_length:.1f}s"
        self.keep_merged_files = keep_merged_files
        
        # Directory for merged files - under inference_dir/merged
        self.merged_dir = self.inference_dir / "merged"
        self.merged_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Segmented evaluation initialized:")
        print(f"  Segment length: {self.segment_str}")
        print(f"  Keep merged files: {self.keep_merged_files}")
        print(f"  Merged files directory: {self.merged_dir}")
    
    def load_csv_data(self):
        """Load segment CSV data and group by original file"""
        try:
            df = pd.read_csv(self.csv_file, encoding='utf-8')
            print(f"Successfully loaded segment CSV: {self.csv_file}")
            print(f"Total segments: {len(df)}")
            
            # Group by original file
            grouped = df.groupby('original_file_name')
            print(f"Unique original files: {len(grouped)}")
            
            # Detect dataset type
            if 'common_voice' in str(self.csv_file).lower():
                self.base_dataset_name = 'CommonVoice'
            else:
                self.base_dataset_name = 'LibriSpeech'
            
            # Set dataset name based on type
            if self.dataset_type == "clean":
                self.dataset_name = self.base_dataset_name
            elif self.dataset_type == "noise":
                self.dataset_name = f"{self.base_dataset_name}_Noise"
            elif self.dataset_type == "blank":
                self.dataset_name = f"{self.base_dataset_name}_Blank"
            
            # Update result CSV path to include segment length
            dataset_suffix = self.base_dataset_name.lower()
            self.result_csv_path = (
                self.result_dir / 
                f"detailed_results_{self.model_name}_{self.segment_str}_{self.dataset_type}_{dataset_suffix}.csv"
            )
            
            print(f"Dataset: {self.dataset_name}")
            print(f"Result CSV: {self.result_csv_path}")
            print(f"Metrics to compute: {', '.join(self.metrics_to_compute)}")
                
            return df, grouped
            
        except Exception as e:
            print(f"Error loading segment CSV: {e}")
            sys.exit(1)
    
    def find_and_merge_segments(self,
                                original_file_name: str,
                                segment_df: pd.DataFrame) -> tuple:
        """
        Find and merge all segments for an original file.
        
        Returns:
            (merged_original_path, merged_inference_path, success)
        """
        
        base_name = Path(original_file_name).stem
        
        # Get segment information from DataFrame
        file_segments = segment_df[
            segment_df['original_file_name'] == original_file_name
        ].sort_values('segment_index')
        
        if len(file_segments) == 0:
            print(f"No segments found for {original_file_name}")
            return None, None, False
        
        # Paths for merged files - now under inference_dir/merged
        merged_original_path = self.merged_dir / "original" / f"{base_name}_merged.wav"
        merged_inference_path = self.merged_dir / "inference" / f"{base_name}_merged_inference.wav"
        
        merged_original_path.parent.mkdir(parents=True, exist_ok=True)
        merged_inference_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Find inference segments in inference directory
        # The inference directory should have the same structure with segment_length subfolder
        # e.g., /path/to/inference/1.0s/file_001_inference.wav
        inference_segment_dir = self.inference_dir / self.segment_str
        
        if not inference_segment_dir.exists():
            # Try without segment subfolder
            inference_segment_dir = self.inference_dir
        
        inference_segments = find_segment_files(
            base_name=base_name,
            search_dir=str(inference_segment_dir),
            suffix="_inference",
            extension="wav"
        )
        
        # Find original segments
        original_segments = []
        for _, seg_row in file_segments.iterrows():
            seg_path = self.resolve_original_path(seg_row['segment_file_path'])
            if seg_path.exists():
                original_segments.append(str(seg_path))
        
        # Validate segments
        is_valid, error_msg = validate_segment_integrity(original_segments, inference_segments)
        
        if not is_valid:
            print(f"  Validation failed for {original_file_name}: {error_msg}")
            return None, None, False
        
        # Merge original segments
        success_original = merge_audio_segments(
            segment_paths=original_segments,
            output_path=str(merged_original_path),
            sample_rate=16000
        )
        
        if not success_original:
            print(f"  Failed to merge original segments for {original_file_name}")
            return None, None, False
        
        # Merge inference segments
        success_inference = merge_audio_segments(
            segment_paths=inference_segments,
            output_path=str(merged_inference_path),
            sample_rate=16000
        )
        
        if not success_inference:
            print(f"  Failed to merge inference segments for {original_file_name}")
            return None, None, False
        
        return merged_original_path, merged_inference_path, True
    
    def cleanup_merged_files(self):
        """Clean up merged files if not keeping them"""
        if not self.keep_merged_files and self.merged_dir.exists():
            try:
                shutil.rmtree(self.merged_dir)
                print(f"Cleaned up merged files directory: {self.merged_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up merged files: {e}")
    
    def run_evaluation(self):
        """Run evaluation on segmented audio files"""
        
        self.start_time = datetime.now().timestamp()
        
        print("=" * 70)
        print("STARTING SEGMENTED AUDIO EVALUATION")
        print("=" * 70)
        
        # Load segment CSV and group by original file
        segment_df, grouped = self.load_csv_data()
        
        # Load existing results if any
        existing_results = self.load_existing_results()
        
        # Initialize evaluator
        from metrics_evaluator import AudioMetricsEvaluator
        evaluator = AudioMetricsEvaluator(
            language=self.language,
            use_gpu=self.use_gpu,
            gpu_id=self.gpu_id
        )
        
        need_asr = ('dcer' in self.metrics_to_compute) or ('dwer' in self.metrics_to_compute)
        need_utmos = 'utmos' in self.metrics_to_compute
        
        if need_asr or need_utmos:
            evaluator.load_models()
        
        # Create empty results DataFrame
        if existing_results is None:
            results_df = self.create_empty_result_dataframe_for_segments(grouped)
        else:
            results_df = existing_results.copy()
        
        new_results = []
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        
        print(f"\nEvaluating {len(grouped)} original files...")
        print(f"Computing metrics: {', '.join(self.metrics_to_compute)}")
        
        evaluation_start = datetime.now().timestamp()
        
        # Process each original file
        for original_file_name, file_segments in tqdm(grouped, desc="Evaluation Progress"):
            
            # Get ground truth from first segment (same for all segments)
            ground_truth = file_segments.iloc[0]['transcription']
            
            # Check if already processed
            should_process = True
            skip_reason = ""
            
            if existing_results is not None:
                existing_row = existing_results[
                    existing_results['file_name'] == original_file_name
                ]
                if not existing_row.empty:
                    should_process, skip_reason = self.should_process_file(
                        existing_row.iloc[0],
                        original_file_name
                    )
            
            if not should_process:
                skipped_count += 1
                if skipped_count <= 5:
                    print(f"  Skipping {original_file_name}: {skip_reason}")
                continue
            
            # Find and merge segments
            merged_original, merged_inference, success = self.find_and_merge_segments(
                original_file_name, file_segments
            )
            
            if not success:
                failed_count += 1
                if failed_count <= 5:
                    print(f"  Failed to merge segments: {original_file_name}")
                continue
            
            try:
                # Evaluate merged files
                metrics_result = self.evaluate_metrics_selectively(
                    evaluator,
                    str(merged_original),
                    str(merged_inference),
                    ground_truth
                )
                
                if metrics_result:
                    result_data = {
                        'file_name': original_file_name,
                        'segment_length': self.segment_str,
                        'num_segments': len(file_segments)
                    }
                    result_data.update(metrics_result)
                    new_results.append(result_data)
                    processed_count += 1
                
            except Exception as e:
                print(f"Error evaluating {original_file_name}: {e}")
                failed_count += 1
                continue
        
        evaluation_time = datetime.now().timestamp() - evaluation_start
        
        print(f"\nEvaluation completed in: {evaluation_time:.2f} seconds")
        print(f"Processed: {processed_count} files")
        print(f"Skipped: {skipped_count} files (metrics already exist)")
        print(f"Failed: {failed_count} files")
        
        # Merge results
        if new_results:
            results_df = self.merge_results(existing_results, new_results)
        
        # Save results
        self.save_results(results_df)
        
        # Generate config and copy files
        if processed_count > 0 or len(results_df) > 0:
            self.generate_config_and_copy_files(results_df)
        
        # Cleanup if requested
        if not self.keep_merged_files:
            self.cleanup_merged_files()
        
        self.end_time = datetime.now().timestamp()
        total_time = self.end_time - self.start_time
        
        print("\n" + "="*70)
        print("SEGMENTED EVALUATION COMPLETED!")
        print("="*70)
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        if processed_count > 0:
            print(f"Average per file: {evaluation_time/processed_count:.2f} seconds")
        print(f"Processed: {processed_count} files")
        print(f"Skipped: {skipped_count} files")
        print(f"Failed: {failed_count} files")
        print(f"Result file: {self.result_csv_path}")
        if self.keep_merged_files:
            print(f"Merged files saved in: {self.merged_dir}")
        
        return results_df
    
    def create_empty_result_dataframe_for_segments(self, grouped) -> pd.DataFrame:
        """Create empty result DataFrame for segmented evaluation"""
        
        columns = [
            'file_name', 'segment_length', 'num_segments',
            'original_path', 'inference_path', 'ground_truth',
            'original_transcript_raw', 'inference_transcript_raw',
            'original_transcript', 'inference_transcript',
            'utmos', 'pesq', 'stoi'
        ]
        
        if 'dwer' in self.metrics_to_compute:
            columns.extend(['original_wer', 'inference_wer', 'dwer'])
        
        if 'dcer' in self.metrics_to_compute:
            columns.extend(['original_cer', 'inference_cer', 'dcer'])
        
        result_df = pd.DataFrame(index=range(len(grouped)))
        for col in columns:
            result_df[col] = np.nan
        
        # Fill in file names
        original_file_names = [name for name, _ in grouped]
        result_df['file_name'] = original_file_names
        
        # Fill in ground truth from first segment of each group
        ground_truths = []
        for name, file_segments in grouped:
            ground_truths.append(file_segments.iloc[0]['transcription'])
        result_df['ground_truth'] = ground_truths
        
        # Fill in segment info
        result_df['segment_length'] = self.segment_str
        num_segments = [len(file_segments) for _, file_segments in grouped]
        result_df['num_segments'] = num_segments
        
        return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Segmented Audio Codec Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate 1.0s segmented LibriSpeech files
  python segmented_evaluation_pipeline.py \\
      --inference_dir /mnt/Internal/jieshiang/Inference_Result/LSCodec/50Hz/librispeech_recon \\
      --segment_csv_file librispeech_test_clean_filtered_1.0s.csv \\
      --segment_length 1.0 \\
      --original_dir /mnt/Internal/ASR \\
      --model_name "LSCodec" \\
      --frequency "50Hz" \\
      --causality "Non-Causal" \\
      --bit_rate "0.45" \\
      --metrics dwer utmos pesq stoi \\
      --use_gpu --gpu_id 0
  
  # Evaluate 2.0s segmented Common Voice files
  python segmented_evaluation_pipeline.py \\
      --inference_dir /path/to/inference/2.0s \\
      --segment_csv_file common_voice_zh_CN_train_filtered_2.0s.csv \\
      --segment_length 2.0 \\
      --original_dir /mnt/Internal/ASR \\
      --model_name "MyCodec" \\
      --frequency "25Hz" \\
      --causality "Causal" \\
      --bit_rate "1.0" \\
      --metrics dcer utmos pesq stoi \\
      --keep_merged_files \\
      --use_gpu --gpu_id 1
        """
    )
    
    # Segment-specific arguments
    parser.add_argument("--segment_csv_file", required=True, type=str,
                       help="Segment CSV filename (e.g., xxx_1.0s.csv in ./csv/ directory)")
    parser.add_argument("--segment_length", required=True, type=float,
                       help="Segment length in seconds (must match CSV)")
    parser.add_argument("--keep_merged_files", action="store_true",
                       help="Keep merged audio files for debugging (default: True)")
    parser.add_argument("--no_keep_merged_files", action="store_true",
                       help="Delete merged audio files after evaluation")
    
    # Standard evaluation arguments
    parser.add_argument("--inference_dir", required=True, type=str,
                       help="Directory containing inference audio segments")
    parser.add_argument("--model_name", required=True, type=str,
                       help="Name of codec model")
    parser.add_argument("--frequency", required=True, type=str,
                       help="Frame rate (e.g., 50Hz)")
    parser.add_argument("--causality", required=True, type=str,
                       choices=["Causal", "Non-Causal"],
                       help="Model causality type")
    parser.add_argument("--bit_rate", required=True, type=str,
                       help="Compression bit rate")
    
    parser.add_argument("--dataset_type", type=str,
                       choices=["clean", "noise", "blank"], default="clean",
                       help="Dataset type (clean/noise/blank)")
    parser.add_argument("--project_dir", type=str,
                       default="/home/jieshiang/Desktop/GitHub/Codec_comparison",
                       help="Project root directory path")
    parser.add_argument("--quantizers", type=str, default="4",
                       help="Number of quantizers")
    parser.add_argument("--codebook_size", type=str, default="1024",
                       help="Codebook size")
    parser.add_argument("--n_params", type=str, default="45M",
                       help="Number of model parameters")
    parser.add_argument("--training_set", type=str, default="Custom Dataset",
                       help="Training dataset description")
    parser.add_argument("--testing_set", type=str, default="Custom Test Set",
                       help="Testing dataset description")
    
    parser.add_argument("--metrics", type=str, nargs='+',
                       choices=["dwer", "dcer", "utmos", "pesq", "stoi"],
                       default=["dwer", "dcer", "utmos", "pesq", "stoi"],
                       help="Metrics to compute")
    
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="Enable GPU acceleration")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU device ID")
    parser.add_argument("--cpu_only", action="store_true",
                       help="Force CPU-only computation")
    parser.add_argument("--original_dir", type=str,
                       help="Root directory path for original audio files")
    parser.add_argument("--language", type=str, choices=["en", "zh"],
                       help="Language for ASR evaluation (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    use_gpu = args.use_gpu and not args.cpu_only
    keep_merged = not args.no_keep_merged_files if args.no_keep_merged_files else True
    
    pipeline = SegmentedEvaluationPipeline(
        segment_csv_file=args.segment_csv_file,
        segment_length=args.segment_length,
        keep_merged_files=keep_merged,
        inference_dir=args.inference_dir,
        csv_file=args.segment_csv_file,  # Will be used by parent class
        model_name=args.model_name,
        frequency=args.frequency,
        causality=args.causality,
        bit_rate=args.bit_rate,
        dataset_type=args.dataset_type,
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
        language=args.language
    )
    
    try:
        results_df = pipeline.run_evaluation()
        print("\nSegmented evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user!")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()