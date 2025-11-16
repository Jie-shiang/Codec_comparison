#!/usr/bin/env python3
"""
Segmented Audio Evaluation Pipeline - Fast Batch Version

Evaluate neural audio codecs on segmented audio files with batch processing,
GPU acceleration, and smart segment merging.
Inherits from FastCodecEvaluationPipeline for optimized batch processing.
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
from typing import Tuple, Optional
import time
import multiprocessing as mp

from fast_evaluation_pipeline import FastCodecEvaluationPipeline
from metrics_evaluator import AudioMetricsEvaluator
from utils.split.segment_utils import (
    find_segment_files,
    merge_audio_segments,
    validate_segment_integrity
)


class SegmentedEvaluationPipeline(FastCodecEvaluationPipeline):
    """Evaluation pipeline for segmented audio files with fast batch processing"""
    
    def __init__(self,
                 segment_csv_file: str,
                 segment_length: float,
                 split_output_dir: str = None,
                 keep_merged_files: bool = True,
                 num_workers: int = 8,
                 asr_batch_size: int = 16,
                 **kwargs):
        """
        Initialize segmented evaluation pipeline.
        
        Args:
            segment_csv_file: CSV file with segment metadata
            segment_length: Segment length in seconds
            split_output_dir: Directory with split segments (auto-detected if None)
            keep_merged_files: Whether to keep merged files
            num_workers: Number of CPU workers for PESQ/STOI
            asr_batch_size: Batch size for ASR processing
            **kwargs: Arguments for FastCodecEvaluationPipeline
        """
        
        kwargs.pop('csv_file', None)
        super().__init__(
            csv_file=segment_csv_file,
            num_workers=num_workers,
            asr_batch_size=asr_batch_size,
            **kwargs
        )
        
        self.segment_length = segment_length
        self.segment_str = f"{segment_length:.1f}s"
        self.keep_merged_files = keep_merged_files
        
        # Load split directory from metadata if not provided
        if split_output_dir is None:
            split_output_dir = self.load_split_output_dir_from_metadata()
        
        if split_output_dir:
            self.split_output_dir = Path(split_output_dir)
        else:
            self.split_output_dir = self.original_dir if self.original_dir else None
        
        # Directory for merged files
        self.merged_dir = self.inference_dir / "merged"
        self.merged_dir.mkdir(parents=True, exist_ok=True)
        
        # Override directories to include segment length
        self.result_dir = self.project_dir / f"result_{self.segment_str}"
        self.audio_dir = self.project_dir / f"audio_{self.segment_str}"
        self.config_dir = self.project_dir / f"configs_{self.segment_str}"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSegmented evaluation initialized:")
        print(f"  Segment length: {self.segment_str}")
        print(f"  Split directory: {self.split_output_dir}")
        print(f"  Keep merged files: {self.keep_merged_files}")
        print(f"  Merged directory: {self.merged_dir}")
        print(f"  Result output: {self.result_dir}")
        print(f"  Audio output: {self.audio_dir}")
        print(f"  Config output: {self.config_dir}")
    
    def load_split_output_dir_from_metadata(self) -> Optional[str]:
        """Load split_output_dir from metadata file"""
        metadata_file = self.csv_file.with_suffix('.metadata.txt')
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                for line in f:
                    if line.startswith('split_output_dir='):
                        split_dir = line.strip().split('=', 1)[1]
                        print(f"Loaded split_output_dir from metadata: {split_dir}")
                        return split_dir
        except Exception as e:
            print(f"Warning: Could not read metadata file: {e}")
        
        return None
    
    def resolve_segment_path(self, csv_path: str) -> Path:
        """Resolve full path to segment file"""
        if self.split_output_dir is None:
            clean_path = csv_path.lstrip('./')
            return self.original_dir / clean_path if self.original_dir else Path(csv_path)
        
        clean_path = csv_path.lstrip('./')
        return self.split_output_dir / clean_path
    
    def load_csv_data(self):
        """Load segment CSV data and group by original file"""
        try:
            df = pd.read_csv(self.csv_file, encoding='utf-8')
            print(f"Successfully loaded segment CSV: {self.csv_file}")
            print(f"Total segments: {len(df)}")
            
            # Verify required columns
            required_cols = ['start_time', 'end_time', 'overlap_with_previous', 'use_duration_for_merge']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: CSV missing smart merge columns: {missing_cols}")
                print(f"Please re-run audio_splitter.py to generate updated CSV")
                sys.exit(1)
            
            # Group by original file
            grouped = df.groupby('original_file_name')
            print(f"Unique original files: {len(grouped)}")
            
            # Count overlapping segments
            overlapping_files = df[df['overlap_with_previous'] > 0]['original_file_name'].nunique()
            if overlapping_files > 0:
                print(f"Files with overlapping segments: {overlapping_files}")
            
            # Detect dataset type
            if 'common_voice' in str(self.csv_file).lower():
                self.base_dataset_name = 'CommonVoice'
            else:
                self.base_dataset_name = 'LibriSpeech'
            
            # Set dataset name
            if self.dataset_type == "clean":
                self.dataset_name = self.base_dataset_name
            elif self.dataset_type == "noise":
                self.dataset_name = f"{self.base_dataset_name}_Noise"
            elif self.dataset_type == "blank":
                self.dataset_name = f"{self.base_dataset_name}_Blank"
            
            # Update result CSV path
            dataset_suffix = self.base_dataset_name.lower()
            self.result_csv_path = (
                self.result_dir / 
                f"detailed_results_{self.model_name}_{self.segment_str}_{self.frequency}_{self.bit_rate}_{self.dataset_type}_{dataset_suffix}.csv"
            )
            
            print(f"Dataset: {self.dataset_name}")
            print(f"Result CSV: {self.result_csv_path}")
                
            return df, grouped
            
        except Exception as e:
            print(f"Error loading segment CSV: {e}")
            sys.exit(1)
    
    def find_and_merge_segments(self,
                                original_file_name: str,
                                segment_df: pd.DataFrame) -> Tuple[Optional[Path], Optional[Path], bool]:
        """
        Find and merge all segments for an original file with smart overlap handling.
        
        Returns:
            (merged_original_path, merged_inference_path, success)
        """
        try:
            # Sort segments by segment_index
            segment_df = segment_df.sort_values('segment_index').reset_index(drop=True)
            
            # Find all segment files
            original_segments = []
            inference_segments = []
            overlap_info = []
            
            for idx, row in segment_df.iterrows():
                try:
                    # Get segment path
                    segment_path = self.resolve_segment_path(row['segment_file_path'])
                    
                    if not segment_path.exists():
                        print(f"Warning: Original segment not found: {segment_path}")
                        return None, None, False
                    
                    # Find inference segment - use segment_file_name (not file_name)
                    segment_file_name = row['segment_file_name']
                    # Remove extension to get base name
                    base_name = Path(segment_file_name).stem
                    inference_segment_path = self.find_inference_audio(base_name)
                    
                    if not inference_segment_path or not inference_segment_path.exists():
                        print(f"Warning: Inference segment not found: {base_name}")
                        print(f"  Looking in: {self.inference_dir}")
                        print(f"  Segment file name: {segment_file_name}")
                        return None, None, False
                    
                    # Append to lists
                    original_segments.append(str(segment_path))
                    inference_segments.append(str(inference_segment_path))
                    overlap_info.append({
                        'overlap_with_previous': row.get('overlap_with_previous', 0.0),
                        'use_duration_for_merge': row.get('use_duration_for_merge', row['end_time'] - row['start_time'])
                    })
                    
                except KeyError as ke:
                    print(f"Error: Missing required column in segment CSV: {ke}")
                    print(f"Available columns: {list(row.index)}")
                    print(f"Row data: {row.to_dict()}")
                    return None, None, False
                except Exception as e:
                    print(f"Error processing segment {idx} for {original_file_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    return None, None, False
            
            # Check if we have any segments
            if len(original_segments) == 0:
                print(f"Error: No segments found for {original_file_name}")
                return None, None, False
            
            # Validate segment integrity
            is_valid, error_msg = validate_segment_integrity(original_segments, inference_segments)
            if not is_valid:
                print(f"Warning: Segment integrity check failed for {original_file_name}: {error_msg}")
                return None, None, False
            
            # Merge segments
            base_name = Path(original_file_name).stem
            merged_original_path = self.merged_dir / f"{base_name}_original_merged.wav"
            merged_inference_path = self.merged_dir / f"{base_name}_inference_merged.wav"
            
            # Merge original segments
            merge_success = merge_audio_segments(
                original_segments,      # segment_paths: List[str]
                overlap_info,           # segment_info: List[Dict]
                str(merged_original_path)  # output_path: str
            )
            
            if not merge_success:
                print(f"Error: Failed to merge original segments for {original_file_name}")
                return None, None, False
            
            # Merge inference segments
            merge_success = merge_audio_segments(
                inference_segments,     # segment_paths: List[str]
                overlap_info,           # segment_info: List[Dict]
                str(merged_inference_path)  # output_path: str
            )
            
            if not merge_success:
                print(f"Error: Failed to merge inference segments for {original_file_name}")
                if merged_original_path.exists():
                    merged_original_path.unlink()
                return None, None, False
            
            return merged_original_path, merged_inference_path, True
            
        except Exception as e:
            print(f"Error merging segments for {original_file_name}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, False
    
    def run_evaluation(self) -> pd.DataFrame:
        """Run complete evaluation pipeline for segmented audio with fast batch processing"""
        print("\n" + "=" * 60)
        print("Starting Segmented Audio Codec Evaluation (Fast Batch)")
        print("=" * 60 + "\n")
        
        self.start_time = time.time()
        
        # Load segment data
        df, grouped = self.load_csv_data()
        
        # Initialize evaluator
        evaluator = AudioMetricsEvaluator(
            language=self.language,
            use_gpu=self.use_gpu,
            gpu_id=self.gpu_id
        )
        
        # Load models if needed
        need_asr = ('dcer' in self.metrics_to_compute) or ('dwer' in self.metrics_to_compute)
        need_utmos = 'utmos' in self.metrics_to_compute
        
        if need_asr or need_utmos:
            evaluator.load_models()
        
        # Build task list: merge all segments first
        print("\n" + "=" * 60)
        print("Phase 1: Merging Segments")
        print("=" * 60)
        tasks = []
        failed_count = 0
        
        for original_file_name, file_segments in tqdm(grouped, desc="Merging segments"):
            ground_truth = file_segments.iloc[0]['transcription']
            
            # Merge segments
            merged_original, merged_inference, merge_success = self.find_and_merge_segments(
                original_file_name,
                file_segments
            )
            
            if not merge_success:
                failed_count += 1
                continue
            
            tasks.append({
                'file_name': original_file_name,
                'ground_truth': ground_truth,
                'original_path': str(merged_original),
                'inference_path': str(merged_inference),
                'num_segments': len(file_segments)
            })
        
        print(f"\nSuccessfully merged {len(tasks)} original files from segments")
        print(f"Failed to merge: {failed_count} files")
        
        if not tasks:
            print("No valid tasks found. Exiting.")
            return None
        
        # Build results dictionary
        results_dict = {t['file_name']: t.copy() for t in tasks}
        
        # --- Phase 2: Batch Evaluation (同 FastCodecEvaluationPipeline) ---
        print("\n" + "=" * 60)
        print("Phase 2: Batch Metric Calculation")
        print("=" * 60)
        
        # 2a. PESQ/STOI (CPU parallel)
        if 'pesq' in self.metrics_to_compute or 'stoi' in self.metrics_to_compute:
            print(f"\nStarting PESQ/STOI calculation with {self.num_workers} workers...")
            pesq_stoi_tasks = [(t['original_path'], t['inference_path']) for t in tasks]
            
            step_start = time.time()
            pesq_results, stoi_results = evaluator.calculate_pesq_stoi_batch(
                pesq_stoi_tasks, 
                num_workers=self.num_workers
            )
            print(f"PESQ/STOI batch calculation finished in {time.time() - step_start:.2f} seconds")
            
            for i, task in enumerate(tasks):
                file_name = task['file_name']
                if 'pesq' in self.metrics_to_compute:
                    results_dict[file_name]['pesq'] = pesq_results[i]
                if 'stoi' in self.metrics_to_compute:
                    results_dict[file_name]['stoi'] = stoi_results[i]
        
        # 2b. ASR (dWER/dCER) (GPU batch)
        if need_asr:
            print(f"\nStarting ASR batch transcription (Batch Size: {self.asr_batch_size})...")
            step_start = time.time()
            
            # Collect all unique audio paths
            original_paths = sorted(list(set([t['original_path'] for t in tasks])))
            inference_paths = sorted(list(set([t['inference_path'] for t in tasks])))
            
            # Batch transcription
            orig_transcripts_map = self._batch_transcribe_fast(
                evaluator, original_paths, "Transcribing Original Audio"
            )
            inf_transcripts_map = self._batch_transcribe_fast(
                evaluator, inference_paths, "Transcribing Inference Audio"
            )
            
            print(f"ASR batch transcription finished in {time.time() - step_start:.2f} seconds")
            
            # Calculate dWER/dCER
            print("Calculating dWER/dCER metrics...")
            for task in tqdm(tasks, desc="Calculating ASR Metrics"):
                file_name = task['file_name']
                
                original_transcript = orig_transcripts_map.get(task['original_path'], "")
                inference_transcript = inf_transcripts_map.get(task['inference_path'], "")
                ground_truth = task['ground_truth']
                
                results_dict[file_name]['original_transcript_raw'] = original_transcript
                results_dict[file_name]['inference_transcript_raw'] = inference_transcript
                
                # Text normalization
                if self.language == 'zh':
                    original_transcript_simplified = evaluator.convert_traditional_to_simplified(original_transcript)
                    inference_transcript_simplified = evaluator.convert_traditional_to_simplified(inference_transcript)
                    ground_truth_simplified = evaluator.convert_traditional_to_simplified(ground_truth)
                else:
                    original_transcript_simplified = original_transcript
                    inference_transcript_simplified = inference_transcript
                    ground_truth_simplified = ground_truth
                
                ground_truth_norm = evaluator.normalize_text(ground_truth_simplified)
                original_norm = evaluator.normalize_text(original_transcript_simplified)
                inference_norm = evaluator.normalize_text(inference_transcript_simplified)
                
                results_dict[file_name]['original_transcript'] = original_norm
                results_dict[file_name]['inference_transcript'] = inference_norm
                
                if self.language == 'zh' and 'dcer' in self.metrics_to_compute:
                    original_cer = evaluator.fast_cer(ground_truth_norm, original_norm)
                    inference_cer = evaluator.fast_cer(ground_truth_norm, inference_norm)
                    results_dict[file_name]['original_cer'] = original_cer
                    results_dict[file_name]['inference_cer'] = inference_cer
                    results_dict[file_name]['dcer'] = inference_cer - original_cer
                    
                elif self.language == 'en' and 'dwer' in self.metrics_to_compute:
                    original_wer = evaluator.fast_wer(ground_truth_norm, original_norm)
                    inference_wer = evaluator.fast_wer(ground_truth_norm, inference_norm)
                    results_dict[file_name]['original_wer'] = original_wer
                    results_dict[file_name]['inference_wer'] = inference_wer
                    results_dict[file_name]['dwer'] = inference_wer - original_wer
            print("dWER/dCER calculation complete.")
        
        # 2c. UTMOS (GPU sequential)
        if 'utmos' in self.metrics_to_compute:
            print("\nStarting UTMOS calculation...")
            step_start = time.time()
            for task in tqdm(tasks, desc="Calculating UTMOS"):
                score = evaluator.calculate_utmos(task['inference_path'])
                results_dict[task['file_name']]['utmos'] = score
            print(f"UTMOS calculation finished in {time.time() - step_start:.2f} seconds")
        
        # 2d. Speaker Similarity (GPU sequential)
        if 'speaker_similarity' in self.metrics_to_compute:
            print("\nStarting Speaker Similarity calculation...")
            step_start = time.time()
            for task in tqdm(tasks, desc="Calculating Speaker Similarity"):
                score = evaluator.calculate_speaker_similarity(
                    task['original_path'], 
                    task['inference_path']
                )
                results_dict[task['file_name']]['speaker_similarity'] = score
            print(f"Speaker Similarity calculation finished in {time.time() - step_start:.2f} seconds")
        
        # --- Phase 3: Convert results to DataFrame ---
        print("\n" + "=" * 60)
        print("Phase 3: Saving Results")
        print("=" * 60)
        
        results_list = []
        for file_name, result in results_dict.items():
            result['segment_length'] = self.segment_str
            results_list.append(result)
        
        results_df = pd.DataFrame(results_list)
        
        # Save results
        self.save_results(results_df)
        
        # Generate config and copy files
        if len(results_df) > 0:
            self.generate_config_and_copy_files(results_df)
        
        # Cleanup merged files if not keeping them
        if not self.keep_merged_files:
            print("\nCleaning up merged files...")
            for task in tasks:
                merged_original = Path(task['original_path'])
                merged_inference = Path(task['inference_path'])
                if merged_original.exists():
                    merged_original.unlink()
                if merged_inference.exists():
                    merged_inference.unlink()
        
        # Cleanup GPU memory
        evaluator.cleanup_gpu_memory()
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        print("\n" + "=" * 60)
        print("Segmented Evaluation Completed!")
        print("=" * 60)
        print(f"Total time: {total_time:.2f}s ({total_time/60:.1f}min)")
        print(f"Processed: {len(tasks)} files")
        print(f"Failed to merge: {failed_count} files")
        print(f"Result file: {self.result_csv_path}")
        print(f"Audio files: {self.audio_dir}")
        print(f"Config files: {self.config_dir}")
        if self.keep_merged_files:
            print(f"Merged files: {self.merged_dir}")
        
        return results_df
    
    def create_empty_result_dataframe_for_segments(self, grouped) -> pd.DataFrame:
        """Create empty result DataFrame for segmented evaluation"""
        
        columns = [
            'file_name', 'segment_length', 'num_segments',
            'original_path', 'inference_path', 'ground_truth',
            'original_transcript_raw', 'inference_transcript_raw',
            'original_transcript', 'inference_transcript',
            'utmos', 'pesq', 'stoi', 'speaker_similarity'
        ]
        
        if 'dwer' in self.metrics_to_compute:
            columns.extend(['original_wer', 'inference_wer', 'dwer'])
        
        if 'dcer' in self.metrics_to_compute:
            columns.extend(['original_cer', 'inference_cer', 'dcer'])
        
        result_df = pd.DataFrame(index=range(len(grouped)))
        for col in columns:
            result_df[col] = np.nan
        
        # Fill in file names and ground truth
        original_file_names = []
        ground_truths = []
        num_segments = []
        
        for name, file_segments in grouped:
            original_file_names.append(name)
            ground_truths.append(file_segments.iloc[0]['transcription'])
            num_segments.append(len(file_segments))
        
        result_df['file_name'] = original_file_names
        result_df['ground_truth'] = ground_truths
        result_df['segment_length'] = self.segment_str
        result_df['num_segments'] = num_segments
        
        return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Segmented Audio Codec Evaluation Pipeline (Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With custom split directory
  python segmented_evaluation_pipeline.py \\
      --inference_dir /path/to/inference \\
      --segment_csv_file librispeech_test_1.0s.csv \\
      --segment_length 1.0 \\
      --split_output_dir /path/to/split \\
      --model_name "LSCodec" \\
      --frequency "50Hz" \\
      --causality "Non-Causal" \\
      --bit_rate "0.45" \\
      --metrics dwer utmos pesq stoi \\
      --use_gpu --gpu_id 0
  
  # Auto-detect split directory
  python segmented_evaluation_pipeline.py \\
      --inference_dir /path/to/inference \\
      --segment_csv_file librispeech_test_1.0s.csv \\
      --segment_length 1.0 \\
      --model_name "LSCodec" \\
      --frequency "50Hz" \\
      --causality "Non-Causal" \\
      --bit_rate "0.45" \\
      --metrics dwer utmos pesq stoi \\
      --use_gpu --gpu_id 0
        """
    )
    
    # Segment-specific arguments
    parser.add_argument("--segment_csv_file", required=True, type=str,
                       help="Segment CSV filename (in ./csv/ directory)")
    parser.add_argument("--segment_length", required=True, type=float,
                       help="Segment length in seconds")
    parser.add_argument("--split_output_dir", type=str, default=None,
                       help="Directory with split segments (auto-detected if not provided)")
    parser.add_argument("--keep_merged_files", action="store_true",
                       help="Keep merged audio files")
    parser.add_argument("--no_keep_merged_files", action="store_true",
                       help="Delete merged audio files after evaluation")
    
    # Standard evaluation arguments
    parser.add_argument("--inference_dir", required=True, type=str,
                       help="Directory containing inference audio segments")
    parser.add_argument("--model_name", required=True, type=str,
                       help="Codec model name")
    parser.add_argument("--frequency", required=True, type=str,
                       help="Frame rate (e.g., 50Hz)")
    parser.add_argument("--causality", required=True, type=str,
                       choices=["Causal", "Non-Causal"],
                       help="Model causality type")
    parser.add_argument("--bit_rate", required=True, type=str,
                       help="Compression bit rate")
    
    parser.add_argument("--dataset_type", type=str,
                       choices=["clean", "noise", "blank"], default="clean",
                       help="Dataset type")
    parser.add_argument("--project_dir", type=str,
                       default="/home/jieshiang/Desktop/GitHub/Codec_comparison",
                       help="Project root directory")
    parser.add_argument("--quantizers", type=str, default="4",
                       help="Number of quantizers")
    parser.add_argument("--codebook_size", type=str, default="1024",
                       help="Codebook size")
    parser.add_argument("--n_params", type=str, default="45M",
                       help="Number of model parameters")
    parser.add_argument("--training_set", type=str, default="Custom Dataset",
                       help="Training dataset")
    parser.add_argument("--testing_set", type=str, default="Custom Test Set",
                       help="Testing dataset")
    
    parser.add_argument("--metrics", type=str, nargs='+',
                       choices=["dwer", "dcer", "utmos", "pesq", "stoi", "speaker_similarity"],
                       default=["dwer", "dcer", "utmos", "pesq", "stoi", "speaker_similarity"],
                       help="Metrics to compute")
    
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="Enable GPU acceleration")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU device ID")
    parser.add_argument("--cpu_only", action="store_true",
                       help="Force CPU-only computation")
    parser.add_argument("--original_dir", type=str,
                       help="Root directory for original audio files")
    parser.add_argument("--language", type=str, choices=["en", "zh"],
                       help="Language for ASR (auto-detected if not specified)")
    
    # Batch processing arguments
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of CPU workers for PESQ/STOI calculation")
    parser.add_argument("--asr_batch_size", type=int, default=16,
                       help="Batch size for ASR transcription")
    
    args = parser.parse_args()
    
    use_gpu = args.use_gpu and not args.cpu_only
    keep_merged = not args.no_keep_merged_files if args.no_keep_merged_files else True
    
    pipeline = SegmentedEvaluationPipeline(
        segment_csv_file=args.segment_csv_file,
        segment_length=args.segment_length,
        split_output_dir=args.split_output_dir,
        keep_merged_files=keep_merged,
        num_workers=args.num_workers,
        asr_batch_size=args.asr_batch_size,
        inference_dir=args.inference_dir,
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