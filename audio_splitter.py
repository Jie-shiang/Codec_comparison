#!/usr/bin/env python3
"""
Audio Splitter Tool

Split audio files into segments with smart overlap handling and customizable output directory.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from segment_utils import split_audio_file, get_segment_duration


class AudioSplitter:
    """Audio file splitting and metadata generation with customizable output directory"""
    
    def __init__(self,
                 csv_file: str,
                 original_dir: str,
                 segment_length: float,
                 split_output_dir: str = None,
                 output_format: str = "wav",
                 sample_rate: int = 16000,
                 project_dir: str = "/home/jieshiang/Desktop/GitHub/Codec_comparison"):
        
        self.csv_file = csv_file
        self.original_dir = Path(original_dir)
        self.segment_length = segment_length
        self.output_format = output_format
        self.sample_rate = sample_rate
        self.project_dir = Path(project_dir)
        
        # Set split output directory
        if split_output_dir:
            self.split_output_dir = Path(split_output_dir)
        else:
            # Default: use original_dir
            self.split_output_dir = self.original_dir
        
        self.csv_path = self.project_dir / "csv" / csv_file
        
        # Create segment length string (e.g., "1.0s")
        self.segment_str = f"{segment_length:.1f}s"
        
        print(f"Initializing Audio Splitter with Smart Overlap Handling:")
        print(f"  Input CSV: {self.csv_path}")
        print(f"  Original directory: {self.original_dir}")
        print(f"  Split output directory: {self.split_output_dir}")
        print(f"  Segment length: {self.segment_length}s")
        print(f"  Output format: {self.output_format}")
        print(f"  Sample rate: {self.sample_rate}Hz")
        print(f"\n  Strategy: All segments will be {self.segment_length}s")
        print(f"  Last segment overlaps with previous if needed to ensure full length")
    
    def load_csv(self) -> pd.DataFrame:
        """Load input CSV file"""
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            print(f"Loaded {len(df)} files from CSV")
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            sys.exit(1)
    
    def resolve_audio_path(self, csv_path: str) -> Path:
        """Resolve full path to audio file"""
        clean_path = csv_path.lstrip('./')
        return self.original_dir / clean_path
    
    def get_output_directory_structure(self, original_file_path: str) -> Path:
        """
        Determine output directory structure for split files.
        
        Structure: {split_output_dir}/{dataset}/{subset}/{segment_length}/...
        Example: /mnt/Internal/jieshiang/Split_Result/LibriSpeech/test-clean/1.0s/8463/287645/
        """
        relative_path = Path(original_file_path.lstrip('./'))
        path_parts = relative_path.parts
        
        # Find the index of the main dataset directory
        dataset_root_idx = None
        dataset_root_keywords = [
            'test-clean', 'test-other', 'train-clean-100', 'train-clean-360', 
            'train-other-500', 'dev-clean', 'dev-other', 'clips'
        ]
        
        for i, part in enumerate(path_parts):
            if part in dataset_root_keywords:
                dataset_root_idx = i
                break
        
        if dataset_root_idx is None:
            # Fallback: just use segment_str as subdirectory
            return self.split_output_dir / self.segment_str
        
        # Extract dataset path up to and including the subset
        # e.g., ['librispeech', 'LibriSpeech', 'test-clean']
        dataset_parts = path_parts[:dataset_root_idx + 1]
        
        # Build output path: split_output_dir / dataset_parts / segment_str / remaining_structure
        output_base = self.split_output_dir / Path(*dataset_parts) / self.segment_str
        
        # Add remaining directory structure (e.g., speaker_id/chapter_id)
        subdir_parts = path_parts[dataset_root_idx + 1:-1]  # Exclude filename
        if subdir_parts:
            output_dir = output_base / Path(*subdir_parts)
        else:
            output_dir = output_base
        
        return output_dir
    
    def split_all_files(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split all audio files and create segment metadata"""
        
        segment_records = []
        failed_files = []
        
        print(f"\nSplitting {len(df)} audio files...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Splitting audio"):
            original_file_name = row['file_name']
            original_file_path = row['file_path']
            transcription = row['transcription']
            original_duration = row['duration']
            
            # Get speaker_id if available (for Common Voice datasets)
            speaker_id = row.get('speaker_id', None)
            
            # Resolve full path to original audio
            full_audio_path = self.resolve_audio_path(original_file_path)
            
            if not full_audio_path.exists():
                failed_files.append(original_file_name)
                continue
            
            # Determine output directory for split segments
            segment_output_dir = self.get_output_directory_structure(original_file_path)
            
            # Split audio file with smart overlap handling
            segments_info = split_audio_file(
                input_path=str(full_audio_path),
                output_dir=str(segment_output_dir),
                segment_length=self.segment_length,
                output_format=self.output_format,
                sample_rate=self.sample_rate
            )
            
            if not segments_info:
                failed_files.append(original_file_name)
                continue
            
            # Create records for each segment
            for seg_info in segments_info:
                seg_path = Path(seg_info['segment_path'])
                
                # Generate relative path for CSV (relative to split_output_dir)
                try:
                    relative_seg_path = seg_path.relative_to(self.split_output_dir)
                    relative_seg_path_str = f"./{relative_seg_path}"
                except ValueError:
                    # If relative path fails, use absolute path
                    relative_seg_path_str = str(seg_path)
                
                record = {
                    'segment_file_name': seg_path.name,
                    'segment_file_path': relative_seg_path_str,
                    'original_file_name': original_file_name,
                    'original_file_path': original_file_path,
                    'segment_index': f"{seg_info['segment_index']:03d}",
                    'segment_duration': seg_info['segment_duration'],
                    'start_time': seg_info['start_time'],
                    'end_time': seg_info['end_time'],
                    'overlap_with_previous': seg_info['overlap_with_previous'],
                    'use_duration_for_merge': seg_info['use_duration_for_merge'],
                    'original_duration': round(original_duration, 3),
                    'transcription': transcription
                }
                
                # Add speaker_id if available
                if speaker_id is not None:
                    record['speaker_id'] = speaker_id
                
                segment_records.append(record)
        
        if failed_files:
            print(f"\nWarning: Failed to split {len(failed_files)} files")
            for fname in failed_files[:10]:
                print(f"  - {fname}")
            if len(failed_files) > 10:
                print(f"  ... and {len(failed_files) - 10} more")
        
        # Create DataFrame
        segment_df = pd.DataFrame(segment_records)
        
        print(f"\nSuccessfully created {len(segment_df)} segment records")
        print(f"Total original files processed: {len(df) - len(failed_files)}")
        
        # Show overlap statistics
        overlapping_segments = segment_df[segment_df['overlap_with_previous'] > 0]
        if len(overlapping_segments) > 0:
            print(f"\nOverlap Statistics:")
            print(f"  Segments with overlap: {len(overlapping_segments)}")
            print(f"  Average overlap: {overlapping_segments['overlap_with_previous'].mean():.3f}s")
            print(f"  Max overlap: {overlapping_segments['overlap_with_previous'].max():.3f}s")
        
        return segment_df
    
    def save_segment_csv(self, segment_df: pd.DataFrame) -> Path:
        """Save segment metadata to CSV"""
        
        # Generate output CSV filename
        base_csv_name = Path(self.csv_file).stem
        output_csv_name = f"{base_csv_name}_{self.segment_str}.csv"
        output_csv_path = self.project_dir / "csv" / output_csv_name
        
        # Add metadata about split output directory to CSV header (as comment)
        # We'll save this info in the first row or as a separate metadata file
        
        # Save CSV
        segment_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        
        # Save metadata file with split_output_dir info
        metadata_file = output_csv_path.with_suffix('.metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write(f"split_output_dir={self.split_output_dir}\n")
            f.write(f"segment_length={self.segment_length}\n")
            f.write(f"output_format={self.output_format}\n")
            f.write(f"sample_rate={self.sample_rate}\n")
        
        print(f"\nSegment CSV saved: {output_csv_path}")
        print(f"Metadata saved: {metadata_file}")
        print(f"Total segments: {len(segment_df)}")
        
        return output_csv_path
    
    def generate_summary_report(self, segment_df: pd.DataFrame):
        """Generate splitting summary report"""
        
        print("\n" + "="*70)
        print("AUDIO SPLITTING SUMMARY")
        print("="*70)
        
        # Count unique original files
        unique_files = segment_df['original_file_name'].nunique()
        print(f"Original files processed: {unique_files}")
        print(f"Total segments created: {len(segment_df)}")
        print(f"Average segments per file: {len(segment_df) / unique_files:.1f}")
        
        print(f"\nOutput directory: {self.split_output_dir}")
        
        # Segment duration statistics
        print(f"\nSegment Duration Statistics:")
        print(f"  Target length: {self.segment_length:.1f}s")
        print(f"  Mean: {segment_df['segment_duration'].mean():.3f}s")
        print(f"  Median: {segment_df['segment_duration'].median():.3f}s")
        print(f"  Min: {segment_df['segment_duration'].min():.3f}s")
        print(f"  Max: {segment_df['segment_duration'].max():.3f}s")
        
        # Overlap statistics
        overlapping = segment_df[segment_df['overlap_with_previous'] > 0]
        if len(overlapping) > 0:
            print(f"\nOverlap Handling:")
            print(f"  Files with last segment overlap: {overlapping['original_file_name'].nunique()}")
            print(f"  Total overlapping segments: {len(overlapping)}")
            print(f"  Average overlap duration: {overlapping['overlap_with_previous'].mean():.3f}s")
        
        # Merge duration statistics
        print(f"\nMerge Duration Statistics:")
        print(f"  Average use_duration: {segment_df['use_duration_for_merge'].mean():.3f}s")
        print(f"  Min use_duration: {segment_df['use_duration_for_merge'].min():.3f}s")
        
        # Count segments per original file
        segments_per_file = segment_df.groupby('original_file_name').size()
        print(f"\nSegments per file:")
        print(f"  Min: {segments_per_file.min()}")
        print(f"  Max: {segments_per_file.max()}")
        print(f"  Most common: {segments_per_file.mode().values[0]}")
        
        print("="*70)
    
    def run(self):
        """Run the complete splitting pipeline"""
        
        print("\n" + "="*70)
        print("STARTING SMART AUDIO SPLITTING PIPELINE")
        print("="*70)
        
        # Load original CSV
        df = self.load_csv()
        
        # Split all files
        segment_df = self.split_all_files(df)
        
        if len(segment_df) == 0:
            print("Error: No segments were created!")
            sys.exit(1)
        
        # Save segment CSV
        output_csv_path = self.save_segment_csv(segment_df)
        
        # Generate summary
        self.generate_summary_report(segment_df)
        
        print("\n" + "="*70)
        print("AUDIO SPLITTING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nOutput CSV: {output_csv_path}")
        print(f"Total segments: {len(segment_df)}")
        print(f"Split files location: {self.split_output_dir}")
        print(f"\nAll segments are {self.segment_length}s (full length)")
        print(f"Overlap information saved in CSV for smart merging")
        
        return segment_df, output_csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Split audio files with smart overlap handling and customizable output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split to custom directory
  python audio_splitter.py \\
      --csv_file librispeech_test_clean_filtered.csv \\
      --original_dir /mnt/Internal/ASR \\
      --split_output_dir /mnt/Internal/jieshiang/Split_Result \\
      --segment_length 1.0
  
  # Split to same directory as original (default)
  python audio_splitter.py \\
      --csv_file librispeech_test_clean_filtered.csv \\
      --original_dir /mnt/Internal/ASR \\
      --segment_length 1.0

Output Structure:
  {split_output_dir}/LibriSpeech/test-clean/1.0s/8463/287645/
  ├── 8463-287645-0001_001.wav
  ├── 8463-287645-0001_002.wav
  └── ...

Segmentation Strategy:
  - All segments will be exactly segment_length seconds
  - If remainder < segment_length, last segment overlaps with previous
  - Example (4.1s file, 1.0s segments):
    Seg 1: 0.0-1.0s (1.0s)
    Seg 2: 1.0-2.0s (1.0s)
    Seg 3: 2.0-3.0s (1.0s)
    Seg 4: 3.1-4.1s (1.0s) ← overlaps 0.1s with Seg 3
  - During merge: only use 0.1s from Seg 4 to reconstruct 4.1s
        """
    )
    
    parser.add_argument("--csv_file", required=True, type=str,
                       help="Input CSV filename (in ./csv/ directory)")
    parser.add_argument("--original_dir", required=True, type=str,
                       help="Root directory for original audio files")
    parser.add_argument("--split_output_dir", type=str, default=None,
                       help="Output directory for split segments (default: same as original_dir)")
    parser.add_argument("--segment_length", required=True, type=float,
                       help="Segment length in seconds (e.g., 1.0, 2.0, 3.0)")
    parser.add_argument("--output_format", type=str, default="wav",
                       choices=["wav", "flac"],
                       help="Output audio format (default: wav)")
    parser.add_argument("--sample_rate", type=int, default=16000,
                       help="Sample rate for output files (default: 16000)")
    parser.add_argument("--project_dir", type=str,
                       default="/home/jieshiang/Desktop/GitHub/Codec_comparison",
                       help="Project root directory")
    
    args = parser.parse_args()
    
    splitter = AudioSplitter(
        csv_file=args.csv_file,
        original_dir=args.original_dir,
        split_output_dir=args.split_output_dir,
        segment_length=args.segment_length,
        output_format=args.output_format,
        sample_rate=args.sample_rate,
        project_dir=args.project_dir
    )
    
    try:
        splitter.run()
        print("\nSplitting completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nSplitting interrupted by user!")
        
    except Exception as e:
        print(f"\nError during splitting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()