#!/usr/bin/env python3
"""
Audio Splitter Tool

Split audio files into segments of specified length and generate metadata CSV.
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
    """Audio file splitting and metadata generation"""
    
    def __init__(self,
                 csv_file: str,
                 original_dir: str,
                 segment_length: float,
                 output_format: str = "wav",
                 sample_rate: int = 16000,
                 project_dir: str = "/home/jieshiang/Desktop/GitHub/Codec_comparison"):
        
        self.csv_file = csv_file
        self.original_dir = Path(original_dir)
        self.segment_length = segment_length
        self.output_format = output_format
        self.sample_rate = sample_rate
        self.project_dir = Path(project_dir)
        
        self.csv_path = self.project_dir / "csv" / csv_file
        
        # Create segment length string (e.g., "1.0s")
        self.segment_str = f"{segment_length:.1f}s"
        
        print(f"Initializing Audio Splitter:")
        print(f"  Input CSV: {self.csv_path}")
        print(f"  Original directory: {self.original_dir}")
        print(f"  Segment length: {self.segment_length}s")
        print(f"  Output format: {self.output_format}")
        print(f"  Sample rate: {self.sample_rate}Hz")
    
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
            
            # Resolve full path
            full_audio_path = self.resolve_audio_path(original_file_path)
            
            if not full_audio_path.exists():
                failed_files.append(original_file_name)
                continue
            
            # Determine output directory
            # e.g., /mnt/Internal/ASR/librispeech/LibriSpeech/test-clean/1.0s/
            audio_dir = full_audio_path.parent
            segment_output_dir = audio_dir / self.segment_str
            
            # Split audio file
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
            for seg_idx, (seg_path, seg_duration) in enumerate(segments_info, start=1):
                seg_path = Path(seg_path)
                
                # Generate relative path for CSV
                # e.g., ./librispeech/LibriSpeech/test-clean/1.0s/61-70968-0000_001.wav
                try:
                    relative_seg_path = seg_path.relative_to(self.original_dir)
                    relative_seg_path_str = f"./{relative_seg_path}"
                except ValueError:
                    # If relative path fails, use absolute path
                    relative_seg_path_str = str(seg_path)
                
                record = {
                    'segment_file_name': seg_path.name,
                    'segment_file_path': relative_seg_path_str,
                    'original_file_name': original_file_name,
                    'original_file_path': original_file_path,
                    'segment_index': f"{seg_idx:03d}",
                    'segment_duration': round(seg_duration, 3),
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
        
        return segment_df
    
    def save_segment_csv(self, segment_df: pd.DataFrame) -> Path:
        """Save segment metadata to CSV"""
        
        # Generate output CSV filename
        # e.g., librispeech_test_clean_filtered_1.0s.csv
        base_csv_name = Path(self.csv_file).stem
        output_csv_name = f"{base_csv_name}_{self.segment_str}.csv"
        output_csv_path = self.project_dir / "csv" / output_csv_name
        
        # Save CSV
        segment_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        
        print(f"\nSegment CSV saved: {output_csv_path}")
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
        
        # Segment duration statistics
        print(f"\nSegment Duration Statistics:")
        print(f"  Target length: {self.segment_length:.1f}s")
        print(f"  Mean: {segment_df['segment_duration'].mean():.3f}s")
        print(f"  Median: {segment_df['segment_duration'].median():.3f}s")
        print(f"  Min: {segment_df['segment_duration'].min():.3f}s")
        print(f"  Max: {segment_df['segment_duration'].max():.3f}s")
        
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
        print("STARTING AUDIO SPLITTING PIPELINE")
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
        
        return segment_df, output_csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Split audio files into segments and generate metadata CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split LibriSpeech files into 1.0s segments
  python audio_splitter.py \\
      --csv_file librispeech_test_clean_filtered.csv \\
      --original_dir /mnt/Internal/ASR \\
      --segment_length 1.0
  
  # Split Common Voice files into 2.0s segments
  python audio_splitter.py \\
      --csv_file common_voice_zh_CN_train_filtered.csv \\
      --original_dir /mnt/Internal/ASR \\
      --segment_length 2.0 \\
      --output_format wav
        """
    )
    
    parser.add_argument("--csv_file", required=True, type=str,
                       help="Input CSV filename (in ./csv/ directory)")
    parser.add_argument("--original_dir", required=True, type=str,
                       help="Root directory for original audio files")
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