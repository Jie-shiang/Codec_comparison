#!/usr/bin/env python3
"""
Audio Segmentation Utilities

Common utilities for audio file splitting and merging operations with smart overlap handling.
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def split_audio_file(
    input_path: str,
    output_dir: str,
    segment_length: float,
    output_format: str = "wav",
    sample_rate: int = 16000
) -> List[Dict]:
    """
    Split an audio file into segments with smart overlap handling.
    
    All segments will be exactly segment_length seconds (or the full file if shorter).
    If the last segment would be shorter than segment_length, it will overlap with 
    the previous segment to ensure full length.
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory to save split segments
        segment_length: Length of each segment in seconds
        output_format: Output file format (wav/flac)
        sample_rate: Target sample rate
        
    Returns:
        List of segment info dictionaries containing:
        - segment_path: Path to saved segment
        - segment_index: Segment number (1-based)
        - segment_duration: Duration of segment (should be segment_length)
        - start_time: Start time in original audio
        - end_time: End time in original audio
        - overlap_with_previous: Overlap duration with previous segment (0 for first segments)
        - use_duration_for_merge: How much of this segment to use when merging
    """
    try:
        # Load audio file
        audio, sr = librosa.load(input_path, sr=sample_rate, mono=True)
        total_duration = len(audio) / sr
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get base filename (without extension)
        base_name = Path(input_path).stem
        
        # Calculate segment parameters
        segment_samples = int(segment_length * sr)
        
        # Calculate number of segments
        num_full_segments = int(total_duration // segment_length)
        remainder = total_duration - (num_full_segments * segment_length)
        
        # Determine total number of segments
        if remainder > 0:
            num_segments = num_full_segments + 1
        else:
            num_segments = num_full_segments
            remainder = 0
        
        segments_info = []
        
        for i in range(num_segments):
            # Calculate start and end for this segment
            if i < num_full_segments:
                # Regular segments: no overlap
                start_sample = i * segment_samples
                end_sample = (i + 1) * segment_samples
                start_time = i * segment_length
                end_time = (i + 1) * segment_length
                overlap = 0.0
                use_duration = segment_length
                
            else:
                # Last segment: take from end to ensure full length
                # This may overlap with previous segment
                end_sample = len(audio)
                start_sample = max(0, end_sample - segment_samples)
                end_time = total_duration
                start_time = end_time - segment_length
                
                # Calculate overlap with previous segment
                if num_full_segments > 0:
                    previous_end_time = num_full_segments * segment_length
                    overlap = previous_end_time - start_time
                else:
                    overlap = 0.0
                
                # For merging, only use the non-overlapping part (remainder)
                use_duration = remainder
            
            # Extract segment
            segment_audio = audio[start_sample:end_sample]
            actual_duration = len(segment_audio) / sr
            
            # Generate output filename: original_001.wav, original_002.wav, etc.
            segment_filename = f"{base_name}_{i+1:03d}.{output_format}"
            segment_path = output_dir / segment_filename
            
            # Save segment
            sf.write(str(segment_path), segment_audio, sr)
            
            # Store segment information
            segment_info = {
                'segment_path': str(segment_path),
                'segment_index': i + 1,
                'segment_duration': round(actual_duration, 3),
                'start_time': round(start_time, 3),
                'end_time': round(end_time, 3),
                'overlap_with_previous': round(overlap, 3),
                'use_duration_for_merge': round(use_duration, 3)
            }
            
            segments_info.append(segment_info)
        
        return segments_info
        
    except Exception as e:
        print(f"Error splitting audio file {input_path}: {e}")
        return []


def merge_audio_segments(
    segment_paths: List[str],
    segment_info: List[Dict],
    output_path: str,
    sample_rate: int = 16000
) -> bool:
    """
    Merge multiple audio segments into a single file with smart overlap handling.
    
    Uses segment_info to determine how much of each segment to use,
    avoiding overlapping regions.
    
    Args:
        segment_paths: List of paths to audio segments (in order)
        segment_info: List of segment info dictionaries from split_audio_file
        output_path: Path to save merged audio
        sample_rate: Sample rate for output file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not segment_paths:
            print("No segments to merge")
            return False
        
        if len(segment_paths) != len(segment_info):
            print(f"Mismatch: {len(segment_paths)} paths but {len(segment_info)} info entries")
            return False
        
        merged_audio = []
        
        for i, (seg_path, info) in enumerate(zip(segment_paths, segment_info)):
            if not Path(seg_path).exists():
                print(f"Segment not found: {seg_path}")
                return False
            
            # Load segment
            audio, sr = librosa.load(seg_path, sr=sample_rate, mono=True)
            
            # Get the duration to use for this segment
            use_duration = info['use_duration_for_merge']
            use_samples = int(use_duration * sr)
            
            # For the last segment, take from the end
            if i == len(segment_paths) - 1 and info['overlap_with_previous'] > 0:
                # Last segment with overlap: take only the non-overlapping part from the end
                segment_to_use = audio[-use_samples:]
            else:
                # Regular segment or first segment: take from beginning
                segment_to_use = audio[:use_samples]
            
            merged_audio.append(segment_to_use)
        
        # Concatenate all segments
        final_audio = np.concatenate(merged_audio)
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save merged audio
        sf.write(str(output_path), final_audio, sample_rate)
        
        return True
        
    except Exception as e:
        print(f"Error merging audio segments: {e}")
        return False


def find_segment_files(
    base_name: str,
    search_dir: str,
    suffix: str = "",
    extension: str = "wav"
) -> List[str]:
    """
    Find all segment files for a given base name.
    
    Args:
        base_name: Base filename (without segment number)
        search_dir: Directory to search in
        suffix: Suffix after segment number (e.g., "_inference")
        extension: File extension
        
    Returns:
        Sorted list of segment file paths
    """
    search_dir = Path(search_dir)
    
    if not search_dir.exists():
        return []
    
    # Pattern: base_name_001suffix.extension, base_name_002suffix.extension, etc.
    pattern = f"{base_name}_[0-9][0-9][0-9]{suffix}.{extension}"
    
    # Find all matching files
    segment_files = list(search_dir.glob(pattern))
    
    # Sort by segment number
    segment_files.sort(key=lambda x: x.name)
    
    return [str(f) for f in segment_files]


def validate_segment_integrity(
    original_segments: List[str],
    inference_segments: List[str]
) -> Tuple[bool, str]:
    """
    Validate that original and inference segments match.
    
    Args:
        original_segments: List of original segment paths
        inference_segments: List of inference segment paths
        
    Returns:
        (is_valid, error_message) tuple
    """
    if len(original_segments) == 0:
        return False, "No original segments found"
    
    if len(inference_segments) == 0:
        return False, "No inference segments found"
    
    if len(original_segments) != len(inference_segments):
        return False, f"Segment count mismatch: {len(original_segments)} original vs {len(inference_segments)} inference"
    
    return True, "OK"


def get_segment_duration(audio_path: str) -> float:
    """
    Get duration of an audio file in seconds.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds, or 0.0 if error
    """
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        return len(audio) / sr
    except Exception as e:
        print(f"Error getting duration for {audio_path}: {e}")
        return 0.0