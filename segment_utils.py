#!/usr/bin/env python3
"""
Audio Segmentation Utilities

Common utilities for audio file splitting and merging operations.
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def split_audio_file(
    input_path: str,
    output_dir: str,
    segment_length: float,
    output_format: str = "wav",
    sample_rate: int = 16000
) -> List[Tuple[str, float]]:
    """
    Split an audio file into segments of specified length.
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory to save split segments
        segment_length: Length of each segment in seconds
        output_format: Output file format (wav/flac)
        sample_rate: Target sample rate
        
    Returns:
        List of (segment_path, segment_duration) tuples
    """
    try:
        # Load audio file
        audio, sr = librosa.load(input_path, sr=sample_rate, mono=True)
        total_duration = len(audio) / sr
        
        # Calculate segment parameters
        segment_samples = int(segment_length * sr)
        num_segments = int(np.ceil(len(audio) / segment_samples))
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get base filename (without extension)
        base_name = Path(input_path).stem
        
        segments_info = []
        
        for i in range(num_segments):
            # Extract segment
            start_sample = i * segment_samples
            end_sample = min((i + 1) * segment_samples, len(audio))
            segment_audio = audio[start_sample:end_sample]
            
            # Calculate actual segment duration
            actual_duration = len(segment_audio) / sr
            
            # Generate output filename: original_001.wav, original_002.wav, etc.
            segment_filename = f"{base_name}_{i+1:03d}.{output_format}"
            segment_path = output_dir / segment_filename
            
            # Save segment
            sf.write(str(segment_path), segment_audio, sr)
            
            segments_info.append((str(segment_path), actual_duration))
            
        return segments_info
        
    except Exception as e:
        print(f"Error splitting audio file {input_path}: {e}")
        return []


def merge_audio_segments(
    segment_paths: List[str],
    output_path: str,
    sample_rate: int = 16000
) -> bool:
    """
    Merge multiple audio segments into a single file.
    
    Args:
        segment_paths: List of paths to audio segments (in order)
        output_path: Path to save merged audio
        sample_rate: Sample rate for output file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not segment_paths:
            print("No segments to merge")
            return False
        
        # Load all segments
        segments = []
        for seg_path in segment_paths:
            if not Path(seg_path).exists():
                print(f"Segment not found: {seg_path}")
                return False
            
            audio, sr = librosa.load(seg_path, sr=sample_rate, mono=True)
            segments.append(audio)
        
        # Concatenate all segments
        merged_audio = np.concatenate(segments)
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save merged audio
        sf.write(str(output_path), merged_audio, sample_rate)
        
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