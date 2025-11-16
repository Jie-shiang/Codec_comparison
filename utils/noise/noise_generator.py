#!/usr/bin/env python3
"""Noise Generator - Add MUSAN noise to clean audio files"""

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import random
from typing import Tuple


class NoiseGenerator:
    """Add MUSAN noise with random SNR (5-15 dB)"""
    
    def __init__(self, musan_noise_dir: str = "/mnt/External/ASR/musan/noise", 
                 target_sr: int = 16000, seed: int = 42):
        self.musan_noise_dir = Path(musan_noise_dir)
        self.target_sr = target_sr
        random.seed(seed)
        np.random.seed(seed)
        
        # Discover all noise files
        self.noise_files = list(self.musan_noise_dir.rglob("*.wav"))
        if not self.noise_files:
            raise ValueError(f"No noise files found in {musan_noise_dir}")
        
        print(f"Loaded {len(self.noise_files)} noise files from MUSAN")
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and resample audio to target SR"""
        audio, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)
        return audio
    
    def get_noise_segment(self, duration: float) -> Tuple[np.ndarray, str]:
        """Get random noise segment of specified duration
        
        Returns:
            (noise_audio, noise_type) where noise_type is 'free-sound' or 'sound-bible'
        """
        noise_file = random.choice(self.noise_files)
        noise_audio = self.load_audio(str(noise_file))
        
        required_samples = int(duration * self.target_sr)
        
        # Loop if noise is shorter
        if len(noise_audio) < required_samples:
            repeats = int(np.ceil(required_samples / len(noise_audio)))
            noise_audio = np.tile(noise_audio, repeats)
        
        # Extract random segment
        start_idx = random.randint(0, max(0, len(noise_audio) - required_samples))
        noise_segment = noise_audio[start_idx:start_idx + required_samples]
        
        # Get noise type from parent directory name
        noise_type = noise_file.parent.name  # e.g., 'free-sound' or 'sound-bible'
        
        return noise_segment, noise_type
    
    def add_noise_at_snr(self, clean_audio: np.ndarray, noise_audio: np.ndarray, 
                         snr_db: float) -> np.ndarray:
        """Add noise at specified SNR level
        
        SNR (dB) = 20 * log10(signal_rms / noise_rms)
        """
        min_len = min(len(clean_audio), len(noise_audio))
        clean_audio = clean_audio[:min_len]
        noise_audio = noise_audio[:min_len]
        
        # Calculate RMS
        clean_rms = np.sqrt(np.mean(clean_audio ** 2))
        noise_rms = np.sqrt(np.mean(noise_audio ** 2))
        
        if noise_rms < 1e-10:
            return clean_audio
        
        # Calculate required noise scaling
        target_noise_rms = clean_rms / (10 ** (snr_db / 20))
        scaled_noise = noise_audio * (target_noise_rms / noise_rms)
        
        # Mix
        noisy_audio = clean_audio + scaled_noise
        
        # Prevent clipping
        max_val = np.abs(noisy_audio).max()
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val * 0.95
        
        return noisy_audio
    
    def process_file(self, clean_path: str, output_path: str, 
                     snr_db: float) -> Tuple[bool, str]:
        """Process one file: add noise and save
        
        Returns:
            (success, noise_type)
        """
        try:
            # Load clean audio
            clean_audio = self.load_audio(clean_path)
            duration = len(clean_audio) / self.target_sr
            
            # Get noise segment
            noise_segment, noise_type = self.get_noise_segment(duration)
            
            # Add noise
            noisy_audio = self.add_noise_at_snr(clean_audio, noise_segment, snr_db)
            
            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, noisy_audio, self.target_sr)
            
            return True, noise_type
            
        except Exception as e:
            print(f"Error processing {clean_path}: {e}")
            return False, ""
