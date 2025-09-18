#!/usr/bin/env python3
"""
Audio Codec Metrics Evaluator

This module provides comprehensive audio quality and ASR accuracy evaluation metrics
for neural audio codec assessment, including dWER/dCER, UTMOS, PESQ, and STOI.
"""

import torch
import librosa
import numpy as np
import re
import string
from transformers import pipeline
import jiwer
from pesq import pesq
from pystoi import stoi
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class AudioMetricsEvaluator:
    """Audio quality and ASR accuracy metrics evaluator"""
    
    def __init__(self, language='en', device=None, use_gpu=True, gpu_id=0):
        self.language = language
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.device = self._setup_device(device)
        self.asr_pipeline = None
        self.utmos_model = None
        
    def _setup_device(self, device=None):
        """Setup computation device with GPU support"""
        if device:
            return device
            
        if not self.use_gpu:
            print("GPU acceleration disabled, using CPU")
            return "cpu"
            
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            return "cpu"
            
        if self.gpu_id >= torch.cuda.device_count():
            print(f"GPU {self.gpu_id} not available, using GPU 0")
            self.gpu_id = 0
            
        device_name = f"cuda:{self.gpu_id}"
        gpu_name = torch.cuda.get_device_name(self.gpu_id)
        gpu_memory = torch.cuda.get_device_properties(self.gpu_id).total_memory / 1024**3
        
        print(f"Using GPU {self.gpu_id}: {gpu_name} ({gpu_memory:.1f}GB)")
        return device_name
    
    def get_gpu_memory_info(self):
        """Get current GPU memory usage"""
        if self.device.startswith('cuda'):
            allocated = torch.cuda.memory_allocated(self.gpu_id) / 1024**3
            reserved = torch.cuda.memory_reserved(self.gpu_id) / 1024**3
            total = torch.cuda.get_device_properties(self.gpu_id).total_memory / 1024**3
            return {
                'allocated': allocated,
                'reserved': reserved, 
                'total': total,
                'free': total - reserved
            }
        return None
        
    def load_models(self):
        """Load ASR and UTMOS models"""
        print(f"Loading models on device: {self.device}")
        
        if self.device.startswith('cuda'):
            mem_info = self.get_gpu_memory_info()
            print(f"Initial GPU memory: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")
        
        print("Loading Whisper-large-v3 ASR model...")
        try:
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-large-v3", 
                device=self.device
            )
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("Falling back to CPU...")
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-large-v3", 
                device="cpu"
            )
            self.device = "cpu"
        
        if self.device.startswith('cuda'):
            mem_info = self.get_gpu_memory_info()
            print(f"After ASR loading: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")
        
        print("Loading UTMOS model...")
        try:
            self.utmos_model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", 
                "utmos22_strong", 
                trust_repo=True
            )
            
            if self.device.startswith('cuda'):
                try:
                    self.utmos_model = self.utmos_model.to(self.device)
                except Exception as e:
                    print(f"Warning: Could not move UTMOS to GPU: {e}")
                    print("UTMOS will run on CPU")
            
        except Exception as e:
            print(f"Error loading UTMOS model: {e}")
            self.utmos_model = None
            
        if self.device.startswith('cuda'):
            mem_info = self.get_gpu_memory_info()
            print(f"After all models loaded: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")
        
        print("All models loaded successfully!")
        
    def normalize_text(self, text: str) -> str:
        """Normalize text for WER/CER calculation"""
        if self.language == 'zh':
            text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)
            text = re.sub(r'\s+', '', text)
            return text.strip()
        else:
            text = text.upper()
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = ' '.join(text.split())
            return text
            
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate (CER)"""
        if not reference or not hypothesis:
            return 1.0
            
        ref_chars = list(reference)
        hyp_chars = list(hypothesis)
        
        return jiwer.wer(ref_chars, hyp_chars)
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using ASR model"""
        try:
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
                
            if self.language == 'zh':
                result = self.asr_pipeline(
                    str(audio_path), 
                    return_timestamps=True,
                    generate_kwargs={
                        "task": "transcribe", 
                        "language": "zh"
                    }
                )
            else:
                result = self.asr_pipeline(
                    str(audio_path), 
                    return_timestamps=True,
                    generate_kwargs={
                        "task": "transcribe", 
                        "language": "en"
                    }
                )
            return result.get("text", "")
        except Exception as e:
            print(f"Error transcribing audio {audio_path}: {e}")
            return ""
    
    def calculate_dwer_dcer(self, original_audio_path: str, inference_audio_path: str, ground_truth: str) -> dict:
        """Calculate dWER or dCER depending on language"""
        try:
            original_transcript = self.transcribe_audio(original_audio_path)
            inference_transcript = self.transcribe_audio(inference_audio_path)
            
            ground_truth_norm = self.normalize_text(ground_truth)
            original_norm = self.normalize_text(original_transcript)
            inference_norm = self.normalize_text(inference_transcript)
            
            if self.language == 'zh':
                original_cer = self.calculate_cer(ground_truth_norm, original_norm)
                inference_cer = self.calculate_cer(ground_truth_norm, inference_norm)
                dcer = inference_cer - original_cer
                
                return {
                    'original_transcript': original_transcript,
                    'inference_transcript': inference_transcript,
                    'original_cer': original_cer,
                    'inference_cer': inference_cer,
                    'dcer': dcer,
                    'metric_name': 'dCER'
                }
            else:
                original_wer = jiwer.wer(ground_truth_norm, original_norm) if ground_truth_norm and original_norm else 1.0
                inference_wer = jiwer.wer(ground_truth_norm, inference_norm) if ground_truth_norm and inference_norm else 1.0
                dwer = inference_wer - original_wer
                
                return {
                    'original_transcript': original_transcript,
                    'inference_transcript': inference_transcript,
                    'original_wer': original_wer,
                    'inference_wer': inference_wer,
                    'dwer': dwer,
                    'metric_name': 'dWER'
                }
                
        except Exception as e:
            print(f"Error calculating dWER/dCER: {e}")
            return None
    
    def calculate_utmos(self, audio_path: str) -> float:
        """Calculate UTMOS score with GPU acceleration"""
        try:
            if self.utmos_model is None:
                return None
                
            wave, sr = librosa.load(str(audio_path), sr=None, mono=True)
            
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                wave_tensor = torch.from_numpy(wave).unsqueeze(0)
                
                # Move to device if model is on GPU
                if hasattr(self.utmos_model, 'device') and str(self.utmos_model.device) != 'cpu':
                    wave_tensor = wave_tensor.to(self.utmos_model.device)
                
                score = self.utmos_model(wave_tensor, sr)
                return score.item()
        except Exception as e:
            print(f"Error calculating UTMOS: {e}")
            return None
    
    def calculate_pesq(self, reference_path: str, degraded_path: str) -> float:
        """Calculate PESQ score (CPU only)"""
        try:
            ref_audio, _ = librosa.load(reference_path, sr=16000, mono=True)
            deg_audio, _ = librosa.load(degraded_path, sr=16000, mono=True)
            
            min_len = min(len(ref_audio), len(deg_audio))
            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]
            
            if min_len < 16000 * 0.1:
                return None
                
            pesq_score = pesq(16000, ref_audio, deg_audio, 'wb')
            return pesq_score
            
        except Exception as e:
            print(f"Error calculating PESQ: {e}")
            return None
    
    def calculate_stoi(self, reference_path: str, degraded_path: str) -> float:
        """Calculate STOI score (CPU only)"""
        try:
            ref_audio, _ = librosa.load(reference_path, sr=16000, mono=True)
            deg_audio, _ = librosa.load(degraded_path, sr=16000, mono=True)
            
            min_len = min(len(ref_audio), len(deg_audio))
            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]
            
            stoi_score = stoi(ref_audio, deg_audio, 16000, extended=False)
            return stoi_score
            
        except Exception as e:
            print(f"Error calculating STOI: {e}")
            return None
    
    def evaluate_audio_pair(self, original_path: str, inference_path: str, ground_truth: str) -> dict:
        """Evaluate a pair of audio files with all metrics"""
        results = {}
        
        asr_result = self.calculate_dwer_dcer(original_path, inference_path, ground_truth)
        if asr_result:
            results.update(asr_result)
        
        results['utmos'] = self.calculate_utmos(inference_path)
        results['pesq'] = self.calculate_pesq(original_path, inference_path)
        results['stoi'] = self.calculate_stoi(original_path, inference_path)
        
        return results