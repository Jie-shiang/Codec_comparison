#!/usr/bin/env python3
"""
Audio Codec Metrics Evaluator

Optimized module for comprehensive audio quality and ASR accuracy evaluation metrics
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
import Levenshtein

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class AudioMetricsEvaluator:
    """Optimized audio quality and ASR accuracy metrics evaluator"""
    
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
        """Load ASR and UTMOS models with optimizations"""
        print(f"Loading models on device: {self.device}")
        
        if self.device.startswith('cuda'):
            mem_info = self.get_gpu_memory_info()
            print(f"Initial GPU memory: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")
        
        print("Loading Whisper-large-v3 ASR model...")
        try:
            torch_dtype = torch.float16 if self.device.startswith('cuda') else torch.float32
            
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-large-v3",
                torch_dtype=torch_dtype,
                device=self.device,
                model_kwargs={"use_flash_attention_2": False}
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
            self.utmos_model = self.utmos_model.cpu()
            print("UTMOS model loaded on CPU")
            
        except Exception as e:
            print(f"Error loading UTMOS model: {e}")
            self.utmos_model = None
            
        if self.device.startswith('cuda'):
            mem_info = self.get_gpu_memory_info()
            print(f"After all models loaded: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")
        
        print("All models loaded successfully!")
        
    def normalize_text(self, text: str) -> str:
        """Optimized text normalization for WER/CER calculation"""
        if not text:
            return ""
            
        if self.language == 'zh':
            # Remove all non-Chinese characters and punctuation
            text = re.sub(r'[^\u4e00-\u9fff\s]', '', text)
            text = re.sub(r'\s+', '', text)
            return text.strip()
        else:
            # Convert to uppercase first
            text = text.upper()
            # Remove all punctuation including commas, periods, etc.
            text = re.sub(r'[^\w\s]', '', text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
    
    def fast_cer(self, reference: str, hypothesis: str) -> float:
        """Fast Character Error Rate calculation using Levenshtein distance"""
        if not reference:
            return 1.0 if hypothesis else 0.0
        if not hypothesis:
            return 1.0
        
        ref_chars = list(reference)
        hyp_chars = list(hypothesis)
        
        distance = Levenshtein.distance(''.join(ref_chars), ''.join(hyp_chars))
        return distance / len(ref_chars)
    
    def fast_wer(self, reference: str, hypothesis: str) -> float:
        """Fast Word Error Rate calculation using Levenshtein distance"""
        if not reference:
            return 1.0 if hypothesis else 0.0
        if not hypothesis:
            return 1.0
        
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        if not ref_words:
            return 1.0 if hyp_words else 0.0
        
        distance = Levenshtein.distance(' '.join(ref_words), ' '.join(hyp_words))
        return distance / len(ref_words)
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Optimized audio transcription using ASR model"""
        try:
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
                
            generate_kwargs = {
                "task": "transcribe",
                "language": "zh" if self.language == 'zh' else "en"
            }
            
            if self.device.startswith('cuda'):
                generate_kwargs["do_sample"] = False
                generate_kwargs["num_beams"] = 1
            
            result = self.asr_pipeline(
                str(audio_path), 
                return_timestamps=False,
                generate_kwargs=generate_kwargs
            )
            return result.get("text", "")
        except Exception as e:
            print(f"Error transcribing audio {audio_path}: {e}")
            return ""
    
    def calculate_dwer_dcer(self, original_audio_path: str, inference_audio_path: str, ground_truth: str) -> dict:
        """Optimized dWER or dCER calculation"""
        try:
            original_transcript = self.transcribe_audio(original_audio_path)
            inference_transcript = self.transcribe_audio(inference_audio_path)
            
            ground_truth_norm = self.normalize_text(ground_truth)
            original_norm = self.normalize_text(original_transcript)
            inference_norm = self.normalize_text(inference_transcript)
            
            if self.language == 'zh':
                original_cer = self.fast_cer(ground_truth_norm, original_norm)
                inference_cer = self.fast_cer(ground_truth_norm, inference_norm)
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
                original_wer = self.fast_wer(ground_truth_norm, original_norm)
                inference_wer = self.fast_wer(ground_truth_norm, inference_norm)
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
        """Calculate UTMOS score with error handling"""
        try:
            if self.utmos_model is None:
                return None
                
            wave, sr = librosa.load(str(audio_path), sr=None, mono=True)
            
            if len(wave) == 0:
                return None
            
            with torch.no_grad():
                wave_tensor = torch.from_numpy(wave).unsqueeze(0)
                score = self.utmos_model(wave_tensor, sr)
                return float(score.item())
        except Exception as e:
            print(f"Error calculating UTMOS: {e}")
            return None
    
    def calculate_pesq(self, reference_path: str, degraded_path: str) -> float:
        """Optimized PESQ score calculation"""
        try:
            ref_audio, _ = librosa.load(reference_path, sr=16000, mono=True)
            deg_audio, _ = librosa.load(degraded_path, sr=16000, mono=True)
            
            min_len = min(len(ref_audio), len(deg_audio))
            if min_len < 16000 * 0.1:
                return None
            
            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]
            
            pesq_score = pesq(16000, ref_audio, deg_audio, 'wb')
            return float(pesq_score)
            
        except Exception as e:
            print(f"Error calculating PESQ: {e}")
            return None
    
    def calculate_stoi(self, reference_path: str, degraded_path: str) -> float:
        """Optimized STOI score calculation"""
        try:
            ref_audio, _ = librosa.load(reference_path, sr=16000, mono=True)
            deg_audio, _ = librosa.load(degraded_path, sr=16000, mono=True)
            
            min_len = min(len(ref_audio), len(deg_audio))
            if min_len < 16000 * 0.1:
                return None
            
            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]
            
            stoi_score = stoi(ref_audio, deg_audio, 16000, extended=False)
            return float(stoi_score)
            
        except Exception as e:
            print(f"Error calculating STOI: {e}")
            return None
    
    def evaluate_audio_pair(self, original_path: str, inference_path: str, ground_truth: str) -> dict:
        """Evaluate audio pair with all metrics"""
        results = {}
        
        asr_result = self.calculate_dwer_dcer(original_path, inference_path, ground_truth)
        if asr_result:
            results.update(asr_result)
        
        results['utmos'] = self.calculate_utmos(inference_path)
        results['pesq'] = self.calculate_pesq(original_path, inference_path)
        results['stoi'] = self.calculate_stoi(original_path, inference_path)
        
        return results