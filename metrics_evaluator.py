#!/usr/bin/env python3
"""
Audio Codec Metrics Evaluator
=============================

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
    
    def __init__(self, language='en', device=None):
        """
        Initialize the metrics evaluator
        
        Args:
            language (str): Language code ('en' for English, 'zh' for Chinese)
            device (str): Device for model inference ('cuda:0', 'cpu', etc.)
        """
        self.language = language
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.asr_pipeline = None
        self.utmos_model = None
        
    def load_models(self):
        """Load ASR and UTMOS models"""
        print(f"Loading models on device: {self.device}")
        
        print("Loading Whisper-large-v3 ASR model...")
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition", 
            model="openai/whisper-large-v3", 
            device=self.device
        )
        
        print("Loading UTMOS model...")
        self.utmos_model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", 
            "utmos22_strong", 
            trust_repo=True
        )
        
        print("All models loaded successfully!")
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for WER/CER calculation
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text
        """
        if self.language == 'zh':
            # Chinese normalization: remove spaces and punctuation
            text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)
            text = re.sub(r'\s+', '', text)
            return text.strip()
        else:
            # English normalization: uppercase, remove punctuation, normalize spaces
            text = text.upper()
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = ' '.join(text.split())
            return text
            
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate (CER)
        
        Args:
            reference (str): Reference text
            hypothesis (str): Hypothesis text
            
        Returns:
            float: CER score
        """
        if not reference or not hypothesis:
            return 1.0
            
        ref_chars = list(reference)
        hyp_chars = list(hypothesis)
        
        return jiwer.wer(ref_chars, hyp_chars)
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio using ASR model
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            str: Transcription text
        """
        try:
            if self.language == 'zh':
                result = self.asr_pipeline(
                    str(audio_path), 
                    return_timestamps=True,
                    generate_kwargs={"task": "transcribe", "language": "zh"}
                )
            else:
                result = self.asr_pipeline(
                    str(audio_path), 
                    return_timestamps=True,
                    generate_kwargs={"task": "transcribe", "language": "en"}
                )
            return result.get("text", "")
        except Exception as e:
            print(f"Error transcribing audio {audio_path}: {e}")
            return ""
    
    def calculate_dwer_dcer(self, original_audio_path: str, inference_audio_path: str, ground_truth: str) -> dict:
        """
        Calculate dWER or dCER depending on language
        
        Args:
            original_audio_path (str): Path to original audio
            inference_audio_path (str): Path to inference audio
            ground_truth (str): Ground truth transcription
            
        Returns:
            dict: Evaluation results containing transcriptions and error rates
        """
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
        """
        Calculate UTMOS score
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            float: UTMOS score or None if error
        """
        try:
            wave, sr = librosa.load(str(audio_path), sr=None, mono=True)
            with torch.no_grad():
                score = self.utmos_model(torch.from_numpy(wave).unsqueeze(0), sr)
                return score.item()
        except Exception as e:
            print(f"Error calculating UTMOS: {e}")
            return None
    
    def calculate_pesq(self, reference_path: str, degraded_path: str) -> float:
        """
        Calculate PESQ score
        
        Args:
            reference_path (str): Path to reference audio
            degraded_path (str): Path to degraded audio
            
        Returns:
            float: PESQ score or None if error
        """
        try:
            ref_audio, _ = librosa.load(reference_path, sr=16000, mono=True)
            deg_audio, _ = librosa.load(degraded_path, sr=16000, mono=True)
            
            min_len = min(len(ref_audio), len(deg_audio))
            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]
            
            if min_len < 16000 * 0.1:  # At least 0.1 seconds
                return None
                
            pesq_score = pesq(16000, ref_audio, deg_audio, 'wb')
            return pesq_score
            
        except Exception as e:
            print(f"Error calculating PESQ: {e}")
            return None
    
    def calculate_stoi(self, reference_path: str, degraded_path: str) -> float:
        """
        Calculate STOI score
        
        Args:
            reference_path (str): Path to reference audio
            degraded_path (str): Path to degraded audio
            
        Returns:
            float: STOI score or None if error
        """
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
        """
        Evaluate a pair of audio files with all metrics
        
        Args:
            original_path (str): Path to original audio
            inference_path (str): Path to inference audio
            ground_truth (str): Ground truth transcription
            
        Returns:
            dict: Complete evaluation results
        """
        results = {}
        
        # ASR evaluation
        asr_result = self.calculate_dwer_dcer(original_path, inference_path, ground_truth)
        if asr_result:
            results.update(asr_result)
        
        # Quality metrics
        results['utmos'] = self.calculate_utmos(inference_path)
        results['pesq'] = self.calculate_pesq(original_path, inference_path)
        results['stoi'] = self.calculate_stoi(original_path, inference_path)
        
        return results