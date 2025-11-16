#!/usr/bin/env python3
"""
Audio Codec Metrics Evaluator - Optimized Version

Comprehensive audio quality and ASR metrics evaluation with GPU acceleration,
batch processing, and optimized I/O operations.

Metrics: dWER/dCER, UTMOS, PESQ, STOI, Speaker Similarity
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import re
import string
from transformers import pipeline
import jiwer
from pesq import pesq
from pystoi import stoi
import warnings
import Levenshtein
import opencc
import pandas as pd
from scipy.spatial.distance import cosine
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool
import multiprocessing as mp

try:
    import cn2an
    CN2AN_AVAILABLE = True
except ImportError:
    CN2AN_AVAILABLE = False
    print("Warning: cn2an not installed. Arabic to Chinese number conversion disabled.")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class AudioMetricsEvaluator:
    """Optimized audio quality and ASR accuracy metrics evaluator with GPU acceleration"""
    
    def __init__(self, language='en', device=None, use_gpu=True, gpu_id=0):
        self.language = language
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.device = self._setup_device(device)
        self.asr_pipeline = None
        self.utmos_model = None
        self.speaker_model = None
        
        # Audio resampler cache
        self.resamplers = {}
        
        # Initialize Traditional to Simplified Chinese converter
        if self.language == 'zh':
            try:
                self.t2s_converter = opencc.OpenCC('t2s')
                print("OpenCC Traditional to Simplified converter initialized")
            except Exception as e:
                print(f"Warning: Could not initialize OpenCC: {e}")
                self.t2s_converter = None
        else:
            self.t2s_converter = None
        
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
    
    def load_audio_optimized(self, path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load and resample audio using torchaudio with GPU acceleration.
        
        Args:
            path: Audio file path
            sr: Target sample rate
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            waveform, sample_rate = torchaudio.load(path)
            
            # Resample if needed
            if sample_rate != sr:
                # Cache resampler for efficiency
                resampler_key = f"{sample_rate}_{sr}"
                if resampler_key not in self.resamplers:
                    self.resamplers[resampler_key] = T.Resample(sample_rate, sr)
                
                resampler = self.resamplers[resampler_key]
                
                # Use GPU for resampling if available
                if self.device.startswith('cuda'):
                    resampler = resampler.to(self.device)
                    waveform = waveform.to(self.device)
                
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Convert to numpy for compatibility
            if self.device.startswith('cuda'):
                waveform = waveform.cpu()
            
            return waveform.squeeze().numpy(), sr
            
        except Exception as e:
            print(f"Error loading audio with torchaudio: {e}")
            return None, None
        
    def load_models(self):
        """Load ASR, UTMOS, and Speaker models with GPU optimization"""
        print(f"Loading models on device: {self.device}")
        
        if self.device.startswith('cuda'):
            mem_info = self.get_gpu_memory_info()
            print(f"Initial GPU memory: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")
        
        # Load Whisper ASR model
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
            print(f"ASR model loaded on {self.device}")
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
        
        # Load UTMOS model with GPU support
        print("Loading UTMOS model...")
        try:
            self.utmos_model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", 
                "utmos22_strong", 
                trust_repo=True
            )
            
            # Use GPU for UTMOS if available
            if self.device.startswith('cuda'):
                self.utmos_model = self.utmos_model.to(self.device)
                print(f"UTMOS model loaded on {self.device}")
            else:
                self.utmos_model = self.utmos_model.cpu()
                print("UTMOS model loaded on CPU")
            
        except Exception as e:
            print(f"Error loading UTMOS model: {e}")
            self.utmos_model = None
        
        if self.device.startswith('cuda'):
            mem_info = self.get_gpu_memory_info()
            print(f"After UTMOS loading: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")
        
        # Load Speaker Embedding model
        print("Loading Speaker Embedding model (ECAPA-TDNN)...")
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            
            self.speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            print(f"Speaker embedding model loaded on {self.device}")
            
        except Exception as e:
            print(f"Error loading speaker embedding model: {e}")
            print("Speaker similarity metric will not be available")
            self.speaker_model = None
            
        if self.device.startswith('cuda'):
            mem_info = self.get_gpu_memory_info()
            print(f"After all models loaded: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")
        
        print("All models loaded successfully!")

    def convert_traditional_to_simplified(self, text: str) -> str:
        """Convert Traditional Chinese to Simplified Chinese"""
        if not text or pd.isna(text):
            return ""
        
        if self.language == 'zh' and self.t2s_converter:
            try:
                return self.t2s_converter.convert(text)
            except Exception as e:
                print(f"Warning: OpenCC conversion failed: {e}")
                return text
        return text

    def convert_arabic_to_chinese_numbers(self, text: str) -> str:
        """Convert Arabic numbers to Chinese numbers"""
        if not CN2AN_AVAILABLE or not text:
            return text
        
        try:
            import re
            pattern = r'\d+'
            
            def replace_number(match):
                num_str = match.group()
                try:
                    return cn2an.an2cn(num_str, "low")
                except:
                    return num_str
            
            return re.sub(pattern, replace_number, text)
        except Exception as e:
            print(f"Warning: Arabic to Chinese number conversion failed: {e}")
            return text

    def normalize_text(self, text: str) -> str:
        """Normalize text for metric calculation"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).lower().strip()
        
        if self.language == 'zh':
            text = re.sub(r'[，。！？；：""''（）《》【】、]', '', text)
            text = re.sub(r'\s+', '', text)
        else:
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def fast_wer(self, reference: str, hypothesis: str) -> float:
        """Fast WER calculation using jiwer"""
        try:
            if not reference or not hypothesis:
                return 1.0
            return jiwer.wer(reference, hypothesis)
        except Exception as e:
            print(f"Error calculating WER: {e}")
            return 1.0

    def fast_cer(self, reference: str, hypothesis: str) -> float:
        """Fast CER calculation using Levenshtein distance"""
        try:
            if not reference or not hypothesis:
                return 1.0
            
            distance = Levenshtein.distance(reference, hypothesis)
            return distance / max(len(reference), 1)
        except Exception as e:
            print(f"Error calculating CER: {e}")
            return 1.0

    def batch_transcribe(self, audio_paths: List[str]) -> List[str]:
        """
        Batch transcribe audio files for improved efficiency.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of transcribed texts
        """
        if not self.asr_pipeline:
            print("ASR pipeline not loaded")
            return [""] * len(audio_paths)
        
        results = []
        for path in audio_paths:
            audio, sr = self.load_audio_optimized(path, sr=16000)
            if audio is None:
                results.append("")
                continue
            
            try:
                result = self.asr_pipeline(
                    audio,
                    generate_kwargs={"language": "zh" if self.language == 'zh' else "en"}
                )
                results.append(result['text'])
            except Exception as e:
                print(f"Error transcribing {path}: {e}")
                results.append("")
        
        return results

    def calculate_dwer_dcer(self, original_path: str, inference_path: str, ground_truth: str) -> Optional[Dict]:
        """Calculate dWER or dCER based on language with optimized audio loading"""
        try:
            if not self.asr_pipeline:
                print("ASR pipeline not loaded")
                return None
            
            # Load audio using optimized method
            original_audio, _ = self.load_audio_optimized(original_path, sr=16000)
            inference_audio, _ = self.load_audio_optimized(inference_path, sr=16000)
            
            if original_audio is None or inference_audio is None:
                return None
            
            # Transcribe
            original_result = self.asr_pipeline(
                original_audio,
                generate_kwargs={"language": "zh" if self.language == 'zh' else "en"}
            )
            inference_result = self.asr_pipeline(
                inference_audio,
                generate_kwargs={"language": "zh" if self.language == 'zh' else "en"}
            )
            
            original_transcript = original_result['text']
            inference_transcript = inference_result['text']
            
            # Convert Traditional to Simplified for Chinese
            if self.language == 'zh':
                original_transcript_simplified = self.convert_traditional_to_simplified(original_transcript)
                inference_transcript_simplified = self.convert_traditional_to_simplified(inference_transcript)
                ground_truth_simplified = self.convert_traditional_to_simplified(ground_truth)
            else:
                original_transcript_simplified = original_transcript
                inference_transcript_simplified = inference_transcript
                ground_truth_simplified = ground_truth
            
            # Normalize
            ground_truth_norm = self.normalize_text(ground_truth_simplified)
            original_norm = self.normalize_text(original_transcript_simplified)
            inference_norm = self.normalize_text(inference_transcript_simplified)
            
            if self.language == 'zh':
                original_cer = self.fast_cer(ground_truth_norm, original_norm)
                inference_cer = self.fast_cer(ground_truth_norm, inference_norm)
                dcer = inference_cer - original_cer
                
                return {
                    'original_transcript_raw': original_transcript,
                    'inference_transcript_raw': inference_transcript,
                    'original_transcript': original_norm,
                    'inference_transcript': inference_norm,
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
                    'original_transcript_raw': original_transcript,
                    'inference_transcript_raw': inference_transcript,
                    'original_transcript': original_norm,
                    'inference_transcript': inference_norm,
                    'original_wer': original_wer,
                    'inference_wer': inference_wer,
                    'dwer': dwer,
                    'metric_name': 'dWER'
                }
                
        except Exception as e:
            print(f"Error calculating dWER/dCER: {e}")
            return None

    def calculate_utmos(self, audio_path: str) -> Optional[float]:
        """Calculate UTMOS score with GPU acceleration"""
        try:
            if self.utmos_model is None:
                return None
            
            # Load audio using optimized method
            wave, sr = self.load_audio_optimized(audio_path, sr=16000)
            
            if wave is None or len(wave) == 0:
                return None
            
            with torch.no_grad():
                wave_tensor = torch.from_numpy(wave).unsqueeze(0)
                
                # Move tensor to GPU if available
                if self.device.startswith('cuda'):
                    wave_tensor = wave_tensor.to(self.device)
                
                score = self.utmos_model(wave_tensor, sr)
                return float(score.item())
                
        except Exception as e:
            print(f"Error calculating UTMOS: {e}")
            return None
    
    def calculate_pesq(self, reference_path: str, degraded_path: str) -> Optional[float]:
        """Calculate PESQ score with optimized audio loading"""
        try:
            # Load audio using optimized method
            ref_audio, _ = self.load_audio_optimized(reference_path, sr=16000)
            deg_audio, _ = self.load_audio_optimized(degraded_path, sr=16000)
            
            if ref_audio is None or deg_audio is None:
                return None
            
            # Ensure same length
            min_len = min(len(ref_audio), len(deg_audio))
            if min_len < 16000 * 0.1:  # At least 0.1 seconds
                return None
            
            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]
            
            pesq_score = pesq(16000, ref_audio, deg_audio, 'wb')
            return float(pesq_score)
            
        except Exception as e:
            print(f"Error calculating PESQ: {e}")
            return None
    
    def calculate_stoi(self, reference_path: str, degraded_path: str) -> Optional[float]:
        """Calculate STOI score with optimized audio loading"""
        try:
            # Load audio using optimized method
            ref_audio, _ = self.load_audio_optimized(reference_path, sr=16000)
            deg_audio, _ = self.load_audio_optimized(degraded_path, sr=16000)
            
            if ref_audio is None or deg_audio is None:
                return None
            
            # Ensure same length
            min_len = min(len(ref_audio), len(deg_audio))
            if min_len < 16000 * 0.1:  # At least 0.1 seconds
                return None
            
            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]
            
            stoi_score = stoi(ref_audio, deg_audio, 16000, extended=False)
            return float(stoi_score)
            
        except Exception as e:
            print(f"Error calculating STOI: {e}")
            return None
    
    @staticmethod
    def _calculate_pesq_worker(args):
        """Worker function for parallel PESQ calculation"""
        reference_path, degraded_path = args
        try:
            import torchaudio
            import numpy as np
            from pesq import pesq
            
            ref_audio, _ = torchaudio.load(reference_path)
            deg_audio, _ = torchaudio.load(degraded_path)
            
            # Resample to 16kHz
            if ref_audio.shape[0] > 1:
                ref_audio = torch.mean(ref_audio, dim=0, keepdim=True)
            if deg_audio.shape[0] > 1:
                deg_audio = torch.mean(deg_audio, dim=0, keepdim=True)
            
            ref_audio = ref_audio.squeeze().numpy()
            deg_audio = deg_audio.squeeze().numpy()
            
            min_len = min(len(ref_audio), len(deg_audio))
            if min_len < 16000 * 0.1:
                return None
            
            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]
            
            return float(pesq(16000, ref_audio, deg_audio, 'wb'))
        except Exception as e:
            return None
    
    @staticmethod
    def _calculate_stoi_worker(args):
        """Worker function for parallel STOI calculation"""
        reference_path, degraded_path = args
        try:
            import torchaudio
            import numpy as np
            from pystoi import stoi
            
            ref_audio, _ = torchaudio.load(reference_path)
            deg_audio, _ = torchaudio.load(degraded_path)
            
            # Convert to mono
            if ref_audio.shape[0] > 1:
                ref_audio = torch.mean(ref_audio, dim=0, keepdim=True)
            if deg_audio.shape[0] > 1:
                deg_audio = torch.mean(deg_audio, dim=0, keepdim=True)
            
            ref_audio = ref_audio.squeeze().numpy()
            deg_audio = deg_audio.squeeze().numpy()
            
            min_len = min(len(ref_audio), len(deg_audio))
            if min_len < 16000 * 0.1:
                return None
            
            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]
            
            return float(stoi(ref_audio, deg_audio, 16000, extended=False))
        except Exception as e:
            return None
    
    def calculate_pesq_stoi_batch(self, file_pairs: List[Tuple[str, str]], num_workers: Optional[int] = None) -> Tuple[List, List]:
        """
        Batch calculation of PESQ and STOI using multiprocessing for CPU parallelization.
        
        Args:
            file_pairs: List of (reference_path, degraded_path) tuples
            num_workers: Number of worker processes (default: CPU count - 1, max 8)
            
        Returns:
            Tuple of (pesq_scores, stoi_scores) lists
        """
        if num_workers is None:
            num_workers = min(mp.cpu_count() - 1, 8)
        
        print(f"Calculating PESQ and STOI with {num_workers} workers...")
        
        with Pool(num_workers) as pool:
            pesq_results = pool.map(self._calculate_pesq_worker, file_pairs)
            stoi_results = pool.map(self._calculate_stoi_worker, file_pairs)
        
        return pesq_results, stoi_results

    def calculate_speaker_similarity(self, reference_path: str, test_path: str) -> Optional[float]:
        """
        Calculate speaker similarity using ECAPA-TDNN embeddings.
        
        Returns cosine similarity score (higher = better identity preservation)
        """
        try:
            if self.speaker_model is None:
                return None
            
            # Load audio files using torchaudio
            ref_signal, ref_sr = torchaudio.load(str(reference_path))
            test_signal, test_sr = torchaudio.load(str(test_path))
            
            # Check minimum duration (at least 0.5 seconds)
            min_samples_ref = int(0.5 * ref_sr)
            min_samples_test = int(0.5 * test_sr)
            
            if ref_signal.shape[1] < min_samples_ref or test_signal.shape[1] < min_samples_test:
                return None
            
            # Move to GPU if available
            if self.device.startswith('cuda'):
                ref_signal = ref_signal.to(self.device)
                test_signal = test_signal.to(self.device)
            
            # Extract speaker embeddings
            with torch.no_grad():
                ref_embedding = self.speaker_model.encode_batch(ref_signal)
                test_embedding = self.speaker_model.encode_batch(test_signal)
                
                # Calculate cosine similarity
                ref_emb_np = ref_embedding.squeeze().cpu().numpy()
                test_emb_np = test_embedding.squeeze().cpu().numpy()
                
                similarity = 1.0 - cosine(ref_emb_np, test_emb_np)
                
                return float(similarity)
                
        except Exception as e:
            print(f"Error calculating speaker similarity: {e}")
            return None
    
    def evaluate_audio_pair(self, original_path: str, inference_path: str, ground_truth: str) -> Dict:
        """Evaluate audio pair with all metrics"""
        results = {}
        
        # ASR metrics
        asr_result = self.calculate_dwer_dcer(original_path, inference_path, ground_truth)
        if asr_result:
            results.update(asr_result)
        
        # Quality metrics
        results['utmos'] = self.calculate_utmos(inference_path)
        results['pesq'] = self.calculate_pesq(original_path, inference_path)
        results['stoi'] = self.calculate_stoi(original_path, inference_path)
        results['speaker_similarity'] = self.calculate_speaker_similarity(original_path, inference_path)
        
        return results
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory cache"""
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
            if self.device.startswith('cuda'):
                mem_info = self.get_gpu_memory_info()
                print(f"GPU memory after cleanup: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")