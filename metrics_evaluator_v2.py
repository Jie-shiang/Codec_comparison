#!/usr/bin/env python3
"""
Audio Codec Metrics Evaluator V2 - Language-Specific Models

New metrics configuration:
English:
  - ASR (WER): Whisper-large-v3
  - Speaker Similarity: ResNet3
  - MOS_Quality: NISQA v2
  - MOS_Naturalness: UTMOS

Chinese:
  - ASR (CER): Paraformer-large
  - Speaker Similarity: CAM++
  - MOS_Quality: NISQA v2
  - MOS_Naturalness: RAMP

Common metrics (unchanged):
  - PESQ
  - STOI
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
import os

try:
    import cn2an
    CN2AN_AVAILABLE = True
except ImportError:
    CN2AN_AVAILABLE = False
    print("Warning: cn2an not installed. Arabic to Chinese number conversion disabled.")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class AudioMetricsEvaluatorV2:
    """V2 Evaluator with language-specific models for improved accuracy"""

    def __init__(self, language='en', device=None, use_gpu=True, gpu_id=0):
        self.language = language
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.device = self._setup_device(device)

        # Model placeholders
        self.asr_pipeline = None
        self.speaker_model = None
        self.mos_quality_model = None  # NISQA v2
        self.mos_naturalness_model = None  # UTMOS (EN) or RAMP (ZH)

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
        """Load all models based on language configuration"""
        print(f"Loading V2 models for language: {self.language}")
        print(f"Device: {self.device}")

        if self.device.startswith('cuda'):
            mem_info = self.get_gpu_memory_info()
            print(f"Initial GPU memory: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")

        # Load ASR model (language-specific)
        self._load_asr_model()

        # Load Speaker Similarity model (language-specific)
        self._load_speaker_model()

        # Load MOS Quality model (NISQA v2 - same for both languages)
        self._load_mos_quality_model()

        # Load MOS Naturalness model (language-specific)
        self._load_mos_naturalness_model()

        if self.device.startswith('cuda'):
            mem_info = self.get_gpu_memory_info()
            print(f"After all models loaded: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")

        print("All V2 models loaded successfully!")

    def _load_asr_model(self):
        """Load ASR model based on language"""
        print(f"\nLoading ASR model for {self.language}...")

        try:
            if self.language == 'en':
                # English: Whisper-large-v3
                print("Loading Whisper-large-v3 for English ASR...")
                torch_dtype = torch.float16 if self.device.startswith('cuda') else torch.float32

                self.asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-large-v3",
                    torch_dtype=torch_dtype,
                    device=self.device,
                    model_kwargs={
                        "attn_implementation": "sdpa"
                    },
                    generate_kwargs={
                        "repetition_penalty": 1.2,
                        "no_repeat_ngram_size": 3,
                        "max_new_tokens": 256
                    }
                )
                print(f"Whisper-large-v3 loaded on {self.device} with repetition suppression")

            elif self.language == 'zh':
                # Chinese: Paraformer-large via FunASR
                print("Loading Paraformer-large for Chinese ASR (FunASR)...")
                from funasr import AutoModel

                # Use FunASR AutoModel for Paraformer with Hugging Face hub
                # Using hub="hf" to download from Hugging Face instead of ModelScope
                # This avoids the "am.mvn download incomplete" error
                self.asr_pipeline = AutoModel(
                    model="paraformer-zh",
                    hub="hf",
                    device=self.device,
                    disable_update=True
                )

                print(f"Paraformer-large loaded successfully on {self.device}")

            if self.device.startswith('cuda'):
                mem_info = self.get_gpu_memory_info()
                print(f"After ASR loading: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")

        except Exception as e:
            print(f"Error loading ASR model: {e}")
            print("Falling back to Whisper on CPU...")
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                device="cpu"
            )

    def _load_speaker_model(self):
        """Load Speaker Similarity model based on language"""
        print(f"\nLoading Speaker Similarity model for {self.language}...")

        try:
            if self.language == 'en':
                # English: ResNet3 (WeSpeaker)
                print("Loading ResNet3 (WeSpeaker) for English Speaker Similarity...")
                import wespeaker

                model_path = "/mnt/Internal/jieshiang/Model/wespeaker_resnet34"

                if not os.path.exists(model_path):
                    print(f"Downloading ResNet3 to {model_path}...")
                    os.makedirs(model_path, exist_ok=True)

                # Load WeSpeaker ResNet model
                self.speaker_model = wespeaker.load_model('english')

                # Move model to GPU
                if self.device.startswith('cuda'):
                    self.speaker_model.set_device(self.gpu_id)
                    print(f"ResNet3 loaded successfully on GPU {self.gpu_id}")
                else:
                    print(f"ResNet3 loaded successfully on CPU")

            elif self.language == 'zh':
                # Chinese: CAM++ via FunASR (more reliable than ModelScope pipeline)
                print("Loading CAM++ for Chinese Speaker Similarity...")
                from funasr import AutoModel

                try:
                    # Use FunASR AutoModel for CAM++ with Hugging Face hub
                    # Model: funasr/campplus on Hugging Face
                    self.speaker_model = AutoModel(
                        model="funasr/campplus",
                        hub="hf",
                        device=self.device,
                        disable_update=True
                    )
                    print(f"CAM++ loaded successfully on {self.device}")
                except Exception as e:
                    print(f"Warning: Could not load CAM++ via FunASR: {e}")
                    print("Trying ModelScope pipeline as fallback...")

                    try:
                        from modelscope.pipelines import pipeline as ms_pipeline
                        model_dir = "/mnt/Internal/jieshiang/Model/speech_campplus_sv_zh-cn_16k-common"

                        if not os.path.exists(model_dir):
                            print(f"Downloading CAM++ to {model_dir}...")
                            # Use string task name instead of Tasks enum to avoid version issues
                            self.speaker_model = ms_pipeline(
                                task='speaker-verification',
                                model='iic/speech_campplus_sv_zh-cn_16k-common'
                            )
                            os.makedirs(model_dir, exist_ok=True)
                        else:
                            print(f"Loading CAM++ from {model_dir}...")
                            self.speaker_model = ms_pipeline(
                                task='speaker-verification',
                                model=model_dir
                            )
                        print("CAM++ loaded successfully via ModelScope")
                    except Exception as e2:
                        print(f"CAM++ loading failed completely: {e2}")
                        self.speaker_model = None
                        raise

            if self.device.startswith('cuda'):
                mem_info = self.get_gpu_memory_info()
                print(f"After Speaker model loading: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")

        except Exception as e:
            print(f"Error loading speaker model: {e}")
            print("Speaker similarity metric will not be available")
            self.speaker_model = None

    def _load_mos_quality_model(self):
        """Load NISQA v2 for MOS Quality (same for both languages)"""
        print("\nLoading NISQA v2 for MOS Quality...")

        try:
            from nisqa.NISQA_model import nisqaModel
            import os

            # NISQA v2 initialization
            # Point to the downloaded model weights
            model_path = '/mnt/Internal/jieshiang/Model/nisqa_weights/nisqa.tar'

            # Create a dummy audio file for initialization (NISQA requires a file during __init__)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                dummy_path = f.name
                # Create 1 second of silence
                sample_rate = 16000
                waveform = torch.zeros(1, sample_rate)
                torchaudio.save(dummy_path, waveform, sample_rate)

            # Load to CPU first to avoid OOM on wrong GPU device
            args = {
                'mode': 'predict_file',
                'pretrained_model': model_path,  # Path to nisqa.tar
                'deg': dummy_path,  # Dummy file for initialization
                'num_workers': 4,  # Use multiple workers for data loading
                'bs': 1,  # Batch size for single file mode
                'tr_bs_val': 32,  # Batch size for validation/prediction (used in batch mode)
                'tr_num_workers': 4,  # Workers for batch processing
                'ms_channel': None,
                'output_dir': None,
                'tr_device': 'cpu',  # Load to CPU first
            }

            self.mos_quality_model = nisqaModel(args)

            # Clean up dummy file
            import os as _os
            if _os.path.exists(dummy_path):
                _os.remove(dummy_path)

            # Move model to the correct GPU device
            if self.device.startswith('cuda'):
                self.mos_quality_model.model = self.mos_quality_model.model.to(self.device)
                self.mos_quality_model.dev = torch.device(self.device)
                print(f"NISQA v2 loaded and moved to {self.device}")
                mem_info = self.get_gpu_memory_info()
                print(f"After NISQA loading: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")
            else:
                print(f"NISQA v2 loaded on CPU")

        except Exception as e:
            print(f"Error loading NISQA v2: {e}")
            import traceback
            traceback.print_exc()
            print("MOS Quality metric will not be available")
            self.mos_quality_model = None

    def _load_mos_naturalness_model(self):
        """Load MOS Naturalness model based on language"""
        print(f"\nLoading MOS Naturalness model for {self.language}...")

        try:
            # UTMOS supports multilingual including Chinese (VoiceMOS Challenge 2022)
            # Reference: UTMOS achieved highest scores on Chinese speech in OOD track
            # See: https://arxiv.org/pdf/2406.04904 (XTTS multilingual TTS evaluation)
            print(f"Loading UTMOS for MOS Naturalness ({self.language})...")
            print("UTMOS supports multilingual evaluation including Chinese speech")

            self.mos_naturalness_model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0",
                "utmos22_strong",
                trust_repo=True
            )

            if self.device.startswith('cuda'):
                self.mos_naturalness_model = self.mos_naturalness_model.to(self.device)
                print(f"UTMOS loaded on {self.device}")
            else:
                self.mos_naturalness_model = self.mos_naturalness_model.cpu()
                print("UTMOS loaded on CPU")

            if self.device.startswith('cuda'):
                mem_info = self.get_gpu_memory_info()
                print(f"After MOS Naturalness loading: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")

        except Exception as e:
            print(f"Error loading MOS Naturalness model: {e}")
            self.mos_naturalness_model = None

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
        """Fast CER calculation using jiwer"""
        try:
            if not reference or not hypothesis:
                return 1.0
            return jiwer.cer(reference, hypothesis)
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

        if self.language == 'en':
            # Whisper pipeline
            for path in audio_paths:
                audio, sr = self.load_audio_optimized(path, sr=16000)
                if audio is None:
                    results.append("")
                    continue

                try:
                    # Enable return_timestamps for long audio (>30s)
                    result = self.asr_pipeline(
                        audio,
                        generate_kwargs={"language": "en"},
                        return_timestamps=True
                    )
                    # Extract text from result
                    if isinstance(result, dict) and 'text' in result:
                        results.append(result['text'])
                    elif isinstance(result, dict) and 'chunks' in result:
                        # Combine chunks for long-form transcription
                        text = ' '.join([chunk['text'] for chunk in result['chunks']])
                        results.append(text)
                    else:
                        results.append("")
                except Exception as e:
                    print(f"Error transcribing {path}: {e}")
                    results.append("")

        elif self.language == 'zh':
            # Paraformer (FunASR) - Process sequentially for stability
            # Batch processing with generate() can cause errors with some audio files
            for path in audio_paths:
                try:
                    result = self.asr_pipeline.generate(input=path)
                    # FunASR returns list of dicts: [{'key': ..., 'text': ...}]
                    if isinstance(result, list) and len(result) > 0:
                        results.append(result[0].get('text', ''))
                    elif isinstance(result, dict) and 'text' in result:
                        results.append(result['text'])
                    else:
                        results.append("")
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

            if self.language == 'en':
                # Whisper: Load audio
                original_audio, _ = self.load_audio_optimized(original_path, sr=16000)
                inference_audio, _ = self.load_audio_optimized(inference_path, sr=16000)

                if original_audio is None or inference_audio is None:
                    return None

                # Transcribe
                original_result = self.asr_pipeline(
                    original_audio,
                    generate_kwargs={"language": "en"}
                )
                inference_result = self.asr_pipeline(
                    inference_audio,
                    generate_kwargs={"language": "en"}
                )

                original_transcript = original_result['text']
                inference_transcript = inference_result['text']

            elif self.language == 'zh':
                # Paraformer (FunASR): Use file paths directly
                original_result = self.asr_pipeline.generate(input=original_path)
                inference_result = self.asr_pipeline.generate(input=inference_path)

                # FunASR returns list of dicts with 'text' key
                if isinstance(original_result, list) and len(original_result) > 0:
                    original_transcript = original_result[0].get('text', '')
                elif isinstance(original_result, dict) and 'text' in original_result:
                    original_transcript = original_result['text']
                else:
                    original_transcript = ''

                if isinstance(inference_result, list) and len(inference_result) > 0:
                    inference_transcript = inference_result[0].get('text', '')
                elif isinstance(inference_result, dict) and 'text' in inference_result:
                    inference_transcript = inference_result['text']
                else:
                    inference_transcript = ''

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
            import traceback
            traceback.print_exc()
            return None

    def calculate_mos_quality(self, audio_path: str) -> Optional[float]:
        """Calculate MOS Quality using NISQA v2 (single file - slow, use batch version if possible)"""
        try:
            if self.mos_quality_model is None:
                return None

            # NISQA v2 predict_file mode expects the file path in args['deg']
            # Update the args and reload dataset
            self.mos_quality_model.args['deg'] = audio_path
            self.mos_quality_model._loadDatasets()  # Reload dataset with new file

            # Ensure model is on correct GPU device before prediction
            if self.device.startswith('cuda'):
                self.mos_quality_model.model = self.mos_quality_model.model.to(self.device)
                self.mos_quality_model.dev = torch.device(self.device)

            # Run prediction (will use GPU if self.dev is cuda)
            results_df = self.mos_quality_model.predict()

            # Extract MOS score
            if isinstance(results_df, pd.DataFrame) and 'mos_pred' in results_df.columns:
                score = results_df['mos_pred'].iloc[0]
                return float(score)
            else:
                return None

        except Exception as e:
            print(f"Error calculating MOS Quality (NISQA): {e}")
            return None

    def calculate_mos_quality_batch(self, audio_paths: List[str]) -> Dict[str, float]:
        """Calculate MOS Quality using NISQA v2 for multiple files (GPU-accelerated batch processing)"""
        try:
            if self.mos_quality_model is None:
                return {path: None for path in audio_paths}

            # Create temporary CSV with all file paths
            import tempfile
            import os

            # Create CSV DataFrame
            df = pd.DataFrame({'deg': audio_paths})

            # Save to temporary CSV file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                csv_path = f.name
                df.to_csv(csv_path, index=False)

            try:
                # Update NISQA args to use CSV mode for batch processing
                original_mode = self.mos_quality_model.args['mode']
                original_csv = self.mos_quality_model.args.get('csv_file', None)
                original_data_dir = self.mos_quality_model.args.get('data_dir', None)
                original_csv_deg = self.mos_quality_model.args.get('csv_deg', None)

                # Set batch mode parameters
                self.mos_quality_model.args['mode'] = 'predict_csv'
                self.mos_quality_model.args['data_dir'] = '/'  # Use absolute paths
                self.mos_quality_model.args['csv_file'] = csv_path
                self.mos_quality_model.args['csv_deg'] = 'deg'

                # Reload dataset with batch files
                self.mos_quality_model._loadDatasets()

                # Ensure model is on correct GPU device
                if self.device.startswith('cuda'):
                    self.mos_quality_model.model = self.mos_quality_model.model.to(self.device)
                    self.mos_quality_model.dev = torch.device(self.device)

                # Run batch prediction (GPU-accelerated)
                results_df = self.mos_quality_model.predict()

                # Restore original args
                self.mos_quality_model.args['mode'] = original_mode
                if original_csv is not None:
                    self.mos_quality_model.args['csv_file'] = original_csv
                if original_data_dir is not None:
                    self.mos_quality_model.args['data_dir'] = original_data_dir
                if original_csv_deg is not None:
                    self.mos_quality_model.args['csv_deg'] = original_csv_deg

                # Parse results into dictionary
                results = {}
                if isinstance(results_df, pd.DataFrame) and 'mos_pred' in results_df.columns:
                    for idx, row in results_df.iterrows():
                        file_path = row['deg']
                        score = row['mos_pred']
                        results[file_path] = float(score)
                else:
                    results = {path: None for path in audio_paths}

                return results

            finally:
                # Clean up temporary CSV file
                if os.path.exists(csv_path):
                    os.remove(csv_path)

        except Exception as e:
            print(f"Error in batch MOS Quality calculation (NISQA): {e}")
            import traceback
            traceback.print_exc()
            return {path: None for path in audio_paths}

    def calculate_mos_naturalness(self, audio_path: str) -> Optional[float]:
        """Calculate MOS Naturalness using UTMOS (EN) or RAMP (ZH)"""
        try:
            if self.mos_naturalness_model is None:
                return None

            if self.language == 'en':
                # UTMOS
                wave, sr = self.load_audio_optimized(audio_path, sr=16000)

                if wave is None or len(wave) == 0:
                    return None

                with torch.no_grad():
                    wave_tensor = torch.from_numpy(wave).unsqueeze(0)

                    if self.device.startswith('cuda'):
                        wave_tensor = wave_tensor.to(self.device)

                    score = self.mos_naturalness_model(wave_tensor, sr)
                    return float(score.item())

            elif self.language == 'zh':
                # RAMP
                result = self.mos_naturalness_model(audio_in=audio_path)

                if isinstance(result, dict) and 'score' in result:
                    return float(result['score'])
                elif isinstance(result, (int, float)):
                    return float(result)
                else:
                    return None

        except Exception as e:
            print(f"Error calculating MOS Naturalness: {e}")
            return None

    def calculate_mos_naturalness_batch(self, audio_paths: List[str], batch_size: int = 32) -> Dict[str, float]:
        """
        Calculate MOS Naturalness for multiple files using batch processing.

        Args:
            audio_paths: List of audio file paths
            batch_size: Number of files to process in each batch (default: 32)

        Returns:
            Dictionary mapping file paths to MOS Naturalness scores
        """
        try:
            if self.mos_naturalness_model is None:
                return {path: None for path in audio_paths}

            results = {}

            # UTMOS - Supports true batch processing with PyTorch (multilingual)
            # Process in batches to avoid OOM
            for i in range(0, len(audio_paths), batch_size):
                batch_paths = audio_paths[i:i + batch_size]
                batch_waves = []
                valid_paths = []

                # Load all audio files in this batch
                for path in batch_paths:
                    wave, sr = self.load_audio_optimized(path, sr=16000)
                    if wave is not None and len(wave) > 0:
                        batch_waves.append(torch.from_numpy(wave))
                        valid_paths.append(path)
                    else:
                        results[path] = None

                if len(batch_waves) == 0:
                    continue

                # Pad sequences to same length
                max_len = max(w.shape[0] for w in batch_waves)
                padded_waves = []
                for wave in batch_waves:
                    if wave.shape[0] < max_len:
                        padding = torch.zeros(max_len - wave.shape[0])
                        wave = torch.cat([wave, padding])
                    padded_waves.append(wave)

                # Stack into batch tensor
                batch_tensor = torch.stack(padded_waves)

                # Move to device and compute
                with torch.no_grad():
                    if self.device.startswith('cuda'):
                        batch_tensor = batch_tensor.to(self.device)

                    # UTMOS expects (batch, samples) and sr
                    scores = self.mos_naturalness_model(batch_tensor, 16000)

                    # Map scores back to file paths
                    for path, score in zip(valid_paths, scores):
                        results[path] = float(score.item())

            return results

        except Exception as e:
            print(f"Error in batch MOS Naturalness calculation: {e}")
            import traceback
            traceback.print_exc()
            return {path: None for path in audio_paths}

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
            import torchaudio.transforms as T
            import numpy as np
            from pesq import pesq

            # Load audio files with original sample rates
            ref_audio, ref_sr = torchaudio.load(reference_path)
            deg_audio, deg_sr = torchaudio.load(degraded_path)

            # Convert to mono first
            if ref_audio.shape[0] > 1:
                ref_audio = torch.mean(ref_audio, dim=0, keepdim=True)
            if deg_audio.shape[0] > 1:
                deg_audio = torch.mean(deg_audio, dim=0, keepdim=True)

            # Resample to 16kHz if needed
            target_sr = 16000
            if ref_sr != target_sr:
                resampler = T.Resample(ref_sr, target_sr)
                ref_audio = resampler(ref_audio)
            if deg_sr != target_sr:
                resampler = T.Resample(deg_sr, target_sr)
                deg_audio = resampler(deg_audio)

            # Convert to numpy
            ref_audio = ref_audio.squeeze().numpy()
            deg_audio = deg_audio.squeeze().numpy()

            # Ensure same length
            min_len = min(len(ref_audio), len(deg_audio))
            if min_len < target_sr * 0.1:  # At least 0.1 seconds
                return None

            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]

            return float(pesq(target_sr, ref_audio, deg_audio, 'wb'))
        except Exception as e:
            return None

    @staticmethod
    def _calculate_stoi_worker(args):
        """Worker function for parallel STOI calculation"""
        reference_path, degraded_path = args
        try:
            import torchaudio
            import torchaudio.transforms as T
            import numpy as np
            from pystoi import stoi

            # Load audio files with original sample rates
            ref_audio, ref_sr = torchaudio.load(reference_path)
            deg_audio, deg_sr = torchaudio.load(degraded_path)

            # Convert to mono first
            if ref_audio.shape[0] > 1:
                ref_audio = torch.mean(ref_audio, dim=0, keepdim=True)
            if deg_audio.shape[0] > 1:
                deg_audio = torch.mean(deg_audio, dim=0, keepdim=True)

            # Resample to 16kHz if needed
            target_sr = 16000
            if ref_sr != target_sr:
                resampler = T.Resample(ref_sr, target_sr)
                ref_audio = resampler(ref_audio)
            if deg_sr != target_sr:
                resampler = T.Resample(deg_sr, target_sr)
                deg_audio = resampler(deg_audio)

            # Convert to numpy
            ref_audio = ref_audio.squeeze().numpy()
            deg_audio = deg_audio.squeeze().numpy()

            # Ensure same length
            min_len = min(len(ref_audio), len(deg_audio))
            if min_len < target_sr * 0.1:  # At least 0.1 seconds
                return None

            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]

            return float(stoi(ref_audio, deg_audio, target_sr, extended=False))
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

        # Use 'spawn' method to avoid CUDA context deadlock when GPU models are loaded
        # Fork method copies CUDA contexts which causes deadlock
        ctx = mp.get_context('spawn')
        with ctx.Pool(num_workers) as pool:
            pesq_results = pool.map(self._calculate_pesq_worker, file_pairs)
            stoi_results = pool.map(self._calculate_stoi_worker, file_pairs)

        return pesq_results, stoi_results

    def calculate_speaker_similarity(self, reference_path: str, test_path: str) -> Optional[float]:
        """
        Calculate speaker similarity using language-specific models.
        ResNet3 (EN) or CAM++ (ZH)

        Returns cosine similarity score (higher = better identity preservation)
        """
        try:
            if self.speaker_model is None:
                return None

            if self.language == 'en':
                # ResNet3 (WeSpeaker)
                import wespeaker

                # Extract embeddings
                ref_embedding = self.speaker_model.extract_embedding(reference_path)
                test_embedding = self.speaker_model.extract_embedding(test_path)

                # Calculate cosine similarity
                similarity = 1.0 - cosine(ref_embedding, test_embedding)
                return float(similarity)

            elif self.language == 'zh':
                # CAM++
                result = self.speaker_model(
                    audio_in1=reference_path,
                    audio_in2=test_path
                )

                if isinstance(result, dict) and 'score' in result:
                    return float(result['score'])
                elif isinstance(result, (int, float)):
                    return float(result)
                else:
                    return None

        except Exception as e:
            print(f"Error calculating speaker similarity: {e}")
            return None

    def _extract_wespeaker_embeddings_batch(self, audio_paths: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Extract WeSpeaker embeddings in true batch mode (GPU accelerated).

        Args:
            audio_paths: List of audio file paths
            batch_size: Number of files to process in each batch

        Returns:
            Dictionary mapping audio paths to embedding vectors
        """
        embeddings_dict = {}

        try:
            import torchaudio

            for i in range(0, len(audio_paths), batch_size):
                batch_paths = audio_paths[i:i + batch_size]
                batch_feats = []
                valid_paths = []

                # Load and compute features for all files in batch
                for audio_path in batch_paths:
                    try:
                        # Load audio
                        pcm, sample_rate = torchaudio.load(audio_path, normalize=self.speaker_model.wavform_norm)

                        # Apply VAD if needed
                        if self.speaker_model.apply_vad:
                            vad_sample_rate = 16000
                            wav = pcm
                            if wav.size(0) > 1:
                                wav = wav.mean(dim=0, keepdim=True)

                            if sample_rate != vad_sample_rate:
                                transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=vad_sample_rate)
                                wav = transform(wav)

                            from wespeaker.utils.vad import get_speech_timestamps
                            segments = get_speech_timestamps(wav, self.speaker_model.vad, return_seconds=True)

                            pcmTotal = torch.Tensor()
                            if len(segments) > 0:
                                for segment in segments:
                                    start = int(segment['start'] * sample_rate)
                                    end = int(segment['end'] * sample_rate)
                                    pcmTemp = pcm[0, start:end]
                                    pcmTotal = torch.cat([pcmTotal, pcmTemp], 0)
                                pcm = pcmTotal.unsqueeze(0)
                            else:
                                embeddings_dict[audio_path] = None
                                continue

                        # Resample if needed
                        pcm = pcm.to(torch.float)
                        if sample_rate != self.speaker_model.resample_rate:
                            pcm = torchaudio.transforms.Resample(
                                orig_freq=sample_rate,
                                new_freq=self.speaker_model.resample_rate)(pcm)

                        # Compute fbank features
                        feats = self.speaker_model.compute_fbank(pcm, sample_rate=self.speaker_model.resample_rate, cmn=True)
                        batch_feats.append(feats)
                        valid_paths.append(audio_path)

                    except Exception as e:
                        print(f"Error loading {audio_path}: {e}")
                        embeddings_dict[audio_path] = None
                        continue

                if len(batch_feats) == 0:
                    continue

                # Pad features to same length
                max_len = max(f.shape[0] for f in batch_feats)
                padded_feats = []
                for feats in batch_feats:
                    if feats.shape[0] < max_len:
                        padding = torch.zeros(max_len - feats.shape[0], feats.shape[1])
                        feats = torch.cat([feats, padding], dim=0)
                    padded_feats.append(feats)

                # Stack into batch and move to GPU
                batch_tensor = torch.stack(padded_feats).to(self.device)

                # Batch inference
                with torch.no_grad():
                    outputs = self.speaker_model.model(batch_tensor)
                    outputs = outputs[-1] if isinstance(outputs, tuple) else outputs

                    # Map embeddings back to paths
                    for path, embedding in zip(valid_paths, outputs):
                        embeddings_dict[path] = embedding.cpu().numpy()

            return embeddings_dict

        except Exception as e:
            print(f"Error in batch WeSpeaker embedding extraction: {e}")
            import traceback
            traceback.print_exc()
            return {path: None for path in audio_paths}

    def calculate_speaker_similarity_batch(self, file_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
        """
        Calculate speaker similarity for multiple file pairs using batch processing.

        Args:
            file_pairs: List of (reference_path, test_path) tuples

        Returns:
            Dictionary mapping file pairs to similarity scores
        """
        try:
            if self.speaker_model is None:
                return {pair: None for pair in file_pairs}

            results = {}

            if self.language == 'en':
                # ResNet3 (WeSpeaker) - Extract embeddings in TRUE batch mode (GPU accelerated)
                import wespeaker

                # Get unique reference and test files
                unique_refs = list(set([pair[0] for pair in file_pairs]))
                unique_tests = list(set([pair[1] for pair in file_pairs]))

                # Batch extract reference embeddings (GPU accelerated)
                ref_embeddings = self._extract_wespeaker_embeddings_batch(unique_refs, batch_size=32)

                # Batch extract test embeddings (GPU accelerated)
                test_embeddings = self._extract_wespeaker_embeddings_batch(unique_tests, batch_size=32)

                # Calculate similarities for all pairs
                for ref_path, test_path in file_pairs:
                    ref_emb = ref_embeddings.get(ref_path)
                    test_emb = test_embeddings.get(test_path)

                    if ref_emb is not None and test_emb is not None:
                        try:
                            similarity = 1.0 - cosine(ref_emb, test_emb)
                            results[(ref_path, test_path)] = float(similarity)
                        except Exception as e:
                            print(f"Error calculating similarity for {ref_path}, {test_path}: {e}")
                            results[(ref_path, test_path)] = None
                    else:
                        results[(ref_path, test_path)] = None

            elif self.language == 'zh':
                # CAM++ (FunASR AutoModel) - Extract embeddings and compute similarity
                import torch
                import torch.nn.functional as F

                for ref_path, test_path in file_pairs:
                    try:
                        # FunASR CAM++ returns embeddings, not scores
                        # Extract embeddings for both files
                        result = self.speaker_model.generate(
                            input=[ref_path, test_path],
                            data_type="sound"
                        )

                        # Result is list of dicts: [{'spk_embedding': tensor}, {'spk_embedding': tensor}]
                        if isinstance(result, list) and len(result) == 2:
                            emb1 = result[0].get('spk_embedding')
                            emb2 = result[1].get('spk_embedding')

                            if emb1 is not None and emb2 is not None:
                                # Compute cosine similarity between embeddings
                                similarity = F.cosine_similarity(emb1, emb2, dim=-1)
                                results[(ref_path, test_path)] = float(similarity.item())
                            else:
                                results[(ref_path, test_path)] = None
                        else:
                            results[(ref_path, test_path)] = None

                    except Exception as e:
                        print(f"Error calculating speaker similarity for {ref_path}, {test_path}: {e}")
                        results[(ref_path, test_path)] = None

            return results

        except Exception as e:
            print(f"Error in batch speaker similarity calculation: {e}")
            import traceback
            traceback.print_exc()
            return {pair: None for pair in file_pairs}

    def evaluate_audio_pair(self, original_path: str, inference_path: str, ground_truth: str) -> Dict:
        """Evaluate audio pair with all V2 metrics"""
        results = {}

        # ASR metrics
        asr_result = self.calculate_dwer_dcer(original_path, inference_path, ground_truth)
        if asr_result:
            results.update(asr_result)

        # Quality metrics
        results['MOS_Quality'] = self.calculate_mos_quality(inference_path)
        results['MOS_Naturalness'] = self.calculate_mos_naturalness(inference_path)
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