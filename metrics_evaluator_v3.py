#!/usr/bin/env python3
"""
Audio Codec Metrics Evaluator V3 - Taiwanese Minnan Language Support

New in V3 (Phase 1):
Taiwanese Minnan (Min):
  - ASR (dCER): Taiwan-Tongues-ASR-CE (Whisper-large-v3 fine-tuned)
  - Speaker Similarity: CAM++ (shared with Chinese)
  - MOS_Quality: NISQA v2
  - MOS_Naturalness: UTMOS (multilingual)
  - VDE (Voicing Decision Error): torchcrepe
  - F0-RMSE & GPE (Gross Pitch Error): torchcrepe + fastdtw
  - TER (Tone Error Rate): taibun + Levenshtein
  - SIM-Sem (Semantic Similarity): WavLM-Large

Phase 2 (TODO):
  - Codebook Perplexity (PPL): Requires codec model interface
  - Token-Tone/Phoneme NMI: Requires MFA Taiwanese model

Supported Languages:
  - English (en): LibriSpeech
  - Chinese (zh): CommonVoice, AISHELL
  - Taiwanese Minnan (min): MinSpeech
  - Cantonese (yue): CommonVoice Cantonese
  - Vietnamese (vi): CommonVoice Vietnamese
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import re
import string
from transformers import pipeline, AutoModel, AutoFeatureExtractor
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

# New imports for V3
try:
    import torchcrepe
    TORCHCREPE_AVAILABLE = True
except ImportError:
    TORCHCREPE_AVAILABLE = False
    print("Warning: torchcrepe not installed. F0-based metrics (VDE, F0-RMSE, GPE) will be unavailable.")

try:
    from fastdtw import fastdtw
    FASTDTW_AVAILABLE = True
except ImportError:
    FASTDTW_AVAILABLE = False
    print("Warning: fastdtw not installed. F0-RMSE and GPE metrics will be unavailable.")

try:
    from taibun import Converter
    TAIBUN_AVAILABLE = True
except ImportError:
    TAIBUN_AVAILABLE = False
    print("Warning: taibun not installed. TER (Tone Error Rate) for Taiwanese will be unavailable.")

try:
    from pypinyin import pinyin, Style
    PYPINYIN_AVAILABLE = True
except ImportError:
    PYPINYIN_AVAILABLE = False
    print("Warning: pypinyin not installed. TER (Tone Error Rate) for Chinese will be unavailable.")

try:
    import pycantonese
    PYCANTONESE_AVAILABLE = True
except ImportError:
    PYCANTONESE_AVAILABLE = False
    print("Warning: pycantonese not installed. TER (Tone Error Rate) for Cantonese will be unavailable.")

try:
    import cn2an
    CN2AN_AVAILABLE = True
except ImportError:
    CN2AN_AVAILABLE = False
    print("Warning: cn2an not installed. Arabic to Chinese number conversion disabled.")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class AudioMetricsEvaluatorV3:
    """V3 Evaluator with Taiwanese Minnan language support and advanced metrics"""

    def __init__(self, language='en', dataset='librispeech', device=None, use_gpu=True, gpu_id=0, taiwanese_asr_model=None):
        """
        Initialize V3 Evaluator

        Args:
            language: 'en' | 'zh' | 'min' (Taiwanese Minnan) | 'yue' (Cantonese) | 'vi' (Vietnamese)
            dataset: 'librispeech' | 'commonvoice' | 'aishell' | 'minspeech'
            device: torch device (auto-detected if None)
            use_gpu: whether to use GPU acceleration
            gpu_id: GPU device ID
            taiwanese_asr_model: Optional Taiwanese ASR model to use (overrides default)
                Options: 'tsukilen', 'whisper-large-v3'
        """
        self.language = language
        self.dataset = dataset
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.taiwanese_asr_model = taiwanese_asr_model
        self.device = self._setup_device(device)

        # Model placeholders
        self.asr_pipeline = None
        self.speaker_model = None
        self.mos_quality_model = None  # NISQA v2
        self.mos_naturalness_model = None  # UTMOS
        self.semantic_model = None  # WavLM for SIM-Sem (V3)
        self.semantic_processor = None

        # Audio resampler cache
        self.resamplers = {}

        # Initialize Traditional to Simplified Chinese converter
        if self.language in ['zh', 'min', 'yue']:
            try:
                self.t2s_converter = opencc.OpenCC('t2s')
                print("OpenCC Traditional to Simplified converter initialized")
            except Exception as e:
                print(f"Warning: Could not initialize OpenCC: {e}")
                self.t2s_converter = None
        else:
            self.t2s_converter = None

        # Initialize Taibun converter for Taiwanese (V3)
        if self.language == 'min' and TAIBUN_AVAILABLE:
            try:
                self.taibun_converter = Converter(system='Tailo', dialect='south')
                print("Taibun converter initialized for Taiwanese TER calculation")
            except Exception as e:
                print(f"Warning: Could not initialize Taibun: {e}")
                self.taibun_converter = None
        else:
            self.taibun_converter = None

        # Language-specific configuration
        self.config = self._get_language_config()

    def _get_language_config(self) -> Dict:
        """Get language-specific configuration"""
        configs = {
            'en': {
                'asr_model': 'openai/whisper-large-v3',
                'asr_metric': 'dWER',
                'speaker_model_type': 'wespeaker',
                'enable_ter': False,
                'enable_vde': False
            },
            'zh': {
                'asr_model': 'paraformer-zh',
                'asr_metric': 'dCER',
                'speaker_model_type': 'campplus',
                'enable_ter': False,
                'enable_vde': False
            },
            'min': {
                'asr_model': 'TSukiLen/whisper-small-chinese-tw-minnan-hanzi',  # Default: TSukiLen
                'asr_metric': 'dCER',
                'speaker_model_type': 'campplus',  # Share with Chinese
                'enable_ter': True,  # Taiwanese-specific
                'enable_vde': True   # Check tone sandhi
            },
            'yue': {
                'asr_model': 'openai/whisper-large-v3',  # Whisper-large-v3 for Cantonese
                'asr_metric': 'dCER',
                'speaker_model_type': 'campplus',  # Share with Chinese
                'enable_ter': True,  # Cantonese has 6-9 tones
                'enable_vde': True   # Check tone preservation
            },
            'vi': {
                'asr_model': 'vinai/PhoWhisper-large',  # PhoWhisper-large for Vietnamese
                'asr_metric': 'dWER',
                'speaker_model_type': 'campplus',  # Use CAM++ for Vietnamese
                'enable_ter': True,  # Vietnamese has 6 tones
                'enable_vde': True   # Check tone preservation
            }
        }
        return configs.get(self.language, configs['en'])

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
        print(f"Loading V3 models for language: {self.language}, dataset: {self.dataset}")
        print(f"Device: {self.device}")

        if self.device.startswith('cuda'):
            mem_info = self.get_gpu_memory_info()
            print(f"Initial GPU memory: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")

        # Load ASR model (language-specific)
        self._load_asr_model()

        # Load Speaker Similarity model (language-specific)
        self._load_speaker_model()

        # Load MOS Quality model (NISQA v2 - same for all languages)
        self._load_mos_quality_model()

        # Load MOS Naturalness model (UTMOS - multilingual)
        self._load_mos_naturalness_model()

        # Load Semantic Similarity model (WavLM - V3 new)
        self._load_semantic_model()

        if self.device.startswith('cuda'):
            mem_info = self.get_gpu_memory_info()
            print(f"After all models loaded: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")

        print("All V3 models loaded successfully!")

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
                print(f"Whisper-large-v3 loaded on {self.device}")

            elif self.language == 'zh':
                # Chinese: Paraformer-large via FunASR
                print("Loading Paraformer-large for Chinese ASR (FunASR)...")
                from funasr import AutoModel

                self.asr_pipeline = AutoModel(
                    model="paraformer-zh",
                    hub="hf",
                    device=self.device,
                    disable_update=True
                )
                print(f"Paraformer-large loaded on {self.device}")

            elif self.language == 'min':
                # Taiwanese Minnan: Use community fine-tuned Whisper models
                print("Loading Taiwanese Minnan ASR model...")
                torch_dtype = torch.float16 if self.device.startswith('cuda') else torch.float32

                # Model selection based on taiwanese_asr_model parameter
                models_to_try = []

                if self.taiwanese_asr_model == 'tsukilen':
                    models_to_try = [
                        ('TSukiLen/whisper-small-chinese-tw-minnan-hanzi', 'TSukiLen', {"language": "zh", "task": "transcribe"})
                    ]
                elif self.taiwanese_asr_model == 'whisper-large-v3':
                    models_to_try = [
                        ('openai/whisper-large-v3', 'Whisper-large-v3', {"language": "zh", "task": "transcribe"})
                    ]
                else:
                    # Default: Use TSukiLen (best for Taiwanese), fallback to Whisper-large-v3
                    models_to_try = [
                        ('TSukiLen/whisper-small-chinese-tw-minnan-hanzi', 'TSukiLen', {"language": "zh", "task": "transcribe"}),
                        ('openai/whisper-large-v3', 'Whisper-large-v3', {"language": "zh", "task": "transcribe"})
                    ]
                    print("Default mode: Will try TSukiLen → Whisper-large-v3")

                loaded = False
                for model_id, model_name, gen_kwargs in models_to_try:
                    try:
                        print(f"Trying {model_name}: {model_id}...")

                        self.asr_pipeline = pipeline(
                            "automatic-speech-recognition",
                            model=model_id,
                            torch_dtype=torch_dtype,
                            device=self.device,
                            generate_kwargs=gen_kwargs if gen_kwargs else None
                        )
                        print(f"✓ {model_name} Taiwanese Minnan model loaded on {self.device} (dtype: {torch_dtype})")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"✗ Error loading {model_name}: {e}")
                        # If float16 fails, try float32
                        if torch_dtype == torch.float16:
                            try:
                                print(f"  Retrying {model_name} with float32...")
                                self.asr_pipeline = pipeline(
                                    "automatic-speech-recognition",
                                    model=model_id,
                                    torch_dtype=torch.float32,
                                    device=self.device,
                                    generate_kwargs=gen_kwargs if gen_kwargs else None
                                )
                                print(f"✓ {model_name} loaded on {self.device} (dtype: float32)")
                                loaded = True
                                break
                            except Exception as e2:
                                print(f"✗ Also failed with float32: {e2}")
                        continue

                if not loaded:
                    raise RuntimeError("Failed to load any Taiwanese ASR model")

            elif self.language == 'yue':
                # Cantonese: Whisper-large-v3 (multilingual, supports Cantonese well)
                print("Loading Whisper-large-v3 for Cantonese ASR...")
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
                        "language": "zh",  # Use Chinese for Cantonese (Whisper detects automatically)
                        "task": "transcribe",
                        "repetition_penalty": 1.2,
                        "no_repeat_ngram_size": 3,
                        "max_new_tokens": 256
                    }
                )
                print(f"Whisper-large-v3 loaded on {self.device} for Cantonese")

            elif self.language == 'vi':
                # Vietnamese: PhoWhisper-large (Vietnamese-specialized)
                print("Loading PhoWhisper-large for Vietnamese ASR...")
                torch_dtype = torch.float16 if self.device.startswith('cuda') else torch.float32

                # Download model to specified directory
                model_cache_dir = "/mnt/Internal/jieshiang/Model/phowhisper-large"
                os.makedirs(model_cache_dir, exist_ok=True)

                try:
                    self.asr_pipeline = pipeline(
                        "automatic-speech-recognition",
                        model="vinai/PhoWhisper-large",
                        torch_dtype=torch_dtype,
                        device=self.device,
                        model_kwargs={
                            "cache_dir": model_cache_dir
                        },
                        generate_kwargs={
                            "language": "vi",
                            "task": "transcribe"
                        }
                    )
                    print(f"PhoWhisper-large loaded on {self.device}")
                except Exception as e:
                    print(f"Failed to load PhoWhisper-large: {e}")
                    print("Falling back to Whisper-large-v3 for Vietnamese...")
                    self.asr_pipeline = pipeline(
                        "automatic-speech-recognition",
                        model="openai/whisper-large-v3",
                        torch_dtype=torch_dtype,
                        device=self.device,
                        generate_kwargs={
                            "language": "vi",
                            "task": "transcribe"
                        }
                    )
                    print(f"Whisper-large-v3 loaded on {self.device} (fallback)")

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
                # English: ResNet34 (WeSpeaker)
                print("Loading ResNet34 (WeSpeaker) for English Speaker Similarity...")
                import wespeaker

                model_path = "/mnt/Internal/jieshiang/Model/wespeaker_resnet34"

                if not os.path.exists(model_path):
                    print(f"Downloading ResNet34 to {model_path}...")
                    os.makedirs(model_path, exist_ok=True)

                # Load WeSpeaker ResNet model
                self.speaker_model = wespeaker.load_model('english')

                # Move model to GPU
                if self.device.startswith('cuda'):
                    self.speaker_model.set_device(self.gpu_id)
                    print(f"ResNet34 loaded on GPU {self.gpu_id}")
                else:
                    print(f"ResNet34 loaded on CPU")

            elif self.language in ['zh', 'min', 'yue', 'vi']:
                # Chinese, Taiwanese, Cantonese & Vietnamese: CAM++ via FunASR
                print(f"Loading CAM++ for {self.language} Speaker Similarity...")
                from funasr import AutoModel

                try:
                    self.speaker_model = AutoModel(
                        model="funasr/campplus",
                        hub="hf",
                        device=self.device,
                        disable_update=True
                    )
                    print(f"CAM++ loaded on {self.device}")
                except Exception as e:
                    print(f"Warning: Could not load CAM++ via FunASR: {e}")
                    print("Trying ModelScope pipeline as fallback...")

                    try:
                        from modelscope.pipelines import pipeline as ms_pipeline
                        model_dir = "/mnt/Internal/jieshiang/Model/speech_campplus_sv_zh-cn_16k-common"

                        if not os.path.exists(model_dir):
                            print(f"Downloading CAM++ to {model_dir}...")
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
        """Load NISQA v2 for MOS Quality (same for all languages)"""
        print("\nLoading NISQA v2 for MOS Quality...")

        try:
            from nisqa.NISQA_model import nisqaModel
            import os

            # NISQA v2 initialization
            model_path = '/mnt/Internal/jieshiang/Model/nisqa_weights/nisqa.tar'

            # Create a dummy audio file for initialization
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                dummy_path = f.name
                sample_rate = 16000
                waveform = torch.zeros(1, sample_rate)
                torchaudio.save(dummy_path, waveform, sample_rate)

            args = {
                'mode': 'predict_file',
                'pretrained_model': model_path,
                'deg': dummy_path,
                'num_workers': 4,
                'bs': 1,
                'tr_bs_val': 32,
                'tr_num_workers': 4,
                'ms_channel': None,
                'output_dir': None,
                'tr_device': 'cpu',
            }

            self.mos_quality_model = nisqaModel(args)

            # Clean up dummy file
            import os as _os
            if _os.path.exists(dummy_path):
                _os.remove(dummy_path)

            # Move model to GPU
            if self.device.startswith('cuda'):
                self.mos_quality_model.model = self.mos_quality_model.model.to(self.device)
                self.mos_quality_model.dev = torch.device(self.device)
                print(f"NISQA v2 loaded on {self.device}")
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
        """Load MOS Naturalness model (UTMOS - multilingual)"""
        print(f"\nLoading MOS Naturalness model for {self.language}...")

        try:
            print(f"Loading UTMOS for MOS Naturalness ({self.language})...")
            print("UTMOS supports multilingual evaluation including Chinese and Taiwanese")

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

    def _load_semantic_model(self):
        """Load WavLM for Semantic Similarity (V3 new)"""
        print("\nLoading WavLM-Large for Semantic Similarity (SIM-Sem)...")

        try:
            from transformers import Wav2Vec2FeatureExtractor

            model_name = "microsoft/wavlm-large"
            cache_dir = "/mnt/Internal/jieshiang/Model/wavlm-large"

            # Create cache directory
            os.makedirs(cache_dir, exist_ok=True)

            # Try to load with safetensors first (avoids torch.load vulnerability)
            try:
                print(f"   Loading from: {cache_dir}")
                print("   Attempting to load with safetensors...")
                self.semantic_model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    use_safetensors=True
                ).to(self.device)
                print("   ✓ Loaded with safetensors")
            except Exception as e:
                print(f"   Safetensors loading failed: {e}")
                print("   Trying without safetensors...")
                self.semantic_model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                ).to(self.device)
                print("   ✓ Loaded without safetensors")

            self.semantic_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )

            self.semantic_model.eval()
            print(f"WavLM-Large loaded on {self.device}")

            if self.device.startswith('cuda'):
                mem_info = self.get_gpu_memory_info()
                print(f"After WavLM loading: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")

        except Exception as e:
            print(f"Error loading WavLM: {e}")
            import traceback
            traceback.print_exc()
            print("Semantic Similarity metric will not be available")
            self.semantic_model = None
            self.semantic_processor = None

    def convert_traditional_to_simplified(self, text: str) -> str:
        """Convert Traditional Chinese to Simplified Chinese"""
        if not text or pd.isna(text):
            return ""

        if self.language in ['zh', 'min', 'yue'] and self.t2s_converter:
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

        if self.language in ['zh', 'min', 'yue']:
            # Chinese-based languages: remove punctuation and spaces
            text = re.sub(r'[，。！？；：""''（）《》【】、]', '', text)
            text = re.sub(r'\s+', '', text)
        elif self.language == 'vi':
            # Vietnamese: keep spaces but remove punctuation (similar to English)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
        else:
            # English and others
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

    def batch_transcribe(self, audio_paths: List[str], batch_size: int = 8) -> List[str]:
        """
        Batch transcribe audio files with true batch processing for improved efficiency.

        Args:
            audio_paths: List of audio file paths
            batch_size: Internal batch size for Whisper pipeline (default: 8)

        Returns:
            List of transcribed texts
        """
        if not self.asr_pipeline:
            print("ASR pipeline not loaded")
            return [""] * len(audio_paths)

        results = []

        if self.language in ['en', 'min', 'yue', 'vi']:
            # Whisper-based models (English, Taiwanese, Cantonese, Vietnamese) - Use true batch processing
            from tqdm import tqdm

            # Determine language code for Whisper
            if self.language == 'en':
                whisper_lang = "en"
            elif self.language == 'vi':
                whisper_lang = "vi"
            else:  # 'min' and 'yue' use 'zh'
                whisper_lang = "zh"

            # Load all audio files first
            batch_audio = []
            valid_indices = []

            print(f"Loading {len(audio_paths)} audio files...")
            for idx, path in enumerate(tqdm(audio_paths, desc="Loading audio")):
                audio, sr = self.load_audio_optimized(path, sr=16000)
                if audio is None:
                    results.append("")
                    continue

                try:
                    # Ensure audio is float32 to avoid LayerNorm issues
                    if isinstance(audio, torch.Tensor):
                        audio = audio.float().cpu().numpy()
                    elif hasattr(audio, 'dtype'):
                        audio = audio.astype('float32')

                    batch_audio.append(audio)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    results.append("")

            # Initialize results with empty strings
            results = [""] * len(audio_paths)

            if len(batch_audio) > 0:
                print(f"Transcribing {len(batch_audio)} audio files using batch processing (batch_size={batch_size})...")
                try:
                    # Try batch processing with Whisper pipeline
                    batch_results = self.asr_pipeline(
                        batch_audio,
                        generate_kwargs={"language": whisper_lang, "task": "transcribe"},
                        batch_size=batch_size,  # Whisper internal batch size
                        return_timestamps=False  # Faster without timestamps
                    )

                    # Map results back to original indices
                    for idx, result in zip(valid_indices, batch_results):
                        if isinstance(result, dict) and 'text' in result:
                            results[idx] = result['text']
                        else:
                            results[idx] = ""

                    print(f"✓ Batch transcription completed successfully")

                except Exception as batch_error:
                    # Batch processing failed, fall back to sequential
                    print(f"⚠ Batch processing failed ({str(batch_error)[:100]}), falling back to sequential processing...")

                    for idx, audio in zip(tqdm(valid_indices, desc="Sequential transcription"), batch_audio):
                        try:
                            result = self.asr_pipeline(
                                audio,
                                generate_kwargs={"language": whisper_lang, "task": "transcribe"},
                                return_timestamps=False
                            )

                            if isinstance(result, dict) and 'text' in result:
                                results[idx] = result['text']
                            elif isinstance(result, dict) and 'chunks' in result:
                                text = ' '.join([chunk['text'] for chunk in result['chunks']])
                                results[idx] = text
                            else:
                                results[idx] = ""
                        except Exception as e:
                            print(f"Error transcribing file {idx}: {e}")
                            results[idx] = ""

        elif self.language == 'zh':
            # Paraformer (FunASR) - Sequential processing (FunASR doesn't support batch well)
            from tqdm import tqdm
            for path in tqdm(audio_paths, desc="Transcribing (Paraformer)"):
                try:
                    result = self.asr_pipeline.generate(input=path)
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

            if self.language in ['en', 'min', 'yue', 'vi']:
                # Whisper-based: Load audio
                original_audio, _ = self.load_audio_optimized(original_path, sr=16000)
                inference_audio, _ = self.load_audio_optimized(inference_path, sr=16000)

                if original_audio is None or inference_audio is None:
                    return None

                # Transcribe
                gen_kwargs = {}
                if self.language == 'en':
                    gen_kwargs["language"] = "en"
                elif self.language == 'vi':
                    gen_kwargs["language"] = "vi"
                else:  # 'min' and 'yue'
                    gen_kwargs["language"] = "zh"

                original_result = self.asr_pipeline(
                    original_audio,
                    generate_kwargs=gen_kwargs
                )
                inference_result = self.asr_pipeline(
                    inference_audio,
                    generate_kwargs=gen_kwargs
                )

                original_transcript = original_result['text']
                inference_transcript = inference_result['text']

            elif self.language == 'zh':
                # Paraformer (FunASR)
                original_result = self.asr_pipeline.generate(input=original_path)
                inference_result = self.asr_pipeline.generate(input=inference_path)

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

            # Convert Traditional to Simplified for Chinese/Taiwanese/Cantonese
            if self.language in ['zh', 'min', 'yue']:
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

            # Use CER for Chinese-based languages (zh, min, yue), WER for others (en, vi)
            if self.language in ['zh', 'min', 'yue']:
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
            else:  # en, vi
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

    # ==================== V3 NEW METRICS ====================

    def calculate_vde(self, ref_path: str, hyp_path: str) -> Optional[float]:
        """
        Calculate VDE (Voicing Decision Error) using torchcrepe for GPU acceleration.

        VDE = (V->UV errors + UV->V errors) / total_frames

        This metric is crucial for Taiwanese to preserve checked tones (入聲).

        Args:
            ref_path: Reference audio path
            hyp_path: Hypothesis (reconstructed) audio path

        Returns:
            VDE score (lower is better)
        """
        if not TORCHCREPE_AVAILABLE:
            print("torchcrepe not available, skipping VDE calculation")
            return None

        try:
            # 1. Load audio
            ref_audio, sr = self.load_audio_optimized(ref_path, sr=16000)
            hyp_audio, _ = self.load_audio_optimized(hyp_path, sr=16000)

            if ref_audio is None or hyp_audio is None:
                return None

            # 2. Extract F0 and confidence using CREPE (GPU-accelerated)
            ref_tensor = torch.from_numpy(ref_audio).unsqueeze(0).to(self.device)
            hyp_tensor = torch.from_numpy(hyp_audio).unsqueeze(0).to(self.device)

            with torch.no_grad():
                ref_f0, ref_confidence = torchcrepe.predict(
                    ref_tensor, sr,
                    hop_length=320,  # 20ms hop (optimized from 10ms for 2x speedup)
                    fmin=50, fmax=550,
                    model='tiny',  # Fast model (optimized from 'full' for 5-10x speedup)
                    batch_size=1024,
                    device=self.device,
                    return_periodicity=True
                )
                hyp_f0, hyp_confidence = torchcrepe.predict(
                    hyp_tensor, sr,
                    hop_length=320,  # 20ms hop (optimized from 10ms for 2x speedup)
                    fmin=50, fmax=550,
                    model='tiny',  # Fast model (optimized from 'full' for 5-10x speedup)
                    batch_size=1024,
                    device=self.device,
                    return_periodicity=True
                )

            # 3. Convert to voicing flags (confidence > 0.5 = voiced)
            ref_voiced = (ref_confidence.squeeze().cpu().numpy() > 0.5).astype(int)
            hyp_voiced = (hyp_confidence.squeeze().cpu().numpy() > 0.5).astype(int)

            # 4. Align sequences to same length
            min_len = min(len(ref_voiced), len(hyp_voiced))
            ref_voiced = ref_voiced[:min_len]
            hyp_voiced = hyp_voiced[:min_len]

            # 5. Calculate VDE
            errors = np.sum(ref_voiced != hyp_voiced)
            vde = errors / len(ref_voiced)

            return float(vde)

        except Exception as e:
            print(f"Error calculating VDE: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_f0_metrics(self, ref_path: str, hyp_path: str) -> Optional[Dict]:
        """
        Calculate F0-RMSE and GPE (Gross Pitch Error) using torchcrepe + DTW.

        These metrics evaluate pitch preservation, critical for tonal languages.

        Args:
            ref_path: Reference audio path
            hyp_path: Hypothesis (reconstructed) audio path

        Returns:
            Dict with 'f0_rmse' (in cents) and 'gpe' (error rate)
        """
        if not TORCHCREPE_AVAILABLE or not FASTDTW_AVAILABLE:
            print("torchcrepe or fastdtw not available, skipping F0 metrics")
            return None

        try:
            # 1. Load audio
            ref_audio, sr = self.load_audio_optimized(ref_path, sr=16000)
            hyp_audio, _ = self.load_audio_optimized(hyp_path, sr=16000)

            if ref_audio is None or hyp_audio is None:
                return None

            # 2. Extract F0 using CREPE (GPU-accelerated)
            ref_tensor = torch.from_numpy(ref_audio).unsqueeze(0).to(self.device)
            hyp_tensor = torch.from_numpy(hyp_audio).unsqueeze(0).to(self.device)

            with torch.no_grad():
                ref_f0, ref_confidence = torchcrepe.predict(
                    ref_tensor, sr,
                    hop_length=320,  # 20ms hop (optimized from 10ms for 2x speedup)
                    fmin=50, fmax=550,
                    model='tiny',  # Fast model (optimized from 'full' for 5-10x speedup)
                    batch_size=1024,
                    device=self.device,
                    return_periodicity=True
                )
                hyp_f0, hyp_confidence = torchcrepe.predict(
                    hyp_tensor, sr,
                    hop_length=320,  # 20ms hop (optimized from 10ms for 2x speedup)
                    fmin=50, fmax=550,
                    model='tiny',  # Fast model (optimized from 'full' for 5-10x speedup)
                    batch_size=1024,
                    device=self.device,
                    return_periodicity=True
                )

            # 3. Convert to numpy and filter by confidence (voiced frames only)
            ref_f0 = ref_f0.squeeze().cpu().numpy()
            hyp_f0 = hyp_f0.squeeze().cpu().numpy()
            ref_conf = ref_confidence.squeeze().cpu().numpy()
            hyp_conf = hyp_confidence.squeeze().cpu().numpy()

            # Align to same length first
            min_len = min(len(ref_f0), len(hyp_f0))
            ref_f0 = ref_f0[:min_len]
            hyp_f0 = hyp_f0[:min_len]
            ref_conf = ref_conf[:min_len]
            hyp_conf = hyp_conf[:min_len]

            # Only use frames where both are voiced
            voiced_mask = (ref_conf > 0.5) & (hyp_conf > 0.5) & (ref_f0 > 0) & (hyp_f0 > 0)

            if np.sum(voiced_mask) < 10:  # Not enough voiced frames
                return {'f0_rmse': None, 'gpe': None}

            ref_f0_voiced = ref_f0[voiced_mask]
            hyp_f0_voiced = hyp_f0[voiced_mask]

            # 4. DTW alignment
            distance, path = fastdtw(
                ref_f0_voiced.reshape(-1, 1),
                hyp_f0_voiced.reshape(-1, 1)
            )

            # 5. Align F0 sequences
            aligned_ref = ref_f0_voiced[[p[0] for p in path]]
            aligned_hyp = hyp_f0_voiced[[p[1] for p in path]]

            # 6. Convert to cents (1200 * log2(f1/f2))
            # Add small epsilon to avoid log(0)
            cents_diff = 1200 * np.log2((aligned_hyp + 1e-8) / (aligned_ref + 1e-8))

            # 7. Calculate F0-RMSE in cents
            f0_rmse = np.sqrt(np.mean(cents_diff ** 2))

            # 8. Calculate GPE (errors > 50 cents threshold)
            threshold_cents = 50
            gpe = np.sum(np.abs(cents_diff) > threshold_cents) / len(cents_diff)

            return {
                'f0_rmse': float(f0_rmse),
                'gpe': float(gpe)
            }

        except Exception as e:
            print(f"Error calculating F0 metrics: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_ter(self, ref_text: str, hyp_text: str) -> Optional[float]:
        """
        Calculate TER (Tone Error Rate) for tonal languages.

        For Chinese (zh): Uses pypinyin to extract tones (1-4)
        For Taiwanese (min): Uses taibun to extract tones (1-8)
        For Cantonese (yue): Uses pycantonese to extract tones (1-6/9)
        For Vietnamese (vi): Extracts Vietnamese tones from diacritics (6 tones)

        TER = Levenshtein distance of tone sequences / reference tone length

        Args:
            ref_text: Reference text (Chinese characters / Vietnamese text)
            hyp_text: Hypothesis text (ASR output)

        Returns:
            TER score (lower is better)
        """
        # Check language and availability
        if self.language == 'min':
            if not TAIBUN_AVAILABLE or not self.taibun_converter:
                print("Taibun not available, skipping TER calculation for Taiwanese")
                return None
        elif self.language == 'zh':
            if not PYPINYIN_AVAILABLE:
                print("pypinyin not available, skipping TER calculation for Chinese")
                return None
        elif self.language == 'yue':
            if not PYCANTONESE_AVAILABLE:
                print("pycantonese not available, skipping TER calculation for Cantonese")
                return None
        elif self.language == 'vi':
            # Vietnamese TER doesn't require external library (uses diacritics)
            pass
        else:
            # TER only supports tonal languages
            return None

        try:
            # Check for empty strings before processing
            if not ref_text or not ref_text.strip():
                return 0.0 if (not hyp_text or not hyp_text.strip()) else 1.0
            if not hyp_text or not hyp_text.strip():
                return 1.0

            # Extract tones based on language
            if self.language == 'zh':
                # Chinese: use pypinyin
                ref_tones = self._extract_chinese_tones(ref_text)
                hyp_tones = self._extract_chinese_tones(hyp_text)
            elif self.language == 'min':
                # Taiwanese: use taibun
                ref_tailo = self.taibun_converter.get(ref_text)
                hyp_tailo = self.taibun_converter.get(hyp_text)
                ref_tones = self._extract_taiwanese_tones(ref_tailo)
                hyp_tones = self._extract_taiwanese_tones(hyp_tailo)
            elif self.language == 'yue':
                # Cantonese: use pycantonese
                ref_tones = self._extract_cantonese_tones(ref_text)
                hyp_tones = self._extract_cantonese_tones(hyp_text)
            elif self.language == 'vi':
                # Vietnamese: extract from diacritics
                ref_tones = self._extract_vietnamese_tones(ref_text)
                hyp_tones = self._extract_vietnamese_tones(hyp_text)

            # Handle empty sequences
            if len(ref_tones) == 0:
                return 0.0 if len(hyp_tones) == 0 else 1.0

            if len(hyp_tones) == 0:
                return 1.0

            # Calculate Levenshtein distance on tone sequences
            ref_str = ''.join(map(str, ref_tones))
            hyp_str = ''.join(map(str, hyp_tones))

            distance = Levenshtein.distance(ref_str, hyp_str)
            ter = distance / len(ref_tones)

            return float(ter)

        except Exception as e:
            print(f"Error calculating TER: {e}")
            print(f"  Ref text: {ref_text}")
            print(f"  Hyp text: {hyp_text}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_chinese_tones(self, text: str) -> List[int]:
        """
        Extract tone numbers from Chinese text using pypinyin.

        Args:
            text: Chinese text

        Returns:
            List of tone numbers (1-4, 5 for neutral tone)
        """
        # Get pinyin with tone numbers
        pinyin_list = pinyin(text, style=Style.TONE3, errors='ignore')

        tones = []
        for syllable in pinyin_list:
            if syllable and len(syllable) > 0:
                py = syllable[0]
                # Extract tone number from pinyin (e.g., "ni3" -> 3)
                tone_match = re.search(r'(\d)$', py)
                if tone_match:
                    tone = int(tone_match.group(1))
                    if tone in [1, 2, 3, 4, 5]:
                        tones.append(tone)
                else:
                    # No tone mark = neutral tone (5) or轻声
                    tones.append(5)

        return tones

    def _extract_taiwanese_tones(self, tailo_text: str) -> List[int]:
        """
        Extract tone numbers from Taiwanese Tailo romanization.

        Args:
            tailo_text: Tailo romanization text

        Returns:
            List of tone numbers (1-8)
        """
        # First try to extract numeric tones (e.g., "a2", "tsia̍h8")
        numeric_tones = re.findall(r'[a-zāáàâēéèêīíìîōóòôūúùû]+(\d)', tailo_text, re.IGNORECASE)
        if numeric_tones:
            return [int(t) for t in numeric_tones if t.isdigit() and int(t) in [1, 2, 3, 4, 5, 7, 8]]

        # Map diacritics to tone numbers
        tone_map = {
            'á': 2, 'é': 2, 'í': 2, 'ó': 2, 'ú': 2,
            'à': 3, 'è': 3, 'ì': 3, 'ò': 3, 'ù': 3,
            'â': 5, 'ê': 5, 'î': 5, 'ô': 5, 'û': 5,
            'ā': 7, 'ē': 7, 'ī': 7, 'ō': 7, 'ū': 7
        }

        tones = []
        # Split by spaces/hyphens
        syllables = re.split(r'[\s\-]+', tailo_text)
        for syl in syllables:
            if not syl or syl == '--':
                continue
            # Check for tone 8 marker (̍)
            if '̍' in syl or 'h' in syl.lower():
                tones.append(8)
            else:
                # Check for diacritics
                tone_found = False
                for char in syl:
                    if char in tone_map:
                        tones.append(tone_map[char])
                        tone_found = True
                        break
                # If no tone marker, default to tone 1 or 4
                if not tone_found and len(syl) > 0:
                    # Default tone depends on ending consonant
                    if syl.endswith(('p', 't', 'k', 'h')):
                        tones.append(4)  # Checked tone
                    else:
                        tones.append(1)  # Level tone

        return tones

    def _extract_cantonese_tones(self, text: str) -> List[int]:
        """
        Extract tone numbers from Cantonese text using pycantonese.

        Args:
            text: Cantonese text (Traditional Chinese characters)

        Returns:
            List of tone numbers (1-6 for standard tones, 7-9 for entering tones)
        """
        try:
            import pycantonese
            import re

            # Parse Cantonese text to get jyutping (romanization)
            # Returns list of tuples: [(chars, jyutping), ...]
            words = pycantonese.characters_to_jyutping(text)

            tones = []
            for item in words:
                if isinstance(item, tuple) and len(item) == 2:
                    chars, jyutping = item
                    # jyutping can be None for unrecognized characters (punctuation, simplified Chinese)
                    if jyutping and isinstance(jyutping, str):
                        # Extract all tone digits from the jyutping string
                        # e.g., "zung1waa4" -> [1, 4], "syu1guk2" -> [1, 2]
                        tone_matches = re.findall(r'(\d)', jyutping)
                        for tone_str in tone_matches:
                            tone = int(tone_str)
                            if 1 <= tone <= 9:
                                tones.append(tone)

            return tones
        except Exception as e:
            print(f"Error extracting Cantonese tones: {e}")
            return []

    def _extract_vietnamese_tones(self, text: str) -> List[int]:
        """
        Extract tone numbers from Vietnamese text using diacritics.

        Vietnamese has 6 tones:
        1. Level (ngang): no diacritic (a, e, i, o, u, y)
        2. Acute (sắc): acute accent (á, é, í, ó, ú, ý)
        3. Grave (huyền): grave accent (à, è, ì, ò, ù, ỳ)
        4. Hook above (hỏi): hook (ả, ẻ, ỉ, ỏ, ủ, ỷ)
        5. Tilde (ngã): tilde (ã, ẽ, ĩ, õ, ũ, ỹ)
        6. Heavy (nặng): dot below (ạ, ệ, ị, ọ, ụ, ỵ)

        Args:
            text: Vietnamese text with diacritics

        Returns:
            List of tone numbers (1-6)
        """
        # Vietnamese tone diacritics mapping
        tone_map = {
            # Tone 1: Level (ngang) - no diacritic
            'a': 1, 'ă': 1, 'â': 1, 'e': 1, 'ê': 1, 'i': 1, 'o': 1, 'ô': 1, 'ơ': 1, 'u': 1, 'ư': 1, 'y': 1,
            'A': 1, 'Ă': 1, 'Â': 1, 'E': 1, 'Ê': 1, 'I': 1, 'O': 1, 'Ô': 1, 'Ơ': 1, 'U': 1, 'Ư': 1, 'Y': 1,

            # Tone 2: Acute (sắc)
            'á': 2, 'ắ': 2, 'ấ': 2, 'é': 2, 'ế': 2, 'í': 2, 'ó': 2, 'ố': 2, 'ớ': 2, 'ú': 2, 'ứ': 2, 'ý': 2,
            'Á': 2, 'Ắ': 2, 'Ấ': 2, 'É': 2, 'Ế': 2, 'Í': 2, 'Ó': 2, 'Ố': 2, 'Ớ': 2, 'Ú': 2, 'Ứ': 2, 'Ý': 2,

            # Tone 3: Grave (huyền)
            'à': 3, 'ằ': 3, 'ầ': 3, 'è': 3, 'ề': 3, 'ì': 3, 'ò': 3, 'ồ': 3, 'ờ': 3, 'ù': 3, 'ừ': 3, 'ỳ': 3,
            'À': 3, 'Ằ': 3, 'Ầ': 3, 'È': 3, 'Ề': 3, 'Ì': 3, 'Ò': 3, 'Ồ': 3, 'Ờ': 3, 'Ù': 3, 'Ừ': 3, 'Ỳ': 3,

            # Tone 4: Hook above (hỏi)
            'ả': 4, 'ẳ': 4, 'ẩ': 4, 'ẻ': 4, 'ể': 4, 'ỉ': 4, 'ỏ': 4, 'ổ': 4, 'ở': 4, 'ủ': 4, 'ử': 4, 'ỷ': 4,
            'Ả': 4, 'Ẳ': 4, 'Ẩ': 4, 'Ẻ': 4, 'Ể': 4, 'Ỉ': 4, 'Ỏ': 4, 'Ổ': 4, 'Ở': 4, 'Ủ': 4, 'Ử': 4, 'Ỷ': 4,

            # Tone 5: Tilde (ngã)
            'ã': 5, 'ẵ': 5, 'ẫ': 5, 'ẽ': 5, 'ễ': 5, 'ĩ': 5, 'õ': 5, 'ỗ': 5, 'ỡ': 5, 'ũ': 5, 'ữ': 5, 'ỹ': 5,
            'Ã': 5, 'Ẵ': 5, 'Ẫ': 5, 'Ẽ': 5, 'Ễ': 5, 'Ĩ': 5, 'Õ': 5, 'Ỗ': 5, 'Ỡ': 5, 'Ũ': 5, 'Ữ': 5, 'Ỹ': 5,

            # Tone 6: Heavy (nặng) - dot below
            'ạ': 6, 'ặ': 6, 'ậ': 6, 'ẹ': 6, 'ệ': 6, 'ị': 6, 'ọ': 6, 'ộ': 6, 'ợ': 6, 'ụ': 6, 'ự': 6, 'ỵ': 6,
            'Ạ': 6, 'Ặ': 6, 'Ậ': 6, 'Ẹ': 6, 'Ệ': 6, 'Ị': 6, 'Ọ': 6, 'Ộ': 6, 'Ợ': 6, 'Ụ': 6, 'Ự': 6, 'Ỵ': 6,
        }

        tones = []
        # Split text into words
        words = text.split()

        for word in words:
            # Find the vowel with tone mark in each word
            tone_found = False
            for char in word:
                if char in tone_map:
                    tones.append(tone_map[char])
                    tone_found = True
                    break  # Only count one tone per word

            # If no diacritic found but word contains vowel, it's tone 1 (level)
            if not tone_found:
                for char in word:
                    if char.lower() in 'aeiouy':
                        tones.append(1)
                        break

        return tones

    def calculate_semantic_similarity(self, ref_path: str, hyp_path: str, layer: int = 9) -> Optional[float]:
        """
        Calculate Semantic Similarity using WavLM embeddings.

        Extracts features from a specific layer and computes cosine similarity.

        Args:
            ref_path: Reference audio path
            hyp_path: Hypothesis (reconstructed) audio path
            layer: Which WavLM layer to extract (default: 9, can use 12 for deeper features)

        Returns:
            Cosine similarity score (0-1, higher is better)
        """
        if self.semantic_model is None or self.semantic_processor is None:
            # print("WavLM not loaded, skipping semantic similarity calculation")
            return None

        try:
            # 1. Load audio
            ref_audio, sr = self.load_audio_optimized(ref_path, sr=16000)
            hyp_audio, _ = self.load_audio_optimized(hyp_path, sr=16000)

            if ref_audio is None or hyp_audio is None:
                return None

            # Ensure audio is numpy array
            if isinstance(ref_audio, torch.Tensor):
                ref_audio = ref_audio.numpy()
            if isinstance(hyp_audio, torch.Tensor):
                hyp_audio = hyp_audio.numpy()

            # 2. Process audio to WavLM input format
            ref_inputs = self.semantic_processor(
                ref_audio,
                sampling_rate=16000,
                return_tensors="pt"
            )
            hyp_inputs = self.semantic_processor(
                hyp_audio,
                sampling_rate=16000,
                return_tensors="pt"
            )

            # Move to device
            ref_inputs = {k: v.to(self.device) for k, v in ref_inputs.items()}
            hyp_inputs = {k: v.to(self.device) for k, v in hyp_inputs.items()}

            # 3. Extract features from specified layer
            with torch.no_grad():
                ref_outputs = self.semantic_model(**ref_inputs, output_hidden_states=True)
                hyp_outputs = self.semantic_model(**hyp_inputs, output_hidden_states=True)

                # Extract layer features and average across time
                ref_features = ref_outputs.hidden_states[layer].mean(dim=1)  # [B, D]
                hyp_features = hyp_outputs.hidden_states[layer].mean(dim=1)

            # 4. Calculate cosine similarity
            import torch.nn.functional as F
            similarity = F.cosine_similarity(ref_features, hyp_features, dim=-1)

            return float(similarity.item())

        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ==================== BATCH PROCESSING FUNCTIONS ====================

    def calculate_vde_batch(self, file_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
        """
        Batch calculate VDE for multiple file pairs.

        Note: VDE requires pairwise comparison, so we process sequentially but
        leverage GPU acceleration for F0 extraction.
        """
        results = {}

        for ref_path, hyp_path in file_pairs:
            vde = self.calculate_vde(ref_path, hyp_path)
            results[(ref_path, hyp_path)] = vde

        return results

    def calculate_f0_metrics_batch(self, file_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Dict]:
        """
        Batch calculate F0-RMSE and GPE for multiple file pairs.
        """
        results = {}

        for ref_path, hyp_path in file_pairs:
            metrics = self.calculate_f0_metrics(ref_path, hyp_path)
            results[(ref_path, hyp_path)] = metrics

        return results

    def calculate_ter_batch(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Batch calculate TER for multiple text pairs.

        Args:
            text_pairs: List of (reference_text, hypothesis_text) tuples

        Returns:
            List of TER scores
        """
        results = []

        for ref_text, hyp_text in text_pairs:
            ter = self.calculate_ter(ref_text, hyp_text)
            results.append(ter)

        return results

    def calculate_semantic_similarity_batch(self, file_pairs: List[Tuple[str, str]], batch_size: int = 8) -> Dict[Tuple[str, str], float]:
        """
        Batch calculate semantic similarity for multiple file pairs.

        Note: WavLM can process batches, but we need to handle varying lengths.
        For simplicity, we process sequentially here.
        """
        results = {}

        for ref_path, hyp_path in file_pairs:
            similarity = self.calculate_semantic_similarity(ref_path, hyp_path)
            results[(ref_path, hyp_path)] = similarity

        return results

    # ==================== INHERITED V2 METHODS ====================
    # (Include all V2 methods for backward compatibility)

    def calculate_mos_quality(self, audio_path: str) -> Optional[float]:
        """Calculate MOS Quality using NISQA v2"""
        try:
            if self.mos_quality_model is None:
                return None

            self.mos_quality_model.args['deg'] = audio_path
            self.mos_quality_model._loadDatasets()

            if self.device.startswith('cuda'):
                self.mos_quality_model.model = self.mos_quality_model.model.to(self.device)
                self.mos_quality_model.dev = torch.device(self.device)

            results_df = self.mos_quality_model.predict()

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

            import tempfile
            import os

            df = pd.DataFrame({'deg': audio_paths})

            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                csv_path = f.name
                df.to_csv(csv_path, index=False)

            try:
                original_mode = self.mos_quality_model.args['mode']
                original_csv = self.mos_quality_model.args.get('csv_file', None)
                original_data_dir = self.mos_quality_model.args.get('data_dir', None)
                original_csv_deg = self.mos_quality_model.args.get('csv_deg', None)

                self.mos_quality_model.args['mode'] = 'predict_csv'
                self.mos_quality_model.args['data_dir'] = '/'
                self.mos_quality_model.args['csv_file'] = csv_path
                self.mos_quality_model.args['csv_deg'] = 'deg'

                self.mos_quality_model._loadDatasets()

                if self.device.startswith('cuda'):
                    self.mos_quality_model.model = self.mos_quality_model.model.to(self.device)
                    self.mos_quality_model.dev = torch.device(self.device)

                results_df = self.mos_quality_model.predict()

                self.mos_quality_model.args['mode'] = original_mode
                if original_csv is not None:
                    self.mos_quality_model.args['csv_file'] = original_csv
                if original_data_dir is not None:
                    self.mos_quality_model.args['data_dir'] = original_data_dir
                if original_csv_deg is not None:
                    self.mos_quality_model.args['csv_deg'] = original_csv_deg

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
                if os.path.exists(csv_path):
                    os.remove(csv_path)

        except Exception as e:
            print(f"Error in batch MOS Quality calculation (NISQA): {e}")
            import traceback
            traceback.print_exc()
            return {path: None for path in audio_paths}

    def calculate_mos_naturalness(self, audio_path: str) -> Optional[float]:
        """Calculate MOS Naturalness using UTMOS"""
        try:
            if self.mos_naturalness_model is None:
                return None

            wave, sr = self.load_audio_optimized(audio_path, sr=16000)

            if wave is None or len(wave) == 0:
                return None

            with torch.no_grad():
                wave_tensor = torch.from_numpy(wave).unsqueeze(0)

                if self.device.startswith('cuda'):
                    wave_tensor = wave_tensor.to(self.device)

                score = self.mos_naturalness_model(wave_tensor, sr)
                return float(score.item())

        except Exception as e:
            print(f"Error calculating MOS Naturalness: {e}")
            return None

    def calculate_mos_naturalness_batch(self, audio_paths: List[str], batch_size: int = 32) -> Dict[str, float]:
        """Calculate MOS Naturalness for multiple files using batch processing"""
        try:
            if self.mos_naturalness_model is None:
                return {path: None for path in audio_paths}

            results = {}

            for i in range(0, len(audio_paths), batch_size):
                batch_paths = audio_paths[i:i + batch_size]
                batch_waves = []
                valid_paths = []

                for path in batch_paths:
                    wave, sr = self.load_audio_optimized(path, sr=16000)
                    if wave is not None and len(wave) > 0:
                        batch_waves.append(torch.from_numpy(wave))
                        valid_paths.append(path)
                    else:
                        results[path] = None

                if len(batch_waves) == 0:
                    continue

                max_len = max(w.shape[0] for w in batch_waves)
                padded_waves = []
                for wave in batch_waves:
                    if wave.shape[0] < max_len:
                        padding = torch.zeros(max_len - wave.shape[0])
                        wave = torch.cat([wave, padding])
                    padded_waves.append(wave)

                batch_tensor = torch.stack(padded_waves)

                with torch.no_grad():
                    if self.device.startswith('cuda'):
                        batch_tensor = batch_tensor.to(self.device)

                    scores = self.mos_naturalness_model(batch_tensor, 16000)

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
            ref_audio, _ = self.load_audio_optimized(reference_path, sr=16000)
            deg_audio, _ = self.load_audio_optimized(degraded_path, sr=16000)

            if ref_audio is None or deg_audio is None:
                return None

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

    def calculate_stoi(self, reference_path: str, degraded_path: str) -> Optional[float]:
        """Calculate STOI score with optimized audio loading"""
        try:
            ref_audio, _ = self.load_audio_optimized(reference_path, sr=16000)
            deg_audio, _ = self.load_audio_optimized(degraded_path, sr=16000)

            if ref_audio is None or deg_audio is None:
                return None

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

    @staticmethod
    def _calculate_pesq_worker(args):
        """Worker function for parallel PESQ calculation"""
        reference_path, degraded_path = args
        try:
            import torchaudio
            import torchaudio.transforms as T
            import numpy as np
            from pesq import pesq

            ref_audio, ref_sr = torchaudio.load(reference_path)
            deg_audio, deg_sr = torchaudio.load(degraded_path)

            if ref_audio.shape[0] > 1:
                ref_audio = torch.mean(ref_audio, dim=0, keepdim=True)
            if deg_audio.shape[0] > 1:
                deg_audio = torch.mean(deg_audio, dim=0, keepdim=True)

            target_sr = 16000
            if ref_sr != target_sr:
                resampler = T.Resample(ref_sr, target_sr)
                ref_audio = resampler(ref_audio)
            if deg_sr != target_sr:
                resampler = T.Resample(deg_sr, target_sr)
                deg_audio = resampler(deg_audio)

            ref_audio = ref_audio.squeeze().numpy()
            deg_audio = deg_audio.squeeze().numpy()

            min_len = min(len(ref_audio), len(deg_audio))
            if min_len < target_sr * 0.1:
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

            ref_audio, ref_sr = torchaudio.load(reference_path)
            deg_audio, deg_sr = torchaudio.load(degraded_path)

            if ref_audio.shape[0] > 1:
                ref_audio = torch.mean(ref_audio, dim=0, keepdim=True)
            if deg_audio.shape[0] > 1:
                deg_audio = torch.mean(deg_audio, dim=0, keepdim=True)

            target_sr = 16000
            if ref_sr != target_sr:
                resampler = T.Resample(ref_sr, target_sr)
                ref_audio = resampler(ref_audio)
            if deg_sr != target_sr:
                resampler = T.Resample(deg_sr, target_sr)
                deg_audio = resampler(deg_audio)

            ref_audio = ref_audio.squeeze().numpy()
            deg_audio = deg_audio.squeeze().numpy()

            min_len = min(len(ref_audio), len(deg_audio))
            if min_len < target_sr * 0.1:
                return None

            ref_audio = ref_audio[:min_len]
            deg_audio = deg_audio[:min_len]

            return float(stoi(ref_audio, deg_audio, target_sr, extended=False))
        except Exception as e:
            return None

    def calculate_pesq_stoi_batch(self, file_pairs: List[Tuple[str, str]], num_workers: Optional[int] = None) -> Tuple[List, List]:
        """Batch calculation of PESQ and STOI using multiprocessing for CPU parallelization"""
        if num_workers is None:
            num_workers = min(mp.cpu_count() - 1, 8)

        print(f"Calculating PESQ and STOI with {num_workers} workers...")

        ctx = mp.get_context('spawn')
        with ctx.Pool(num_workers) as pool:
            pesq_results = pool.map(self._calculate_pesq_worker, file_pairs)
            stoi_results = pool.map(self._calculate_stoi_worker, file_pairs)

        return pesq_results, stoi_results

    def calculate_speaker_similarity(self, reference_path: str, test_path: str) -> Optional[float]:
        """Calculate speaker similarity using language-specific models"""
        try:
            if self.speaker_model is None:
                return None

            if self.language == 'en':
                import wespeaker

                ref_embedding = self.speaker_model.extract_embedding(reference_path)
                test_embedding = self.speaker_model.extract_embedding(test_path)

                similarity = 1.0 - cosine(ref_embedding, test_embedding)
                return float(similarity)

            elif self.language in ['zh', 'min', 'yue', 'vi']:
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
        """Extract WeSpeaker embeddings in batch mode (GPU accelerated)"""
        embeddings_dict = {}

        try:
            import torchaudio

            for i in range(0, len(audio_paths), batch_size):
                batch_paths = audio_paths[i:i + batch_size]
                batch_feats = []
                valid_paths = []

                for audio_path in batch_paths:
                    try:
                        pcm, sample_rate = torchaudio.load(audio_path, normalize=self.speaker_model.wavform_norm)

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

                        pcm = pcm.to(torch.float)
                        if sample_rate != self.speaker_model.resample_rate:
                            pcm = torchaudio.transforms.Resample(
                                orig_freq=sample_rate,
                                new_freq=self.speaker_model.resample_rate)(pcm)

                        feats = self.speaker_model.compute_fbank(pcm, sample_rate=self.speaker_model.resample_rate, cmn=True)
                        batch_feats.append(feats)
                        valid_paths.append(audio_path)

                    except Exception as e:
                        print(f"Error loading {audio_path}: {e}")
                        embeddings_dict[audio_path] = None
                        continue

                if len(batch_feats) == 0:
                    continue

                max_len = max(f.shape[0] for f in batch_feats)
                padded_feats = []
                for feats in batch_feats:
                    if feats.shape[0] < max_len:
                        padding = torch.zeros(max_len - feats.shape[0], feats.shape[1])
                        feats = torch.cat([feats, padding], dim=0)
                    padded_feats.append(feats)

                batch_tensor = torch.stack(padded_feats).to(self.device)

                with torch.no_grad():
                    outputs = self.speaker_model.model(batch_tensor)
                    outputs = outputs[-1] if isinstance(outputs, tuple) else outputs

                    for path, embedding in zip(valid_paths, outputs):
                        embeddings_dict[path] = embedding.cpu().numpy()

            return embeddings_dict

        except Exception as e:
            print(f"Error in batch WeSpeaker embedding extraction: {e}")
            import traceback
            traceback.print_exc()
            return {path: None for path in audio_paths}

    def calculate_speaker_similarity_batch(self, file_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
        """Calculate speaker similarity for multiple file pairs using batch processing"""
        try:
            if self.speaker_model is None:
                return {pair: None for pair in file_pairs}

            results = {}

            if self.language == 'en':
                import wespeaker

                unique_refs = list(set([pair[0] for pair in file_pairs]))
                unique_tests = list(set([pair[1] for pair in file_pairs]))

                ref_embeddings = self._extract_wespeaker_embeddings_batch(unique_refs, batch_size=32)
                test_embeddings = self._extract_wespeaker_embeddings_batch(unique_tests, batch_size=32)

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

            elif self.language in ['zh', 'min', 'yue', 'vi']:
                import torch
                import torch.nn.functional as F

                for ref_path, test_path in file_pairs:
                    try:
                        result = self.speaker_model.generate(
                            input=[ref_path, test_path],
                            data_type="sound"
                        )

                        if isinstance(result, list) and len(result) == 2:
                            emb1 = result[0].get('spk_embedding')
                            emb2 = result[1].get('spk_embedding')

                            if emb1 is not None and emb2 is not None:
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
        """Evaluate audio pair with all V3 metrics"""
        results = {}

        # ASR metrics
        asr_result = self.calculate_dwer_dcer(original_path, inference_path, ground_truth)
        if asr_result:
            results.update(asr_result)

        # Quality metrics (V2)
        results['MOS_Quality'] = self.calculate_mos_quality(inference_path)
        results['MOS_Naturalness'] = self.calculate_mos_naturalness(inference_path)
        results['pesq'] = self.calculate_pesq(original_path, inference_path)
        results['stoi'] = self.calculate_stoi(original_path, inference_path)
        results['speaker_similarity'] = self.calculate_speaker_similarity(original_path, inference_path)

        # V3 New Metrics
        if self.config['enable_vde']:
            results['vde'] = self.calculate_vde(original_path, inference_path)

        f0_metrics = self.calculate_f0_metrics(original_path, inference_path)
        if f0_metrics:
            results.update(f0_metrics)

        results['semantic_similarity'] = self.calculate_semantic_similarity(original_path, inference_path)

        # TER (requires ground truth text and ASR output)
        if self.config['enable_ter'] and asr_result:
            results['ter'] = self.calculate_ter(
                ground_truth,
                asr_result.get('inference_transcript_raw', '')
            )

        return results

    def cleanup_gpu_memory(self):
        """Clean up GPU memory cache"""
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
            if self.device.startswith('cuda'):
                mem_info = self.get_gpu_memory_info()
                print(f"GPU memory after cleanup: {mem_info['allocated']:.1f}GB allocated, {mem_info['free']:.1f}GB free")


# ==================== PHASE 2 PLACEHOLDER ====================

class CodebookPPLEvaluator:
    """
    TODO: Phase 2 implementation

    Evaluates codec token perplexity using a trained language model.
    Requires codec model interface with encode() method.
    """
    def __init__(self, num_codebooks=8, vocab_size=1024, device='cuda'):
        raise NotImplementedError("Phase 2: Codebook PPL evaluation not yet implemented")

    def calculate_ppl(self, codec_model, audio_paths):
        """Calculate perplexity on codec tokens"""
        raise NotImplementedError("Phase 2: PPL calculation requires codec model interface")


class TokenNMIEvaluator:
    """
    TODO: Phase 2 implementation

    Calculates Normalized Mutual Information between codec tokens and phoneme/tone labels.
    Requires Montreal Forced Aligner with Taiwanese acoustic model.
    """
    def __init__(self, mfa_model_path=None, device='cuda'):
        raise NotImplementedError("Phase 2: Token-NMI evaluation requires MFA Taiwanese model")

    def calculate_token_phoneme_nmi(self, codec_model, audio_path, text):
        """Calculate NMI between tokens and phonemes"""
        raise NotImplementedError("Phase 2: Requires frame-level phoneme alignment")

    def calculate_token_tone_nmi(self, codec_model, audio_path, text):
        """Calculate NMI between tokens and tones"""
        raise NotImplementedError("Phase 2: Requires frame-level tone alignment")
