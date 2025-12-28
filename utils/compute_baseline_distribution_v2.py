#!/usr/bin/env python3
"""
Speaker Similarity Baseline Distribution Calculator V2

使用與 metrics_evaluator_v2.py 相同的語言特定模型計算基線分布：
- English: ResNet3 (WeSpeaker)
- Chinese: CAM++ (FunASR)

計算資料集中正樣本（同speaker）和負樣本（不同speaker）的相似度分布
用於建立 codec 評估的基線標準
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns

# 設定隨機種子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class SpeakerSimilarityCalculatorV2:
    """使用語言特定模型計算 speaker similarity 的類別"""

    def __init__(self, language: str = 'en', gpu_id: int = 0, use_gpu: bool = True):
        """
        初始化 Speaker Similarity Calculator V2

        Args:
            language: 'en' for English (ResNet3), 'zh' for Chinese (CAM++)
            gpu_id: GPU ID to use (0, 1, 2, ...)
            use_gpu: 是否使用 GPU
        """
        self.language = language
        self.gpu_id = gpu_id
        self.use_gpu = use_gpu
        self.device = self._setup_device()
        self.speaker_model = None

        print(f"Initializing SpeakerSimilarityCalculatorV2")
        print(f"  Language: {self.language}")
        print(f"  Device: {self.device}")
        self._load_speaker_model()

    def _setup_device(self) -> str:
        """設定計算設備（GPU 或 CPU）"""
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

    def _load_speaker_model(self):
        """載入語言特定的 speaker embedding model"""
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
                # Chinese: CAM++ via FunASR
                print("Loading CAM++ for Chinese Speaker Similarity...")
                from funasr import AutoModel

                try:
                    # Use FunASR AutoModel for CAM++ with Hugging Face hub
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
            else:
                raise ValueError(f"Unsupported language: {self.language}")

            print(f"Speaker model loaded successfully")

        except Exception as e:
            print(f"Error loading speaker model: {e}")
            print("Please install required packages:")
            if self.language == 'en':
                print("  pip install wespeaker")
            elif self.language == 'zh':
                print("  pip install funasr modelscope")
            raise

    def calculate_similarity(self, audio_path1: str, audio_path2: str) -> Optional[float]:
        """
        計算兩個音檔的 speaker similarity
        使用與 metrics_evaluator_v2.py 相同的方法

        Args:
            audio_path1: 第一個音檔路徑
            audio_path2: 第二個音檔路徑

        Returns:
            Similarity score (0-1, higher = more similar)
        """
        try:
            if self.speaker_model is None:
                return None

            if self.language == 'en':
                # ResNet3 (WeSpeaker)
                # Extract embeddings
                ref_embedding = self.speaker_model.extract_embedding(audio_path1)
                test_embedding = self.speaker_model.extract_embedding(audio_path2)

                # Calculate cosine similarity
                similarity = 1.0 - cosine(ref_embedding, test_embedding)
                return float(similarity)

            elif self.language == 'zh':
                # CAM++ - Extract embeddings separately then calculate similarity
                # Using generate() method which returns a list with one dict containing 'spk_embedding'

                # Generate embeddings for both audio files
                emb1_result = self.speaker_model.generate(input=audio_path1)
                emb2_result = self.speaker_model.generate(input=audio_path2)

                # Extract embedding arrays from results
                # Result format: list with one dict: [{'spk_embedding': array}]
                def extract_embedding(result):
                    """Extract embedding from FunASR CAM++ result"""
                    if isinstance(result, list) and len(result) > 0:
                        result = result[0]  # Get first element

                    if isinstance(result, dict) and 'spk_embedding' in result:
                        emb = result['spk_embedding']
                        # Convert to numpy array if it's a tensor
                        if hasattr(emb, 'cpu'):
                            emb = emb.cpu().numpy()
                        return np.array(emb).flatten()

                    return None

                emb1 = extract_embedding(emb1_result)
                emb2 = extract_embedding(emb2_result)

                if emb1 is None or emb2 is None:
                    return None

                # Calculate cosine similarity
                similarity = 1.0 - cosine(emb1, emb2)
                return float(similarity)

        except Exception as e:
            print(f"Error calculating similarity for {audio_path1} and {audio_path2}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def cleanup(self):
        """清理 GPU 記憶體"""
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
            print(f"GPU memory cleaned")


class BaselineDistributionCalculatorV2:
    """計算正負樣本分布的主要類別 (V2)"""

    def __init__(
        self,
        csv_path: str,
        audio_base_path: str,
        dataset_name: str,
        language: str,
        output_dir: str,
        gpu_id: int = 0,
        num_negative_speakers: int = 5,
        max_positive_pairs: int = 20
    ):
        """
        初始化 Baseline Distribution Calculator V2

        Args:
            csv_path: CSV 檔案路徑
            audio_base_path: 音檔基礎目錄
            dataset_name: 資料集名稱 (commonvoice 或 librispeech)
            language: 語言 ('en' 或 'zh')
            output_dir: 輸出目錄
            gpu_id: GPU ID
            num_negative_speakers: 負樣本要選擇多少個其他 speaker
            max_positive_pairs: 每個 speaker 最多取多少對正樣本
        """
        self.csv_path = csv_path
        self.audio_base_path = audio_base_path
        self.dataset_name = dataset_name
        self.language = language
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gpu_id = gpu_id
        self.num_negative_speakers = num_negative_speakers
        self.max_positive_pairs = max_positive_pairs

        # 初始化 similarity calculator
        self.similarity_calc = SpeakerSimilarityCalculatorV2(
            language=language,
            gpu_id=gpu_id,
            use_gpu=True
        )

        # 資料結構
        self.df = None
        self.speaker_to_files = defaultdict(list)
        self.all_speakers = []

    def load_data(self):
        """載入 CSV 並建立 speaker -> files 的映射"""
        print(f"\nLoading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} records")

        # 根據資料集類型處理 speaker_id
        if self.dataset_name == 'librispeech':
            # LibriSpeech: 從 file_name 提取 speaker_id
            # 例如: 61-70968-0013 -> speaker_id = 61
            self.df['speaker_id'] = self.df['file_name'].str.split('-').str[0]
            print("Extracted speaker_id from file_name for LibriSpeech")
        elif self.dataset_name == 'commonvoice':
            # Common Voice: 直接使用 speaker_id 欄位
            if 'speaker_id' not in self.df.columns:
                raise ValueError("Common Voice CSV must have 'speaker_id' column")
            print("Using existing speaker_id column for Common Voice")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # 建立 speaker -> audio files 的映射
        for idx, row in self.df.iterrows():
            speaker_id = row['speaker_id']
            file_path = row['file_path']

            # 組合完整路徑
            full_path = os.path.join(self.audio_base_path, file_path)

            # 檢查檔案是否存在
            if os.path.exists(full_path):
                self.speaker_to_files[speaker_id].append(full_path)
            else:
                print(f"Warning: File not found: {full_path}")

        # 過濾掉只有 1 個或 0 個音檔的 speaker（無法計算正樣本）
        self.speaker_to_files = {
            spk: files for spk, files in self.speaker_to_files.items()
            if len(files) >= 2
        }

        self.all_speakers = list(self.speaker_to_files.keys())

        print(f"\nDataset Statistics:")
        print(f"  Total speakers with ≥2 files: {len(self.all_speakers)}")
        print(f"  Total audio files: {sum(len(files) for files in self.speaker_to_files.values())}")

        # 顯示音檔數量分布
        file_counts = [len(files) for files in self.speaker_to_files.values()]
        print(f"  Files per speaker - Min: {min(file_counts)}, Max: {max(file_counts)}, Mean: {np.mean(file_counts):.1f}")

    def compute_positive_samples(self) -> List[Dict]:
        """
        計算正樣本（同一個 speaker 的不同音檔）
        每個 speaker 最多取 max_positive_pairs 對

        Returns:
            List of dicts with keys: speaker_id, file1, file2, similarity
        """
        print(f"\n{'='*60}")
        print("Computing Positive Samples (Same Speaker)")
        print(f"{'='*60}")
        print(f"Max pairs per speaker: {self.max_positive_pairs}")

        positive_samples = []

        for speaker_id in tqdm(self.all_speakers, desc="Processing speakers"):
            files = self.speaker_to_files[speaker_id]

            # 生成所有可能的配對
            all_pairs = list(combinations(files, 2))

            # 限制配對數量
            if len(all_pairs) > self.max_positive_pairs:
                selected_pairs = random.sample(all_pairs, self.max_positive_pairs)
            else:
                selected_pairs = all_pairs

            # 計算每對的相似度
            for file1, file2 in selected_pairs:
                similarity = self.similarity_calc.calculate_similarity(file1, file2)

                if similarity is not None:
                    positive_samples.append({
                        'speaker_id': speaker_id,
                        'file1': file1,
                        'file2': file2,
                        'similarity': similarity
                    })

        print(f"Computed {len(positive_samples)} positive pairs from {len(self.all_speakers)} speakers")

        return positive_samples

    def compute_negative_samples(self) -> List[Dict]:
        """
        計算負樣本（不同 speaker 的音檔）
        對每個 speaker，隨機選擇 num_negative_speakers 個其他 speaker
        每個被選中的 speaker 選 1 個音檔

        Returns:
            List of dicts with keys: speaker_id1, speaker_id2, file1, file2, similarity
        """
        print(f"\n{'='*60}")
        print("Computing Negative Samples (Different Speakers)")
        print(f"{'='*60}")
        print(f"Negative speakers per target speaker: {self.num_negative_speakers}")

        negative_samples = []

        for speaker_id in tqdm(self.all_speakers, desc="Processing speakers"):
            # 取得當前 speaker 的一個音檔
            target_file = self.speaker_to_files[speaker_id][0]

            # 選擇其他 speaker
            other_speakers = [s for s in self.all_speakers if s != speaker_id]

            if len(other_speakers) < self.num_negative_speakers:
                selected_speakers = other_speakers
            else:
                selected_speakers = random.sample(other_speakers, self.num_negative_speakers)

            # 對每個被選中的 speaker，選一個音檔並計算相似度
            for other_speaker in selected_speakers:
                other_file = random.choice(self.speaker_to_files[other_speaker])

                similarity = self.similarity_calc.calculate_similarity(target_file, other_file)

                if similarity is not None:
                    negative_samples.append({
                        'speaker_id1': speaker_id,
                        'speaker_id2': other_speaker,
                        'file1': target_file,
                        'file2': other_file,
                        'similarity': similarity
                    })

        print(f"Computed {len(negative_samples)} negative pairs")

        return negative_samples

    def compute_statistics(self, samples: List[float]) -> Dict:
        """計算統計指標"""
        if not samples:
            return {}

        arr = np.array(samples)

        return {
            'count': len(samples),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'percentile_5': float(np.percentile(arr, 5)),
            'percentile_25': float(np.percentile(arr, 25)),
            'percentile_75': float(np.percentile(arr, 75)),
            'percentile_95': float(np.percentile(arr, 95))
        }

    def save_results(self, positive_samples: List[Dict], negative_samples: List[Dict]):
        """儲存結果到檔案"""
        print(f"\n{'='*60}")
        print("Saving Results")
        print(f"{'='*60}")

        # 提取相似度分數
        positive_sims = [s['similarity'] for s in positive_samples]
        negative_sims = [s['similarity'] for s in negative_samples]

        # 檢查是否有有效數據
        if not positive_sims:
            print("ERROR: No positive samples were successfully computed!")
            print("Please check if the audio files exist and the model is working correctly.")
            raise ValueError("No positive samples computed")

        if not negative_sims:
            print("WARNING: No negative samples were successfully computed!")
            print("Will proceed with positive samples only.")

        # 計算統計
        positive_stats = self.compute_statistics(positive_sims)
        negative_stats = self.compute_statistics(negative_sims) if negative_sims else {}

        # 1. 儲存統計結果為 JSON
        stats_dict = {
            'dataset': self.dataset_name,
            'language': self.language,
            'model': 'ResNet3 (WeSpeaker)' if self.language == 'en' else 'CAM++ (FunASR)',
            'csv_path': self.csv_path,
            'num_speakers': len(self.all_speakers),
            'max_positive_pairs': self.max_positive_pairs,
            'num_negative_speakers': self.num_negative_speakers,
            'positive_distribution': positive_stats,
            'negative_distribution': negative_stats
        }

        json_path = self.output_dir / f"baseline_statistics_{self.dataset_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved statistics to: {json_path}")

        # 2. 儲存正樣本原始資料為 CSV
        pos_df = pd.DataFrame(positive_samples)
        pos_csv_path = self.output_dir / f"positive_samples_{self.dataset_name}.csv"
        pos_df.to_csv(pos_csv_path, index=False)
        print(f"✓ Saved positive samples to: {pos_csv_path}")

        # 3. 儲存負樣本原始資料為 CSV
        neg_df = pd.DataFrame(negative_samples)
        neg_csv_path = self.output_dir / f"negative_samples_{self.dataset_name}.csv"
        neg_df.to_csv(neg_csv_path, index=False)
        print(f"✓ Saved negative samples to: {neg_csv_path}")

        # 4. 繪製分布圖
        if negative_sims:
            self.plot_distributions(positive_sims, negative_sims, positive_stats, negative_stats)
        else:
            print("Skipping distribution plot due to no negative samples")

        # 5. 生成文字報告
        if negative_sims:
            self.generate_report(positive_stats, negative_stats)
        else:
            self.generate_report_positive_only(positive_stats)

    def plot_distributions(
        self,
        positive_sims: List[float],
        negative_sims: List[float],
        positive_stats: Dict,
        negative_stats: Dict
    ):
        """繪製正負樣本分布圖"""
        plt.figure(figsize=(12, 6))

        # 設定風格
        sns.set_style("whitegrid")

        # 繪製直方圖
        plt.hist(positive_sims, bins=50, alpha=0.5, label='Positive (Same Speaker)',
                 color='green', edgecolor='black', density=True)
        plt.hist(negative_sims, bins=50, alpha=0.5, label='Negative (Different Speakers)',
                 color='red', edgecolor='black', density=True)

        # 標註統計線
        plt.axvline(positive_stats['mean'], color='green', linestyle='--',
                   linewidth=2, label=f"Pos Mean: {positive_stats['mean']:.3f}")
        plt.axvline(negative_stats['mean'], color='red', linestyle='--',
                   linewidth=2, label=f"Neg Mean: {negative_stats['mean']:.3f}")

        plt.axvline(positive_stats['percentile_5'], color='orange', linestyle=':',
                   linewidth=1.5, label=f"Pos 5th percentile: {positive_stats['percentile_5']:.3f}")

        model_name = 'ResNet3 (WeSpeaker)' if self.language == 'en' else 'CAM++ (FunASR)'
        plt.xlabel('Speaker Similarity', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'Speaker Similarity Distribution - {self.dataset_name.upper()}\n{model_name}',
                 fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)

        # 儲存圖表
        plot_path = self.output_dir / f"distribution_{self.dataset_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved distribution plot to: {plot_path}")

    def generate_report(self, positive_stats: Dict, negative_stats: Dict):
        """生成人類可讀的文字報告"""
        model_name = 'ResNet3 (WeSpeaker)' if self.language == 'en' else 'CAM++ (FunASR)'

        report_lines = [
            f"{'='*70}",
            f"Speaker Similarity Baseline Report V2 - {self.dataset_name.upper()}",
            f"{'='*70}",
            "",
            f"Dataset: {self.csv_path}",
            f"Language: {self.language}",
            f"Model: {model_name}",
            f"Number of speakers: {len(self.all_speakers)}",
            f"Max positive pairs per speaker: {self.max_positive_pairs}",
            f"Negative speakers sampled per speaker: {self.num_negative_speakers}",
            "",
            f"{'-'*70}",
            "POSITIVE SAMPLES (Same Speaker - Intra-speaker)",
            f"{'-'*70}",
            f"  Count: {positive_stats['count']:,} pairs",
            f"  Mean: {positive_stats['mean']:.4f} ± {positive_stats['std']:.4f}",
            f"  Median: {positive_stats['median']:.4f}",
            f"  Range: [{positive_stats['min']:.4f}, {positive_stats['max']:.4f}]",
            "",
            "  Percentiles:",
            f"    5th:  {positive_stats['percentile_5']:.4f}",
            f"    25th: {positive_stats['percentile_25']:.4f}",
            f"    75th: {positive_stats['percentile_75']:.4f}",
            f"    95th: {positive_stats['percentile_95']:.4f}",
            "",
            f"{'-'*70}",
            "NEGATIVE SAMPLES (Different Speakers - Inter-speaker)",
            f"{'-'*70}",
            f"  Count: {negative_stats['count']:,} pairs",
            f"  Mean: {negative_stats['mean']:.4f} ± {negative_stats['std']:.4f}",
            f"  Median: {negative_stats['median']:.4f}",
            f"  Range: [{negative_stats['min']:.4f}, {negative_stats['max']:.4f}]",
            "",
            "  Percentiles:",
            f"    5th:  {negative_stats['percentile_5']:.4f}",
            f"    25th: {negative_stats['percentile_25']:.4f}",
            f"    75th: {negative_stats['percentile_75']:.4f}",
            f"    95th: {negative_stats['percentile_95']:.4f}",
            "",
            f"{'='*70}",
            "SUGGESTED THRESHOLDS FOR CODEC EVALUATION",
            f"{'='*70}",
            f"  Conservative (Pos mean - 1*std): {positive_stats['mean'] - positive_stats['std']:.4f}",
            f"  Moderate (Pos mean - 2*std):     {positive_stats['mean'] - 2*positive_stats['std']:.4f}",
            f"  Percentile-based (Pos 5th):      {positive_stats['percentile_5']:.4f}",
            "",
            "Interpretation:",
            "  - Codec similarity should ideally be > Conservative threshold",
            "  - Values below Moderate threshold indicate significant degradation",
            "  - Compare codec mean similarity against these thresholds",
            f"{'='*70}",
        ]

        report_text = '\n'.join(report_lines)

        # 儲存報告
        report_path = self.output_dir / f"baseline_report_{self.dataset_name}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"✓ Saved report to: {report_path}")

        # 同時印出到終端
        print("\n" + report_text)

    def generate_report_positive_only(self, positive_stats: Dict):
        """生成僅包含正樣本的報告（當負樣本計算失敗時）"""
        model_name = 'ResNet3 (WeSpeaker)' if self.language == 'en' else 'CAM++ (FunASR)'

        report_lines = [
            f"{'='*70}",
            f"Speaker Similarity Baseline Report V2 - {self.dataset_name.upper()}",
            f"{'='*70}",
            "",
            f"Dataset: {self.csv_path}",
            f"Language: {self.language}",
            f"Model: {model_name}",
            f"Number of speakers: {len(self.all_speakers)}",
            f"Max positive pairs per speaker: {self.max_positive_pairs}",
            "",
            "WARNING: Negative samples calculation failed!",
            "This report contains only positive (same-speaker) samples.",
            "",
            f"{'-'*70}",
            "POSITIVE SAMPLES (Same Speaker - Intra-speaker)",
            f"{'-'*70}",
            f"  Count: {positive_stats['count']:,} pairs",
            f"  Mean: {positive_stats['mean']:.4f} ± {positive_stats['std']:.4f}",
            f"  Median: {positive_stats['median']:.4f}",
            f"  Range: [{positive_stats['min']:.4f}, {positive_stats['max']:.4f}]",
            "",
            "  Percentiles:",
            f"    5th:  {positive_stats['percentile_5']:.4f}",
            f"    25th: {positive_stats['percentile_25']:.4f}",
            f"    75th: {positive_stats['percentile_75']:.4f}",
            f"    95th: {positive_stats['percentile_95']:.4f}",
            "",
            f"{'='*70}",
            "SUGGESTED THRESHOLDS FOR CODEC EVALUATION",
            f"{'='*70}",
            f"  Conservative (Pos mean - 1*std): {positive_stats['mean'] - positive_stats['std']:.4f}",
            f"  Moderate (Pos mean - 2*std):     {positive_stats['mean'] - 2*positive_stats['std']:.4f}",
            f"  Percentile-based (Pos 5th):      {positive_stats['percentile_5']:.4f}",
            "",
            "Interpretation:",
            "  - Codec similarity should ideally be > Conservative threshold",
            "  - Values below Moderate threshold indicate significant degradation",
            "  - Compare codec mean similarity against these thresholds",
            f"{'='*70}",
        ]

        report_text = '\n'.join(report_lines)

        # 儲存報告
        report_path = self.output_dir / f"baseline_report_{self.dataset_name}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"✓ Saved report to: {report_path}")

        # 同時印出到終端
        print("\n" + report_text)

    def run(self):
        """執行完整的計算流程"""
        print(f"\n{'='*70}")
        print(f"Starting Baseline Distribution Calculation V2")
        print(f"Dataset: {self.dataset_name}")
        print(f"Language: {self.language}")
        print(f"GPU: {self.gpu_id}")
        print(f"{'='*70}")

        # 1. 載入資料
        self.load_data()

        # 2. 計算正樣本
        positive_samples = self.compute_positive_samples()

        # 3. 計算負樣本
        negative_samples = self.compute_negative_samples()

        # 4. 儲存結果
        self.save_results(positive_samples, negative_samples)

        # 5. 清理 GPU 記憶體
        self.similarity_calc.cleanup()

        print(f"\n{'='*70}")
        print("✓ Baseline distribution calculation V2 completed successfully!")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate speaker similarity baseline distribution V2 for codec evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Common Voice (中文)
  python compute_baseline_distribution_v2.py \\
      --dataset commonvoice \\
      --language zh \\
      --csv_path /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/common_voice_zh_CN_train_filtered.csv \\
      --audio_base_path /mnt/Internal/ASR \\
      --output_dir /home/jieshiang/Desktop/GitHub/Codec_comparison/common_voice_spk_sim_result \\
      --gpu_id 0

  # LibriSpeech (英文)
  python compute_baseline_distribution_v2.py \\
      --dataset librispeech \\
      --language en \\
      --csv_path /home/jieshiang/Desktop/GitHub/Codec_comparison/csv/librispeech_test_clean_filtered.csv \\
      --audio_base_path /mnt/Internal/ASR \\
      --output_dir /home/jieshiang/Desktop/GitHub/Codec_comparison/librispeech_spk_sim_result \\
      --gpu_id 1
        """
    )

    parser.add_argument('--dataset', type=str, required=True, choices=['commonvoice', 'librispeech'],
                       help='Dataset type (commonvoice or librispeech)')
    parser.add_argument('--language', type=str, required=True, choices=['en', 'zh'],
                       help='Language (en for English, zh for Chinese)')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV file with audio metadata')
    parser.add_argument('--audio_base_path', type=str, required=True,
                       help='Base path for audio files (e.g., /mnt/Internal/ASR)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use (default: 0)')
    parser.add_argument('--num_negative_speakers', type=int, default=5,
                       help='Number of other speakers to sample for negative pairs (default: 5)')
    parser.add_argument('--max_positive_pairs', type=int, default=20,
                       help='Maximum positive pairs per speaker (default: 20)')

    args = parser.parse_args()

    # 執行計算
    calculator = BaselineDistributionCalculatorV2(
        csv_path=args.csv_path,
        audio_base_path=args.audio_base_path,
        dataset_name=args.dataset,
        language=args.language,
        output_dir=args.output_dir,
        gpu_id=args.gpu_id,
        num_negative_speakers=args.num_negative_speakers,
        max_positive_pairs=args.max_positive_pairs
    )

    calculator.run()


if __name__ == "__main__":
    main()
