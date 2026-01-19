#!/usr/bin/env python3
"""
從 *_filtered_clean.csv 計算 metrics 並產生視覺化報告

此腳本的工作流程：
1. 讀取 *_filtered_clean.csv（只包含 file_name, file_path, duration, transcription, speaker_id）
2. 使用 metrics_evaluator_v3.py 計算 WER/CER、MOS_Quality、MOS_Naturalness
3. 產生完整的 *_filtered.csv（包含所有 metrics）
4. 產生分佈圖（CER/WER、MOS_Quality、MOS_Naturalness）
5. 產生統計報告

使用範例：
    # 處理 Hokkien 資料集 (新加入)
    python compute_metrics_from_clean.py --dataset hokkien --gpu_id 0

    # 處理 AISHELL 資料集
    python compute_metrics_from_clean.py --dataset aishell --gpu_id 0

    # 處理 MinSpeech 資料集
    python compute_metrics_from_clean.py --dataset minspeech --gpu_id 0

    # 處理 LibriSpeech 資料集
    python compute_metrics_from_clean.py --dataset librispeech --gpu_id 0

    # 使用 CPU（較慢）
    python compute_metrics_from_clean.py --dataset aishell --use_cpu
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings

# 加入 metrics evaluator v3 的路徑
sys.path.insert(0, '/home/jieshiang/Desktop/GitHub/Codec_comparison')
from metrics_evaluator_v3 import AudioMetricsEvaluatorV3

warnings.filterwarnings('ignore')

# Matplotlib 中文設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MetricsComputer:
    """從 clean CSV 計算 metrics 並產生報告"""

    def __init__(self, dataset_name, gpu_id=0, use_cpu=False, batch_size=32, taiwanese_asr_model=None):
        self.dataset_name = dataset_name.lower()
        self.gpu_id = gpu_id
        self.use_cpu = use_cpu
        self.batch_size = batch_size
        self.taiwanese_asr_model = taiwanese_asr_model

        # 資料集設定（使用 metrics_evaluator_v3）
        # metrics 選項: 'asr' (包含 cer/wer), 'mos_quality', 'mos_naturalness'
        self.configs = {
            'aishell': {
                'language': 'zh',
                'dataset': 'aishell',
                'metric_name': 'cer',
                'metric_label': 'Character Error Rate (CER)',
                'base_path': '/mnt/Internal/ASR',
                'clean_csv': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/aishell_filtered_clean.csv',
                'output_dir': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/aishell',
                'metrics': ['asr', 'mos_quality', 'mos_naturalness'],
            },
            'commonvoice': {
                'language': 'zh',
                'dataset': 'commonvoice',
                'metric_name': 'cer',
                'metric_label': 'Character Error Rate (CER)',
                'base_path': '/mnt/Internal/ASR',
                'clean_csv': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/commonvoice_filtered_clean.csv',
                'output_dir': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/commonvoice',
                'metrics': ['asr', 'mos_quality', 'mos_naturalness'],
            },
            'librispeech': {
                'language': 'en',
                'dataset': 'librispeech',
                'metric_name': 'wer',
                'metric_label': 'Word Error Rate (WER)',
                'base_path': '/mnt/Internal/ASR',
                'clean_csv': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/librispeech_filtered_clean.csv',
                'output_dir': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/librispeech',
                'metrics': ['asr', 'mos_quality', 'mos_naturalness'],
            },
            'minspeech': {
                'language': 'min',
                'dataset': 'minspeech',
                'metric_name': 'cer',
                'metric_label': 'Character Error Rate (CER)',
                'base_path': '/mnt/Internal/ASR',
                'clean_csv': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/minspeech_filtered_clean.csv',
                'output_dir': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/minspeech',
                'metrics': ['asr', 'mos_quality', 'mos_naturalness', 'ter'],
            },
            'hokkien': {
                'language': 'min',
                'dataset': 'hokkien',
                'metric_name': 'cer',
                'metric_label': 'Character Error Rate (CER)',
                'base_path': '/mnt/Internal/ASR',
                #'clean_csv': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/hokkien_filtered_clean.csv',
                'clean_csv': '/mnt/Internal/ASR/hokkien/hokkien_clean.csv',
                'output_dir': '/home/jieshiang/Desktop/GitHub/Codec_comparison/csv/analysis/hokkien_full',
                'metrics': ['asr', 'mos_quality', 'mos_naturalness', 'ter'],
            }
        }

        if self.dataset_name not in self.configs:
            raise ValueError(f"未知的資料集: {dataset_name}. 請選擇: {list(self.configs.keys())}")

        self.config = self.configs[self.dataset_name]
        self.output_dir = Path(self.config['output_dir'])
        self.plot_dir = self.output_dir / 'plot'

        # 建立輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def load_clean_csv(self):
        """載入 clean CSV 檔案"""
        clean_csv_path = self.config['clean_csv']

        if not os.path.exists(clean_csv_path):
            raise FileNotFoundError(f"找不到 clean CSV 檔案: {clean_csv_path}")

        df = pd.read_csv(clean_csv_path)
        print(f"✓ 已載入 {len(df)} 個樣本，來源: {clean_csv_path}")

        # 驗證必要欄位
        required_columns = ['file_name', 'file_path', 'duration', 'transcription', 'speaker_id']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"CSV 缺少必要欄位: {missing_columns}")

        return df

    def build_full_audio_paths(self, df):
        """建立完整的音檔路徑"""
        base_path = self.config['base_path']
        full_paths = []

        for _, row in df.iterrows():
            relative_path = row['file_path']
            # 移除開頭的 './'
            if relative_path.startswith('./'):
                relative_path = relative_path[2:]

            full_path = os.path.join(base_path, relative_path)
            full_paths.append(full_path)

        return full_paths

    def calculate_error_rate(self, evaluator, asr_transcript, ground_truth):
        """計算 WER 或 CER（根據語言）- 與 analyze_dataset.py 相同的處理方式"""
        try:
            if not asr_transcript or not ground_truth:
                return None

            # 轉換和正規化（與 analyze_dataset.py 相同）
            if self.config['language'] in ['zh', 'min']:
                asr_simplified = evaluator.convert_traditional_to_simplified(asr_transcript)
                gt_simplified = evaluator.convert_traditional_to_simplified(ground_truth)
                asr_norm = evaluator.normalize_text(asr_simplified)
                gt_norm = evaluator.normalize_text(gt_simplified)
                return evaluator.fast_cer(gt_norm, asr_norm)
            else:
                asr_norm = evaluator.normalize_text(asr_transcript)
                gt_norm = evaluator.normalize_text(ground_truth)
                return evaluator.fast_wer(gt_norm, asr_norm)

        except Exception as e:
            print(f"計算錯誤率時發生錯誤: {e}")
            return None

    def calculate_metrics(self, evaluator, df, audio_paths):
        """計算所有 metrics（根據 config 中的 metrics 設定）"""
        results = []
        transcriptions = df['transcription'].tolist()
        enabled_metrics = self.config['metrics']

        print(f"\n總共要處理的檔案數: {len(audio_paths)}")
        print(f"要計算的 metrics: {', '.join(enabled_metrics)}")

        # 初始化結果字典
        asr_transcripts = [None] * len(audio_paths)
        error_scores = [None] * len(audio_paths)
        mos_quality_scores = [None] * len(audio_paths)
        mos_naturalness_scores = [None] * len(audio_paths)
        pesq_scores = [None] * len(audio_paths)
        stoi_scores = [None] * len(audio_paths)
        speaker_similarity_scores = [None] * len(audio_paths)
        vde_scores = [None] * len(audio_paths)
        f0_rmse_scores = [None] * len(audio_paths)
        gpe_scores = [None] * len(audio_paths)
        ter_scores = [None] * len(audio_paths)
        semantic_similarity_scores = [None] * len(audio_paths)

        metric_count = len(enabled_metrics)
        current_metric = 0

        # === 1. ASR & Error Rate ===
        if 'asr' in enabled_metrics:
            current_metric += 1
            metric_name = self.config['metric_name'].upper()
            print(f"\n=== [{current_metric}/{metric_count}] 執行 ASR 並計算 {metric_name} ===")

            # Batch transcribe with progress bar
            print("批次轉錄音訊檔案...")
            asr_transcripts = evaluator.batch_transcribe(audio_paths)

            error_scores = []
            for gt_transcript, asr_transcript in zip(transcriptions, asr_transcripts):
                error = self.calculate_error_rate(evaluator, asr_transcript, gt_transcript)
                error_scores.append(error)

            valid_errors = [x for x in error_scores if x is not None]
            if valid_errors:
                print(f"完成 {metric_name}: {len(valid_errors)}/{len(error_scores)} 個有效分數, 平均值: {np.mean(valid_errors):.4f}")

        # === 2. MOS Quality ===
        if 'mos_quality' in enabled_metrics:
            current_metric += 1
            print(f"\n=== [{current_metric}/{metric_count}] 計算 MOS Quality (NISQA v2) ===")
            mos_quality_dict = evaluator.calculate_mos_quality_batch(audio_paths)
            mos_quality_scores = [mos_quality_dict.get(path) for path in audio_paths]

            valid_mos_q = [x for x in mos_quality_scores if x is not None]
            if valid_mos_q:
                print(f"完成 MOS Quality: {len(valid_mos_q)}/{len(mos_quality_scores)} 個有效分數, 平均值: {np.mean(valid_mos_q):.4f}")

        # === 3. MOS Naturalness ===
        if 'mos_naturalness' in enabled_metrics:
            current_metric += 1
            print(f"\n=== [{current_metric}/{metric_count}] 計算 MOS Naturalness (UTMOS) ===")
            mos_naturalness_dict = evaluator.calculate_mos_naturalness_batch(audio_paths, batch_size=self.batch_size)
            mos_naturalness_scores = [mos_naturalness_dict.get(path) for path in audio_paths]

            valid_mos_n = [x for x in mos_naturalness_scores if x is not None]
            if valid_mos_n:
                print(f"完成 MOS Naturalness: {len(valid_mos_n)}/{len(mos_naturalness_scores)} 個有效分數, 平均值: {np.mean(valid_mos_n):.4f}")

        # === 4. PESQ ===
        if 'pesq' in enabled_metrics:
            current_metric += 1
            print(f"\n=== [{current_metric}/{metric_count}] 計算 PESQ ===")
            print("註：PESQ 需要 reference audio，clean dataset 無法計算，跳過")
            pesq_scores = [None] * len(audio_paths)

        # === 5. STOI ===
        if 'stoi' in enabled_metrics:
            current_metric += 1
            print(f"\n=== [{current_metric}/{metric_count}] 計算 STOI ===")
            print("註：STOI 需要 reference audio，clean dataset 無法計算，跳過")
            stoi_scores = [None] * len(audio_paths)

        # === 6. Speaker Similarity ===
        if 'speaker_similarity' in enabled_metrics:
            current_metric += 1
            print(f"\n=== [{current_metric}/{metric_count}] 計算 Speaker Similarity ===")
            print("註：Speaker Similarity 需要 reference audio，clean dataset 無法計算，跳過")
            speaker_similarity_scores = [None] * len(audio_paths)

        # === 7. VDE (Voicing Decision Error) ===
        if 'vde' in enabled_metrics:
            current_metric += 1
            print(f"\n=== [{current_metric}/{metric_count}] 計算 VDE (Voicing Decision Error) ===")
            print("註：VDE 需要 reference audio，clean dataset 無法計算，跳過")
            vde_scores = [None] * len(audio_paths)

        # === 8. F0-RMSE ===
        if 'f0_rmse' in enabled_metrics:
            current_metric += 1
            print(f"\n=== [{current_metric}/{metric_count}] 計算 F0-RMSE ===")
            print("註：F0-RMSE 需要 reference audio，clean dataset 無法計算，跳過")
            f0_rmse_scores = [None] * len(audio_paths)

        # === 9. GPE (Gross Pitch Error) ===
        if 'gpe' in enabled_metrics:
            current_metric += 1
            print(f"\n=== [{current_metric}/{metric_count}] 計算 GPE (Gross Pitch Error) ===")
            print("註：GPE 需要 reference audio，clean dataset 無法計算，跳過")
            gpe_scores = [None] * len(audio_paths)

        # === 10. TER (Tone Error Rate) ===
        if 'ter' in enabled_metrics:
            current_metric += 1
            print(f"\n=== [{current_metric}/{metric_count}] 計算 TER (Tone Error Rate) ===")

            if self.config['language'] in ['zh', 'min']:
                print("計算台語/中文聲調錯誤率...")
                ter_scores = []
                for gt_transcript, asr_transcript in tqdm(zip(transcriptions, asr_transcripts), total=len(transcriptions), desc="TER"):
                    if asr_transcript and gt_transcript:
                        ter = evaluator.calculate_ter(gt_transcript, asr_transcript)
                        ter_scores.append(ter)
                    else:
                        ter_scores.append(None)

                valid_ter = [x for x in ter_scores if x is not None]
                if valid_ter:
                    print(f"完成 TER: {len(valid_ter)}/{len(ter_scores)} 個有效分數, 平均值: {np.mean(valid_ter):.4f}")
            else:
                print("TER 僅支援中文和台語，跳過")
                ter_scores = [None] * len(audio_paths)

        # === 11. Semantic Similarity ===
        if 'semantic_similarity' in enabled_metrics:
            current_metric += 1
            print(f"\n=== [{current_metric}/{metric_count}] 計算 Semantic Similarity (WavLM) ===")
            print("註：Semantic Similarity 需要 reference audio，clean dataset 無法計算，跳過")
            semantic_similarity_scores = [None] * len(audio_paths)

        # 合併結果
        print("\n=== 合併結果 ===")
        for i, row in df.iterrows():
            result_dict = {
                'file_name': row['file_name'],
                'file_path': row['file_path'],
                'duration': row['duration'],
                'transcription': row['transcription'],
                'speaker_id': row['speaker_id'],
            }

            # 只加入有計算的 metrics
            if 'asr' in enabled_metrics:
                result_dict['asr_result'] = asr_transcripts[i] if i < len(asr_transcripts) else None
                result_dict[self.config['metric_name']] = error_scores[i]

            if 'mos_quality' in enabled_metrics:
                result_dict['MOS_Quality'] = mos_quality_scores[i]

            if 'mos_naturalness' in enabled_metrics:
                result_dict['MOS_Naturalness'] = mos_naturalness_scores[i]

            if 'pesq' in enabled_metrics:
                result_dict['PESQ'] = pesq_scores[i]

            if 'stoi' in enabled_metrics:
                result_dict['STOI'] = stoi_scores[i]

            if 'speaker_similarity' in enabled_metrics:
                result_dict['Speaker_Similarity'] = speaker_similarity_scores[i]

            if 'vde' in enabled_metrics:
                result_dict['VDE'] = vde_scores[i]

            if 'f0_rmse' in enabled_metrics:
                result_dict['F0_RMSE'] = f0_rmse_scores[i]

            if 'gpe' in enabled_metrics:
                result_dict['GPE'] = gpe_scores[i]

            if 'ter' in enabled_metrics:
                result_dict['TER'] = ter_scores[i]

            if 'semantic_similarity' in enabled_metrics:
                result_dict['Semantic_Similarity'] = semantic_similarity_scores[i]

            results.append(result_dict)

        return results

    def create_distribution_plot(self, data, metric_name, output_path, xlabel=None, title=None):
        """建立分佈直方圖（包含平均值線）"""
        data = [x for x in data if x is not None and not np.isnan(x)]

        if len(data) == 0:
            print(f"⚠ {metric_name} 沒有有效資料，跳過圖表生成")
            return

        mean_val = np.mean(data)

        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'平均值: {mean_val:.4f}')

        plt.xlabel(xlabel or metric_name, fontsize=12)
        plt.ylabel('頻率', fontsize=12)
        plt.title(title or f'{metric_name} 分佈', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 已儲存圖表: {output_path}")

    def run(self):
        """執行完整的處理流程"""
        print("="*80)
        print(f"從 Clean CSV 計算 Metrics: {self.dataset_name.upper()}")
        print("="*80)
        print(f"語言: {self.config['language']}")
        print(f"資料集: {self.config['dataset']}")
        print(f"指標: {self.config['metric_name'].upper()}")
        print(f"GPU ID: {self.gpu_id}")
        print(f"使用 CPU: {self.use_cpu}")
        print(f"批次大小: {self.batch_size}")
        print(f"輸出目錄: {self.output_dir}")
        print("="*80)

        # [1/4] 載入 clean CSV
        print("\n[1/4] 載入 clean CSV...")
        df_clean = self.load_clean_csv()
        print(f"找到 {len(df_clean)} 個樣本")
        print(f"共 {df_clean['speaker_id'].nunique()} 位說話者")

        # 建立完整音檔路徑
        audio_paths = self.build_full_audio_paths(df_clean)

        # [2/4] 載入模型（使用 metrics_evaluator_v3）
        print("\n[2/4] 載入模型...")
        if self.taiwanese_asr_model:
            print(f"指定使用台語 ASR 模型: {self.taiwanese_asr_model}")

        evaluator = AudioMetricsEvaluatorV3(
            language=self.config['language'],
            dataset=self.config['dataset'],
            device=None,
            use_gpu=not self.use_cpu,
            gpu_id=self.gpu_id,
            taiwanese_asr_model=self.taiwanese_asr_model
        )
        evaluator.load_models()
        print(f"✓ 已載入 AudioMetricsEvaluatorV3 (語言: {self.config['language']}, 資料集: {self.config['dataset']})")

        # [3/4] 計算 metrics
        print("\n[3/4] 計算 metrics...")
        results = self.calculate_metrics(evaluator, df_clean, audio_paths)

        # 儲存完整的 filtered CSV
        df_results = pd.DataFrame(results)
        filtered_csv_path = self.output_dir / f"{self.dataset_name}_filtered.csv"
        df_results.to_csv(filtered_csv_path, index=False, encoding='utf-8')
        print(f"\n✓ 已儲存完整結果: {filtered_csv_path}")

        # [4/4] 產生視覺化和統計報告
        print("\n[4/4] 產生視覺化和統計報告...")

        # 計算摘要統計
        spk_num = df_results['speaker_id'].nunique()
        avg_utts = len(df_results) / spk_num

        # 只計算有啟用的 metrics
        enabled_metrics = self.config['metrics']
        metric_name = self.config['metric_name']
        metric_means = {}

        # 根據啟用的 metrics 計算平均值
        metrics_to_summarize = []
        if 'asr' in enabled_metrics:
            metrics_to_summarize.append(metric_name)
        if 'mos_quality' in enabled_metrics:
            metrics_to_summarize.append('MOS_Quality')
        if 'mos_naturalness' in enabled_metrics:
            metrics_to_summarize.append('MOS_Naturalness')

        for metric in metrics_to_summarize:
            if metric in df_results.columns:
                values = df_results[metric].dropna()
                if len(values) > 0:
                    metric_means[f'{metric}_mean'] = values.mean()
                else:
                    metric_means[f'{metric}_mean'] = None

        # 產生摘要 CSV
        summary_data = {
            'base_path': [self.config['base_path']],
            'spk_num': [spk_num],
            'avg_utts': [avg_utts],
            **{k: [v] for k, v in metric_means.items()}
        }

        df_summary = pd.DataFrame(summary_data)
        summary_csv_path = self.output_dir / f"{self.dataset_name}_summary.csv"
        df_summary.to_csv(summary_csv_path, index=False, encoding='utf-8')
        print(f"✓ 已儲存摘要結果: {summary_csv_path}")

        # 列印摘要統計
        print("\n" + "="*80)
        print("摘要統計")
        print("="*80)
        print(f"資料集: {self.dataset_name.upper()}")
        print(f"基礎路徑: {self.config['base_path']}")
        print(f"總說話者數: {spk_num}")
        print(f"總語句數: {len(df_results)}")
        print(f"每位說話者平均語句數: {avg_utts:.2f}")
        print(f"總時長: {df_results['duration'].sum()/3600:.2f} 小時")
        print(f"\n指標平均值:")
        for metric, value in metric_means.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: N/A")

        # 產生分佈圖（只產生有啟用的 metrics）
        print("\n=== 產生分佈圖 ===")
        plot_configs = []

        if 'asr' in enabled_metrics:
            plot_configs.append((
                metric_name,
                f"{'Word Error Rate (WER)' if metric_name == 'wer' else 'Character Error Rate (CER)'}",
                f"{metric_name.upper()} 分佈 - {self.dataset_name.upper()}"
            ))

        if 'mos_quality' in enabled_metrics:
            plot_configs.append((
                'MOS_Quality',
                'MOS Quality Score',
                f'MOS Quality 分佈 - {self.dataset_name.upper()}'
            ))

        if 'mos_naturalness' in enabled_metrics:
            plot_configs.append((
                'MOS_Naturalness',
                'MOS Naturalness Score',
                f'MOS Naturalness 分佈 - {self.dataset_name.upper()}'
            ))

        for metric, xlabel, title in plot_configs:
            if metric in df_results.columns:
                output_path = self.plot_dir / f"{metric}_distribution_filtered.png"
                self.create_distribution_plot(
                    df_results[metric].values,
                    metric,
                    output_path,
                    xlabel=xlabel,
                    title=title
                )

        print("\n" + "="*80)
        print("處理完成！")
        print("="*80)
        print(f"\n輸出檔案:")
        print(f"  - 完整 CSV: {filtered_csv_path}")
        print(f"  - 摘要 CSV: {summary_csv_path}")
        print(f"  - 圖表: {self.plot_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='從 *_filtered_clean.csv 計算 metrics 並產生報告',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 處理 Hokkien 資料集
  python compute_metrics_from_clean.py --dataset hokkien --gpu_id 0

  # 處理 AISHELL 資料集
  python compute_metrics_from_clean.py --dataset aishell --gpu_id 0

  # 處理 MinSpeech 資料集
  python compute_metrics_from_clean.py --dataset minspeech --gpu_id 1

  # 使用 CPU
  python compute_metrics_from_clean.py --dataset aishell --use_cpu
        """
    )

    parser.add_argument('--dataset', type=str, required=True,
                       choices=['aishell', 'commonvoice', 'librispeech', 'minspeech', 'hokkien'],
                       help='要處理的資料集')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID (預設: 0)')
    parser.add_argument('--use_cpu', action='store_true',
                       help='使用 CPU 而非 GPU')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='MOS Naturalness 的批次大小 (預設: 32)')
    parser.add_argument('--taiwanese_asr_model', type=str, default=None,
                       choices=['tsukilen', 'whisper-large-v3'],
                       help='指定台語 ASR 模型 (預設: TSukiLen，失敗時回退到 Whisper-large-v3)')

    args = parser.parse_args()

    computer = MetricsComputer(
        dataset_name=args.dataset,
        gpu_id=args.gpu_id,
        use_cpu=args.use_cpu,
        batch_size=args.batch_size,
        taiwanese_asr_model=args.taiwanese_asr_model
    )

    computer.run()


if __name__ == "__main__":
    main()