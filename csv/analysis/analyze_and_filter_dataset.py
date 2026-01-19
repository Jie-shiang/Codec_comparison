#!/usr/bin/env python3
"""
Dataset Analysis and Filtering Tool
分析數據集的指標分布，過濾異常值，生成乾淨的 train/test 數據集
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """數據集分析和過濾工具"""

    def __init__(self, csv_path: str, language: str):
        """
        初始化分析器

        Args:
            csv_path: CSV 文件路徑
            language: 語言類型 ('vi' for Vietnamese, 'yue' for Cantonese)
        """
        self.csv_path = Path(csv_path)
        self.language = language
        self.df = None
        self.stats = {}

        # 根據語言定義指標列名
        if language == 'vi':
            self.error_metric = 'wer'
            self.dataset_name = 'vivos'
        elif language == 'yue':
            self.error_metric = 'cer'
            self.dataset_name = 'mdcc'
        else:
            raise ValueError(f"Unsupported language: {language}")

        # 定義要分析的指標（調整為更寬鬆的閾值）
        self.metrics = {
            'duration': {'min': 1.0, 'max': 15.0, 'type': 'duration'},
            self.error_metric: {'min': 0.0, 'max': 1.5, 'type': 'error'},  # 允許 WER/CER <= 1.5
            'MOS_Quality': {'min': 2.0, 'max': 5.0, 'type': 'quality'},
            'MOS_Naturalness': {'min': 1.5, 'max': 4.5, 'type': 'quality'},
            'TER': {'min': 0.0, 'max': 0.5, 'type': 'error'}  # 允許 TER <= 0.5
        }

    def load_data(self) -> pd.DataFrame:
        """載入 CSV 數據"""
        logger.info(f"Loading data from {self.csv_path}")
        # 使用 quoting 參數正確處理 CSV 中的引號
        self.df = pd.read_csv(self.csv_path, quoting=1)  # QUOTE_ALL
        logger.info(f"Loaded {len(self.df)} records")
        logger.info(f"Columns: {list(self.df.columns)}")

        # 確保數值列是正確的數值類型
        numeric_cols = ['duration', self.error_metric, 'MOS_Quality', 'MOS_Naturalness', 'TER']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        logger.info(f"Data types: {dict(self.df.dtypes)}")
        return self.df

    def analyze_distribution(self, output_dir: Path = None) -> Dict:
        """分析各指標的分布"""
        logger.info("Analyzing metric distributions...")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        stats = {}

        for metric, config in self.metrics.items():
            if metric not in self.df.columns:
                logger.warning(f"Metric {metric} not found in dataset")
                continue

            data = self.df[metric].dropna()

            metric_stats = {
                'count': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'median': data.median(),
                'q25': data.quantile(0.25),
                'q75': data.quantile(0.75),
                'q95': data.quantile(0.95),
                'q99': data.quantile(0.99)
            }

            stats[metric] = metric_stats

            logger.info(f"\n{metric}:")
            logger.info(f"  Count: {metric_stats['count']}")
            logger.info(f"  Mean: {metric_stats['mean']:.4f}")
            logger.info(f"  Median: {metric_stats['median']:.4f}")
            logger.info(f"  Std: {metric_stats['std']:.4f}")
            logger.info(f"  Range: [{metric_stats['min']:.4f}, {metric_stats['max']:.4f}]")
            logger.info(f"  Q95: {metric_stats['q95']:.4f}, Q99: {metric_stats['q99']:.4f}")

        self.stats = stats

        # 繪製分布圖
        if output_dir:
            self._plot_distributions(output_dir)

        return stats

    def _plot_distributions(self, output_dir: Path):
        """繪製指標分布圖"""
        logger.info("Plotting distributions...")

        n_metrics = len([m for m in self.metrics.keys() if m in self.df.columns])
        fig, axes = plt.subplots(n_metrics, 2, figsize=(15, 4 * n_metrics))

        if n_metrics == 1:
            axes = axes.reshape(1, -1)

        for idx, (metric, config) in enumerate(self.metrics.items()):
            if metric not in self.df.columns:
                continue

            data = self.df[metric].dropna()

            # 直方圖
            axes[idx, 0].hist(data, bins=50, edgecolor='black', alpha=0.7)
            axes[idx, 0].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.3f}')
            axes[idx, 0].axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.3f}')
            axes[idx, 0].set_xlabel(metric)
            axes[idx, 0].set_ylabel('Frequency')
            axes[idx, 0].set_title(f'{metric} Distribution')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)

            # 箱型圖
            axes[idx, 1].boxplot(data, vert=False)
            axes[idx, 1].set_xlabel(metric)
            axes[idx, 1].set_title(f'{metric} Boxplot')
            axes[idx, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / f'{self.dataset_name}_distributions.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved distribution plots to {plot_path}")
        plt.close()

    def filter_outliers(self, method: str = 'threshold', threshold_multiplier: float = 3.0) -> pd.DataFrame:
        """
        過濾異常值

        Args:
            method: 過濾方法 ('threshold', 'iqr', 'percentile')
            threshold_multiplier: IQR 方法的倍數

        Returns:
            過濾後的 DataFrame
        """
        logger.info(f"Filtering outliers using method: {method}")

        filtered_df = self.df.copy()
        initial_count = len(filtered_df)

        for metric, config in self.metrics.items():
            if metric not in filtered_df.columns:
                continue

            if method == 'threshold':
                # 使用預定義的閾值
                min_val = config['min']
                max_val = config['max']

                before = len(filtered_df)
                filtered_df = filtered_df[
                    (filtered_df[metric] >= min_val) &
                    (filtered_df[metric] <= max_val)
                ]
                removed = before - len(filtered_df)
                logger.info(f"  {metric}: Removed {removed} samples outside [{min_val}, {max_val}]")

            elif method == 'iqr':
                # 使用 IQR 方法
                Q1 = filtered_df[metric].quantile(0.25)
                Q3 = filtered_df[metric].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold_multiplier * IQR
                upper_bound = Q3 + threshold_multiplier * IQR

                before = len(filtered_df)
                filtered_df = filtered_df[
                    (filtered_df[metric] >= lower_bound) &
                    (filtered_df[metric] <= upper_bound)
                ]
                removed = before - len(filtered_df)
                logger.info(f"  {metric}: Removed {removed} samples outside [{lower_bound:.4f}, {upper_bound:.4f}]")

            elif method == 'percentile':
                # 使用百分位數
                lower_bound = filtered_df[metric].quantile(0.01)
                upper_bound = filtered_df[metric].quantile(0.99)

                before = len(filtered_df)
                filtered_df = filtered_df[
                    (filtered_df[metric] >= lower_bound) &
                    (filtered_df[metric] <= upper_bound)
                ]
                removed = before - len(filtered_df)
                logger.info(f"  {metric}: Removed {removed} samples outside [{lower_bound:.4f}, {upper_bound:.4f}]")

        final_count = len(filtered_df)
        logger.info(f"Filtering complete: {initial_count} -> {final_count} ({initial_count - final_count} removed, {final_count/initial_count*100:.2f}% retained)")

        return filtered_df

    def check_audio_exists(self, df: pd.DataFrame, base_audio_dir: str = None) -> pd.DataFrame:
        """檢查音頻文件是否存在"""
        if base_audio_dir is None:
            logger.info("Skipping audio file existence check")
            return df

        logger.info(f"Checking audio file existence in {base_audio_dir}")
        base_path = Path(base_audio_dir)

        def file_exists(row):
            file_path = base_path / row['file_path']
            return file_path.exists()

        df['audio_exists'] = df.apply(file_exists, axis=1)

        existing_count = df['audio_exists'].sum()
        logger.info(f"Found {existing_count}/{len(df)} existing audio files")

        df_filtered = df[df['audio_exists']].drop(columns=['audio_exists'])
        return df_filtered

    def split_train_test(
        self,
        df: pd.DataFrame,
        test_size: int = 1000,
        existing_test_csv: str = None,
        sort_by: str = None,
        ascending: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        分割訓練集和測試集

        Args:
            df: 要分割的 DataFrame
            test_size: 測試集大小
            existing_test_csv: 現有測試集 CSV (用於排除)
            sort_by: 排序依據的列名 (例如 'wer' 或 'cer')
            ascending: 是否升序排列

        Returns:
            (train_df, test_df)
        """
        logger.info("Splitting train/test sets...")

        # 排除現有測試集中的樣本
        if existing_test_csv and Path(existing_test_csv).exists():
            logger.info(f"Excluding samples from existing test set: {existing_test_csv}")
            existing_test = pd.read_csv(existing_test_csv)
            existing_test_files = set(existing_test['file_name'].values)
            df = df[~df['file_name'].isin(existing_test_files)]
            logger.info(f"After exclusion: {len(df)} samples remaining")

        # 按指標排序（選擇最乾淨的數據）
        if sort_by and sort_by in df.columns:
            logger.info(f"Sorting by {sort_by} ({'ascending' if ascending else 'descending'})")
            df = df.sort_values(by=sort_by, ascending=ascending)

        # 分割
        test_df = df.head(test_size).copy()
        train_df = df.iloc[test_size:].copy()

        logger.info(f"Split complete: {len(train_df)} train, {len(test_df)} test")

        return train_df, test_df

    def save_datasets(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path
    ):
        """保存訓練集和測試集"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / f'{self.dataset_name}_filtered_train.csv'
        test_path = output_dir / f'{self.dataset_name}_filtered_test.csv'

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Saved train set to {train_path}")
        logger.info(f"Saved test set to {test_path}")

        # 生成統計摘要
        self._save_summary(train_df, test_df, output_dir)

    def _save_summary(self, train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path):
        """保存統計摘要"""
        summary_path = output_dir / f'{self.dataset_name}_filter_summary.txt'

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Dataset Filtering Summary - {self.dataset_name.upper()}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Original dataset: {self.csv_path}\n")
            f.write(f"Total samples after filtering: {len(train_df) + len(test_df)}\n")
            f.write(f"  - Train: {len(train_df)}\n")
            f.write(f"  - Test: {len(test_df)}\n\n")

            for dataset_name, dataset_df in [('Train', train_df), ('Test', test_df)]:
                f.write(f"\n{dataset_name} Set Statistics:\n")
                f.write("-" * 80 + "\n")

                for metric in self.metrics.keys():
                    if metric not in dataset_df.columns:
                        continue

                    data = dataset_df[metric].dropna()
                    f.write(f"\n{metric}:\n")
                    f.write(f"  Count: {len(data)}\n")
                    f.write(f"  Mean: {data.mean():.4f}\n")
                    f.write(f"  Median: {data.median():.4f}\n")
                    f.write(f"  Std: {data.std():.4f}\n")
                    f.write(f"  Min: {data.min():.4f}\n")
                    f.write(f"  Max: {data.max():.4f}\n")
                    f.write(f"  Q25: {data.quantile(0.25):.4f}\n")
                    f.write(f"  Q75: {data.quantile(0.75):.4f}\n")

        logger.info(f"Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze and filter dataset')
    parser.add_argument('--csv', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--language', type=str, required=True, choices=['vi', 'yue'],
                        help='Language: vi (Vietnamese) or yue (Cantonese)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--test_size', type=int, default=1000, help='Test set size')
    parser.add_argument('--existing_test', type=str, default=None,
                        help='Existing test CSV to exclude from train set')
    parser.add_argument('--filter_method', type=str, default='threshold',
                        choices=['threshold', 'iqr', 'percentile'],
                        help='Outlier filtering method')
    parser.add_argument('--plot', action='store_true', help='Generate distribution plots')
    parser.add_argument('--audio_dir', type=str, default=None,
                        help='Base audio directory to check file existence')

    args = parser.parse_args()

    # 創建分析器
    analyzer = DatasetAnalyzer(args.csv, args.language)

    # 載入數據
    analyzer.load_data()

    # 分析分布
    output_dir = Path(args.output_dir)
    if args.plot:
        analyzer.analyze_distribution(output_dir=output_dir)
    else:
        analyzer.analyze_distribution()

    # 過濾異常值
    filtered_df = analyzer.filter_outliers(method=args.filter_method)

    # 檢查音頻文件存在性（可選）
    if args.audio_dir:
        filtered_df = analyzer.check_audio_exists(filtered_df, args.audio_dir)

    # 分割訓練集和測試集
    sort_metric = 'wer' if args.language == 'vi' else 'cer'
    train_df, test_df = analyzer.split_train_test(
        filtered_df,
        test_size=args.test_size,
        existing_test_csv=args.existing_test,
        sort_by=sort_metric,
        ascending=True  # 選擇錯誤率最低的作為測試集
    )

    # 保存數據集
    analyzer.save_datasets(train_df, test_df, output_dir)

    logger.info("Processing complete!")


if __name__ == '__main__':
    main()
