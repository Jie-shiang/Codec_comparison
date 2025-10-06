#!/usr/bin/env python3
"""
Enhanced Neural Audio Codec Evaluation Pipeline

Optimized evaluation pipeline with selective metric calculation, incremental CSV updates,
and support for both English (dWER) and Chinese (dCER) evaluation.
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time

from metrics_evaluator import AudioMetricsEvaluator


class EnhancedCodecEvaluationPipeline:
    """Enhanced evaluation pipeline with selective metric calculation"""
    
    def __init__(self, 
                 inference_dir: str,
                 csv_file: str, 
                 model_name: str,
                 frequency: str,
                 causality: str,
                 bit_rate: str,
                 dataset_type: str = "clean",
                 project_dir: str = "/home/jieshiang/Desktop/GitHub/Codec_comparison",
                 quantizers: str = "4",
                 codebook_size: str = "1024", 
                 n_params: str = "45M",
                 training_set: str = "Custom Dataset",
                 testing_set: str = "Custom Test Set",
                 metrics_to_compute: list = None,
                 use_gpu: bool = True,
                 gpu_id: int = 0,
                 original_dir: str = None,
                 language: str = None):
        
        self.inference_dir = Path(inference_dir)
        self.csv_file = Path(project_dir) / "csv" / csv_file
        self.model_name = model_name
        self.frequency = frequency
        self.causality = causality
        self.bit_rate = bit_rate
        self.dataset_type = dataset_type.lower()
        self.quantizers = quantizers
        self.codebook_size = codebook_size
        self.n_params = n_params
        self.training_set = training_set
        self.testing_set = testing_set
        
        self.metrics_to_compute = metrics_to_compute or ['dwer', 'dcer', 'utmos', 'pesq', 'stoi']
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.original_dir = Path(original_dir) if original_dir else None
        
        # Auto-detect language if not specified
        if language:
            self.language = language
        else:
            # Auto-detect from CSV filename
            csv_file_str = str(csv_file).lower()
            if 'common_voice' in csv_file_str or 'zh' in csv_file_str:
                self.language = 'zh'
            else:
                self.language = 'en'
        
        self.project_dir = Path(project_dir)
        self.result_dir = self.project_dir / "result"
        self.audio_dir = self.project_dir / "audio" 
        self.config_dir = self.project_dir / "configs"
        
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = None
        self.end_time = None
        
        self.base_dataset_name = None
        self.dataset_name = None
        self.result_csv_path = None
        
        print(f"Initializing enhanced evaluation pipeline:")
        print(f"  Model: {self.model_name}")
        print(f"  Frequency: {self.frequency}")
        print(f"  Dataset type: {self.dataset_type}")
        print(f"  Language: {self.language}")
        print(f"  Metrics to compute: {', '.join(self.metrics_to_compute)}")
        print(f"  GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        if self.use_gpu:
            print(f"  GPU ID: {self.gpu_id}")
        if self.original_dir:
            print(f"  Original files directory: {self.original_dir}")
        print(f"  Inference directory: {self.inference_dir}")
        
    def load_csv_data(self):
        try:
            df = pd.read_csv(self.csv_file, encoding='utf-8')
            print(f"Successfully loaded CSV: {self.csv_file}")
            print(f"Number of samples: {len(df)}")
            
            if 'common_voice' in str(self.csv_file).lower():
                self.base_dataset_name = 'CommonVoice'
            else:
                self.base_dataset_name = 'LibriSpeech'
            
            if self.dataset_type == "clean":
                self.dataset_name = self.base_dataset_name
            elif self.dataset_type == "noise":
                self.dataset_name = f"{self.base_dataset_name}_Noise"
            elif self.dataset_type == "blank":
                self.dataset_name = f"{self.base_dataset_name}_Blank"
            else:
                raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
            
            dataset_suffix = self.base_dataset_name.lower()
            self.result_csv_path = self.result_dir / f"detailed_results_{self.model_name}_{self.dataset_type}_{dataset_suffix}.csv"
            
            print(f"Dataset name: {self.dataset_name}")
            print(f"Result CSV will be saved as: {self.result_csv_path}")
            print(f"Metrics to compute: {', '.join(self.metrics_to_compute)}")
                
            return df
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            sys.exit(1)
    
    def load_existing_results(self) -> pd.DataFrame:
        if self.result_csv_path and self.result_csv_path.exists():
            try:
                existing_df = pd.read_csv(self.result_csv_path, encoding='utf-8')
                print(f"Found existing results CSV with {len(existing_df)} records")
                return existing_df
            except Exception as e:
                print(f"Error loading existing CSV: {e}")
                return None
        else:
            print("No existing results CSV found, will create new one")
            return None
    
    def create_empty_result_dataframe(self, input_df: pd.DataFrame) -> pd.DataFrame:
        columns = [
            'file_name', 'original_path', 'inference_path', 'ground_truth',
            'original_transcript', 'inference_transcript',
            'utmos', 'pesq', 'stoi'
        ]
        
        if 'dwer' in self.metrics_to_compute:
            columns.extend(['original_wer', 'inference_wer', 'dwer'])
        
        if 'dcer' in self.metrics_to_compute:
            columns.extend(['original_cer', 'inference_cer', 'dcer'])
        
        result_df = pd.DataFrame(index=range(len(input_df)))
        for col in columns:
            result_df[col] = np.nan
        
        result_df['file_name'] = input_df['file_name'].values
        result_df['ground_truth'] = input_df['transcription'].values
        result_df['original_path'] = input_df['file_path'].values
        
        return result_df
    
    def resolve_original_path(self, csv_path: str) -> Path:
        if self.original_dir:
            clean_path = csv_path.lstrip('./')
            return self.original_dir / clean_path
        else:
            return Path(csv_path)
    
    def find_inference_audio(self, original_filename: str) -> Path:
        base_name = Path(original_filename).stem
        
        possible_names = [
            f"{base_name}_inference.wav",
            f"{base_name}_inference.flac",
            f"{base_name}.wav",
            f"{base_name}.flac"
        ]
        
        for name in possible_names:
            inference_path = self.inference_dir / name
            if inference_path.exists():
                return inference_path
                
        return None
    
    def should_process_file(self, existing_row: pd.Series, file_name: str) -> tuple:
        if existing_row.empty:
            return True, "No existing data for this file"
        
        metric_map = {
            'dcer': 'dcer',
            'dwer': 'dwer',
            'utmos': 'utmos',
            'pesq': 'pesq',
            'stoi': 'stoi'
        }
        
        missing_metrics = []
        for metric in self.metrics_to_compute:
            if metric in metric_map:
                col_name = metric_map[metric]
                
                if col_name not in existing_row.index:
                    missing_metrics.append(metric)
                elif pd.isna(existing_row[col_name]):
                    missing_metrics.append(metric)
        
        if missing_metrics:
            return True, f"Missing metrics: {', '.join(missing_metrics)}"
        else:
            return False, "All requested metrics already exist"
    
    def merge_results(self, existing_df: pd.DataFrame, new_results: list) -> pd.DataFrame:
        new_df = pd.DataFrame(new_results)
        
        if existing_df is None or len(existing_df) == 0:
            return new_df
        
        for _, new_row in new_df.iterrows():
            file_name = new_row['file_name']
            mask = existing_df['file_name'] == file_name
            
            if mask.any():
                for col in new_df.columns:
                    if pd.notna(new_row[col]) and col != 'file_name':
                        existing_df.loc[mask, col] = new_row[col]
            else:
                existing_df = pd.concat([existing_df, new_row.to_frame().T], ignore_index=True)
        
        return existing_df
    
    def evaluate_metrics_selectively(self, evaluator: AudioMetricsEvaluator, 
                                   original_path: str, inference_path: str, 
                                   ground_truth: str) -> dict:
        results = {}
        
        results['original_path'] = str(original_path)
        results['inference_path'] = str(inference_path)
        results['ground_truth'] = ground_truth
        
        # Check if we need ASR metrics
        compute_dwer = 'dwer' in self.metrics_to_compute
        compute_dcer = 'dcer' in self.metrics_to_compute
        need_asr = compute_dwer or compute_dcer
        
        if need_asr:
            # Call calculate_dwer_dcer with proper parameters
            asr_result = evaluator.calculate_dwer_dcer(
                str(original_path), 
                str(inference_path), 
                ground_truth
            )
            
            if asr_result:
                # Always store transcripts
                results['original_transcript'] = asr_result.get('original_transcript', '')
                results['inference_transcript'] = asr_result.get('inference_transcript', '')
                
                # Store metrics based on what was requested and what language was detected
                metric_name = asr_result.get('metric_name', 'dWER')
                
                if metric_name == 'dWER':
                    if compute_dwer:
                        results['original_wer'] = asr_result.get('original_wer', np.nan)
                        results['inference_wer'] = asr_result.get('inference_wer', np.nan)
                        results['dwer'] = asr_result.get('dwer', np.nan)
                elif metric_name == 'dCER':
                    if compute_dcer:
                        results['original_cer'] = asr_result.get('original_cer', np.nan)
                        results['inference_cer'] = asr_result.get('inference_cer', np.nan)
                        results['dcer'] = asr_result.get('dcer', np.nan)
                
                # If user requested both metrics, we would need to compute with both languages
                # For now, we compute based on the evaluator's language setting
                # If you need both, you'd need to create two evaluators or modify the evaluator class
        
        if 'utmos' in self.metrics_to_compute:
            results['utmos'] = evaluator.calculate_utmos(inference_path)
        
        if 'pesq' in self.metrics_to_compute:
            results['pesq'] = evaluator.calculate_pesq(original_path, inference_path)
        
        if 'stoi' in self.metrics_to_compute:
            results['stoi'] = evaluator.calculate_stoi(original_path, inference_path)
        
        return results
    
    def save_results(self, results_df: pd.DataFrame) -> None:
        detailed_csv_path = self.result_csv_path
        results_df.to_csv(detailed_csv_path, index=False, encoding='utf-8')
        print(f"Detailed results saved: {detailed_csv_path}")
        
        dataset_suffix = self.base_dataset_name.lower()
        summary_csv_path = self.result_dir / f"summary_results_{self.model_name}_{self.dataset_type}_{dataset_suffix}.csv"
        summary_data = self.create_summary_statistics(results_df)
        
        if summary_data:
            if summary_csv_path.exists():
                try:
                    existing_summary = pd.read_csv(summary_csv_path, encoding='utf-8')
                    updated_summary = self.update_summary_data(existing_summary, summary_data)
                    updated_summary.to_csv(summary_csv_path, index=False, encoding='utf-8')
                    print(f"Summary results updated: {summary_csv_path}")
                except Exception as e:
                    print(f"Warning: Could not update existing summary: {e}")
                    summary_df = pd.DataFrame([summary_data])
                    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
                    print(f"Summary results saved: {summary_csv_path}")
            else:
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
                print(f"Summary results saved: {summary_csv_path}")
            
            self.print_summary_statistics(summary_data)
    
    def create_summary_statistics(self, results_df: pd.DataFrame) -> dict:
        summary_data = {
            'Model': self.model_name,
            'Frequency': self.frequency,
            'Dataset': self.dataset_name,
            'Dataset_Type': self.dataset_type,
            'Language': self.language,
            'Timestamp': self.timestamp,
            'Total_Files': len(results_df),
            'Valid_Files': len(results_df.dropna(subset=['file_name']))
        }
        
        metric_columns = []
        if 'dwer' in self.metrics_to_compute:
            metric_columns.append('dwer')
        if 'dcer' in self.metrics_to_compute:
            metric_columns.append('dcer')
        if 'utmos' in self.metrics_to_compute:
            metric_columns.append('utmos')
        if 'pesq' in self.metrics_to_compute:
            metric_columns.append('pesq')
        if 'stoi' in self.metrics_to_compute:
            metric_columns.append('stoi')
        
        for metric in metric_columns:
            if metric in results_df.columns:
                valid_values = results_df[metric].dropna()
                if len(valid_values) > 0:
                    summary_data.update({
                        f'{metric.upper()}_Count': len(valid_values),
                        f'{metric.upper()}_Mean': round(valid_values.mean(), 4),
                        f'{metric.upper()}_Std': round(valid_values.std(), 4),
                        f'{metric.upper()}_Min': round(valid_values.min(), 4),
                        f'{metric.upper()}_Max': round(valid_values.max(), 4),
                        f'{metric.upper()}_Median': round(valid_values.median(), 4),
                        f'{metric.upper()}_Q25': round(valid_values.quantile(0.25), 4),
                        f'{metric.upper()}_Q75': round(valid_values.quantile(0.75), 4)
                    })
                else:
                    summary_data.update({
                        f'{metric.upper()}_Count': 0,
                        f'{metric.upper()}_Mean': np.nan,
                        f'{metric.upper()}_Std': np.nan,
                        f'{metric.upper()}_Min': np.nan,
                        f'{metric.upper()}_Max': np.nan,
                        f'{metric.upper()}_Median': np.nan,
                        f'{metric.upper()}_Q25': np.nan,
                        f'{metric.upper()}_Q75': np.nan
                    })
        
        return summary_data
    
    def update_summary_data(self, existing_df: pd.DataFrame, new_data: dict) -> pd.DataFrame:
        mask = (
            (existing_df['Model'] == new_data['Model']) &
            (existing_df['Frequency'] == new_data['Frequency']) &
            (existing_df['Dataset'] == new_data['Dataset']) &
            (existing_df['Dataset_Type'] == new_data['Dataset_Type'])
        )
        
        if mask.any():
            for key, value in new_data.items():
                if pd.notna(value):
                    existing_df.loc[mask, key] = value
        else:
            new_row_df = pd.DataFrame([new_data])
            existing_df = pd.concat([existing_df, new_row_df], ignore_index=True)
        
        return existing_df
    
    def print_summary_statistics(self, summary_data: dict) -> None:
        print(f"\n" + "="*70)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*70)
        print(f"Model: {summary_data['Model']} ({summary_data['Frequency']})")
        print(f"Dataset: {summary_data['Dataset']} - {summary_data['Dataset_Type']}")
        print(f"Language: {summary_data['Language']}")
        print(f"Total files: {summary_data['Total_Files']}")
        print(f"Valid evaluations: {summary_data['Valid_Files']}")
        print(f"Timestamp: {summary_data['Timestamp']}")
        
        print(f"\nDETAILED METRICS STATISTICS:")
        print("-" * 70)
        
        metrics_order = []
        if 'DWER_Count' in summary_data:
            metrics_order.append('DWER')
        if 'DCER_Count' in summary_data:
            metrics_order.append('DCER')
        if 'UTMOS_Count' in summary_data:
            metrics_order.append('UTMOS')
        if 'PESQ_Count' in summary_data:
            metrics_order.append('PESQ')
        if 'STOI_Count' in summary_data:
            metrics_order.append('STOI')
        
        for metric in metrics_order:
            count_key = f'{metric}_Count'
            if count_key in summary_data and summary_data[count_key] > 0:
                print(f"\n{metric}:")
                print(f"  Count: {summary_data[f'{metric}_Count']}")
                print(f"  Mean:  {summary_data[f'{metric}_Mean']:.4f}")
                print(f"  Std:   {summary_data[f'{metric}_Std']:.4f}")
                print(f"  Min:   {summary_data[f'{metric}_Min']:.4f}")
                print(f"  Max:   {summary_data[f'{metric}_Max']:.4f}")
                print(f"  Median:{summary_data[f'{metric}_Median']:.4f}")
                print(f"  Q25:   {summary_data[f'{metric}_Q25']:.4f}")
                print(f"  Q75:   {summary_data[f'{metric}_Q75']:.4f}")
            elif count_key in summary_data:
                print(f"\n{metric}: Not computed (0 samples)")
        
        print(f"\n" + "="*70)
    
    def generate_config_and_copy_files(self, results_df: pd.DataFrame) -> None:
        if len(results_df) == 0:
            print("No results to generate config from")
            return
            
        config_path = self.config_dir / f"{self.model_name}_{self.frequency}_config.json"
        config = self.generate_or_update_json_config(results_df, config_path)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"JSON config saved: {config_path}")
        
        self.copy_sample_audio_files(results_df)
    
    def generate_or_update_json_config(self, results_df: pd.DataFrame, config_path: Path) -> dict:
        existing_config = {}
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
                print(f"Loading existing config from: {config_path}")
            except Exception as e:
                print(f"Warning: Could not load existing config: {e}")
        
        # Determine which metrics are available
        available_metrics = []
        if 'dwer' in results_df.columns and results_df['dwer'].notna().any():
            available_metrics.append('dwer')
        if 'dcer' in results_df.columns and results_df['dcer'].notna().any():
            available_metrics.append('dcer')
        
        if not available_metrics:
            print("Warning: No WER/CER metrics found in results")
            # Still create config but without ASR metrics
            primary_metric = 'dWER' if self.language == 'en' else 'dCER'
            metric_col = 'dwer' if self.language == 'en' else 'dcer'
        else:
            # Use first available metric as primary
            metric_col = available_metrics[0]
            primary_metric = 'dWER' if metric_col == 'dwer' else 'dCER'
        
        valid_results = results_df.dropna(subset=['utmos', 'pesq', 'stoi'])
        
        if len(valid_results) == 0:
            print("Warning: No valid results found for config generation")
            return existing_config
        
        current_dataset_data = self.generate_dataset_section(valid_results, primary_metric, metric_col, available_metrics)
        
        config = {
            "model_info": {
                "modelName": self.model_name,
                "causality": self.causality,
                "trainingSet": self.training_set,
                "testingSet": self.testing_set,
                "bitRate": self.bit_rate,
                "parameters": {
                    "frameRate": self.frequency.replace('Hz', ''),
                    "quantizers": self.quantizers,
                    "codebookSize": self.codebook_size,
                    "nParams": self.n_params
                }
            }
        }
        
        all_datasets = ["LibriSpeech", "LibriSpeech_Noise", "LibriSpeech_Blank", 
                       "CommonVoice", "CommonVoice_Noise", "CommonVoice_Blank"]
        
        for dataset in all_datasets:
            if dataset == self.dataset_name:
                config[dataset] = current_dataset_data
                print(f"Updated {dataset} section with evaluation results")
            elif dataset in existing_config:
                config[dataset] = existing_config[dataset]
                print(f"Preserved existing {dataset} section")
            else:
                config[dataset] = self.create_dataset_specific_placeholder(dataset)
                print(f"Created placeholder {dataset} section")
        
        return config
    
    def generate_dataset_section(self, valid_results: pd.DataFrame, metric_name: str, metric_col: str, available_metrics: list = None) -> dict:
        total_stats = {
            'UTMOS': f"{valid_results['utmos'].mean():.1f}",
            'PESQ': f"{valid_results['pesq'].mean():.1f}",
            'STOI': f"{valid_results['stoi'].mean():.2f}"
        }
        
        # Add computed metrics to total stats with numeric conversion
        if 'dwer' in valid_results.columns:
            dwer_numeric = pd.to_numeric(valid_results['dwer'], errors='coerce')
            if dwer_numeric.notna().any():
                total_stats['dWER'] = f"{dwer_numeric.mean():.2f}"
        
        if 'dcer' in valid_results.columns:
            dcer_numeric = pd.to_numeric(valid_results['dcer'], errors='coerce')
            if dcer_numeric.notna().any():
                total_stats['dCER'] = f"{dcer_numeric.mean():.2f}"
        
        selected_samples = self.select_diverse_samples(valid_results)
        
        samples = {}
        for sample_name, row in selected_samples:
            if sample_name.startswith('Sample_'):
                sample_data = {
                    'File_name': Path(row['file_name']).stem,
                    'Transcription': row['ground_truth'],
                    'Origin': row.get('original_transcript', 'N/A'),
                    'Inference': row.get('inference_transcript', 'N/A'),
                    'UTMOS': f"{row['utmos']:.1f}",
                    'PESQ': f"{row['pesq']:.1f}",
                    'STOI': f"{row['stoi']:.2f}"
                }
                
                # Add computed metrics to samples with numeric conversion
                if 'dwer' in row.index:
                    dwer_val = pd.to_numeric(row['dwer'], errors='coerce')
                    if pd.notna(dwer_val):
                        sample_data['dWER'] = f"{dwer_val:.2f}"
                
                if 'dcer' in row.index:
                    dcer_val = pd.to_numeric(row['dcer'], errors='coerce')
                    if pd.notna(dcer_val):
                        sample_data['dCER'] = f"{dcer_val:.2f}"
                
                samples[sample_name] = sample_data
        
        error_sample_data = {}
        for sample_name, row in selected_samples:
            if sample_name == 'Error_Sample_1':
                error_sample_data = {
                    'File_name': Path(row['file_name']).stem,
                    'Transcription': row['ground_truth'],
                    'Origin': row.get('original_transcript', 'N/A'),
                    'Inference': row.get('inference_transcript', 'N/A'),
                    'UTMOS': f"{row['utmos']:.1f}",
                    'PESQ': f"{row['pesq']:.1f}",
                    'STOI': f"{row['stoi']:.2f}"
                }
                
                # Add computed metrics to error sample with numeric conversion
                if 'dwer' in row.index:
                    dwer_val = pd.to_numeric(row['dwer'], errors='coerce')
                    if pd.notna(dwer_val):
                        error_sample_data['dWER'] = f"{dwer_val:.2f}"
                
                if 'dcer' in row.index:
                    dcer_val = pd.to_numeric(row['dcer'], errors='coerce')
                    if pd.notna(dcer_val):
                        error_sample_data['dCER'] = f"{dcer_val:.2f}"
                
                break
        
        dataset_section = {
            "Total": total_stats,
            **samples,
            "Error_Sample_1": error_sample_data
        }
        
        return dataset_section
    
    def create_dataset_specific_placeholder(self, dataset_name: str) -> dict:
        # Determine which metrics to include based on dataset
        metrics_dict = {}
        
        # If both metrics were requested, include both in placeholder
        if 'dwer' in self.metrics_to_compute or dataset_name.startswith('LibriSpeech'):
            metrics_dict['dWER'] = "N/A"
        if 'dcer' in self.metrics_to_compute or dataset_name.startswith('CommonVoice'):
            metrics_dict['dCER'] = "N/A"
            
        placeholder_section = {
            "Total": {
                **metrics_dict,
                'UTMOS': "N/A",
                'PESQ': "N/A",
                'STOI': "N/A"
            }
        }
        
        for i in range(1, 6):
            placeholder_section[f"Sample_{i}"] = {
                'File_name': "N/A",
                'Transcription': "N/A",
                'Origin': "N/A",
                'Inference': "N/A",
                **metrics_dict,
                'UTMOS': "N/A",
                'PESQ': "N/A",
                'STOI': "N/A"
            }
        
        placeholder_section["Error_Sample_1"] = {
            'File_name': "N/A",
            'Transcription': "N/A",
            'Origin': "N/A",
            'Inference': "N/A",
            **metrics_dict,
            'UTMOS': "N/A",
            'PESQ': "N/A",
            'STOI': "N/A"
        }
        
        return placeholder_section
    
    def get_dataset_path_for_audio(self) -> str:
        if self.dataset_type == "clean":
            return self.base_dataset_name
        elif self.dataset_type == "noise":
            return f"{self.base_dataset_name}/Noise"
        elif self.dataset_type == "blank":
            return f"{self.base_dataset_name}/Blank"
        else:
            return self.base_dataset_name
    
    def copy_sample_audio_files(self, results_df: pd.DataFrame) -> None:
        audio_dataset_path = self.get_dataset_path_for_audio()
        dataset_dir = self.audio_dir / audio_dataset_path
        
        original_dir = dataset_dir / "original"
        inference_dir = dataset_dir / self.model_name / self.frequency
        
        original_dir.mkdir(parents=True, exist_ok=True)
        inference_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Copying sample audio files to {dataset_dir}...")
        
        # Determine which metric to use for filtering
        metric_cols = []
        if 'dcer' in results_df.columns:
            metric_cols.append('dcer')
        if 'dwer' in results_df.columns:
            metric_cols.append('dwer')
        
        # If no ASR metrics available, just use other metrics
        if not metric_cols:
            valid_results = results_df.dropna(subset=['utmos', 'pesq', 'stoi'])
        else:
            # Use first available ASR metric
            valid_results = results_df.dropna(subset=[metric_cols[0], 'utmos', 'pesq', 'stoi'])
        
        if len(valid_results) == 0:
            print("Warning: No valid results found for file copying")
            return
        
        selected_samples = self.select_diverse_samples(valid_results)
        
        copied_count = 0
        for i, (sample_name, row) in enumerate(selected_samples):
            file_name = row['file_name']
            base_name = Path(file_name).stem
            
            original_path = Path(row['original_path'])
            inference_path = Path(row['inference_path'])
            
            if original_path.exists() and inference_path.exists():
                try:
                    original_dest = original_dir / f"{base_name}.flac"
                    shutil.copy2(original_path, original_dest)
                    
                    inference_dest = inference_dir / f"{base_name}.wav"
                    shutil.copy2(inference_path, inference_dest)
                    
                    if sample_name.startswith('Sample_'):
                        print(f"  Copied sample (JSON key: {sample_name}, File_name: {base_name}): {base_name}")
                    else:
                        print(f"  Copied {sample_name}: {base_name}")
                    copied_count += 1
                    
                except Exception as e:
                    print(f"  Warning: Failed to copy {base_name}: {e}")
            else:
                print(f"  Warning: Could not find files for {base_name}")
        
        print(f"Successfully copied {copied_count} sample pairs to audio directory")
    
    def select_diverse_samples(self, valid_results: pd.DataFrame) -> list:
        selected_samples = []
        
        if self.base_dataset_name == 'LibriSpeech':
            results_with_keys = valid_results.copy()
            results_with_keys['sort_key'] = results_with_keys['file_name'].apply(
                lambda x: self.parse_librispeech_id(x)
            )
            results_sorted = results_with_keys.sort_values('sort_key')
            
            speakers_seen = set()
            sample_count = 0
            
            for idx, row in results_sorted.iterrows():
                speaker_id = row['file_name'].split('-')[0]
                if speaker_id not in speakers_seen and sample_count < 5:
                    speakers_seen.add(speaker_id)
                    sample_count += 1
                    selected_samples.append((f'Sample_{sample_count}', row))
                    
        elif self.base_dataset_name == 'CommonVoice':
            if 'speaker_id' in valid_results.columns:
                speakers_seen = set()
                sample_count = 0
                
                for idx, row in valid_results.iterrows():
                    speaker_id = row.get('speaker_id', 'unknown')
                    if speaker_id not in speakers_seen and sample_count < 5:
                        speakers_seen.add(speaker_id)
                        sample_count += 1
                        selected_samples.append((f'Sample_{sample_count}', row))
            else:
                for i in range(min(5, len(valid_results))):
                    row = valid_results.iloc[i]
                    selected_samples.append((f'Sample_{i+1}', row))
        else:
            for i in range(min(5, len(valid_results))):
                row = valid_results.iloc[i]
                selected_samples.append((f'Sample_{i+1}', row))
        
        while len(selected_samples) < 5 and len(selected_samples) < len(valid_results):
            remaining_idx = len(selected_samples)
            if remaining_idx < len(valid_results):
                row = valid_results.iloc[remaining_idx]
                selected_samples.append((f'Sample_{len(selected_samples)+1}', row))
        
        # Find error sample based on available metrics
        error_idx = None
        try:
            # Try to find the maximum error metric
            if 'dcer' in valid_results.columns:
                # Ensure numeric type and drop NaN values
                dcer_numeric = pd.to_numeric(valid_results['dcer'], errors='coerce')
                if dcer_numeric.notna().any():
                    error_idx = dcer_numeric.idxmax()
            
            if error_idx is None and 'dwer' in valid_results.columns:
                # Ensure numeric type and drop NaN values
                dwer_numeric = pd.to_numeric(valid_results['dwer'], errors='coerce')
                if dwer_numeric.notna().any():
                    error_idx = dwer_numeric.idxmax()
            
            if error_idx is None:
                # Fallback to first row
                error_idx = valid_results.index[0]
                
        except Exception as e:
            print(f"Warning: Could not find error sample using max metric: {e}")
            error_idx = valid_results.index[0]
        
        error_sample = valid_results.loc[error_idx]
        selected_samples.append(('Error_Sample_1', error_sample))
        
        return selected_samples
    
    def parse_librispeech_id(self, utt_id: str) -> tuple:
        try:
            parts = utt_id.split('-')
            if len(parts) >= 3:
                speaker = int(parts[0])
                chapter = int(parts[1])
                utterance = int(parts[2])
                return (speaker, chapter, utterance)
            else:
                return (0, 0, 0)
        except:
            return (0, 0, 0)
    
    def run_evaluation(self):
        self.start_time = time.time()
        
        print("=" * 60)
        print("Starting Enhanced Neural Audio Codec Evaluation")
        print("=" * 60)
        
        step_start = time.time()
        df = self.load_csv_data()
        print(f"Data loading completed in: {time.time() - step_start:.2f} seconds")
        
        step_start = time.time()
        existing_results = self.load_existing_results()
        print(f"Existing results check completed in: {time.time() - step_start:.2f} seconds")
        
        step_start = time.time()
        evaluator = AudioMetricsEvaluator(
            language=self.language,
            use_gpu=self.use_gpu,
            gpu_id=self.gpu_id
        )
        
        need_asr = ('dcer' in self.metrics_to_compute) or ('dwer' in self.metrics_to_compute)
        need_utmos = 'utmos' in self.metrics_to_compute
        
        if need_asr or need_utmos:
            evaluator.load_models()
        
        print(f"Model loading completed in: {time.time() - step_start:.2f} seconds")
        
        if existing_results is None:
            results_df = self.create_empty_result_dataframe(df)
        else:
            results_df = existing_results.copy()
        
        new_results = []
        processed_count = 0
        skipped_count = 0
        
        print(f"Starting evaluation of {len(df)} audio files...")
        print(f"Computing metrics: {', '.join(self.metrics_to_compute)}")
        evaluation_start = time.time()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluation Progress"):
            file_name = row['file_name']
            ground_truth = row['transcription']
            original_path = self.resolve_original_path(row['file_path'])
            
            inference_path = self.find_inference_audio(file_name)
            
            if not original_path.exists():
                continue
                
            if not inference_path or not inference_path.exists():
                continue
            
            should_process = True
            skip_reason = ""
            
            if existing_results is not None:
                existing_row = existing_results[existing_results['file_name'] == file_name]
                if not existing_row.empty:
                    should_process, skip_reason = self.should_process_file(
                        existing_row.iloc[0], 
                        file_name
                    )
            
            if not should_process:
                skipped_count += 1
                if skipped_count <= 5:
                    print(f"  Skipping {file_name}: {skip_reason}")
                continue
            
            try:
                metrics_result = self.evaluate_metrics_selectively(
                    evaluator, str(original_path), str(inference_path), ground_truth
                )
                
                if metrics_result:
                    result_data = {'file_name': file_name}
                    result_data.update(metrics_result)
                    new_results.append(result_data)
                    processed_count += 1
                
            except Exception as e:
                print(f"Error evaluating file {file_name}: {e}")
                continue
        
        evaluation_time = time.time() - evaluation_start
        print(f"Audio evaluation completed in: {evaluation_time:.2f} seconds")
        print(f"Processed: {processed_count} files")
        print(f"Skipped: {skipped_count} files (all requested metrics already exist)")
        
        if new_results:
            step_start = time.time()
            results_df = self.merge_results(existing_results, new_results)
            print(f"Result merging completed in: {time.time() - step_start:.2f} seconds")
        
        step_start = time.time()
        self.save_results(results_df)
        print(f"Result saving completed in: {time.time() - step_start:.2f} seconds")
        
        if processed_count > 0 or len(results_df) > 0:
            step_start = time.time()
            self.generate_config_and_copy_files(results_df)
            print(f"Config generation and file copying completed in: {time.time() - step_start:.2f} seconds")
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        print("\n" + "=" * 60)
        print("Enhanced Evaluation Completed Successfully!")
        print("=" * 60)
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        if processed_count > 0:
            print(f"Average per file: {evaluation_time/processed_count:.2f} seconds")
        print(f"Processed: {processed_count} files")
        print(f"Skipped: {skipped_count} files")
        print(f"Result file: {self.result_csv_path}")
        
        return results_df


def main():
    parser = argparse.ArgumentParser(description="Enhanced Neural Audio Codec Evaluation Pipeline")
    
    parser.add_argument("--inference_dir", required=True, type=str,
                       help="Directory path containing inference audio files")
    parser.add_argument("--csv_file", required=True, type=str,
                       help="CSV dataset filename (must be in ./csv/ directory)")
    parser.add_argument("--model_name", required=True, type=str,
                       help="Name of codec model")
    parser.add_argument("--frequency", required=True, type=str,
                       help="Frame rate (e.g., 50Hz)")
    parser.add_argument("--causality", required=True, type=str,
                       choices=["Causal", "Non-Causal"],
                       help="Model causality type")
    parser.add_argument("--bit_rate", required=True, type=str,
                       help="Compression bit rate")
    
    parser.add_argument("--dataset_type", type=str, 
                       choices=["clean", "noise", "blank"], default="clean",
                       help="Dataset type (clean/noise/blank)")
    parser.add_argument("--project_dir", type=str,
                       default="/home/jieshiang/Desktop/GitHub/Codec_comparison",
                       help="Project root directory path")
    parser.add_argument("--quantizers", type=str, default="4",
                       help="Number of quantizers")
    parser.add_argument("--codebook_size", type=str, default="1024",
                       help="Codebook size")
    parser.add_argument("--n_params", type=str, default="45M",
                       help="Number of model parameters")
    parser.add_argument("--training_set", type=str, default="Custom Dataset",
                       help="Training dataset description")
    parser.add_argument("--testing_set", type=str, default="Custom Test Set",
                       help="Testing dataset description")
    
    parser.add_argument("--metrics", type=str, nargs='+', 
                       choices=["dwer", "dcer", "utmos", "pesq", "stoi"],
                       default=["dwer", "dcer", "utmos", "pesq", "stoi"],
                       help="Metrics to compute")
    
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="Enable GPU acceleration")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU device ID")
    parser.add_argument("--cpu_only", action="store_true",
                       help="Force CPU-only computation")
    parser.add_argument("--original_dir", type=str,
                       help="Root directory path for original audio files")
    parser.add_argument("--language", type=str, choices=["en", "zh"],
                       help="Language for ASR evaluation (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    use_gpu = args.use_gpu and not args.cpu_only
    
    pipeline = EnhancedCodecEvaluationPipeline(
        inference_dir=args.inference_dir,
        csv_file=args.csv_file,
        model_name=args.model_name,
        frequency=args.frequency,
        causality=args.causality,
        bit_rate=args.bit_rate,
        dataset_type=args.dataset_type,
        project_dir=args.project_dir,
        quantizers=args.quantizers,
        codebook_size=args.codebook_size,
        n_params=args.n_params,
        training_set=args.training_set,
        testing_set=args.testing_set,
        metrics_to_compute=args.metrics,
        use_gpu=use_gpu,
        gpu_id=args.gpu_id,
        original_dir=args.original_dir,
        language=args.language
    )
    
    try:
        results_df = pipeline.run_evaluation()
        print("\nProgram executed successfully!")
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user!")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()