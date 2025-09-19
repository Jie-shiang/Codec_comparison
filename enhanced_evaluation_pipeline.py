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
                 original_dir: str = None):
        
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
        
        # Language and dataset will be detected when loading CSV
        self.language = None
        self.base_dataset_name = None
        self.dataset_name = None
        self.result_csv_path = None
        
        print(f"Initializing enhanced evaluation pipeline:")
        print(f"  Model: {self.model_name}")
        print(f"  Frequency: {self.frequency}")
        print(f"  Dataset type: {self.dataset_type}")
        print(f"  Metrics to compute: {', '.join(self.metrics_to_compute)}")
        print(f"  GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        if self.use_gpu:
            print(f"  GPU ID: {self.gpu_id}")
        if self.original_dir:
            print(f"  Original files directory: {self.original_dir}")
        print(f"  Inference directory: {self.inference_dir}")
        
    def load_csv_data(self):
        """Load CSV dataset file and detect language"""
        try:
            df = pd.read_csv(self.csv_file, encoding='utf-8')
            print(f"Successfully loaded CSV: {self.csv_file}")
            print(f"Number of samples: {len(df)}")
            
            # Detect language and dataset type
            if 'common_voice' in str(self.csv_file).lower():
                self.language = 'zh'
                self.base_dataset_name = 'CommonVoice'
                print("Detected Chinese dataset, will use dCER evaluation")
            else:
                self.language = 'en'
                self.base_dataset_name = 'LibriSpeech'
                print("Detected English dataset, will use dWER evaluation")
            
            # Set dataset name based on type
            if self.dataset_type == "clean":
                self.dataset_name = self.base_dataset_name
            elif self.dataset_type == "noise":
                self.dataset_name = f"{self.base_dataset_name}_Noise"
            elif self.dataset_type == "blank":
                self.dataset_name = f"{self.base_dataset_name}_Blank"
            else:
                raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
            
            # Set result file paths now that we know the dataset
            dataset_suffix = self.base_dataset_name.lower()
            self.result_csv_path = self.result_dir / f"detailed_results_{self.model_name}_{self.dataset_type}_{dataset_suffix}.csv"
            
            print(f"Complete dataset name: {self.dataset_name}")
            print(f"Result CSV will be saved as: {self.result_csv_path}")
            
            # Auto-add correct primary metric for detected language
            if self.language == 'zh' and 'dcer' not in self.metrics_to_compute:
                self.metrics_to_compute.append('dcer')
                print(f"Auto-added dCER for Chinese dataset")
            elif self.language == 'en' and 'dwer' not in self.metrics_to_compute:
                self.metrics_to_compute.append('dwer')
                print(f"Auto-added dWER for English dataset")
                
            return df
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            sys.exit(1)
    
    def load_existing_results(self) -> pd.DataFrame:
        """Load existing results CSV if it exists"""
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
        """Create empty result DataFrame with all required columns"""
        columns = [
            'file_name', 'original_path', 'inference_path', 'ground_truth',
            'original_transcript', 'inference_transcript',
            'utmos', 'pesq', 'stoi'
        ]
        
        if self.language == 'zh':
            columns.extend(['original_cer', 'inference_cer', 'dcer'])
        else:
            columns.extend(['original_wer', 'inference_wer', 'dwer'])
        
        result_df = pd.DataFrame(index=range(len(input_df)))
        for col in columns:
            result_df[col] = np.nan
        
        result_df['file_name'] = input_df['file_name'].values
        result_df['ground_truth'] = input_df['transcription'].values
        result_df['original_path'] = input_df['file_path'].values
        
        return result_df
    
    def resolve_original_path(self, csv_path: str) -> Path:
        """Resolve original file path from CSV relative path"""
        if self.original_dir:
            clean_path = csv_path.lstrip('./')
            return self.original_dir / clean_path
        else:
            return Path(csv_path)
    
    def find_inference_audio(self, original_filename: str) -> Path:
        """Find corresponding inference audio file"""
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
    
    def merge_results(self, existing_df: pd.DataFrame, new_results: list) -> pd.DataFrame:
        """Merge new results with existing DataFrame"""
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
        """Evaluate only selected metrics"""
        results = {}
        
        results['original_path'] = str(original_path)
        results['inference_path'] = str(inference_path)
        results['ground_truth'] = ground_truth
        
        # Check if ASR metrics are needed
        need_asr = (self.language == 'zh' and 'dcer' in self.metrics_to_compute) or \
                   (self.language == 'en' and 'dwer' in self.metrics_to_compute)
        
        if need_asr:
            asr_result = evaluator.calculate_dwer_dcer(original_path, inference_path, ground_truth)
            if asr_result:
                results.update({k: v for k, v in asr_result.items() 
                              if k in ['original_transcript', 'inference_transcript']})
                
                if self.language == 'zh':
                    results.update({
                        'original_cer': asr_result.get('original_cer', np.nan),
                        'inference_cer': asr_result.get('inference_cer', np.nan),
                        'dcer': asr_result.get('dcer', np.nan)
                    })
                else:
                    results.update({
                        'original_wer': asr_result.get('original_wer', np.nan),
                        'inference_wer': asr_result.get('inference_wer', np.nan),
                        'dwer': asr_result.get('dwer', np.nan)
                    })
        
        if 'utmos' in self.metrics_to_compute:
            results['utmos'] = evaluator.calculate_utmos(inference_path)
        
        if 'pesq' in self.metrics_to_compute:
            results['pesq'] = evaluator.calculate_pesq(original_path, inference_path)
        
        if 'stoi' in self.metrics_to_compute:
            results['stoi'] = evaluator.calculate_stoi(original_path, inference_path)
        
        return results
    
    def save_results(self, results_df: pd.DataFrame) -> None:
        """Save detailed and summary results"""
        
        # Save detailed results
        detailed_csv_path = self.result_csv_path
        results_df.to_csv(detailed_csv_path, index=False, encoding='utf-8')
        print(f"Detailed results saved: {detailed_csv_path}")
        
        # Save summary results with correct naming
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
        """Create comprehensive summary statistics"""
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
        
        metric_columns = ['utmos', 'pesq', 'stoi']
        
        if self.language == 'zh':
            metric_columns.append('dcer')
        else:
            metric_columns.append('dwer')
        
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
        """Update existing summary with new metric data"""
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
        """Print comprehensive summary statistics"""
        print(f"\n" + "="*70)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*70)
        print(f"Model: {summary_data['Model']} ({summary_data['Frequency']})")
        print(f"Dataset: {summary_data['Dataset']} ({summary_data['Language']}) - {summary_data['Dataset_Type']}")
        print(f"Total files: {summary_data['Total_Files']}")
        print(f"Valid evaluations: {summary_data['Valid_Files']}")
        print(f"Timestamp: {summary_data['Timestamp']}")
        
        print(f"\nDETAILED METRICS STATISTICS:")
        print("-" * 70)
        
        primary_metric = 'DCER' if summary_data['Language'] == 'zh' else 'DWER'
        metrics_order = [primary_metric, 'UTMOS', 'PESQ', 'STOI']
        
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
        """Generate JSON config and copy sample audio files"""
        if len(results_df) == 0:
            print("No results to generate config from")
            return
            
        # Generate JSON config file (update existing or create new)
        config_path = self.config_dir / f"{self.model_name}_{self.frequency}_config.json"
        config = self.generate_or_update_json_config(results_df, config_path)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"JSON config saved: {config_path}")
        
        # Copy sample audio files for web interface
        self.copy_sample_audio_files(results_df)
    
    def generate_or_update_json_config(self, results_df: pd.DataFrame, config_path: Path) -> dict:
        """Generate or update JSON configuration file"""
        # Load existing config if it exists
        existing_config = {}
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
                print(f"Loading existing config from: {config_path}")
            except Exception as e:
                print(f"Warning: Could not load existing config: {e}")
        
        # Calculate statistics for current dataset
        metric_col = 'dcer' if self.language == 'zh' else 'dwer'
        metric_name = 'dCER' if self.language == 'zh' else 'dWER'
        
        valid_results = results_df.dropna(subset=[metric_col, 'utmos', 'pesq', 'stoi'])
        
        if len(valid_results) == 0:
            print("Warning: No valid results found for config generation")
            return existing_config
        
        # Generate current dataset section
        current_dataset_data = self.generate_dataset_section(valid_results, metric_name, metric_col)
        
        # Create complete config structure
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
        
        # Create all dataset sections (preserve existing data)
        all_datasets = ["LibriSpeech", "LibriSpeech_Noise", "LibriSpeech_Blank", 
                       "CommonVoice", "CommonVoice_Noise", "CommonVoice_Blank"]
        
        for dataset in all_datasets:
            if dataset == self.dataset_name:
                # Update current dataset with real data
                config[dataset] = current_dataset_data
                print(f"Updated {dataset} section with evaluation results")
            elif dataset in existing_config:
                # Keep existing data
                config[dataset] = existing_config[dataset]
                print(f"Preserved existing {dataset} section")
            else:
                # Create placeholder
                config[dataset] = self.create_dataset_specific_placeholder(dataset)
                print(f"Created placeholder {dataset} section")
        
        return config
    
    def generate_dataset_section(self, valid_results: pd.DataFrame, metric_name: str, metric_col: str) -> dict:
        """Generate dataset section with real evaluation results"""
        # Calculate total statistics
        total_stats = {
            metric_name: f"{valid_results[metric_col].mean():.2f}",
            'UTMOS': f"{valid_results['utmos'].mean():.1f}",
            'PESQ': f"{valid_results['pesq'].mean():.1f}",
            'STOI': f"{valid_results['stoi'].mean():.2f}"
        }
        
        # Select diverse samples
        selected_samples = self.select_diverse_samples(valid_results)
        
        # Generate sample entries using file names as keys
        samples = {}
        for sample_name, row in selected_samples:
            if sample_name.startswith('Sample_'):
                file_key = Path(row['file_name']).stem
                samples[file_key] = {
                    'Transcription': row['ground_truth'],
                    metric_name: f"{row[metric_col]:.2f}",
                    'UTMOS': f"{row['utmos']:.1f}",
                    'PESQ': f"{row['pesq']:.1f}",
                    'STOI': f"{row['stoi']:.2f}"
                }
        
        # Add error sample
        error_sample_data = {}
        for sample_name, row in selected_samples:
            if sample_name == 'Error_Sample_1':
                error_sample_data = {
                    'Transcription': row['ground_truth'],
                    metric_name: f"{row[metric_col]:.2f}",
                    'UTMOS': f"{row['utmos']:.1f}",
                    'PESQ': f"{row['pesq']:.1f}",
                    'STOI': f"{row['stoi']:.2f}"
                }
                break
        
        dataset_section = {
            "Total": total_stats,
            **samples,
            "Error_Sample_1": error_sample_data
        }
        
        return dataset_section
    
    def create_dataset_specific_placeholder(self, dataset_name: str) -> dict:
        """Create placeholder section with correct metric name for specific dataset"""
        # Determine metric based on dataset
        if 'CommonVoice' in dataset_name:
            metric_name = "dCER"
        else:
            metric_name = "dWER"
            
        placeholder_section = {
            "Total": {
                metric_name: "N/A",
                'UTMOS': "N/A",
                'PESQ': "N/A",
                'STOI': "N/A"
            }
        }
        
        # Add sample entries
        for i in range(1, 6):
            placeholder_section[f"Sample_{i}"] = {
                'Transcription': "N/A",
                metric_name: "N/A",
                'UTMOS': "N/A",
                'PESQ': "N/A",
                'STOI': "N/A"
            }
        
        placeholder_section["Error_Sample_1"] = {
            'Transcription': "N/A",
            metric_name: "N/A",
            'UTMOS': "N/A",
            'PESQ': "N/A",
            'STOI': "N/A"
        }
        
        return placeholder_section
    
    def get_dataset_path_for_audio(self) -> str:
        """Get the correct audio directory path based on dataset type"""
        if self.dataset_type == "clean":
            return self.base_dataset_name
        elif self.dataset_type == "noise":
            return f"{self.base_dataset_name}/Noise"
        elif self.dataset_type == "blank":
            return f"{self.base_dataset_name}/Blank"
        else:
            return self.base_dataset_name
    
    def copy_sample_audio_files(self, results_df: pd.DataFrame) -> None:
        """Copy sample audio files for web interface - organized by dataset type"""
        # Get correct audio directory path based on dataset type
        audio_dataset_path = self.get_dataset_path_for_audio()
        dataset_dir = self.audio_dir / audio_dataset_path
        
        original_dir = dataset_dir / "original"
        inference_dir = dataset_dir / self.model_name / self.frequency
        
        original_dir.mkdir(parents=True, exist_ok=True)
        inference_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Copying sample audio files to {dataset_dir}...")
        
        # Get valid results for file selection
        metric_col = 'dcer' if self.language == 'zh' else 'dwer'
        valid_results = results_df.dropna(subset=[metric_col, 'utmos', 'pesq', 'stoi'])
        
        if len(valid_results) == 0:
            print("Warning: No valid results found for file copying")
            return
        
        # Select files to copy - prioritize different speakers for diversity
        selected_samples = self.select_diverse_samples(valid_results)
        
        # Copy files
        copied_count = 0
        for i, (sample_name, row) in enumerate(selected_samples):
            file_name = row['file_name']
            base_name = Path(file_name).stem
            
            original_path = Path(row['original_path'])
            inference_path = Path(row['inference_path'])
            
            if original_path.exists() and inference_path.exists():
                try:
                    # Copy original file
                    original_dest = original_dir / f"{base_name}.flac"
                    shutil.copy2(original_path, original_dest)
                    
                    # Copy inference file
                    inference_dest = inference_dir / f"{base_name}.wav"
                    shutil.copy2(inference_path, inference_dest)
                    
                    # Show which key will be used in JSON
                    if sample_name.startswith('Sample_'):
                        print(f"  Copied sample (JSON key: {base_name}): {base_name}")
                    else:
                        print(f"  Copied {sample_name}: {base_name}")
                    copied_count += 1
                    
                except Exception as e:
                    print(f"  Warning: Failed to copy {base_name}: {e}")
            else:
                print(f"  Warning: Could not find files for {base_name}")
        
        print(f"Successfully copied {copied_count} sample pairs to audio directory")
    
    def select_diverse_samples(self, valid_results: pd.DataFrame) -> list:
        """Select diverse samples - first utterance from different speakers when possible"""
        selected_samples = []
        
        if self.base_dataset_name == 'LibriSpeech':
            # For LibriSpeech: select first utterance from different speakers
            results_with_keys = valid_results.copy()
            results_with_keys['sort_key'] = results_with_keys['file_name'].apply(
                lambda x: self.parse_librispeech_id(x)
            )
            results_sorted = results_with_keys.sort_values('sort_key')
            
            speakers_seen = set()
            sample_count = 0
            
            # First, try to get first utterance from different speakers
            for idx, row in results_sorted.iterrows():
                speaker_id = row['file_name'].split('-')[0]
                if speaker_id not in speakers_seen and sample_count < 5:
                    speakers_seen.add(speaker_id)
                    sample_count += 1
                    selected_samples.append((f'Sample_{sample_count}', row))
                    print(f"Selected Sample_{sample_count}: {row['file_name']} (Speaker {speaker_id})")
                    
        elif self.base_dataset_name == 'CommonVoice':
            # For CommonVoice: select from different speakers if speaker_id available
            if 'speaker_id' in valid_results.columns:
                speakers_seen = set()
                sample_count = 0
                
                for idx, row in valid_results.iterrows():
                    speaker_id = row.get('speaker_id', 'unknown')
                    if speaker_id not in speakers_seen and sample_count < 5:
                        speakers_seen.add(speaker_id)
                        sample_count += 1
                        selected_samples.append((f'Sample_{sample_count}', row))
                        print(f"Selected Sample_{sample_count}: {row['file_name']} (Speaker {speaker_id})")
            else:
                # Fallback: select first 5
                for i in range(min(5, len(valid_results))):
                    row = valid_results.iloc[i]
                    selected_samples.append((f'Sample_{i+1}', row))
        else:
            # Default: select first 5
            for i in range(min(5, len(valid_results))):
                row = valid_results.iloc[i]
                selected_samples.append((f'Sample_{i+1}', row))
        
        # If we don't have 5 samples yet, fill with remaining samples
        while len(selected_samples) < 5 and len(selected_samples) < len(valid_results):
            remaining_idx = len(selected_samples)
            if remaining_idx < len(valid_results):
                row = valid_results.iloc[remaining_idx]
                selected_samples.append((f'Sample_{len(selected_samples)+1}', row))
        
        # Add error sample (highest error rate)
        metric_col = 'dcer' if self.language == 'zh' else 'dwer'
        error_idx = valid_results[metric_col].idxmax()
        error_sample = valid_results.loc[error_idx]
        selected_samples.append(('Error_Sample_1', error_sample))
        
        return selected_samples
    
    def parse_librispeech_id(self, utt_id: str) -> tuple:
        """Parse LibriSpeech utterance ID and return sort key for speaker-chapter-utterance order"""
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
        """Execute selective evaluation pipeline"""
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
        
        need_asr = (self.language == 'zh' and 'dcer' in self.metrics_to_compute) or \
                   (self.language == 'en' and 'dwer' in self.metrics_to_compute)
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
        
        print(f"Starting evaluation of {len(df)} audio files...")
        print(f"Language detected: {self.language}")
        print(f"Primary metric: {'dCER' if self.language == 'zh' else 'dWER'}")
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
            
            # Simple check: process if no existing results or primary metric missing
            should_process = True
            if existing_results is not None:
                existing_row = existing_results[existing_results['file_name'] == file_name]
                if not existing_row.empty:
                    primary_metric_col = 'dcer' if self.language == 'zh' else 'dwer'
                    if primary_metric_col in existing_row.columns:
                        if not pd.isna(existing_row[primary_metric_col].iloc[0]):
                            should_process = False
            
            if not should_process:
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
        print(f"Processed {processed_count} files")
        
        if new_results:
            step_start = time.time()
            results_df = self.merge_results(existing_results, new_results)
            print(f"Result merging completed in: {time.time() - step_start:.2f} seconds")
        
        step_start = time.time()
        self.save_results(results_df)
        print(f"Result saving completed in: {time.time() - step_start:.2f} seconds")
        
        # Generate config file and copy sample audio files
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
        original_dir=args.original_dir
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