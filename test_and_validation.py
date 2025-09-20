#!/usr/bin/env python3
"""
Neural Audio Codec Test and Validation Module

Comprehensive testing and validation module for neural audio codec evaluation pipeline.
Includes test mode with 20 samples and file validation functionality.
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
import librosa

from metrics_evaluator import AudioMetricsEvaluator


class CodecTestValidator:
    """Test and validation class for codec evaluation pipeline"""
    
    def __init__(self, 
                 inference_dir: str,
                 csv_file: str,
                 model_name: str = "TestCodec",
                 frequency: str = "50Hz",
                 project_dir: str = "/home/jieshiang/Desktop/GitHub/Codec_comparison",
                 use_gpu: bool = True,
                 gpu_id: int = 0,
                 original_dir: str = None):
        
        self.inference_dir = Path(inference_dir)
        self.csv_file = Path(project_dir) / "csv" / csv_file
        self.model_name = model_name
        self.frequency = frequency
        self.project_dir = Path(project_dir)
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.original_dir = Path(original_dir) if original_dir else None
        
        self.result_dir = self.project_dir / "result" / "test_results"
        self.audio_dir = self.project_dir / "audio" / "test_audio"
        self.config_dir = self.project_dir / "configs" / "test_configs"
        
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Initializing test validator:")
        print(f"  Model: {self.model_name}")
        print(f"  Frequency: {self.frequency}")
        print(f"  GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        if self.use_gpu:
            print(f"  GPU ID: {self.gpu_id}")
        if self.original_dir:
            print(f"  Original files directory: {self.original_dir}")
        print(f"  Inference directory: {self.inference_dir}")
        print(f"  Test results directory: {self.result_dir}")
    
    def load_test_data(self, num_samples: int = 20):
        """Load first N samples from CSV for testing"""
        try:
            df = pd.read_csv(self.csv_file, encoding='utf-8')
            print(f"Successfully loaded CSV: {self.csv_file}")
            print(f"Total samples available: {len(df)}")
            
            test_df = df.head(num_samples)
            print(f"Selected {len(test_df)} samples for testing")
            
            if 'common_voice' in str(self.csv_file).lower():
                self.language = 'zh'
                self.base_dataset_name = 'CommonVoice'
                print("Detected Chinese dataset")
            else:
                self.language = 'en'
                self.base_dataset_name = 'LibriSpeech'
                print("Detected English dataset")
                
            return test_df
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
    
    def validate_file_naming(self) -> dict:
        """Validate inference file naming conventions"""
        print("\n" + "="*50)
        print("FILE NAMING VALIDATION")
        print("="*50)
        
        validation_report = {
            'correct_files': [],
            'needs_processing': [],
            'other_files': [],
            'missing_files': []
        }
        
        df = pd.read_csv(self.csv_file, encoding='utf-8')
        expected_files = set(Path(fname).stem for fname in df['file_name'])
        
        print(f"Expected {len(expected_files)} inference files based on CSV")
        
        inference_files = []
        for ext in ['*.wav', '*.flac', '*.mp3']:
            inference_files.extend(self.inference_dir.glob(ext))
        
        print(f"Found {len(inference_files)} audio files in inference directory")
        
        for file_path in inference_files:
            file_name = file_path.name
            base_name = file_path.stem
            
            if base_name.endswith('_inference'):
                original_name = base_name[:-10]
            else:
                original_name = base_name
            
            if original_name in expected_files:
                if base_name.endswith('_inference'):
                    validation_report['correct_files'].append({
                        'file': file_name,
                        'status': 'Correct naming format',
                        'original_name': original_name
                    })
                else:
                    validation_report['needs_processing'].append({
                        'file': file_name,
                        'status': 'Needs _inference suffix',
                        'original_name': original_name,
                        'suggested_name': f"{original_name}_inference{file_path.suffix}"
                    })
            else:
                validation_report['other_files'].append({
                    'file': file_name,
                    'status': 'Not found in CSV',
                    'action': 'Please check if this file should be included'
                })
        
        found_originals = set()
        for file_path in inference_files:
            base_name = file_path.stem
            if base_name.endswith('_inference'):
                found_originals.add(base_name[:-10])
            else:
                found_originals.add(base_name)
        
        missing = expected_files - found_originals
        for missing_file in missing:
            validation_report['missing_files'].append({
                'original_name': missing_file,
                'status': 'No corresponding inference file found'
            })
        
        print(f"\nVALIDATION REPORT:")
        print(f"Correct files: {len(validation_report['correct_files'])}")
        print(f"Needs processing: {len(validation_report['needs_processing'])}")
        print(f"Other files: {len(validation_report['other_files'])}")
        print(f"Missing files: {len(validation_report['missing_files'])}")
        
        if validation_report['correct_files']:
            print(f"\nCORRECT FILES ({len(validation_report['correct_files'])}):")
            for item in validation_report['correct_files'][:5]:
                print(f"  • {item['file']} -> {item['original_name']}")
            if len(validation_report['correct_files']) > 5:
                print(f"  ... and {len(validation_report['correct_files']) - 5} more")
        
        if validation_report['needs_processing']:
            print(f"\nNEEDS PROCESSING ({len(validation_report['needs_processing'])}):")
            for item in validation_report['needs_processing'][:5]:
                print(f"  • {item['file']} -> should be: {item['suggested_name']}")
            if len(validation_report['needs_processing']) > 5:
                print(f"  ... and {len(validation_report['needs_processing']) - 5} more")
        
        if validation_report['other_files']:
            print(f"\nOTHER FILES ({len(validation_report['other_files'])}):")
            for item in validation_report['other_files'][:5]:
                print(f"  • {item['file']} - {item['action']}")
            if len(validation_report['other_files']) > 5:
                print(f"  ... and {len(validation_report['other_files']) - 5} more")
        
        if validation_report['missing_files']:
            print(f"\nMISSING FILES ({len(validation_report['missing_files'])}):")
            for item in validation_report['missing_files'][:5]:
                print(f"  • {item['original_name']} - {item['status']}")
            if len(validation_report['missing_files']) > 5:
                print(f"  ... and {len(validation_report['missing_files']) - 5} more")
        
        report_path = self.result_dir / f"file_validation_report_{self.timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, ensure_ascii=False, indent=2)
        print(f"\nValidation report saved: {report_path}")
        
        return validation_report
    
    def auto_fix_file_naming(self, validation_report: dict, dry_run: bool = True) -> None:
        """Automatically fix file naming issues"""
        if not validation_report['needs_processing']:
            print("No files need processing.")
            return
        
        print(f"\nAUTO-FIX FILE NAMING (Dry run: {dry_run})")
        print("="*50)
        
        for item in validation_report['needs_processing']:
            old_path = self.inference_dir / item['file']
            new_path = self.inference_dir / item['suggested_name']
            
            if dry_run:
                print(f"Would rename: {item['file']} -> {item['suggested_name']}")
            else:
                try:
                    old_path.rename(new_path)
                    print(f"Renamed: {item['file']} -> {item['suggested_name']}")
                except Exception as e:
                    print(f"Failed to rename {item['file']}: {e}")
        
        if dry_run:
            print(f"\nTo actually perform the renaming, run with --fix_naming flag")
    
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
    
    def resolve_original_path(self, csv_path: str) -> Path:
        """Resolve original file path from CSV relative path"""
        if self.original_dir:
            clean_path = csv_path.lstrip('./')
            return self.original_dir / clean_path
        else:
            return Path(csv_path)
    
    def validate_audio_files(self, test_df: pd.DataFrame) -> dict:
        """Validate audio file integrity and properties"""
        print("\n" + "="*50)
        print("AUDIO FILE VALIDATION")
        print("="*50)
        
        validation_results = {
            'valid_files': [],
            'corrupted_files': [],
            'missing_files': [],
            'format_issues': []
        }
        
        for idx, row in test_df.iterrows():
            file_name = row['file_name']
            original_path = self.resolve_original_path(row['file_path'])
            inference_path = self.find_inference_audio(file_name)
            
            original_status = self.check_audio_file(original_path, "original")
            
            if inference_path and inference_path.exists():
                inference_status = self.check_audio_file(inference_path, "inference")
            else:
                inference_status = {
                    'valid': False,
                    'error': 'File not found',
                    'path': 'N/A'
                }
            
            file_result = {
                'file_name': file_name,
                'original': original_status,
                'inference': inference_status
            }
            
            if original_status['valid'] and inference_status['valid']:
                validation_results['valid_files'].append(file_result)
            elif not inference_path:
                validation_results['missing_files'].append(file_result)
            elif not original_status['valid'] or not inference_status['valid']:
                validation_results['corrupted_files'].append(file_result)
        
        print(f"Valid pairs: {len(validation_results['valid_files'])}")
        print(f"Corrupted files: {len(validation_results['corrupted_files'])}")
        print(f"Missing inference files: {len(validation_results['missing_files'])}")
        
        if validation_results['corrupted_files']:
            print(f"\nCORRUPTED FILES:")
            for item in validation_results['corrupted_files'][:3]:
                print(f"  • {item['file_name']}")
                if not item['original']['valid']:
                    print(f"    Original: {item['original']['error']}")
                if not item['inference']['valid']:
                    print(f"    Inference: {item['inference']['error']}")
        
        if validation_results['missing_files']:
            print(f"\nMISSING INFERENCE FILES:")
            for item in validation_results['missing_files'][:3]:
                print(f"  • {item['file_name']}")
        
        if not validation_results['valid_files'] and self.original_dir is None:
            print(f"\nNOTE: No original directory specified with --original_dir")
            print(f"      Original file validation skipped. Only inference files were checked.")
            print(f"      To validate original files, use: --original_dir /path/to/original/files")
        
        return validation_results
    
    def check_audio_file(self, file_path: Path, file_type: str) -> dict:
        """Check individual audio file validity"""
        try:
            if not file_path.exists():
                return {'valid': False, 'error': 'File does not exist', 'path': str(file_path)}
            
            audio, sr = librosa.load(str(file_path), sr=None, mono=True)
            
            if len(audio) == 0:
                return {'valid': False, 'error': 'Empty audio file', 'path': str(file_path)}
            
            if sr <= 0:
                return {'valid': False, 'error': 'Invalid sample rate', 'path': str(file_path)}
            
            duration = len(audio) / sr
            if duration < 0.1:
                return {'valid': False, 'error': f'Too short: {duration:.3f}s', 'path': str(file_path)}
            
            return {
                'valid': True,
                'duration': duration,
                'sample_rate': sr,
                'channels': 1,
                'path': str(file_path)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e), 'path': str(file_path)}
    
    def run_test_evaluation(self, num_samples: int = 20) -> dict:
        """Run test evaluation on first N samples"""
        print("\n" + "="*60)
        print(f"RUNNING TEST EVALUATION ({num_samples} SAMPLES)")
        print("="*60)
        
        start_time = time.time()
        
        test_df = self.load_test_data(num_samples)
        if test_df is None:
            return None
        
        print("\nLoading evaluation models...")
        model_start = time.time()
        evaluator = AudioMetricsEvaluator(
            language=self.language,
            use_gpu=self.use_gpu,
            gpu_id=self.gpu_id
        )
        evaluator.load_models()
        model_time = time.time() - model_start
        print(f"Models loaded in: {model_time:.2f} seconds")
        
        results = []
        eval_start = time.time()
        
        print(f"\nEvaluating {len(test_df)} test samples...")
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test Evaluation"):
            file_name = row['file_name']
            ground_truth = row['transcription']
            original_path = self.resolve_original_path(row['file_path'])
            
            inference_path = self.find_inference_audio(file_name)
            
            if not original_path.exists() or not inference_path:
                continue
            
            try:
                metrics_result = evaluator.evaluate_audio_pair(
                    str(original_path), 
                    str(inference_path), 
                    ground_truth
                )
                
                if metrics_result:
                    result_data = {
                        'file_name': file_name,
                        'original_path': str(original_path),
                        'inference_path': str(inference_path),
                        'ground_truth': ground_truth,
                        'utmos': metrics_result.get('utmos', 0.0) or 0.0,
                        'pesq': metrics_result.get('pesq', 0.0) or 0.0,
                        'stoi': metrics_result.get('stoi', 0.0) or 0.0
                    }
                    
                    if self.language == 'zh':
                        result_data.update({
                            'original_cer': metrics_result.get('original_cer', 0.0),
                            'inference_cer': metrics_result.get('inference_cer', 0.0),
                            'dcer': metrics_result.get('dcer', 0.0)
                        })
                        metric_name = 'dCER'
                    else:
                        result_data.update({
                            'original_wer': metrics_result.get('original_wer', 0.0),
                            'inference_wer': metrics_result.get('inference_wer', 0.0),
                            'dwer': metrics_result.get('dwer', 0.0)
                        })
                        metric_name = 'dWER'
                    
                    result_data.update({k: v for k, v in metrics_result.items() 
                                      if k in ['original_transcript', 'inference_transcript']})
                    
                    results.append(result_data)
                
            except Exception as e:
                print(f"Error evaluating {file_name}: {e}")
                continue
        
        eval_time = time.time() - eval_start
        total_time = time.time() - start_time
        
        if not results:
            print("No files were successfully evaluated!")
            return None
        
        results_df = pd.DataFrame(results)
        
        # Generate file paths with dataset suffix
        dataset_suffix = self.base_dataset_name.lower()
        test_csv_path = self.result_dir / f"test_detailed_results_{self.model_name}_{dataset_suffix}_{self.timestamp}.csv"
        results_df.to_csv(test_csv_path, index=False, encoding='utf-8')
        
        summary_data = self.create_test_summary_statistics(results_df)
        summary_csv_path = self.result_dir / f"test_summary_results_{self.model_name}_{dataset_suffix}_{self.timestamp}.csv"
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
        
        test_config = self.generate_test_config(results_df)
        config_path = self.config_dir / f"test_config_{self.model_name}_{dataset_suffix}_{self.timestamp}.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(test_config, f, ensure_ascii=False, indent=2)
        
        self.copy_test_audio_files(results_df[:5])
        
        stats = self.calculate_test_statistics(results_df)
        validation_results = self.validate_test_results(results_df, stats)
        
        print("\n" + "="*60)
        print("TEST EVALUATION COMPLETED")
        print("="*60)
        print(f"Successfully evaluated: {len(results)}/{len(test_df)} files")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Model loading: {model_time:.2f} seconds")
        print(f"Evaluation: {eval_time:.2f} seconds")
        print(f"Average per file: {eval_time/len(results):.2f} seconds")
        
        print(f"\nMETRICS SUMMARY:")
        primary_metric = 'dCER' if self.language == 'zh' else 'dWER'
        primary_key = 'dcer' if self.language == 'zh' else 'dwer'
        print(f"{primary_metric}: {stats[primary_key]['mean']:.4f} ± {stats[primary_key]['std']:.4f}")
        print(f"UTMOS: {stats['utmos']['mean']:.3f} ± {stats['utmos']['std']:.3f}")
        print(f"PESQ: {stats['pesq']['mean']:.3f} ± {stats['pesq']['std']:.3f}")
        print(f"STOI: {stats['stoi']['mean']:.3f} ± {stats['stoi']['std']:.3f}")
        
        print(f"\nOUTPUT FILES:")
        print(f"Detailed CSV: {test_csv_path}")
        print(f"Summary CSV: {summary_csv_path}")
        print(f"Config JSON: {config_path}")
        print(f"Audio files: {self.audio_dir}")
        
        if validation_results['issues']:
            print(f"\nVALIDATION ISSUES DETECTED:")
            for issue in validation_results['issues']:
                print(f"  • {issue}")
        else:
            print(f"\nAll validation checks passed!")
        
        return {
            'results_df': results_df,
            'statistics': stats,
            'validation': validation_results,
            'timing': {
                'total_time': total_time,
                'model_loading_time': model_time,
                'evaluation_time': eval_time,
                'avg_per_file': eval_time/len(results)
            },
            'files': {
                'detailed_csv': test_csv_path,
                'summary_csv': summary_csv_path,
                'config': config_path,
                'audio_dir': self.audio_dir
            }
        }
    
    def create_test_summary_statistics(self, results_df: pd.DataFrame) -> dict:
        """Create comprehensive test summary statistics"""
        summary_data = {
            'Model': self.model_name,
            'Frequency': self.frequency,
            'Dataset': self.base_dataset_name,
            'Language': self.language,
            'Test_Mode': True,
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
        
        return summary_data
    
    def calculate_test_statistics(self, results_df: pd.DataFrame) -> dict:
        """Calculate test statistics"""
        stats = {}
        
        metric_cols = ['utmos', 'pesq', 'stoi']
        if self.language == 'zh':
            metric_cols.append('dcer')
        else:
            metric_cols.append('dwer')
        
        for col in metric_cols:
            if col in results_df.columns:
                values = results_df[col].dropna()
                stats[col] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'count': len(values)
                }
        
        return stats
    
    def validate_test_results(self, results_df: pd.DataFrame, stats: dict) -> dict:
        """Validate test results for correctness"""
        issues = []
        
        if 'utmos' in stats:
            if stats['utmos']['mean'] < 1.0 or stats['utmos']['mean'] > 5.0:
                issues.append(f"UTMOS mean ({stats['utmos']['mean']:.3f}) outside expected range [1.0, 5.0]")
        
        if 'pesq' in stats:
            if stats['pesq']['mean'] < 0.5 or stats['pesq']['mean'] > 4.5:
                issues.append(f"PESQ mean ({stats['pesq']['mean']:.3f}) outside expected range [0.5, 4.5]")
        
        if 'stoi' in stats:
            if stats['stoi']['mean'] < 0.0 or stats['stoi']['mean'] > 1.0:
                issues.append(f"STOI mean ({stats['stoi']['mean']:.3f}) outside expected range [0.0, 1.0]")
        
        if 'dwer' in stats:
            if stats['dwer']['mean'] > 1.0:
                issues.append(f"dWER mean ({stats['dwer']['mean']:.3f}) seems too high (>1.0)")
        
        if 'dcer' in stats:
            if stats['dcer']['mean'] > 1.0:
                issues.append(f"dCER mean ({stats['dcer']['mean']:.3f}) seems too high (>1.0)")
        
        expected_count = len(results_df)
        for metric, stat in stats.items():
            if stat['count'] < expected_count * 0.8:
                issues.append(f"{metric.upper()} has low coverage: {stat['count']}/{expected_count} samples")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def generate_test_config(self, results_df: pd.DataFrame) -> dict:
        """Generate test configuration JSON"""
        metric_name = 'dCER' if self.language == 'zh' else 'dWER'
        metric_col = 'dcer' if self.language == 'zh' else 'dwer'
        
        total_stats = {
            metric_name: f"{results_df[metric_col].mean():.2f}",
            'UTMOS': f"{results_df['utmos'].mean():.1f}",
            'PESQ': f"{results_df['pesq'].mean():.1f}",
            'STOI': f"{results_df['stoi'].mean():.2f}"
        }
        
        samples = {}
        for i in range(min(5, len(results_df))):
            row = results_df.iloc[i]
            samples[f'Sample_{i+1}'] = {
                'File_name': Path(row['file_name']).stem,  # 新增 File_name 欄位
                'Transcription': row['ground_truth'][:50] + '...' if len(row['ground_truth']) > 50 else row['ground_truth'],
                metric_name: f"{row[metric_col]:.2f}",
                'UTMOS': f"{row['utmos']:.1f}",
                'PESQ': f"{row['pesq']:.1f}",
                'STOI': f"{row['stoi']:.2f}"
            }
        
        max_error_idx = results_df[metric_col].idxmax()
        error_sample = results_df.loc[max_error_idx]
        
        config = {
            "model_info": {
                "modelName": self.model_name,
                "causality": "Non-Causal",
                "trainingSet": "Test Dataset",
                "testingSet": "Test Samples",
                "bitRate": "1.5",
                "parameters": {
                    "frameRate": self.frequency.replace('Hz', ''),
                    "quantizers": "4",
                    "codebookSize": "1024",
                    "nParams": "Test"
                }
            },
            f"{self.base_dataset_name}": {
                "Total": total_stats,
                **samples,
                "Error_Sample_1": {
                    'File_name': Path(error_sample['file_name']).stem,  # 新增 File_name 欄位
                    'Transcription': error_sample['ground_truth'][:50] + '...' if len(error_sample['ground_truth']) > 50 else error_sample['ground_truth'],
                    metric_name: f"{error_sample[metric_col]:.2f}",
                    'UTMOS': f"{error_sample['utmos']:.1f}",
                    'PESQ': f"{error_sample['pesq']:.1f}",
                    'STOI': f"{error_sample['stoi']:.2f}"
                }
            }
        }
        
        return config
    
    def sort_by_speaker_chapter(self, utt_id: str) -> tuple:
        """Sort audio files by speaker and chapter (LibriSpeech format)"""
        try:
            parts = utt_id.split('-')
            if len(parts) >= 2:
                speaker = int(parts[0])
                chapter = int(parts[1])
                return (speaker, chapter, utt_id)
            else:
                return (0, 0, utt_id)
        except:
            return (0, 0, utt_id)

    def copy_test_audio_files(self, results_df: pd.DataFrame) -> None:
        """Copy test audio files for web interface"""
        dataset_dir = self.audio_dir / self.base_dataset_name
        original_dir = dataset_dir / "original"
        inference_dir = dataset_dir / self.model_name / self.frequency
        
        original_dir.mkdir(parents=True, exist_ok=True)
        inference_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying test audio files...")
        
        selected_samples = []
        
        if self.base_dataset_name == 'LibriSpeech':
            sorted_df = results_df.copy()
            sorted_df['sort_key'] = sorted_df['file_name'].apply(self.sort_by_speaker_chapter)
            sorted_df = sorted_df.sort_values('sort_key')
            
            speakers_seen = set()
            
            for idx, row in sorted_df.iterrows():
                speaker_id = row['file_name'].split('-')[0]
                if speaker_id not in speakers_seen and len(selected_samples) < 5:
                    speakers_seen.add(speaker_id)
                    selected_samples.append(row)
                    print(f"Selected: {row['file_name']} (Speaker {speaker_id})")
        else:
            selected_samples = results_df.head(5).to_dict('records')
        
        for row in selected_samples:
            file_name = row['file_name']
            base_name = Path(file_name).stem
            
            original_path = Path(row['original_path'])
            inference_path = Path(row['inference_path'])
            
            if original_path.exists() and inference_path.exists():
                original_dest = original_dir / f"{base_name}.flac"
                shutil.copy2(original_path, original_dest)
                
                inference_dest = inference_dir / f"{base_name}.wav"
                shutil.copy2(inference_path, inference_dest)
                
                print(f"  Copied: {base_name}")
            else:
                print(f"  Warning: Could not copy {base_name} - files not found")


def main():
    parser = argparse.ArgumentParser(description="Neural Audio Codec Test and Validation Tool")
    
    parser.add_argument("--inference_dir", required=True, type=str,
                       help="Directory path containing inference audio files")
    parser.add_argument("--csv_file", required=True, type=str,
                       help="CSV dataset filename (must be in ./csv/ directory)")
    parser.add_argument("--model_name", type=str, default="TestCodec",
                       help="Name of codec model for testing")
    parser.add_argument("--frequency", type=str, default="50Hz",
                       help="Frame rate (e.g., 50Hz)")
    parser.add_argument("--project_dir", type=str,
                       default="/home/jieshiang/Desktop/GitHub/Codec_comparison",
                       help="Project root directory path")
    parser.add_argument("--original_dir", type=str,
                       help="Root directory path for original audio files")
    
    parser.add_argument("--mode", type=str, choices=["test", "validate", "both"], 
                       default="both", help="Mode: test evaluation, file validation, or both")
    parser.add_argument("--num_samples", type=int, default=20,
                       help="Number of samples for test evaluation")
    parser.add_argument("--fix_naming", action="store_true",
                       help="Actually fix file naming issues (default: dry run)")
    
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="Enable GPU acceleration (default: True)")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU device ID to use (default: 0)")
    parser.add_argument("--cpu_only", action="store_true",
                       help="Force CPU-only computation")
    
    args = parser.parse_args()
    
    use_gpu = args.use_gpu and not args.cpu_only
    
    validator = CodecTestValidator(
        inference_dir=args.inference_dir,
        csv_file=args.csv_file,
        model_name=args.model_name,
        frequency=args.frequency,
        project_dir=args.project_dir,
        use_gpu=use_gpu,
        gpu_id=args.gpu_id,
        original_dir=args.original_dir
    )
    
    try:
        if args.mode in ["validate", "both"]:
            print("Starting file validation...")
            validation_report = validator.validate_file_naming()
            validator.auto_fix_file_naming(validation_report, dry_run=not args.fix_naming)
            
            test_df = validator.load_test_data(args.num_samples)
            if test_df is not None:
                audio_validation = validator.validate_audio_files(test_df)
        
        if args.mode in ["test", "both"]:
            print("\nStarting test evaluation...")
            test_results = validator.run_test_evaluation(args.num_samples)
            
            if test_results:
                print(f"\nTest evaluation completed successfully!")
            else:
                print(f"\nTest evaluation failed!")
        
        print("\nAll operations completed!")
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user!")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()