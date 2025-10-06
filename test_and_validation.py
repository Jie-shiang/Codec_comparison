#!/usr/bin/env python3
"""
Neural Audio Codec Test and Validation Module

Comprehensive testing and validation module using EnhancedCodecEvaluationPipeline
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import librosa

from enhanced_evaluation_pipeline import EnhancedCodecEvaluationPipeline


class CodecTestValidator:
    """Test and validation class using EnhancedCodecEvaluationPipeline"""
    
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
        self.csv_file = csv_file
        self.model_name = model_name
        self.frequency = frequency
        self.project_dir = Path(project_dir)
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.original_dir = original_dir
        
        self.result_dir = self.project_dir / "result" / "test_results"
        self.audio_dir = self.project_dir / "audio" / "test_audio"
        self.config_dir = self.project_dir / "configs" / "test_configs"
        
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_path = self.project_dir / "csv" / csv_file
        if 'common_voice' in str(csv_path).lower():
            self.language = 'zh'
            self.base_dataset_name = 'CommonVoice'
        else:
            self.language = 'en'
            self.base_dataset_name = 'LibriSpeech'
        
        print(f"Initializing test validator:")
        print(f"  Model: {self.model_name}")
        print(f"  Frequency: {self.frequency}")
        print(f"  Dataset: {self.csv_file}")
        print(f"  Language: {self.language}")
        print(f"  GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        if self.use_gpu:
            print(f"  GPU ID: {self.gpu_id}")
        if self.original_dir:
            print(f"  Original files directory: {self.original_dir}")
        print(f"  Inference directory: {self.inference_dir}")
        print(f"  Test results directory: {self.result_dir}")
    
    def validate_file_naming(self) -> dict:
        print("\n" + "="*50)
        print("FILE NAMING VALIDATION")
        print("="*50)
        
        validation_report = {
            'correct_files': [],
            'needs_processing': [],
            'other_files': [],
            'missing_files': []
        }
        
        csv_path = self.project_dir / "csv" / self.csv_file
        df = pd.read_csv(csv_path, encoding='utf-8')
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
    
    def validate_audio_files(self, num_samples: int = 20) -> dict:
        print("\n" + "="*50)
        print("AUDIO FILE VALIDATION")
        print("="*50)
        
        csv_path = self.project_dir / "csv" / self.csv_file
        df = pd.read_csv(csv_path, encoding='utf-8')
        test_df = df.head(num_samples)
        
        validation_results = {
            'valid_files': [],
            'corrupted_files': [],
            'missing_files': []
        }
        
        for idx, row in test_df.iterrows():
            file_name = row['file_name']
            base_name = Path(file_name).stem
            
            possible_names = [
                f"{base_name}_inference.wav",
                f"{base_name}_inference.flac",
                f"{base_name}.wav",
                f"{base_name}.flac"
            ]
            
            inference_path = None
            for name in possible_names:
                test_path = self.inference_dir / name
                if test_path.exists():
                    inference_path = test_path
                    break
            
            if inference_path and inference_path.exists():
                inference_status = self.check_audio_file(inference_path)
                
                if inference_status['valid']:
                    validation_results['valid_files'].append({
                        'file_name': file_name,
                        'inference': inference_status
                    })
                else:
                    validation_results['corrupted_files'].append({
                        'file_name': file_name,
                        'inference': inference_status
                    })
            else:
                validation_results['missing_files'].append({
                    'file_name': file_name,
                    'status': 'No inference file found'
                })
        
        print(f"Valid files: {len(validation_results['valid_files'])}")
        print(f"Corrupted files: {len(validation_results['corrupted_files'])}")
        print(f"Missing inference files: {len(validation_results['missing_files'])}")
        
        if validation_results['corrupted_files']:
            print(f"\nCORRUPTED FILES:")
            for item in validation_results['corrupted_files'][:3]:
                print(f"  • {item['file_name']}")
                print(f"    Inference: {item['inference']['error']}")
        
        if validation_results['missing_files']:
            print(f"\nMISSING INFERENCE FILES:")
            for item in validation_results['missing_files'][:3]:
                print(f"  • {item['file_name']}")
        
        return validation_results
    
    def check_audio_file(self, file_path: Path) -> dict:
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
        print("\n" + "="*60)
        print(f"RUNNING TEST EVALUATION ({num_samples} SAMPLES)")
        print("="*60)
        
        csv_path = self.project_dir / "csv" / self.csv_file
        full_df = pd.read_csv(csv_path, encoding='utf-8')
        test_df = full_df.head(num_samples)
        
        temp_csv = self.project_dir / "csv" / f"temp_test_{self.language}_{self.model_name}_{self.timestamp}.csv"
        test_df.to_csv(temp_csv, index=False, encoding='utf-8')
        
        try:
            # Determine which metrics to compute based on language
            metrics_to_compute = ['utmos', 'pesq', 'stoi']
            if self.language == 'zh':
                metrics_to_compute.append('dcer')
            else:
                metrics_to_compute.append('dwer')
            
            pipeline = EnhancedCodecEvaluationPipeline(
                inference_dir=str(self.inference_dir),
                csv_file=temp_csv.name,
                model_name=self.model_name,
                frequency=self.frequency,
                causality="Non-Causal",
                bit_rate="1.5",
                dataset_type="clean",
                project_dir=str(self.project_dir),
                quantizers="4",
                codebook_size="1024",
                n_params="Test",
                training_set="Test Dataset",
                testing_set="Test Samples",
                metrics_to_compute=metrics_to_compute,
                use_gpu=self.use_gpu,
                gpu_id=self.gpu_id,
                original_dir=self.original_dir,
                language=self.language  # Pass language explicitly
            )
            
            results_df = pipeline.run_evaluation()
            
            dataset_suffix = self.base_dataset_name.lower()
            test_csv_path = self.result_dir / f"test_detailed_results_{self.model_name}_{dataset_suffix}_{self.timestamp}.csv"
            results_df.to_csv(test_csv_path, index=False, encoding='utf-8')
            
            summary_data = self.create_test_summary(results_df)
            summary_csv_path = self.result_dir / f"test_summary_results_{self.model_name}_{dataset_suffix}_{self.timestamp}.csv"
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
            
            test_config = self.generate_test_config(results_df)
            config_path = self.config_dir / f"test_config_{self.model_name}_{dataset_suffix}_{self.timestamp}.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(test_config, f, ensure_ascii=False, indent=2)
            
            self.copy_test_audio_files(results_df[:5])
            
            print("\n" + "="*60)
            print("TEST EVALUATION COMPLETED")
            print("="*60)
            print(f"Successfully evaluated: {len(results_df)} files")
            
            if len(results_df) > 0:
                print(f"\nMETRICS SUMMARY:")
                primary_metric = 'dCER' if self.language == 'zh' else 'dWER'
                primary_key = 'dcer' if self.language == 'zh' else 'dwer'
                
                valid_primary = results_df[primary_key].dropna()
                if len(valid_primary) > 0:
                    print(f"{primary_metric}: {valid_primary.mean():.4f} ± {valid_primary.std():.4f}")
                
                for metric in ['utmos', 'pesq', 'stoi']:
                    valid_values = results_df[metric].dropna()
                    if len(valid_values) > 0:
                        print(f"{metric.upper()}: {valid_values.mean():.3f} ± {valid_values.std():.3f}")
            
            print(f"\nOUTPUT FILES:")
            print(f"Detailed CSV: {test_csv_path}")
            print(f"Summary CSV: {summary_csv_path}")
            print(f"Config JSON: {config_path}")
            print(f"Audio files: {self.audio_dir}")
            
            return {
                'results_df': results_df,
                'summary': summary_data,
                'files': {
                    'detailed_csv': test_csv_path,
                    'summary_csv': summary_csv_path,
                    'config': config_path,
                    'audio_dir': self.audio_dir
                }
            }
            
        except Exception as e:
            print(f"Error during test evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            if temp_csv.exists():
                temp_csv.unlink()
    
    def create_test_summary(self, results_df: pd.DataFrame) -> dict:
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
                        f'{metric.upper()}_Median': round(valid_values.median(), 4)
                    })
        
        return summary_data
    
    def generate_test_config(self, results_df: pd.DataFrame) -> dict:
        metric_name = 'dCER' if self.language == 'zh' else 'dWER'
        metric_col = 'dcer' if self.language == 'zh' else 'dwer'
        
        # Filter out rows with missing inference_path (NaN values)
        valid_df = results_df.copy()
        if 'inference_path' in valid_df.columns:
            valid_df = valid_df[valid_df['inference_path'].notna()]
        
        if len(valid_df) == 0:
            print(f"Warning: No valid data with inference paths found")
            return {}
        
        if metric_col not in valid_df.columns:
            print(f"Warning: {metric_col} column not found in results")
            return {}
        
        valid_results = valid_df.dropna(subset=[metric_col, 'utmos', 'pesq', 'stoi'])
        
        if len(valid_results) == 0:
            print("Warning: No valid results with all metrics")
            # Try without ASR metric
            valid_results = valid_df.dropna(subset=['utmos', 'pesq', 'stoi'])
            if len(valid_results) == 0:
                return {}
        
        # Calculate total stats
        total_stats = {
            'UTMOS': f"{valid_results['utmos'].mean():.1f}",
            'PESQ': f"{valid_results['pesq'].mean():.1f}",
            'STOI': f"{valid_results['stoi'].mean():.2f}"
        }
        
        # Convert metric column to numeric and calculate mean
        metric_values = pd.to_numeric(valid_results[metric_col], errors='coerce')
        if metric_values.notna().any():
            total_stats[metric_name] = f"{metric_values.mean():.2f}"
        
        samples = {}
        for i in range(min(5, len(valid_results))):
            row = valid_results.iloc[i]
            sample_data = {
                'File_name': Path(row['file_name']).stem,
                'Transcription': row['ground_truth'][:100] + '...' if len(row['ground_truth']) > 100 else row['ground_truth'],
                'Origin': row.get('original_transcript', 'N/A')[:100] + '...' if isinstance(row.get('original_transcript'), str) and len(row.get('original_transcript', '')) > 100 else row.get('original_transcript', 'N/A'),
                'Inference': row.get('inference_transcript', 'N/A')[:100] + '...' if isinstance(row.get('inference_transcript'), str) and len(row.get('inference_transcript', '')) > 100 else row.get('inference_transcript', 'N/A'),
                'UTMOS': f"{row['utmos']:.1f}",
                'PESQ': f"{row['pesq']:.1f}",
                'STOI': f"{row['stoi']:.2f}"
            }
            
            # Add metric if it's numeric
            metric_val = pd.to_numeric(row[metric_col], errors='coerce')
            if pd.notna(metric_val):
                sample_data[metric_name] = f"{metric_val:.2f}"
            
            samples[f'Sample_{i+1}'] = sample_data
        
        # Find error sample with proper error handling
        try:
            metric_numeric = pd.to_numeric(valid_results[metric_col], errors='coerce')
            if metric_numeric.notna().any():
                max_error_idx = metric_numeric.idxmax()
            else:
                max_error_idx = valid_results.index[0]
        except Exception as e:
            print(f"Warning: Could not find max error: {e}")
            max_error_idx = valid_results.index[0]
        
        error_sample = valid_results.loc[max_error_idx]
        
        error_sample_data = {
            'File_name': Path(error_sample['file_name']).stem,
            'Transcription': error_sample['ground_truth'][:100] + '...' if len(error_sample['ground_truth']) > 100 else error_sample['ground_truth'],
            'Origin': error_sample.get('original_transcript', 'N/A')[:100] + '...' if isinstance(error_sample.get('original_transcript'), str) and len(error_sample.get('original_transcript', '')) > 100 else error_sample.get('original_transcript', 'N/A'),
            'Inference': error_sample.get('inference_transcript', 'N/A')[:100] + '...' if isinstance(error_sample.get('inference_transcript'), str) and len(error_sample.get('inference_transcript', '')) > 100 else error_sample.get('inference_transcript', 'N/A'),
            'UTMOS': f"{error_sample['utmos']:.1f}",
            'PESQ': f"{error_sample['pesq']:.1f}",
            'STOI': f"{error_sample['stoi']:.2f}"
        }
        
        # Add error metric if numeric
        error_metric_val = pd.to_numeric(error_sample[metric_col], errors='coerce')
        if pd.notna(error_metric_val):
            error_sample_data[metric_name] = f"{error_metric_val:.2f}"
        
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
                "Error_Sample_1": error_sample_data
            }
        }
        
        return config
    
    def copy_test_audio_files(self, results_df: pd.DataFrame) -> None:
        dataset_dir = self.audio_dir / self.base_dataset_name
        original_dir = dataset_dir / "original"
        inference_dir = dataset_dir / self.model_name / self.frequency
        
        original_dir.mkdir(parents=True, exist_ok=True)
        inference_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying test audio files...")
        
        copied_count = 0
        skipped_count = 0
        
        for idx, row in results_df.iterrows():
            file_name = row['file_name']
            base_name = Path(file_name).stem
            
            # Check if paths exist and are not NaN
            if 'original_path' not in row or 'inference_path' not in row:
                skipped_count += 1
                continue
                
            # Handle NaN values
            if pd.isna(row['original_path']) or pd.isna(row['inference_path']):
                skipped_count += 1
                if skipped_count <= 3:
                    print(f"  Skipping {base_name}: Missing path information")
                continue
            
            try:
                original_path = Path(row['original_path'])
                inference_path = Path(row['inference_path'])
                
                if original_path.exists() and inference_path.exists():
                    try:
                        original_dest = original_dir / f"{base_name}.flac"
                        shutil.copy2(original_path, original_dest)
                        
                        inference_dest = inference_dir / f"{base_name}.wav"
                        shutil.copy2(inference_path, inference_dest)
                        
                        print(f"  Copied: {base_name}")
                        copied_count += 1
                    except Exception as e:
                        print(f"  Warning: Failed to copy {base_name}: {e}")
                else:
                    if not original_path.exists():
                        print(f"  Warning: Original file not found: {base_name}")
                    if not inference_path.exists():
                        print(f"  Warning: Inference file not found: {base_name}")
                    skipped_count += 1
            except Exception as e:
                print(f"  Error processing {base_name}: {e}")
                skipped_count += 1
                continue
        
        if skipped_count > 3:
            print(f"  ... and {skipped_count - 3} more files skipped")
        
        print(f"Successfully copied {copied_count} sample pairs")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} files due to missing paths or files")


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
            
            audio_validation = validator.validate_audio_files(args.num_samples)
        
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