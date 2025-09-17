#!/usr/bin/env python3
"""
Neural Audio Codec Evaluation Pipeline
======================================

Automated evaluation pipeline for neural audio codecs with comprehensive 
metrics assessment and web interface file generation.
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


class CodecEvaluationPipeline:
    """Main evaluation pipeline for neural audio codecs"""
    
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
                 testing_set: str = "Custom Test Set"):
        
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
        
        print(f"Initializing evaluation pipeline:")
        print(f"  Model: {self.model_name}")
        print(f"  Frequency: {self.frequency}")
        print(f"  Dataset type: {self.dataset_type}")
        print(f"  Inference directory: {self.inference_dir}")
        print(f"  Project directory: {self.project_dir}")
        
    def load_csv_data(self):
        """Load CSV dataset file"""
        try:
            df = pd.read_csv(self.csv_file, encoding='utf-8')
            print(f"Successfully loaded CSV: {self.csv_file}")
            print(f"Number of samples: {len(df)}")
            
            if 'common_voice' in str(self.csv_file).lower():
                self.language = 'zh'
                self.base_dataset_name = 'CommonVoice'
                print("Detected Chinese dataset, will use dCER evaluation")
            else:
                self.language = 'en'
                self.base_dataset_name = 'LibriSpeech'
                print("Detected English dataset, will use dWER evaluation")
            
            if self.dataset_type == "clean":
                self.dataset_name = self.base_dataset_name
                self.dataset_suffix = ""
            elif self.dataset_type == "noise":
                self.dataset_name = f"{self.base_dataset_name}_Noise"
                self.dataset_suffix = "/Noise"
            elif self.dataset_type == "blank":
                self.dataset_name = f"{self.base_dataset_name}_Blank"
                self.dataset_suffix = "/Blank"
            else:
                raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
                
            print(f"Complete dataset name: {self.dataset_name}")
                
            return df
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            sys.exit(1)
    
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
    
    def select_samples_for_json(self, results_df: pd.DataFrame) -> dict:
        """Select representative samples for JSON configuration"""
        if self.language == 'zh':
            metric_col = 'dcer'
            metric_name = 'dWER'
        else:
            metric_col = 'dwer'
            metric_name = 'dWER'
        
        total_stats = {
            metric_name: f"{results_df[metric_col].mean():.2f}",
            'UTMOS': f"{results_df['utmos'].mean():.1f}",
            'PESQ': f"{results_df['pesq'].mean():.1f}",
            'STOI': f"{results_df['stoi'].mean():.2f}"
        }
        
        samples = {}
        selected_files = {}
        
        if self.base_dataset_name == 'LibriSpeech':
            sorted_df = results_df.sort_values('file_name', key=lambda x: x.map(self.sort_by_speaker_chapter))
            
            speakers_seen = set()
            sample_count = 0
            
            for idx, row in sorted_df.iterrows():
                speaker_id = row['file_name'].split('-')[0]
                if speaker_id not in speakers_seen and sample_count < 5:
                    speakers_seen.add(speaker_id)
                    sample_count += 1
                    
                    sample_key = f'Sample_{sample_count}'
                    samples[sample_key] = {
                        'Transcription': row['ground_truth'],
                        metric_name: f"{row[metric_col]:.2f}",
                        'UTMOS': f"{row['utmos']:.1f}",
                        'PESQ': f"{row['pesq']:.1f}",
                        'STOI': f"{row['stoi']:.2f}"
                    }
                    selected_files[sample_key] = row['file_name']
        else:
            for i in range(min(5, len(results_df))):
                row = results_df.iloc[i]
                sample_key = f'Sample_{i+1}'
                samples[sample_key] = {
                    'Transcription': row['ground_truth'],
                    metric_name: f"{row[metric_col]:.2f}",
                    'UTMOS': f"{row['utmos']:.1f}",
                    'PESQ': f"{row['pesq']:.1f}",
                    'STOI': f"{row['stoi']:.2f}"
                }
                selected_files[sample_key] = row['file_name']
        
        max_error_idx = results_df[metric_col].idxmax()
        error_sample = results_df.loc[max_error_idx]
        
        error_sample_data = {
            'Transcription': error_sample['ground_truth'],
            metric_name: f"{error_sample[metric_col]:.2f}",
            'UTMOS': f"{error_sample['utmos']:.1f}",
            'PESQ': f"{error_sample['pesq']:.1f}",
            'STOI': f"{error_sample['stoi']:.2f}"
        }
        selected_files['Error_Sample_1'] = error_sample['file_name']
        
        return {
            'total_stats': total_stats,
            'samples': samples,
            'error_sample': error_sample_data,
            'selected_files': selected_files
        }
    
    def copy_audio_files(self, selected_data: dict, original_df: pd.DataFrame) -> None:
        """Copy selected audio files to project directory"""
        dataset_dir = self.audio_dir / (self.base_dataset_name + self.dataset_suffix)
        original_dir = dataset_dir / "original"
        inference_dir = dataset_dir / self.model_name / self.frequency
        
        original_dir.mkdir(parents=True, exist_ok=True)
        inference_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Copying audio files to: {dataset_dir}")
        
        file_mapping = selected_data['selected_files']
        
        for sample_name, actual_file_name in file_mapping.items():
            original_row = original_df[original_df['file_name'] == actual_file_name].iloc[0]
            original_path = Path(original_row['file_path'])
            
            inference_path = self.find_inference_audio(actual_file_name)
            
            if original_path.exists() and inference_path and inference_path.exists():
                original_dest = original_dir / f"{actual_file_name}.flac"
                shutil.copy2(original_path, original_dest)
                
                inference_dest = inference_dir / f"{actual_file_name}.wav"
                shutil.copy2(inference_path, inference_dest)
                
                print(f"Copied {sample_name}: {actual_file_name}")
            else:
                print(f"Warning: Could not find files for {actual_file_name}")
    
    def generate_json_config(self, selected_data: dict) -> None:
        """Generate JSON configuration file for web interface"""
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
            },
            self.dataset_name: {
                "Total": selected_data['total_stats']
            }
        }
        
        config[self.dataset_name].update(selected_data['samples'])
        config[self.dataset_name]["Error_Sample_1"] = selected_data['error_sample']
        
        if self.dataset_type == "clean":
            noise_dataset = f"{self.base_dataset_name}_Noise"
            blank_dataset = f"{self.base_dataset_name}_Blank"
            
            placeholder_data = {
                "Total": {"dWER": "0.5", "UTMOS": "3.5", "PESQ": "2.5", "STOI": "0.85"},
                "Sample_1": {"Transcription": "X", "dWER": "N/A", "UTMOS": "X", "PESQ": "X", "STOI": "X"},
                "Sample_2": {"Transcription": "X", "dWER": "N/A", "UTMOS": "X", "PESQ": "X", "STOI": "X"},
                "Sample_3": {"Transcription": "X", "dWER": "N/A", "UTMOS": "X", "PESQ": "X", "STOI": "X"},
                "Sample_4": {"Transcription": "X", "dWER": "N/A", "UTMOS": "X", "PESQ": "X", "STOI": "X"},
                "Sample_5": {"Transcription": "X", "dWER": "N/A", "UTMOS": "X", "PESQ": "X", "STOI": "X"},
                "Error_Sample_1": {"Transcription": "X", "dWER": "N/A", "UTMOS": "X", "PESQ": "X", "STOI": "X"}
            }
            
            config[noise_dataset] = placeholder_data.copy()
            config[blank_dataset] = placeholder_data.copy()
        
        config_filename = f"{self.model_name}_{self.frequency}_{self.dataset_type}_config.json"
        config_path = self.config_dir / config_filename
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        print(f"JSON configuration generated: {config_path}")
    
    def save_detailed_results(self, results_df: pd.DataFrame) -> None:
        """Save detailed evaluation results"""
        detailed_csv = self.result_dir / f"detailed_results_{self.model_name}_{self.timestamp}.csv"
        results_df.to_csv(detailed_csv, index=False, encoding='utf-8')
        print(f"Detailed results saved: {detailed_csv}")
        
        if self.language == 'zh':
            metric_col = 'dcer'
            metric_name = 'dCER'
        else:
            metric_col = 'dwer'
            metric_name = 'dWER'
            
        summary_data = {
            'Model': self.model_name,
            'Frequency': self.frequency,
            'Dataset': self.dataset_name,
            'Language': self.language,
            'Total_Samples': len(results_df),
            f'{metric_name}_Mean': results_df[metric_col].mean(),
            f'{metric_name}_Std': results_df[metric_col].std(),
            f'{metric_name}_Min': results_df[metric_col].min(),
            f'{metric_name}_Max': results_df[metric_col].max(),
            'UTMOS_Mean': results_df['utmos'].mean(),
            'PESQ_Mean': results_df['pesq'].mean(),
            'STOI_Mean': results_df['stoi'].mean()
        }
        
        summary_df = pd.DataFrame([summary_data])
        summary_csv = self.result_dir / f"summary_results_{self.model_name}_{self.timestamp}.csv"
        summary_df.to_csv(summary_csv, index=False, encoding='utf-8')
        print(f"Summary results saved: {summary_csv}")
        
        print(f"\n=== Evaluation Summary ===")
        print(f"Model: {self.model_name} ({self.frequency})")
        print(f"Dataset: {self.dataset_name} ({self.language})")
        print(f"Total samples: {len(results_df)}")
        print(f"{metric_name}: {results_df[metric_col].mean():.4f} ± {results_df[metric_col].std():.4f}")
        print(f"UTMOS: {results_df['utmos'].mean():.4f}")
        print(f"PESQ: {results_df['pesq'].mean():.4f}")
        print(f"STOI: {results_df['stoi'].mean():.4f}")
    
    def run_evaluation(self):
        """Execute complete evaluation pipeline"""
        self.start_time = time.time()
        
        print("=" * 50)
        print("Starting Neural Audio Codec Evaluation")
        print("=" * 50)
        
        step_start = time.time()
        df = self.load_csv_data()
        print(f"Data loading completed in: {time.time() - step_start:.2f} seconds")
        
        step_start = time.time()
        evaluator = AudioMetricsEvaluator(language=self.language)
        evaluator.load_models()
        print(f"Model loading completed in: {time.time() - step_start:.2f} seconds")
        
        results = []
        
        print(f"\nStarting evaluation of {len(df)} audio files...")
        evaluation_start = time.time()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluation Progress"):
            file_name = row['file_name']
            ground_truth = row['transcription']
            original_path = Path(row['file_path'])
            
            inference_path = self.find_inference_audio(file_name)
            
            if not original_path.exists():
                print(f"Warning: Original audio not found: {original_path}")
                continue
                
            if not inference_path or not inference_path.exists():
                print(f"Warning: Inference audio not found: {file_name}")
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
                    
                    result_data.update({k: v for k, v in metrics_result.items() 
                                      if k in ['original_transcript', 'inference_transcript']})
                    
                    if self.language == 'zh':
                        result_data.update({
                            'original_cer': metrics_result.get('original_cer', 0.0),
                            'inference_cer': metrics_result.get('inference_cer', 0.0),
                            'dcer': metrics_result.get('dcer', 0.0)
                        })
                    else:
                        result_data.update({
                            'original_wer': metrics_result.get('original_wer', 0.0),
                            'inference_wer': metrics_result.get('inference_wer', 0.0),
                            'dwer': metrics_result.get('dwer', 0.0)
                        })
                    
                    results.append(result_data)
                
            except Exception as e:
                print(f"Error evaluating file {file_name}: {e}")
                continue
        
        evaluation_time = time.time() - evaluation_start
        print(f"Audio evaluation completed in: {evaluation_time:.2f} seconds")
        
        if not results:
            print("Error: No files were successfully evaluated!")
            return
            
        step_start = time.time()
        results_df = pd.DataFrame(results)
        self.save_detailed_results(results_df)
        print(f"Result saving completed in: {time.time() - step_start:.2f} seconds")
        
        step_start = time.time()
        selected_data = self.select_samples_for_json(results_df)
        print(f"Sample selection completed in: {time.time() - step_start:.2f} seconds")
        
        step_start = time.time()
        self.copy_audio_files(selected_data, df)
        print(f"File copying completed in: {time.time() - step_start:.2f} seconds")
        
        step_start = time.time()
        self.generate_json_config(selected_data)
        print(f"JSON configuration completed in: {time.time() - step_start:.2f} seconds")
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        print("\n" + "=" * 50)
        print("Evaluation Completed Successfully!")
        print("=" * 50)
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"Average per file: {total_time/len(results):.2f} seconds") if results else None
        print(f"Successfully evaluated: {len(results)}/{len(df)} files")
        print(f"Detailed reports: {self.result_dir}")
        print(f"Audio files: {self.audio_dir}")
        print(f"Configuration files: {self.config_dir}")
        
        return results_df


def main():
    parser = argparse.ArgumentParser(description="Neural Audio Codec Evaluation Pipeline")
    
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
    
    args = parser.parse_args()
    
    pipeline = CodecEvaluationPipeline(
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
        testing_set=args.testing_set
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