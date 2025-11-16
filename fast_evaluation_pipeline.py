#!/usr/bin/env python3
"""
Fast Neural Audio Codec Evaluation Pipeline (Batch-First)

Optimized for first-run evaluation speed by batching all metrics.
- PESQ/STOI are run in parallel on CPU.
- ASR (dWER/dCER) is run in batches on GPU.

This script inherits all helper functions (saving, config generation)
from the enhanced pipeline but overrides the core execution logic.
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
import multiprocessing as mp

# 導入原始的 Evaluator 和 Pipeline 類別
# 我們將繼承 EnhancedCodecEvaluationPipeline 以重用其所有輔助方法
from metrics_evaluator import AudioMetricsEvaluator
from enhanced_evaluation_pipeline import EnhancedCodecEvaluationPipeline

class FastCodecEvaluationPipeline(EnhancedCodecEvaluationPipeline):
    """
    批次優先的評估流程，繼承自 EnhancedCodecEvaluationPipeline
    以重用 CSV 讀取、結果儲存和 JSON 生成邏輯。
    
    此類別重寫了 'run_evaluation' 方法以實現最大批次速度。
    """
    
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
                 language: str = None,
                 # --- 新增參數 ---
                 num_workers: int = 8,
                 asr_batch_size: int = 16):
        
        # 呼叫父類別的 __init__
        super().__init__(
            inference_dir=inference_dir,
            csv_file=csv_file,
            model_name=model_name,
            frequency=frequency,
            causality=causality,
            bit_rate=bit_rate,
            dataset_type=dataset_type,
            project_dir=project_dir,
            quantizers=quantizers,
            codebook_size=codebook_size,
            n_params=n_params,
            training_set=training_set,
            testing_set=testing_set,
            metrics_to_compute=metrics_to_compute,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
            original_dir=original_dir,
            language=language
        )
        
        # 儲存新參數
        self.num_workers = min(num_workers, mp.cpu_count())
        self.asr_batch_size = asr_batch_size
        print(f"  ASR Batch Size: {self.asr_batch_size}")
        print(f"  PESQ/STOI CPU Workers: {self.num_workers}")
        print("-" * 30)

    def _batch_transcribe_fast(self, evaluator: AudioMetricsEvaluator, 
                               audio_paths: list, desc: str) -> dict:
        """
        使用 evaluator 的 ASR pipeline 進行真正的批次轉錄。
        返回一個 {path: text} 的字典。
        """
        transcripts_map = {}
        
        for i in tqdm(range(0, len(audio_paths), self.asr_batch_size), desc=desc):
            batch_paths = audio_paths[i:i + self.asr_batch_size]
            
            # 批次載入音檔
            batch_audio = []
            valid_paths_in_batch = []
            for path in batch_paths:
                audio, sr = evaluator.load_audio_optimized(path, 16000)
                if audio is not None:
                    batch_audio.append(audio)
                    valid_paths_in_batch.append(path)
                else:
                    transcripts_map[path] = "" # 記錄載入失敗
            
            if not batch_audio:
                continue
            
            # 執行 GPU 批次轉錄
            try:
                batch_results = evaluator.asr_pipeline(
                    batch_audio,
                    generate_kwargs={"language": "zh" if self.language == 'zh' else "en"},
                    batch_size=len(batch_audio)
                )
                
                # 將結果映射回原始路徑
                for path, result in zip(valid_paths_in_batch, batch_results):
                    transcripts_map[path] = result['text']
                    
            except Exception as e:
                print(f"Error during ASR batch: {e}")
                for path in valid_paths_in_batch:
                    transcripts_map[path] = "" # 記錄轉錄失敗

        return transcripts_map

    def run_evaluation(self):
        """
        *** 重寫的核心評估方法 (Batch-First) ***
        
        此方法取代了 enhanced_evaluation_pipeline.py 中的 'run_evaluation'
        """
        self.start_time = time.time()
        
        print("=" * 60)
        print("Starting FAST Batch-First Evaluation")
        print("=" * 60)
        
        # --- 1. 載入資料和模型 (與原版相同) ---
        step_start = time.time()
        df = self.load_csv_data()
        print(f"Data loading completed in: {time.time() - step_start:.2f} seconds")

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

        # --- 2. 建立任務列表 (新邏輯) ---
        # **注意**: 此版本假定為「首次運行」，不檢查現有結果
        
        print("Building task lists...")
        tasks = []
        for idx, row in df.iterrows():
            file_name = row['file_name']
            ground_truth = row['transcription']
            original_path = str(self.resolve_original_path(row['file_path']))
            inference_path_obj = self.find_inference_audio(file_name)
            
            if not os.path.exists(original_path):
                continue
            if not inference_path_obj or not inference_path_obj.exists():
                continue
            
            tasks.append({
                'file_name': file_name,
                'ground_truth': ground_truth,
                'original_path': original_path,
                'inference_path': str(inference_path_obj)
            })

        print(f"Found {len(tasks)} valid file pairs for evaluation.")
        if not tasks:
            print("No valid tasks found. Exiting.")
            return None

        # 建立一個以 file_name 為 key 的字典來儲存結果
        results_dict = {t['file_name']: t.copy() for t in tasks}

        # --- 3. 執行批次計算 (新邏輯) ---
        
        # 3a. PESQ/STOI (CPU 並行)
        if 'pesq' in self.metrics_to_compute or 'stoi' in self.metrics_to_compute:
            print(f"\nStarting PESQ/STOI calculation with {self.num_workers} workers...")
            pesq_stoi_tasks = [(t['original_path'], t['inference_path']) for t in tasks]
            
            step_start = time.time()
            pesq_results, stoi_results = evaluator.calculate_pesq_stoi_batch(
                pesq_stoi_tasks, 
                num_workers=self.num_workers
            )
            print(f"PESQ/STOI batch calculation finished in {time.time() - step_start:.2f} seconds")
            
            # 將結果合併回字典
            for i, task in enumerate(tasks):
                file_name = task['file_name']
                if 'pesq' in self.metrics_to_compute:
                    results_dict[file_name]['pesq'] = pesq_results[i]
                if 'stoi' in self.metrics_to_compute:
                    results_dict[file_name]['stoi'] = stoi_results[i]

        # 3b. ASR (dWER/dCER) (GPU 批次)
        if need_asr:
            print(f"\nStarting ASR batch transcription (Batch Size: {self.asr_batch_size})...")
            step_start = time.time()
            
            # 收集所有獨特的音檔路徑
            original_paths = sorted(list(set([t['original_path'] for t in tasks])))
            inference_paths = sorted(list(set([t['inference_path'] for t in tasks])))
            
            # 批次轉錄
            orig_transcripts_map = self._batch_transcribe_fast(
                evaluator, original_paths, "Transcribing Original Audio"
            )
            inf_transcripts_map = self._batch_transcribe_fast(
                evaluator, inference_paths, "Transcribing Inference Audio"
            )
            
            print(f"ASR batch transcription finished in {time.time() - step_start:.2f} seconds")
            
            # 計算 dWER/dCER (這部分很快，在 CPU 上進行)
            print("Calculating dWER/dCER metrics...")
            for task in tqdm(tasks, desc="Calculating ASR Metrics"):
                file_name = task['file_name']
                
                # 從 map 中獲取轉錄稿
                original_transcript = orig_transcripts_map.get(task['original_path'], "")
                inference_transcript = inf_transcripts_map.get(task['inference_path'], "")
                ground_truth = task['ground_truth']

                # 儲存原始稿
                results_dict[file_name]['original_transcript_raw'] = original_transcript
                results_dict[file_name]['inference_transcript_raw'] = inference_transcript
                
                # --- 複製自 metrics_evaluator.py 的標準化邏輯 ---
                if self.language == 'zh':
                    original_transcript_simplified = evaluator.convert_traditional_to_simplified(original_transcript)
                    inference_transcript_simplified = evaluator.convert_traditional_to_simplified(inference_transcript)
                    ground_truth_simplified = evaluator.convert_traditional_to_simplified(ground_truth)
                else:
                    original_transcript_simplified = original_transcript
                    inference_transcript_simplified = inference_transcript
                    ground_truth_simplified = ground_truth
                
                ground_truth_norm = evaluator.normalize_text(ground_truth_simplified)
                original_norm = evaluator.normalize_text(original_transcript_simplified)
                inference_norm = evaluator.normalize_text(inference_transcript_simplified)
                
                results_dict[file_name]['original_transcript'] = original_norm
                results_dict[file_name]['inference_transcript'] = inference_norm
                
                if self.language == 'zh' and 'dcer' in self.metrics_to_compute:
                    original_cer = evaluator.fast_cer(ground_truth_norm, original_norm)
                    inference_cer = evaluator.fast_cer(ground_truth_norm, inference_norm)
                    results_dict[file_name]['original_cer'] = original_cer
                    results_dict[file_name]['inference_cer'] = inference_cer
                    results_dict[file_name]['dcer'] = inference_cer - original_cer
                    
                elif self.language == 'en' and 'dwer' in self.metrics_to_compute:
                    original_wer = evaluator.fast_wer(ground_truth_norm, original_norm)
                    inference_wer = evaluator.fast_wer(ground_truth_norm, inference_norm)
                    results_dict[file_name]['original_wer'] = original_wer
                    results_dict[file_name]['inference_wer'] = inference_wer
                    results_dict[file_name]['dwer'] = inference_wer - original_wer
            print("dWER/dCER calculation complete.")

        # 3c. UTMOS (GPU 序列執行 - 已經很快)
        if 'utmos' in self.metrics_to_compute:
            print("\nStarting UTMOS calculation...")
            step_start = time.time()
            for task in tqdm(tasks, desc="Calculating UTMOS"):
                score = evaluator.calculate_utmos(task['inference_path'])
                results_dict[task['file_name']]['utmos'] = score
            print(f"UTMOS calculation finished in {time.time() - step_start:.2f} seconds")

        # 3d. Speaker Similarity (GPU 序列執行 - 已經很快)
        if 'speaker_similarity' in self.metrics_to_compute:
            print("\nStarting Speaker Similarity calculation...")
            step_start = time.time()
            for task in tqdm(tasks, desc="Calculating Speaker Similarity"):
                score = evaluator.calculate_speaker_similarity(
                    task['original_path'], 
                    task['inference_path']
                )
                results_dict[task['file_name']]['speaker_similarity'] = score
            print(f"Speaker Similarity calculation finished in {time.time() - step_start:.2f} seconds")

        # --- 4. 整合結果 (新邏輯) ---
        step_start = time.time()
        # 將結果字典轉換回 DataFrame，確保順序與原始 CSV 一致
        final_results_list = []
        for idx, row in df.iterrows():
            file_name = row['file_name']
            if file_name in results_dict:
                final_results_list.append(results_dict[file_name])
        
        results_df = pd.DataFrame(final_results_list)
        print(f"\nResult consolidation finished in {time.time() - step_start:.2f} seconds")

        # --- 5. 儲存與產出 (使用繼承的函式) ---
        step_start = time.time()
        self.save_results(results_df)
        print(f"Result saving completed in: {time.time() - step_start:.2f} seconds")
        
        step_start = time.time()
        self.generate_config_and_copy_files(results_df)
        print(f"Config generation and file copying completed in: {time.time() - step_start:.2f} seconds")
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        print("\n" + "=" * 60)
        print("FAST Batch-First Evaluation Completed Successfully!")
        print("=" * 60)
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"Total files processed: {len(tasks)}")
        if tasks:
            print(f"Average time per file: {total_time/len(tasks):.2f} seconds")
        print(f"Result file: {self.result_csv_path}")
        
        return results_df

def main():
    parser = argparse.ArgumentParser(description="FAST Batch-First Neural Audio Codec Evaluation Pipeline")
    
    # --- 從 enhanced_evaluation_pipeline.py 複製所有參數 ---
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
                       choices=["dwer", "dcer", "utmos", "pesq", "stoi", "speaker_similarity"],
                       default=["dwer", "dcer", "utmos", "pesq", "stoi", "speaker_similarity"],
                       help="Metrics to compute (dwer/dcer for ASR, utmos/pesq/stoi for quality, speaker_similarity for identity preservation)")
    
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

    # --- 新增的批次大小參數 ---
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of CPU workers for PESQ/STOI (default: 8)")
    parser.add_argument("--asr_batch_size", type=int, default=16,
                       help="Batch size for ASR transcription on GPU (default: 16)")
    
    args = parser.parse_args()
    
    use_gpu = args.use_gpu and not args.cpu_only
    
    # *** 實例化新的 FastCodecEvaluationPipeline ***
    pipeline = FastCodecEvaluationPipeline(
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
        language=args.language,
        # 傳入新參數
        num_workers=args.num_workers,
        asr_batch_size=args.asr_batch_size
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