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
import logging

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
                 use_v2_metrics: bool = False,
                 output_base_dir: str = None,
                 # --- Batch processing parameters ---
                 num_workers: int = 8,
                 asr_batch_size: int = 16,
                 # --- Logging parameters ---
                 enable_logging: bool = False,
                 log_dir: str = None):

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
            language=language,
            use_v2_metrics=use_v2_metrics,
            output_base_dir=output_base_dir
        )
        
        # 儲存新參數
        self.num_workers = min(num_workers, mp.cpu_count())
        self.asr_batch_size = asr_batch_size
        self.enable_logging = enable_logging

        # Initialize logger (will be set up after parent init completes)
        self.logger = None

        # Setup logging after parent class has set all attributes
        if self.enable_logging:
            self._setup_logging(log_dir)

        self._log_and_print(f"  ASR Batch Size: {self.asr_batch_size}")
        self._log_and_print(f"  PESQ/STOI CPU Workers: {self.num_workers}")
        self._log_and_print("-" * 30)

    def _setup_logging(self, log_dir: str = None):
        """Setup logging configuration"""
        # Determine log directory
        if log_dir:
            base_log_dir = Path(log_dir)
        elif hasattr(self, 'output_base_dir') and self.output_base_dir:
            base_log_dir = Path(self.output_base_dir) / "logs"
        elif hasattr(self, 'project_dir'):
            base_log_dir = Path(self.project_dir) / "logs"
        else:
            base_log_dir = Path.cwd() / "logs"

        base_log_dir.mkdir(parents=True, exist_ok=True)

        # Generate log filename with same naming pattern as CSV results
        dataset_name = Path(self.csv_file).stem.replace('_filtered', '')
        log_filename = f"log_{self.model_name}_{self.frequency}_{self.dataset_type}_{dataset_name}.log"
        log_path = base_log_dir / log_filename

        # Configure logger
        self.logger = logging.getLogger(f"FastEval_{self.model_name}_{self.frequency}")
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        self.logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        self.log_path = log_path
        self._log_and_print(f"Logging enabled: {log_path}")

    def _log_and_print(self, message: str, level: str = "info"):
        """Log message and print to console"""
        print(message)
        if self.logger:
            if level == "debug":
                self.logger.debug(message)
            elif level == "info":
                self.logger.info(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "error":
                self.logger.error(message)
            elif level == "critical":
                self.logger.critical(message)

    def _batch_transcribe_fast(self, evaluator,
                               audio_paths: list, desc: str) -> dict:
        """
        使用 evaluator 的 ASR pipeline 進行真正的批次轉錄。
        返回一個 {path: text} 的字典。
        """
        import gc
        import torch

        transcripts_map = {}

        self._log_and_print(f"Starting {desc} - Total files: {len(audio_paths)}, Batch size: {self.asr_batch_size}")

        for i in tqdm(range(0, len(audio_paths), self.asr_batch_size), desc=desc):
            batch_paths = audio_paths[i:i + self.asr_batch_size]
            batch_start_time = time.time()

            # 批次載入音檔
            batch_audio = []
            valid_paths_in_batch = []
            audio_durations = []

            for path in batch_paths:
                audio, sr = evaluator.load_audio_optimized(path, 16000)
                if audio is not None:
                    batch_audio.append(audio)
                    valid_paths_in_batch.append(path)
                    audio_durations.append(len(audio) / sr)  # Duration in seconds
                else:
                    transcripts_map[path] = "" # 記錄載入失敗

            if not batch_audio:
                continue

            # Log batch statistics
            avg_duration = sum(audio_durations) / len(audio_durations) if audio_durations else 0
            total_duration = sum(audio_durations)
            self._log_and_print(f"Batch {i//self.asr_batch_size + 1}: {len(batch_audio)} files, "
                               f"Avg duration: {avg_duration:.2f}s, Total: {total_duration:.2f}s",
                               level="debug")

            # 執行 GPU 批次轉錄
            try:
                if self.language == 'zh':
                    # Paraformer (FunASR): Process each file sequentially
                    # FunASR's batch processing can be unstable, so process one by one
                    for path in valid_paths_in_batch:
                        try:
                            result = evaluator.asr_pipeline.generate(input=path)
                            # FunASR returns list of dicts: [{'key': ..., 'text': ...}]
                            if isinstance(result, list) and len(result) > 0:
                                transcripts_map[path] = result[0].get('text', '')
                            elif isinstance(result, dict) and 'text' in result:
                                transcripts_map[path] = result['text']
                            else:
                                transcripts_map[path] = ""
                        except Exception as e:
                            self._log_and_print(f"Error transcribing {path}: {e}", level="error")
                            transcripts_map[path] = ""
                else:
                    # Whisper: True batch processing
                    batch_results = evaluator.asr_pipeline(
                        batch_audio,
                        generate_kwargs={"language": "en"},
                        batch_size=8,  # Whisper 內部批量大小，8-16 是最佳值
                        return_timestamps=False  # 不需要時間戳，加快速度
                    )

                    # 將結果映射回原始路徑
                    for path, result in zip(valid_paths_in_batch, batch_results):
                        transcripts_map[path] = result['text']

            except Exception as e:
                error_msg = f"Error during ASR batch {i//self.asr_batch_size + 1}: {e}"
                self._log_and_print(error_msg, level="error")
                for path in valid_paths_in_batch:
                    transcripts_map[path] = "" # 記錄轉錄失敗

            # Log batch processing time
            batch_time = time.time() - batch_start_time
            self._log_and_print(f"Batch {i//self.asr_batch_size + 1} completed in {batch_time:.2f}s "
                               f"({batch_time/len(batch_audio):.2f}s per file)",
                               level="debug")

            # 釋放記憶體 - 關鍵修復！
            del batch_audio
            del valid_paths_in_batch
            del audio_durations
            if 'batch_results' in locals():
                del batch_results

            # 清理 GPU 緩存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 強制垃圾回收
            gc.collect()

        return transcripts_map

    def run_evaluation(self):
        """
        *** 重寫的核心評估方法 (Batch-First) ***
        
        此方法取代了 enhanced_evaluation_pipeline.py 中的 'run_evaluation'
        """
        self.start_time = time.time()

        self._log_and_print("=" * 60)
        self._log_and_print("Starting FAST Batch-First Evaluation")
        self._log_and_print("=" * 60)

        # --- 1. 載入資料和模型 (與原版相同) ---
        step_start = time.time()
        df = self.load_csv_data()
        self._log_and_print(f"Data loading completed in: {time.time() - step_start:.2f} seconds")

        step_start = time.time()
        evaluator = self.EvaluatorClass(
            language=self.language,
            use_gpu=self.use_gpu,
            gpu_id=self.gpu_id
        )

        need_asr = ('dcer' in self.metrics_to_compute) or ('dwer' in self.metrics_to_compute)
        need_utmos = 'utmos' in self.metrics_to_compute
        need_mos_quality = 'MOS_Quality' in self.metrics_to_compute
        need_mos_naturalness = 'MOS_Naturalness' in self.metrics_to_compute
        need_speaker_similarity = 'speaker_similarity' in self.metrics_to_compute

        if need_asr or need_utmos or need_mos_quality or need_mos_naturalness or need_speaker_similarity:
            evaluator.load_models()
        self._log_and_print(f"Model loading completed in: {time.time() - step_start:.2f} seconds")

        # --- 2. 建立任務列表 (新邏輯) ---
        # **注意**: 此版本假定為「首次運行」，不檢查現有結果

        self._log_and_print("Building task lists...")
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

        self._log_and_print(f"Found {len(tasks)} valid file pairs for evaluation.")
        if not tasks:
            self._log_and_print("No valid tasks found. Exiting.", level="warning")
            return None

        # 建立一個以 file_name 為 key 的字典來儲存結果
        results_dict = {t['file_name']: t.copy() for t in tasks}

        # --- 3. 執行批次計算 (新邏輯) ---
        
        # 3a. PESQ/STOI (CPU 並行)
        if 'pesq' in self.metrics_to_compute or 'stoi' in self.metrics_to_compute:
            self._log_and_print(f"\nStarting PESQ/STOI calculation with {self.num_workers} workers...")
            pesq_stoi_tasks = [(t['original_path'], t['inference_path']) for t in tasks]

            step_start = time.time()
            pesq_results, stoi_results = evaluator.calculate_pesq_stoi_batch(
                pesq_stoi_tasks,
                num_workers=self.num_workers
            )
            self._log_and_print(f"PESQ/STOI batch calculation finished in {time.time() - step_start:.2f} seconds")
            
            # 將結果合併回字典
            for i, task in enumerate(tasks):
                file_name = task['file_name']
                if 'pesq' in self.metrics_to_compute:
                    results_dict[file_name]['pesq'] = pesq_results[i]
                if 'stoi' in self.metrics_to_compute:
                    results_dict[file_name]['stoi'] = stoi_results[i]

        # 3b. ASR (dWER/dCER) (GPU 批次)
        if need_asr:
            self._log_and_print(f"\nStarting ASR batch transcription (Batch Size: {self.asr_batch_size})...")
            step_start = time.time()

            # 收集所有獨特的音檔路徑
            original_paths = sorted(list(set([t['original_path'] for t in tasks])))
            inference_paths = sorted(list(set([t['inference_path'] for t in tasks])))

            self._log_and_print(f"Total unique original audio files: {len(original_paths)}")
            self._log_and_print(f"Total unique inference audio files: {len(inference_paths)}")

            # 批次轉錄
            orig_transcripts_map = self._batch_transcribe_fast(
                evaluator, original_paths, "Transcribing Original Audio"
            )
            inf_transcripts_map = self._batch_transcribe_fast(
                evaluator, inference_paths, "Transcribing Inference Audio"
            )

            self._log_and_print(f"ASR batch transcription finished in {time.time() - step_start:.2f} seconds")

            # 計算 dWER/dCER (這部分很快，在 CPU 上進行)
            self._log_and_print("Calculating dWER/dCER metrics...")
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
            self._log_and_print("dWER/dCER calculation complete.")

        # 3c. UTMOS (GPU 序列執行 - 已經很快) - V1 only
        if 'utmos' in self.metrics_to_compute:
            self._log_and_print("\nStarting UTMOS calculation...")
            step_start = time.time()
            for task in tqdm(tasks, desc="Calculating UTMOS"):
                score = evaluator.calculate_utmos(task['inference_path'])
                results_dict[task['file_name']]['utmos'] = score
            self._log_and_print(f"UTMOS calculation finished in {time.time() - step_start:.2f} seconds")

        # 3c-v2. MOS Quality (NISQA v2) - V2 only (GPU-accelerated batch processing)
        if 'MOS_Quality' in self.metrics_to_compute:
            self._log_and_print("\nStarting MOS Quality (NISQA v2) calculation (batch mode)...")
            step_start = time.time()

            # Collect all audio paths for batch processing
            audio_paths = [task['inference_path'] for task in tasks]

            # Batch prediction (much faster with GPU acceleration!)
            batch_results = evaluator.calculate_mos_quality_batch(audio_paths)

            # Map results back to tasks
            for task in tasks:
                score = batch_results.get(task['inference_path'], None)
                results_dict[task['file_name']]['MOS_Quality'] = score

            self._log_and_print(f"MOS Quality calculation finished in {time.time() - step_start:.2f} seconds")

        # 3c-v3. MOS Naturalness (UTMOS/RAMP) - V2 only - BATCH MODE
        if 'MOS_Naturalness' in self.metrics_to_compute:
            self._log_and_print("\nStarting MOS Naturalness calculation (batch mode)...")
            step_start = time.time()

            # Prepare audio paths for batch processing
            audio_paths = [task['inference_path'] for task in tasks]

            # Batch calculate MOS Naturalness
            naturalness_scores = evaluator.calculate_mos_naturalness_batch(audio_paths)

            # Map results back to tasks
            for task in tasks:
                score = naturalness_scores.get(task['inference_path'])
                results_dict[task['file_name']]['MOS_Naturalness'] = score

            self._log_and_print(f"MOS Naturalness calculation finished in {time.time() - step_start:.2f} seconds")

        # 3d. Speaker Similarity (批次處理)
        if 'speaker_similarity' in self.metrics_to_compute:
            self._log_and_print("\nStarting Speaker Similarity calculation (batch mode)...")
            step_start = time.time()

            # Prepare file pairs for batch processing
            file_pairs = [(task['original_path'], task['inference_path']) for task in tasks]

            # Batch calculate speaker similarity
            similarity_scores = evaluator.calculate_speaker_similarity_batch(file_pairs)

            # Map results back to tasks
            for task in tasks:
                file_pair = (task['original_path'], task['inference_path'])
                score = similarity_scores.get(file_pair)
                results_dict[task['file_name']]['speaker_similarity'] = score

            self._log_and_print(f"Speaker Similarity calculation finished in {time.time() - step_start:.2f} seconds")

        # --- 4. 整合結果 (新邏輯) ---
        step_start = time.time()
        # 將結果字典轉換回 DataFrame，確保順序與原始 CSV 一致
        final_results_list = []
        for idx, row in df.iterrows():
            file_name = row['file_name']
            if file_name in results_dict:
                final_results_list.append(results_dict[file_name])
        
        results_df = pd.DataFrame(final_results_list)
        self._log_and_print(f"\nResult consolidation finished in {time.time() - step_start:.2f} seconds")

        # --- 5. 儲存與產出 (使用繼承的函式) ---
        step_start = time.time()
        self.save_results(results_df)
        self._log_and_print(f"Result saving completed in: {time.time() - step_start:.2f} seconds")

        step_start = time.time()
        self.generate_config_and_copy_files(results_df)
        self._log_and_print(f"Config generation and file copying completed in: {time.time() - step_start:.2f} seconds")

        self.end_time = time.time()
        total_time = self.end_time - self.start_time

        self._log_and_print("\n" + "=" * 60)
        self._log_and_print("FAST Batch-First Evaluation Completed Successfully!")
        self._log_and_print("=" * 60)
        self._log_and_print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        self._log_and_print(f"Total files processed: {len(tasks)}")
        if tasks:
            self._log_and_print(f"Average time per file: {total_time/len(tasks):.2f} seconds")
        self._log_and_print(f"Result file: {self.result_csv_path}")

        if self.enable_logging:
            self._log_and_print(f"Log file: {self.log_path}")

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
                       choices=["dwer", "dcer", "utmos", "MOS_Quality", "MOS_Naturalness", "pesq", "stoi", "speaker_similarity"],
                       default=None,  # Will be set based on use_v2_metrics
                       help="Metrics to compute (dwer/dcer for ASR, utmos/MOS_Quality/MOS_Naturalness for quality, pesq/stoi for signal quality, speaker_similarity for identity preservation)")

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

    # V2 metrics support
    parser.add_argument("--use_v2_metrics", action="store_true",
                       help="Use V2 metrics with language-specific models (ResNet3/CAM++ for speaker, Paraformer for Chinese ASR, NISQA v2 for quality, RAMP/UTMOS for naturalness)")
    parser.add_argument("--output_base_dir", type=str,
                       help="Custom output base directory (for V2 metrics testing)")

    # --- Batch processing parameters ---
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of CPU workers for PESQ/STOI (default: 8)")
    parser.add_argument("--asr_batch_size", type=int, default=16,
                       help="Batch size for ASR transcription on GPU (default: 16)")

    # --- Logging parameters ---
    parser.add_argument("--enable_logging", action="store_true",
                       help="Enable logging to file")
    parser.add_argument("--log_dir", type=str,
                       help="Custom log directory (default: {output_base_dir}/logs or {project_dir}/logs)")

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
        use_v2_metrics=args.use_v2_metrics,
        output_base_dir=args.output_base_dir,
        # Batch processing parameters
        num_workers=args.num_workers,
        asr_batch_size=args.asr_batch_size,
        # Logging parameters
        enable_logging=args.enable_logging,
        log_dir=args.log_dir
    )
    
    try:
        results_df = pipeline.run_evaluation()
        pipeline._log_and_print("\nProgram executed successfully!")

    except KeyboardInterrupt:
        pipeline._log_and_print("\nProgram interrupted by user!", level="warning")

    except Exception as e:
        error_msg = f"\nError during execution: {e}"
        pipeline._log_and_print(error_msg, level="error")
        import traceback
        if pipeline.enable_logging:
            pipeline.logger.error(traceback.format_exc())
        traceback.print_exc()

if __name__ == "__main__":
    main()