# Codec Inference Pipeline Scripts

此目錄包含所有codec的inference和完整pipeline腳本。

## 腳本結構

每個codec都有兩個腳本：

1. **inference_[codec].sh** - 產生推論檔案
2. **run_full_[codec]_pipeline.sh** - 完整pipeline（推論 → 評估 → 清理）

## 腳本列表

### 1. BigCodec
- **inference_bigcodec.sh** - 產生BigCodec 80kbps推論檔案
- **run_full_bigcodec_pipeline.sh** - 完整pipeline
- **環境**: `codec_eval`
- **配置**: 1個 (80Hz/80kbps)
- **資料集**: 3個 (LibriSpeech, Common Voice, Aishell)

### 2. LSCodec
- **inference_lscodec.sh** - 產生LSCodec 25Hz和50Hz推論檔案
- **run_full_lscodec_pipeline.sh** - 完整pipeline
- **環境**: `lscodec`
- **配置**: 2個 (25Hz, 50Hz)
- **資料集**: 3個 (LibriSpeech, Common Voice, Aishell)

### 3. FocalCodec
- **inference_focalcodec.sh** - 產生FocalCodec 12.5Hz, 25Hz, 50Hz推論檔案
- **run_full_focalcodec_pipeline.sh** - 完整pipeline
- **環境**: `focalcodec`
- **配置**: 3個 (12.5Hz, 25Hz, 50Hz)
- **資料集**: 3個 (LibriSpeech, Common Voice, Aishell)

### 4. FocalCodec-S
- **inference_focalcodec-S.sh** - 產生FocalCodec-S streaming版本推論檔案
- **run_full_focalcodec-S_pipeline.sh** - 完整pipeline
- **環境**: `focalcodec`
- **配置**: 3個 (50Hz_2k, 50Hz_4k, 50Hz_65k)
- **資料集**: 3個 (LibriSpeech, Common Voice, Aishell)

### 5. EnCodec
- **inference_encodec.sh** - 產生EnCodec 24khz多頻寬推論檔案
- **run_full_encodec_pipeline.sh** - 完整pipeline
- **環境**: `codec_eval`
- **配置**: 4個 (3.0kbps, 6.0kbps, 12.0kbps, 24.0kbps)
- **資料集**: 3個 (LibriSpeech, Common Voice, Aishell)

### 6. DAC
- **inference_dac.sh** - 產生DAC多採樣率推論檔案
- **run_full_dac_pipeline.sh** - 完整pipeline
- **環境**: `codec_eval`
- **配置**: 3個 (16kHz, 24kHz, 44kHz)
- **資料集**: 3個 (LibriSpeech, Common Voice, Aishell)

### 7. Mimi
- **inference_mimi.sh** - 產生Mimi兩種配置推論檔案
- **run_full_mimi_pipeline.sh** - 完整pipeline
- **環境**: `mimi_env`
- **配置**: 2個 (12.5Hz_8k, 12.5Hz_16k)
- **資料集**: 3個 (LibriSpeech, Common Voice, Aishell)

### 8. NanoCodec
- **inference_nanocodec.sh** - 產生NanoCodec三種變體推論檔案
- **run_full_nanocodec_pipeline.sh** - 完整pipeline
- **環境**: `nemo`
- **配置**: 3個 (12.5Hz/2K, 12.5Hz/4K, 21.5Hz/2K)
- **資料集**: 3個 (LibriSpeech, Common Voice, Aishell)

### 9. SpeechTokenizer
- **inference_speechtokenizer.sh** - 產生SpeechTokenizer推論檔案
- **run_full_speechtokenizer_pipeline.sh** - 完整pipeline
- **環境**: `codec_eval`
- **配置**: 1個 (50Hz)
- **資料集**: 3個 (LibriSpeech, Common Voice, Aishell)

## 使用方式

### 方式1: 僅產生推論檔案

```bash
# 產生BigCodec推論檔案
bash inference_bigcodec.sh

# 產生LSCodec推論檔案
bash inference_lscodec.sh

# ... 以此類推
```

### 方式2: 執行完整pipeline（推薦）

```bash
# BigCodec完整pipeline
bash run_full_bigcodec_pipeline.sh

# LSCodec完整pipeline
bash run_full_lscodec_pipeline.sh

# ... 以此類推
```

完整pipeline會自動執行：
1. 產生推論檔案
2. 執行評估（使用對應的run_[codec]_evaluations.sh）
3. 刪除推論檔案（保留資料夾結構）

## Pipeline流程

每個 `run_full_[codec]_pipeline.sh` 的執行流程：

```
Step 1: 產生推論檔案
├─ 啟動對應的conda環境
├─ 執行inference_[codec].sh
└─ 輸出到 /mnt/Internal/jieshiang/Inference_Result/[Codec]/[Config]/[Dataset]

Step 2: 執行評估
├─ 切換到codec_eval_pip_py39環境
├─ 執行對應的run_[codec]_evaluations.sh
└─ 結果輸出到 /home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison

Step 3: 清理推論檔案
├─ 刪除所有 *_inference.wav 檔案
└─ 保留資料夾結構
```

## 輸出位置

### 推論檔案
```
/mnt/Internal/jieshiang/Inference_Result/
├── BigCodec/80kbps/{librispeech,commonvoice,aishell}/
├── LSCodec/{25Hz,50Hz}/{librispeech,commonvoice,aishell}_recon/
├── FocalCodec/{12.5HZ,25HZ,50HZ,50HZ_2K,50HZ_4K,50HZ_65K}/{librispeech,commonvoice,aishell}/
├── EnCodec/24khz_{3.0,6.0,12.0,24.0}kbps/{librispeech,commonvoice,aishell}/
├── DAC/{16khz,24khz,44khz}_8kbps/{librispeech,commonvoice,aishell}/
├── MimiCodec/{12.5Hz_8k,12.5Hz_16k}/complete/{librispeech,commonvoice,aishell}/
├── NanoCodec/{12.5Hz/2K,12.5Hz/4K,21.5Hz/2K}/{librispeech,commonvoice,aishell}/
└── SpeechTokenizer/50Hz/complete/{librispeech,commonvoice,aishell}/
```

### 評估結果
```
/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/Codec_comparison/
├── detailed_results_[Codec]_[Config]_clean_[dataset].csv
├── summary_results_[Codec]_[Config]_clean_[dataset].csv
└── evaluation_logs_[timestamp]/
```

## 環境需求

| Codec | 環境名稱 | Python版本 |
|-------|---------|-----------|
| BigCodec | codec_eval | 3.8.20 |
| LSCodec | lscodec | 3.10.18 |
| FocalCodec | focalcodec | 3.10.19 |
| FocalCodec-S | focalcodec | 3.10.19 |
| EnCodec | codec_eval | 3.8.20 |
| DAC | codec_eval | 3.8.20 |
| Mimi | mimi_env | 3.10.19 |
| NanoCodec | nemo | 3.10.12 |
| SpeechTokenizer | codec_eval | 3.8.20 |
| **評估環境** | codec_eval_pip_py39 | 3.9.x |

## 總測試數

| Codec | 配置數 | 資料集數 | 總測試數 |
|-------|--------|---------|---------|
| BigCodec | 1 | 3 | 3 |
| LSCodec | 2 | 3 | 6 |
| FocalCodec | 3 | 3 | 9 |
| FocalCodec-S | 3 | 3 | 9 |
| EnCodec | 4 | 3 | 12 |
| DAC | 3 | 3 | 9 |
| Mimi | 2 | 3 | 6 |
| NanoCodec | 3 | 3 | 9 |
| SpeechTokenizer | 1 | 3 | 3 |
| **總計** | **22** | **3** | **66** |

## CSV檔案位置

所有inference腳本使用以下CSV檔案：
```
/home/jieshiang/Desktop/GitHub/Codec_comparison_tools/backup/
├── librispeech_test_clean_filtered.csv
├── common_voice_zh_CN_train_filtered.csv
└── aishell_filtered.csv
```

## 注意事項

1. **環境切換**: 推論和評估使用不同的環境，腳本會自動處理
2. **GPU使用**: 所有推論預設使用 `cuda:0`，可在腳本中修改
3. **磁碟空間**: 推論檔案會在評估後自動刪除，但評估期間需要足夠空間
4. **錯誤處理**: 所有腳本使用 `set -e`，任何錯誤會立即停止執行
5. **並行執行**: 不建議同時執行多個pipeline，避免GPU資源衝突

## 執行所有codec

如需執行所有codec的完整pipeline，可以創建一個master script：

```bash
#!/bin/bash
# run_all_codecs.sh

set -e

for codec in bigcodec lscodec focalcodec focalcodec-S encodec dac mimi nanocodec speechtokenizer; do
    echo "========================================="
    echo "Starting ${codec} pipeline..."
    echo "========================================="
    bash run_full_${codec}_pipeline.sh
    echo ""
done

echo "All codecs completed!"
```

## 疑難排解

### 1. 環境找不到
```bash
conda env list  # 確認環境存在
```

### 2. CSV檔案找不到
確認CSV檔案在正確位置：
```bash
ls -la /home/jieshiang/Desktop/GitHub/Codec_comparison_tools/backup/*.csv
```

### 3. 輸出目錄權限問題
```bash
mkdir -p /mnt/Internal/jieshiang/Inference_Result
chmod 755 /mnt/Internal/jieshiang/Inference_Result
```

### 4. GPU記憶體不足
修改腳本中的 `--device` 參數，或減少batch size

## 更新日誌

- 2026-01-02: 初始版本，包含所有9個codec的inference和pipeline腳本
- BigCodec環境已刪除，改用codec_eval環境
- Mimi批次腳本已更新支援file_path欄位
