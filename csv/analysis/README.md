# Audio Evaluation - 使用說明

## ✅ 準備就緒

兩個資料集已重置並準備開始評估：
- **粵語 (MDCC)**: 13,481 筆
- **越南語 (VIVOS)**: 12,420 筆

## 快速開始

### 粵語評估

```bash
cd mdcc
bash run_evaluation_batch.sh 0 128
```

### 越南語評估

```bash
cd vivois
bash run_evaluation_batch.sh 0 128
```

## 參數說明

```bash
bash run_evaluation_batch.sh [GPU_ID] [BATCH_SIZE]
```

- `GPU_ID`: GPU編號（預設0）
- `BATCH_SIZE`: 批次大小（預設128）

## 監控進度

### 查看Log

```bash
# 粵語
tail -f mdcc/evaluation_cantonese_*.log

# 越南語
tail -f vivois/evaluation_vietnamese_*.log
```

### 查看已處理筆數

```bash
# 粵語
python3 << EOF
import pandas as pd
df = pd.read_csv('mdcc/mdcc_filtered_full.csv')
print(f"Processed: {df['asr_result'].notna().sum()}/13481")
EOF

# 越南語
python3 << EOF
import pandas as pd
df = pd.read_csv('vivois/vivos_filtered_full.csv')
print(f"Processed: {df['asr_result'].notna().sum()}/12420")
EOF
```

## 重置資料

如需重新開始：

```bash
# 粵語
cd mdcc
python3 reset_results.py mdcc_filtered_full.csv

# 越南語
cd vivois
python3 reset_results.py vivos_filtered_full.csv
```

## 技術細節

### 粵語 (Cantonese)
- **ASR**: Whisper-large-v3 (語言: zh)
- **CER**: fast_cer() from metrics_evaluator_v3.py
- **TER**: calculate_ter() from metrics_evaluator_v3.py (pycantonese)
- **MOS**: NISQA v2 + UTMOS

### 越南語 (Vietnamese)
- **ASR**: PhoWhisper-large (語言: vi)
- **WER**: fast_wer() from metrics_evaluator_v3.py
- **TER**: calculate_ter() from metrics_evaluator_v3.py (Vietnamese diacritics)
- **MOS**: NISQA v2 + UTMOS

## 檔案結構

```
mdcc/
├── evaluate_cantonese_batch.py  # 評估腳本
├── run_evaluation_batch.sh      # Shell wrapper
├── reset_results.py              # 重置工具
└── mdcc_filtered_full.csv       # 資料檔

vivois/
├── evaluate_vietnamese_batch.py # 評估腳本
├── run_evaluation_batch.sh      # Shell wrapper
├── reset_results.py              # 重置工具
└── vivos_filtered_full.csv      # 資料檔
```

---

**修正完成日期**: 2026-01-12
