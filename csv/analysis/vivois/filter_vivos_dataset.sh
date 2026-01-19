#!/bin/bash
# Filter Vietnamese (Vivos) Dataset
# 過濾越南語數據集並生成乾淨的 train/test 集

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_SCRIPT="${SCRIPT_DIR}/../analyze_and_filter_dataset.py"
INPUT_CSV="${SCRIPT_DIR}/vivos_filtered_full.csv"
OUTPUT_DIR="${SCRIPT_DIR}"
EXISTING_TEST="${SCRIPT_DIR}/vivos_filtered_test_clean.csv"

echo "========================================="
echo "Vivos Dataset Filtering"
echo "========================================="
echo ""
echo "Input CSV: ${INPUT_CSV}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Test Set Size: 1000"
echo ""

# 檢查文件是否存在
if [ ! -f "${INPUT_CSV}" ]; then
    echo "Error: Input CSV not found: ${INPUT_CSV}"
    exit 1
fi

# 執行分析和過濾
python3 "${ANALYSIS_SCRIPT}" \
    --csv "${INPUT_CSV}" \
    --language vi \
    --output_dir "${OUTPUT_DIR}" \
    --test_size 1000 \
    --existing_test "${EXISTING_TEST}" \
    --filter_method threshold \
    --plot

echo ""
echo "========================================="
echo "Filtering Complete!"
echo "========================================="
echo ""
echo "Generated files:"
echo "  - vivos_filtered_train.csv"
echo "  - vivos_filtered_test.csv"
echo "  - vivos_filter_summary.txt"
echo "  - vivos_distributions.png"
