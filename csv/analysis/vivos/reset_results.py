#!/usr/bin/env python3
"""
Reset evaluation results to force recalculation from scratch
"""
import pandas as pd
import sys

def reset_results(csv_file):
    """Clear evaluation result columns to force recalculation"""
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)

    print(f"Total rows: {len(df)}")

    # Check which columns exist
    result_cols = ['asr_result', 'cer', 'MOS_Quality', 'MOS_Naturalness', 'TER']
    existing_cols = [col for col in result_cols if col in df.columns]

    if not existing_cols:
        print("No result columns found. File is already clean.")
        return

    print(f"\nFound result columns: {existing_cols}")

    # Count non-null values before reset
    for col in existing_cols:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null} processed")

    # Reset all result columns to None
    for col in existing_cols:
        df[col] = None

    print(f"\nClearing all result columns...")

    # Save back
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved to {csv_file}")
    print(f"✓ All {len(df)} rows reset - ready for fresh evaluation")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python reset_results.py <csv_file>")
        print("Example: python reset_results.py mdcc_filtered_full.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    reset_results(csv_file)
