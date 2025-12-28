#!/usr/bin/env python3
"""
Check and clean up unused audio files in original folders
"""

import os
from pathlib import Path
from collections import defaultdict
import argparse


def get_audio_files(directory, return_stems=False):
    """Get audio files from directory, optionally return stems only"""
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

    if not os.path.exists(directory):
        return {} if return_stems else set()

    if return_stems:
        audio_files = {}
        for file in Path(directory).iterdir():
            if file.is_file() and file.suffix.lower() in audio_extensions:
                audio_files[file.stem] = file.name
        return audio_files
    else:
        audio_files = []
        for file in Path(directory).iterdir():
            if file.is_file() and file.suffix.lower() in audio_extensions:
                audio_files.append(file.name)
        return set(audio_files)


def scan_processed_files(base_dir, dataset_name):
    """Scan all subdirectories except 'original' and collect used filenames (stems only)"""
    used_files = set()
    processed_dirs = []

    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"⚠️  Directory not found: {base_dir}")
        return used_files

    for item in base_path.iterdir():
        if not item.is_dir():
            continue

        if item.name == 'original':
            continue

        for root, dirs, files in os.walk(item):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                    file_stem = Path(file).stem
                    used_files.add(file_stem)

                    rel_path = os.path.relpath(root, base_dir)
                    if rel_path not in processed_dirs:
                        processed_dirs.append(rel_path)

    print(f"\n📁 [{dataset_name}] Found {len(processed_dirs)} processing directories:")
    for dir_path in sorted(processed_dirs):
        file_count = len([f for f in Path(base_dir, dir_path).glob('*')
                         if f.is_file() and f.suffix.lower() in {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}])
        print(f"   - {dir_path} ({file_count} files)")

    print(f"\n📊 [{dataset_name}] Total {len(used_files)} unique files used")

    return used_files


def check_and_cleanup(base_dir, dataset_name, dry_run=True):
    """Check original folder and optionally delete unused files"""
    print(f"\n{'='*80}")
    print(f"🔍 Checking: {dataset_name}")
    print(f"{'='*80}")

    used_file_stems = scan_processed_files(base_dir, dataset_name)

    original_dir = Path(base_dir) / 'original'
    original_files_dict = get_audio_files(original_dir, return_stems=True)

    if not original_files_dict:
        print(f"\n⚠️  [{dataset_name}] original folder is empty or not found")
        return {
            'dataset': dataset_name,
            'total_original': 0,
            'used': 0,
            'unused': 0,
            'missing': 0
        }

    original_file_stems = set(original_files_dict.keys())

    print(f"\n📂 [{dataset_name}] original folder contains {len(original_files_dict)} files")

    # Find unused files
    unused_file_stems = original_file_stems - used_file_stems
    unused_files = [original_files_dict[stem] for stem in unused_file_stems]

    # Find missing files
    missing_file_stems = used_file_stems - original_file_stems
    stats = {
        'dataset': dataset_name,
        'total_original': len(original_files_dict),
        'used': len(used_file_stems & original_file_stems),
        'unused': len(unused_files),
        'missing': len(missing_file_stems),
        'unused_files': sorted(unused_files),
        'missing_files': sorted(missing_file_stems)
    }

    print(f"\n📈 [{dataset_name}] Statistics:")
    print(f"   ✓ In use: {stats['used']} files")
    print(f"   ✗ Unused: {stats['unused']} files")
    print(f"   ⚠ Missing: {stats['missing']} files")

    if unused_files:
        print(f"\n🗑️  [{dataset_name}] Unused files ({len(unused_files)}):")
        for i, file in enumerate(sorted(unused_files)[:10], 1):
            print(f"   {i}. {file}")
        if len(unused_files) > 10:
            print(f"   ... and {len(unused_files) - 10} more")

    if missing_file_stems:
        print(f"\n❌ [{dataset_name}] Missing files ({len(missing_file_stems)}):")
        for i, file in enumerate(sorted(missing_file_stems)[:10], 1):
            print(f"   {i}. {file}")
        if len(missing_file_stems) > 10:
            print(f"   ... and {len(missing_file_stems) - 10} more")

    if unused_files and not dry_run:
        print(f"\n🗑️  [{dataset_name}] Deleting unused files...")
        deleted_count = 0
        for file in unused_files:
            file_path = original_dir / file
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"   ⚠️  Failed to delete: {file} - {e}")
        print(f"   ✓ Deleted {deleted_count} files")
        stats['deleted'] = deleted_count
    elif unused_files and dry_run:
        print(f"\n⚠️  [{dataset_name}] Dry run mode: files will not be deleted")
        print(f"   Tip: Use --execute to actually delete files")

    return stats


def save_report(all_stats, output_file):
    """Save detailed report to file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Audio File Cleanup Report\n")
        f.write("=" * 80 + "\n\n")

        for stats in all_stats:
            f.write(f"Dataset: {stats['dataset']}\n")
            f.write(f"  Total original: {stats['total_original']}\n")
            f.write(f"  In use: {stats['used']}\n")
            f.write(f"  Unused: {stats['unused']}\n")
            f.write(f"  Missing: {stats['missing']}\n")

            if stats.get('unused_files'):
                f.write(f"\n  Unused files:\n")
                for file in stats['unused_files']:
                    f.write(f"    - {file}\n")

            if stats.get('missing_files'):
                f.write(f"\n  Missing files:\n")
                for file in stats['missing_files']:
                    f.write(f"    - {file}\n")

            f.write("\n" + "-" * 80 + "\n\n")

        total_original = sum(s['total_original'] for s in all_stats)
        total_unused = sum(s['unused'] for s in all_stats)
        total_missing = sum(s['missing'] for s in all_stats)

        f.write("Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total original files: {total_original}\n")
        f.write(f"Total unused files: {total_unused}\n")
        f.write(f"Total missing files: {total_missing}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Check and clean up audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (no deletion)
  python organize_audio_files.py

  # Execute deletion
  python organize_audio_files.py --execute

  # Check specific dataset only
  python organize_audio_files.py --dataset CommonVoice

  # Save detailed report
  python organize_audio_files.py --report report.txt
        """
    )

    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually delete files (default: dry run)'
    )

    parser.add_argument(
        '--dataset',
        choices=['CommonVoice', 'LibriSpeech', 'all'],
        default='all',
        help='Dataset to check (default: all)'
    )

    parser.add_argument(
        '--report',
        type=str,
        help='Save detailed report to file'
    )

    parser.add_argument(
        '--base-path',
        type=str,
        default='/home/jieshiang/Desktop/GitHub/Codec_comparison/audio',
        help='Base path for audio files'
    )

    args = parser.parse_args()

    datasets = []
    if args.dataset == 'all':
        datasets = ['CommonVoice', 'LibriSpeech']
    else:
        datasets = [args.dataset]

    mode = "EXECUTE MODE 🔴" if args.execute else "DRY RUN MODE 🟡"
    print(f"\n{'='*80}")
    print(f"🎵 Audio File Cleanup Tool - {mode}")
    print(f"{'='*80}")

    if not args.execute:
        print("⚠️  Dry run mode: files will not be deleted")
        print("   Use --execute to actually delete files\n")

    all_stats = []
    for dataset in datasets:
        dataset_path = Path(args.base_path) / dataset
        stats = check_and_cleanup(
            dataset_path,
            dataset,
            dry_run=not args.execute
        )
        all_stats.append(stats)

    print(f"\n{'='*80}")
    print("📊 Summary")
    print(f"{'='*80}")

    total_original = sum(s['total_original'] for s in all_stats)
    total_used = sum(s['used'] for s in all_stats)
    total_unused = sum(s['unused'] for s in all_stats)
    total_missing = sum(s['missing'] for s in all_stats)

    print(f"\nAll datasets:")
    print(f"  📂 Total original files: {total_original}")
    print(f"  ✓ In use: {total_used}")
    print(f"  ✗ Unused: {total_unused}")
    print(f"  ⚠ Missing: {total_missing}")

    if args.execute and total_unused > 0:
        total_deleted = sum(s.get('deleted', 0) for s in all_stats)
        print(f"  🗑️  Deleted: {total_deleted}")

    if args.report:
        save_report(all_stats, args.report)
        print(f"\n✓ Report saved to: {args.report}")

    print(f"\n{'='*80}\n")

    if total_missing > 0:
        print("⚠️  Warning: Some files are missing!")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
