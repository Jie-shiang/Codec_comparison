#!/usr/bin/env python3
"""
Test Files Cleanup Script

This script cleans up test files, directories, and __pycache__ folders.
"""

import os
import shutil
import argparse
from pathlib import Path
import glob

def cleanup_test_files(project_dir: str, confirm: bool = True):
    """Clean up test files, directories, and __pycache__ folders"""
    project_path = Path(project_dir)
    
    # Define paths to clean
    paths_to_clean = [
        project_path / "result" / "test_results",
        project_path / "audio" / "test_audio", 
        project_path / "configs" / "test_configs"
    ]
    
    # Find files and directories to clean
    files_to_clean = []
    dirs_to_clean = []
    
    for path in paths_to_clean:
        if path.exists():
            if path.is_dir():
                dirs_to_clean.append(path)
                # Count files in directory
                file_count = len(list(path.rglob('*'))) if path.exists() else 0
                files_to_clean.extend(list(path.rglob('*')))
            else:
                files_to_clean.append(path)
    
    # Also clean any test CSV files in result directory
    result_dir = project_path / "result"
    if result_dir.exists():
        test_csvs = list(result_dir.glob("test_*.csv"))
        files_to_clean.extend(test_csvs)

    # Find __pycache__ directories
    pycache_dirs = list(project_path.rglob('__pycache__'))
    
    print("TEST FILES AND CACHE CLEANUP")
    print("="*50)
    print(f"Project directory: {project_path}")
    print(f"Directories to remove: {len(dirs_to_clean)}")
    print(f"Files to remove: {len([f for f in files_to_clean if f.is_file()])}")
    print(f"__pycache__ directories to remove: {len(pycache_dirs)}")

    if dirs_to_clean:
        print(f"\nTest Directories:")
        for dir_path in dirs_to_clean:
            file_count = len([f for f in dir_path.rglob('*') if f.is_file()]) if dir_path.exists() else 0
            print(f"  • {dir_path} ({file_count} files)")

    if pycache_dirs:
        print(f"\n__pycache__ Directories:")
        for dir_path in pycache_dirs[:5]:
            print(f"  • {dir_path.relative_to(project_path)}")
        if len(pycache_dirs) > 5:
            print(f"  ... and {len(pycache_dirs) - 5} more")

    test_csvs = [f for f in files_to_clean if f.name.startswith('test_') and f.name.endswith('.csv')]
    if test_csvs:
        print(f"\nTest CSV files:")
        for csv_file in test_csvs[:5]:  # Show first 5
            print(f"  • {csv_file.name}")
        if len(test_csvs) > 5:
            print(f"  ... and {len(test_csvs) - 5} more")
    
    if not dirs_to_clean and not files_to_clean and not pycache_dirs:
        print("\nNo test files or cache directories found to clean.")
        return
    
    if confirm:
        print(f"\nThis will permanently delete all listed files and directories.")
        response = input("Do you want to proceed? (y/N): ").lower()
        if response != 'y' and response != 'yes':
            print("Cleanup cancelled.")
            return
    
    # Perform cleanup
    removed_dirs = 0
    removed_files = 0
    removed_pycache = 0
    errors = []
    
    print(f"\nCleaning up...")
    
    # Remove test directories
    for dir_path in dirs_to_clean:
        try:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                removed_dirs += 1
                print(f"Removed directory: {dir_path.name}")
        except Exception as e:
            errors.append(f"Could not remove directory {dir_path}: {e}")
    
    # Remove __pycache__ directories
    for cache_dir in pycache_dirs:
        try:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                removed_pycache += 1
                print(f"Removed __pycache__ directory: {cache_dir.relative_to(project_path)}")
        except Exception as e:
            errors.append(f"Could not remove directory {cache_dir}: {e}")
            
    # Remove individual test CSV files
    for csv_file in test_csvs:
        try:
            if csv_file.exists():
                csv_file.unlink()
                removed_files += 1
                print(f"Removed file: {csv_file.name}")
        except Exception as e:
            errors.append(f"Could not remove file {csv_file}: {e}")
    
    # Summary
    print(f"\nCleanup completed!")
    print(f"Removed {removed_dirs} test directories")
    print(f"Removed {removed_files} test files")
    print(f"Removed {removed_pycache} __pycache__ directories")
    
    if errors:
        print(f"\nErrors encountered:")
        for error in errors:
            print(f"  • {error}")
    else:
        print("All specified files and directories cleaned successfully!")

def main():
    parser = argparse.ArgumentParser(description="Clean up test files and __pycache__ directories.")
    
    parser.add_argument("--project_dir", type=str,
                       default="/home/jieshiang/Desktop/GitHub/Codec_comparison",
                       help="Project root directory path")
    parser.add_argument("--yes", action="store_true",
                       help="Skip confirmation prompt")
    parser.add_argument("--dry_run", action="store_true", 
                       help="Show what would be deleted without actually deleting")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be deleted")
        cleanup_test_files(args.project_dir, confirm=False)
        print("\nDry run completed. Use without --dry_run to actually delete files.")
    else:
        cleanup_test_files(args.project_dir, confirm=not args.yes)

if __name__ == "__main__":
    main()