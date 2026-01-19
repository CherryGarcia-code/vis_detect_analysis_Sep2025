"""
Script to verify integrity of files moved to 'to_delete' against their matches in the parent folder.
Calculates MD5 checksums to ensure 100% identical content before permanent deletion.
Supports parallel processing to speed up file hashing.

Usage:
    python scripts/data_management/verify_deletion_integrity.py --csv table_output/backup_redundancy_check.csv --backup-root X:/.../raw_backup --to-delete-root X:/.../to_delete --n_workers 8
"""

import argparse
import hashlib
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

def calculate_md5(filepath, chunk_size=1024*1024):
    """Calculates MD5 hash of a file. Default chunk 1MB."""
    md5 = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
    except OSError:
        return None

def compare_partial(file1, file2, head_tail_bytes=1024*1024):
    """
    Compares the first and last N bytes of two files.
    Returns True if identical, False otherwise.
    """
    try:
        s1 = file1.stat().st_size
        s2 = file2.stat().st_size
        if s1 != s2:
            return False
            
        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
            # Compare Head
            if f1.read(head_tail_bytes) != f2.read(head_tail_bytes):
                return False
                
            # If file is smaller than 2*head_tail, we basically compared the whole thing or overlapped, so done.
            if s1 < 2 * head_tail_bytes:
                return True
                
            # Compare Tail
            f1.seek(-head_tail_bytes, 2)
            f2.seek(-head_tail_bytes, 2)
            if f1.read(head_tail_bytes) != f2.read(head_tail_bytes):
                return False
                
        return True
    except OSError:
        return False

def verify_worker(task_args):
    """
    Worker function to process a single file verification.
    """
    backup_file_str, matches_str, backup_root_str, to_delete_root_str, use_quick_mode = task_args
    
    backup_root = Path(backup_root_str)
    to_delete_root = Path(to_delete_root_str)
    original_src = Path(backup_file_str)
    
    # 1. Determine current path in 'to_delete'
    try:
        rel_path = original_src.relative_to(backup_root)
        current_path = to_delete_root / rel_path
    except ValueError:
        return ('ERROR', str(original_src), "Path relative check failed (not inside backup root)")
        
    if not current_path.exists():
        return ('MISSING', str(current_path), None)

    # 2. Get list of matches in parent
    if not isinstance(matches_str, str):
        matches = []
    else:
        matches = matches_str.split(';')
        
    if not matches or matches == ['']:
        return ('FAILED', str(current_path), "No matches listed in CSV")

    # 4. Check against matches
    match_found = False
    
    current_hash = None
    if not use_quick_mode:
        current_hash = calculate_md5(current_path)
        if current_hash is None:
            return ('ERROR', str(current_path), "Read Error")
    
    for match_path_str in matches:
        match_path = Path(match_path_str)
        if not match_path.exists():
            continue
            
        if use_quick_mode:
            # Quick Check: Partial Compare
            if compare_partial(current_path, match_path):
                match_found = True
                break
        else:
            # Full Check: MD5
            match_hash = calculate_md5(match_path)
            if current_hash == match_hash:
                match_found = True
                break
    
    if match_found:
        return ('PASSED', str(current_path), None)
    else:
        return ('FAILED', str(current_path), "Content mismatch with all candidates")

def verify_files(df, backup_root, to_delete_root, n_workers=4, quick=False):
    backup_root = str(Path(backup_root).resolve())
    to_delete_root = str(Path(to_delete_root).resolve())
    
    # Filter for FOUND files
    found_files = df[df['status'] == 'FOUND'].copy()
    mode_str = "Quick Mode (Head/Tail)" if quick else "Full MD5 Mode"
    print(f"Verifying {len(found_files)} files using {n_workers} workers. Mode: {mode_str}")
    
    tasks = []
    # Prepare tasks
    for idx, row in found_files.iterrows():
        tasks.append((
            str(row['backup_file']),
            row['matches'],
            backup_root,
            to_delete_root,
            quick
        ))

        
    passed_count = 0
    failed_count = 0
    missing_count = 0
    error_count = 0
    
    failures = []
    
    # Run in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(verify_worker, task): task for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), unit="file"):
            status, path, msg = future.result()
            
            if status == 'PASSED':
                passed_count += 1
            elif status == 'MISSING':
                missing_count += 1
                if missing_count <= 1:
                     failures.append((path, "FILE MISSING (Not found in to_delete folder)"))
            elif status == 'FAILED':
                failed_count += 1
                failures.append((path, msg))
            elif status == 'ERROR':
                error_count += 1
                failures.append((path, msg))

    print(f"\nVerification Complete.")
    print(f"  Verified Identical: {passed_count}")
    print(f"  Failed Verification: {failed_count}")
    print(f"  Missing (not in to_delete): {missing_count}")
    if error_count > 0:
        print(f"  Errors (Read/Path): {error_count}")
    
    if failures:
        print("\nWARNING: Issues detected (first 50 shown):")
        for i, (path, reason) in enumerate(failures):
            if i >= 50: 
                print(f"  ... and {len(failures) - 50} more.")
                break
            print(f"  {path} -> {reason}")
        print("\nDO NOT DELETE the 'to_delete' folder until resolved.")
    elif passed_count == 0:
        print("\nWARNING: No files were verified (0 passed). Check your paths.")
    else:
        print("\nSUCCESS: All checked files have identical copies in the parent folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify integrity of files in to_delete.")
    parser.add_argument('--csv', required=True)
    parser.add_argument('--backup-root', required=True)
    parser.add_argument('--to-delete-root', required=True)
    parser.add_argument('--n_workers', type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument('--quick', action='store_true', help="Use quick mode (check size + bitwise compare of first/last 1MB only).")
    
    args = parser.parse_args()
    
    if not Path(args.csv).exists():
        sys.exit(f"CSV not found: {args.csv}")
        
    df = pd.read_csv(args.csv)
    verify_files(df, args.backup_root, args.to_delete_root, n_workers=args.n_workers, quick=args.quick)
