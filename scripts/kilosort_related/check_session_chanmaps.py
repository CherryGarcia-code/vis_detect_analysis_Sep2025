import os
import csv
import re
from pathlib import Path
from collections import defaultdict

# Example usage: python check_session_chanmaps.py BG_031
# Looks for meta files in the expected folder structure for the given subject

def find_meta_files(subject_root):
    """
    Recursively find all .meta files matching *_imec0.ap.meta under subject_root.
    Returns list of (session_name, meta_file_path)
    """
    meta_files = []
    for root, dirs, files in os.walk(subject_root):
        for f in files:
            if f.endswith('imec0.ap.meta'):
                # Session name is the first part of the file name before the first '_g' (e.g. BG_031_25062025)
                match = re.match(r'(.*?)_g', f)
                session_name = match.group(1) if match else f.split('_g')[0]
                meta_files.append((session_name, os.path.join(root, f)))
    return meta_files

def extract_imro_file(meta_path):
    """
    Extract the imroFile field from a meta file, returning only the filename part.
    """
    try:
        with open(meta_path, 'r') as f:
            for line in f:
                if line.startswith('imroFile='):
                    # Get only the filename
                    imro_full = line.strip().split('=', 1)[1]
                    return os.path.basename(imro_full.replace('\\', '/'))
    except Exception:
        pass
    return ''

def main(subject):
    # Example path: X:/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_031
    # You may need to adjust this root path for your system
    # For now, assume data is not in repo, so user must provide full path
    subject_root = input(f"Enter full path to subject folder for {subject}: ")
    meta_files = find_meta_files(subject_root)
    results = []
    for session_name, meta_path in meta_files:
        imro_file = extract_imro_file(meta_path)
        results.append({'session': session_name, 'imro_file': imro_file})

    # Group by imro_file
    grouped_results = defaultdict(list)
    for row in results:
        grouped_results[row['imro_file']].append(row)

    # Output directory: data/subject_session_imro_matching/{subject}/
    out_dir = Path('data/subject_session_imro_matching') / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    for imro_file, rows in grouped_results.items():
        # Create a safe filename from the imro file name
        if imro_file:
            imro_stem = Path(imro_file).stem
            safe_imro_name = re.sub(r'[^\w\-. ]', '_', imro_stem)
            out_csv = out_dir / f'{subject}_{safe_imro_name}_sessions.csv'
        else:
            out_csv = out_dir / f'{subject}_unknown_imro_sessions.csv'

        with open(out_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['session', 'imro_file'])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Wrote summary to {out_csv}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python check_session_chanmaps.py <subject>")
    else:
        main(sys.argv[1])