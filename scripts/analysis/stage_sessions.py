
import os
import sys
import glob
import pandas as pd
import numpy as np
from scipy.stats import norm

# Ensure project root is in path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from visdetect.core.session import load_session

def calculate_dprime(hit_rate, fa_rate):
    # Clip rates to avoid infinity
    hit_rate = np.clip(hit_rate, 0.01, 0.99)
    fa_rate = np.clip(fa_rate, 0.01, 0.99)
    return norm.ppf(hit_rate) - norm.ppf(fa_rate)

def stage_sessions(subject_dir, subject_name, output_csv):
    pkl_files = glob.glob(os.path.join(subject_dir, f"{subject_name}_*.pkl"))
    pkl_files.sort()
    
    records = []
    
    log_file = open('staging_log.txt', 'w')
    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()
    
    log(f"Found {len(pkl_files)} sessions for {subject_name}")
    
    for pkl_path in pkl_files:
        try:
            session = load_session(pkl_path)
            
            # Extract basic metrics
            hits = 0
            misses = 0
            fas = 0
            crs = 0
            aborts = 0
            
            for t in session.trials:
                outcome = t.trialoutcome
                if outcome == 'Hit':
                    hits += 1
                elif outcome == 'Miss':
                    misses += 1
                elif outcome == 'FA':
                    fas += 1
                elif outcome == 'Ref':
                    crs += 1
                elif outcome == 'abort':
                    aborts += 1
            
            # Simple Psychometrics
            n_go = hits + misses
            n_catch = fas + crs
            
            # QC: Minimum trial constraints
            MIN_GO = 20
            MIN_CATCH = 20
            
            hit_rate = hits / n_go if n_go > 0 else 0
            fa_rate = fas / n_catch if n_catch > 0 else 0
            
            # Calculate d' only if QC passes
            if n_go >= MIN_GO and n_catch >= MIN_CATCH:
                d_prime = calculate_dprime(hit_rate, fa_rate)
                qc_fail = False
            else:
                d_prime = np.nan
                qc_fail = True
                log(f"QC Fail for {session.session_name}: n_go={n_go}, n_catch={n_catch}")
            
            records.append({
                'session_name': session.session_name or os.path.basename(pkl_path).replace('.pkl',''),
                'date': session.session_name.split('_')[-1] if session.session_name else "Unknown",
                'path': pkl_path,
                'hits': hits,
                'misses': misses,
                'fas': fas,
                'crs': crs,
                'n_go': n_go,
                'n_catch': n_catch,
                'hit_rate': hit_rate,
                'fa_rate': fa_rate,
                'd_prime': d_prime,
                'qc_fail': qc_fail
            })
            if not qc_fail:
                log(f"Processed {session.session_name}: d'={d_prime:.2f}")
            
        except Exception as e:
            log(f"Error processing {pkl_path}: {e}")
            continue

    log_file.close()

    df = pd.DataFrame(records)
    
    # Staging Logic: Quartiles of d'
    # Sort by d' to find thresholds? Or sort by date and assume improvement?
    # User says "Option A: compute session-wise d' ... Naïve = bottom 25% ... Expert = top 25%" relative to performance distribution?
    # Or "Naive = first k sessions"?
    # Usually "Naive" means "Early in training". "Expert" means "High performance".
    # But if performance fluctuates, "Naive" label on a late bad day is weird.
    # I'll implement: 
    # 1. Sort by Date.
    # 2. Check trend.
    # 3. Use performance quantiles to define thresholds, but classify based on *performance* primarily?
    # Prompt says: "Define Naïve = bottom 25% of sessions (or first k sessions)".
    # I will use performance quantiles on the whole dataset to categorize.
    
    # Filter valid d'
    valid_df = df.dropna(subset=['d_prime'])
    
    if len(valid_df) > 0:
        q25 = valid_df['d_prime'].quantile(0.25)
        q75 = valid_df['d_prime'].quantile(0.75)
        
        def classify(row):
            if pd.isna(row['d_prime']): return 'Excluded'
            if row['d_prime'] <= q25: return 'Naive'
            if row['d_prime'] >= q75: return 'Expert'
            return 'Learning'
        
        df['stage'] = df.apply(classify, axis=1)
    
    df.to_csv(output_csv, index=False)
    print(f"Saved manifest to {output_csv}")
    print("Staging Summary:")
    print(df['stage'].value_counts())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_dir', default='data/pkls/BG_046')
    parser.add_argument('--subject_name', default='BG_046')
    parser.add_argument('--output', default='data/BG_046_staging_manifest.csv')
    args = parser.parse_args()
    
    stage_sessions(args.subject_dir, args.subject_name, args.output)

