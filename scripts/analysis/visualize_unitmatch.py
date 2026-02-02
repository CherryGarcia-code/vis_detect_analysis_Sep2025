
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Try importing UnitMatchPy for visualization utils if available
try:
    import UnitMatchPy.utils as util
except ImportError:
    pass

def visualize_unitmatch_results(output_dir):
    output_dir = Path(output_dir)
    print(f"Visualizing results from {output_dir}")
    
    # Load data
    try:
        # Load MatchTable (CSV) which contains the assignments
        match_table_path = output_dir / "MatchTable.csv"
        if match_table_path.exists():
            df = pd.read_csv(match_table_path)
            print("Loaded MatchTable.csv")
            print(df.head())
            
            # MatchTable columns based on check:
            # ID1, ID2, RecSes 1, RecSes 2, UID1, UID2, etc.
            # UnitMatch typically outputs pairs. 
            # We need to look at 'UID1' or 'UID2'.
            # Or perhaps construct a clean table of Unit -> UID.
            
            # Use 'UID Conservative 1' as the main UID for now (or UID Conservative 2)
            # Actually, this table lists Pairs?
            # If so, visualizing "Units per Session" using this table is tricky if it repeats units.
            
            # Let's try to reconstruct the Unit ID -> Unique ID mapping.
            # We can use ID1 (Unit) and RecSes 1 (Session) and UID Conservative 1 (UID).
            
            df_u1 = df[['ID1', 'RecSes 1', 'UID Conservative 1']].rename(columns={'ID1': 'UnitID', 'RecSes 1': 'Session', 'UID Conservative 1': 'UID'})
            # We might duplicates if units are involved in multiple pairs. Drop duplicates.
            df_unique = df_u1.drop_duplicates()
            
            # Simple stats
            n_unique = df_unique['UID'].nunique()
            print(f"Found {n_unique} unique neurons across sessions.")
            
            # Plot units per session
            counts = df_unique.groupby('Session').size()
            print("Units per session:")
            print(counts)
            
            # Use this clean dataframe for finding best unit
            df = df_unique

        # Load Probability Matrix
        prob_path = output_dir / "MatchProb.npy"
        if prob_path.exists():
            prob_matrix = np.load(prob_path)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(prob_matrix, aspect='auto', cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(label='Match Probability')
            plt.title('Unit Match Probability Matrix')
            plt.xlabel('Unit Index')
            plt.ylabel('Unit Index')
            
            plt.savefig(output_dir / "match_probability_matrix.png")
            print(f"Saved probability matrix to {output_dir / 'match_probability_matrix.png'}")
            plt.close()

        # Load Waveform Info for plotting matched examples
        wf_info_path = output_dir / "WaveformInfo.npz"
        clus_info_path = output_dir / "ClusInfo.pickle"
        
        if wf_info_path.exists() and clus_info_path.exists() and match_table_path.exists():
            wf_data = np.load(wf_info_path)
            # Keys usually: 'avg_waveform', 'avg_centroid', etc.
            avg_waveforms = wf_data['avg_waveform'] # Shape (N_units, Time, Channels) ?? Or (N_units, Channels, Time)?
            # Usually (N_units, TimePoints, N_Channels) or similar.
            
            import pickle
            with open(clus_info_path, 'rb') as f:
                clus_info = pickle.load(f)
                
            # Find a UID that exists in all 3 sessions (or max sessions)
            uids_counts = df['UID'].value_counts()
            best_uid = uids_counts.idxmax()
            print(f"Visualizing best tracked unit: UID {best_uid} (found in {uids_counts[best_uid]} sessions)")
            
            # Get the unit entries for this UID
            unit_entries = df[df['UID'] == best_uid]
            
            # Reconstruct Global Index Map
            # Assumes order of units in MatchTable/Waveforms matches sorted(ID) per session, concatenated by Session 1, 2, 3...
            
            # Note: `df` here is `df_unique`, which has renamed columns 'Session' and 'UnitID'.
            # We need to rely on the original frame or mapped names.
            
            # Using mapped names:
            cleaned_df = df[['Session', 'UnitID']].drop_duplicates().sort_values(['Session', 'UnitID'])
            
            # Check consistency with 350 total
            if len(cleaned_df) == avg_waveforms.shape[1]:
                 print(f"Index alignment check passed: {len(cleaned_df)} units.")
                 global_map = {}
                 current_idx = 0
                 for _, row in cleaned_df.iterrows():
                     global_map[(row['Session'], row['UnitID'])] = current_idx
                     current_idx += 1
            else:
                 print(f"WARNING: Unit count mismatch. DF has {len(cleaned_df)}, WFS has {avg_waveforms.shape[1]}. Trying to infer...")
                 # Fallback: maybe just iterate assuming sorted order?
                 global_map = {}
                 current_idx = 0
                 for _, row in cleaned_df.iterrows():
                     global_map[(row['Session'], row['UnitID'])] = current_idx
                     current_idx += 1

            # Check waveform shape 
            print(f"Waveforms shape: {avg_waveforms.shape}") 
            # Expected (Time, Units, Splits) -> (82, 350, 2)
            
            plt.figure(figsize=(10, 6))
            
            for _, row in unit_entries.iterrows():
                # Use mapped names
                sess = row['Session']
                u_id = row['UnitID']
                
                if (sess, u_id) in global_map:
                    g_idx = global_map[(sess, u_id)]
                    
                    if g_idx < avg_waveforms.shape[1]: 
                        # Extract waveform: (Time, Units, Splits) -> (Time, Splits)
                        # We averaged over splits (or plot both)
                        wf_splits = avg_waveforms[:, g_idx, :] # (82, 2)
                        wf_mean = wf_splits.mean(axis=1)
                        
                        plt.plot(wf_mean, linewidth=2, label=f"Session {sess} (Clus {u_id})")
                        
                        # Optional: Plot individual splits faint?
                        # plt.plot(wf_splits[:,0], '--', alpha=0.5, color=plt.gca().lines[-1].get_color())
            
            plt.legend()
            plt.title(f"Tracked Unit (UID {best_uid}) Waveforms")
            plt.xlabel("Time (samples)")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.3)
            
            out_file = output_dir / f"UID_{best_uid}_waveforms.png"
            plt.savefig(out_file)
            print(f"Saved matched waveform plot to {out_file}")
            plt.close()

    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    visualize_unitmatch_results("e:/python_analysis/git_repos/vis_detect_analysis_Sep2025/data/unit_match/output/BG_046")
