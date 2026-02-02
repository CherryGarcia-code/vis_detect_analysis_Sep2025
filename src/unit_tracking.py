import pandas as pd
import numpy as np
from pathlib import Path

def stitch_registries(file_list):
    """
    Stitches multiple overlapping CellRegistry.csv files into a single Global Registry.
    
    Args:
        file_list (list): List of Path objects pointing to CellRegistry.csv files.
        
    Returns:
        pd.DataFrame: Global registry with unique Global_UIDs.
    """
    loaded_batches = []
    
    print(f"Loading {len(file_list)} batches for stitching...")
    
    for p in file_list:
        try:
            df = pd.read_csv(p, index_col=0)
            # Parse columns to datetime to establish range
            # Columns are DDMMYYYY
            cols = [pd.to_datetime(c, format="%d%m%Y") for c in df.columns]
            df.columns = cols # Use Timestamps as columns for reliable matching
            
            start_date = min(cols)
            end_date = max(cols)
            
            loaded_batches.append({
                'df': df,
                'start': start_date,
                'end': end_date,
                'path': p
            })
        except Exception as e:
            print(f"Error loading {p}: {e}")
            
    # Sort by start date
    loaded_batches.sort(key=lambda x: x['start'])
    print("Batches sorted chronologically.")
    
    # 2. Iterative Stitching
    # Start with the first batch
    global_df = loaded_batches[0]['df'].copy()
    
    # Assign temporary Global IDs (index)
    global_df.index.name = 'Global_UID'
    global_df = global_df.reset_index(drop=True)
    
    for i in range(1, len(loaded_batches)):
        next_batch = loaded_batches[i]
        next_df = next_batch['df']
        
        # Find overlapping columns
        common_cols = list(set(global_df.columns) & set(next_df.columns))
        
        if not common_cols:
            print(f"Warning: Batch {i} has no overlap with current global chain. Appending as new.")
            # If no overlap, just join outer
            global_df = global_df.join(next_df, how='outer', lsuffix='_old', rsuffix='_new')
            continue
            
        # Match rows
        # Build Lookup Table from current Global Registry
        # Key: (SessionDate, ClusterID), Value: GlobalRowIndex
        lookup = {}
        for idx, row in global_df.iterrows():
            for col in common_cols:
                val = row[col]
                if pd.notna(val):
                    lookup[(col, val)] = idx
                    
        # Identify matches for next_df rows
        next_to_global_map = {} # next_idx -> global_idx
        new_rows = []
        
        for idx, row in next_df.iterrows():
            matches = []
            for col in common_cols:
                val = row[col]
                if pd.notna(val):
                    key = (col, val)
                    if key in lookup:
                        matches.append(lookup[key])
            
            # Decide
            if len(matches) > 0:
                # Take the most frequent match
                best_match = max(set(matches), key=matches.count)
                next_to_global_map[idx] = best_match
            else:
                # This is a new unit entering the window
                new_rows.append(idx)
                
        # Merge Information
        # 1. Update existing global rows with new data from next_df (extended columns)
        new_cols = [c for c in next_df.columns if c not in global_df.columns]
        
        # Add new columns to global_df (fill nan)
        for c in new_cols:
            global_df[c] = np.nan
            
        # Update matched rows
        for next_idx, global_idx in next_to_global_map.items():
            row_data = next_df.loc[next_idx]
            original_row = global_df.loc[global_idx]
            
            # Optimization: Update just the new columns
            global_df.loc[global_idx, new_cols] = row_data[new_cols]
            
            # Also fill gaps in overlap if global was NaN but next has value?
            for c in common_cols:
                if pd.isna(global_df.at[global_idx, c]) and pd.notna(row_data[c]):
                     global_df.at[global_idx, c] = row_data[c]
                     
        # 2. Append new units
        if new_rows:
            rows_to_add = next_df.loc[new_rows]
            # Align columns
            rows_to_add = rows_to_add.reindex(columns=global_df.columns)
            
            # Reset index to simple range to avoid conflicts and extend
            rows_to_add = rows_to_add.reset_index(drop=True)
            
            # We need to append. `concat` creates new index usually
            global_df = pd.concat([global_df, rows_to_add], ignore_index=True)

    # Sort columns chronologically finally
    global_df = global_df.reindex(sorted(global_df.columns), axis=1)
    
    print(f"Stitching Complete.")
    print(f"Final Global Registry Shape: {global_df.shape} (Units x Sessions)")
    return global_df

class TFAnalyzer:
    """
    Helper class for analyzing Temporal Frequency responsiveness.
    """
    _cache = {}
    
    @classmethod
    def get_tf_metric(cls, date_obj, cluster_id, tf_root, metric='status', threshold=2.0):
        """
        Retrieves TF responsiveness metric for a given unit.
        
        Args:
            date_obj (datetime): Session date.
            cluster_id (str|int): Unit ID (can be merged "123;124").
            tf_root (Path): Path to FIGURES/tf/ folder.
            metric (str): 'status' (0=None, 1=Present, 2=Responsive) or 'raw_z' (float).
            threshold (float): Z-score threshold for 'status'.
            
        Returns:
            int or float: The requested metric.
        """
        date_str = date_obj.strftime("%d%m%Y")
        session_name = f"BG_046_{date_str}"
        f_path = Path(tf_root) / session_name / "tf_pulse_grid_both.csv"
        
        if not f_path.exists():
            return np.nan 
        
        # Cache Loading
        if date_str not in cls._cache:
            try:
                cls._cache[date_str] = pd.read_csv(f_path)
            except Exception as e:
                return np.nan
        
        df_tf = cls._cache[date_str]
        
        # Handle Merged IDs (e.g. "558;561")
        ids_to_check = []
        if isinstance(cluster_id, str) and ';' in cluster_id:
            ids_to_check = [int(x) for x in cluster_id.split(';')]
        else:
            try:
                ids_to_check = [int(cluster_id)]
            except:
                return 0 if metric == 'status' else np.nan
        
        z_values = []
        for cid in ids_to_check:
            row = df_tf[df_tf['cluster_id'] == cid]
            if row.empty:
                continue
            
            z_cols = ['z_max_fast', 'z_min_fast', 'z_max_slow', 'z_min_slow']
            available_cols = [c for c in z_cols if c in row.columns]
            
            if not available_cols:
                continue
                
            vals = row[available_cols].values.flatten()
            vals = vals[~pd.isna(vals)]
            
            if len(vals) > 0:
                z_values.append(np.max(np.abs(vals)))
        
        if not z_values:
            return 0 if metric == 'status' else np.nan
            
        max_z = max(z_values)
        
        if metric == 'raw_z':
            return max_z
        else:
            # Status Logic
            if max_z > threshold:
                return 2
            else:
                return 1
