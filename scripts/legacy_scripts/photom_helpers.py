import pandas as pd
import os
import pandas as pd
import numpy as np
import json
from glob import glob

#  Define a function to calculate performance and standard error of the mean
def calculate_session_performance(session_df):
    """
    Calculates the performance and CI95 of for a given session DataFrame.
    Parameters:
    - session_df: The DataFrame containing the session data.

    Returns:
    - performance_dict: A dictionary containing a tuple of the performance and CI95 for each change size.
    """
    performance_dict = {}
    for change_size in sorted(session_df['change_sizes_TF'].unique()):  # Sort the change sizes
        outcomes = session_df[session_df['change_sizes_TF'] == change_size]['outcomes']
        hits = outcomes.value_counts().get('Hit', 0)
        misses = outcomes.value_counts().get('Miss', 0)
        total = hits + misses
        if total > 0:
            performance = hits / total
            # Calculate standard error of the mean (SEM)
            sem = np.sqrt(performance * (1 - performance) / total)
            ci95 = 1.96 * np.sqrt(performance * (1 - performance) / total)
        else:
            performance = np.nan
            sem = np.nan
            ci95 = np.nan
        # performance_dict[float(change_size)] = (performance, sem)
        performance_dict[float(change_size)] = (performance, ci95)
    return performance_dict

#  Define a function to calculate performance and standard error of the mean
def calculate_session_performance_laser(session_df):
    """
    Calculates the performance and CI95 of for a given session DataFrame.
    Parameters:
    - session_df: The DataFrame containing the session data.

    Returns:
    - performance_dict: A dictionary containing a tuple of the performance and CI95 for each change size and laser state.
    """
    performance_dict = {'laser_on': {}, 'laser_off': {}}
    
    for laser_state in [True, False]:
        laser_state_key = 'laser_on' if laser_state else 'laser_off'
        filtered_df = session_df[session_df['laser_states'] == laser_state]
        
        for change_size in sorted(filtered_df['change_sizes_TF'].unique()):  # Sort the change sizes
            outcomes = filtered_df[filtered_df['change_sizes_TF'] == change_size]['outcomes']
            hits = outcomes.value_counts().get('Hit', 0)
            misses = outcomes.value_counts().get('Miss', 0)
            total = hits + misses
            if total > 0:
                performance = hits / total
                # Calculate standard error of the mean (SEM)
                sem = np.sqrt(performance * (1 - performance) / total)
                ci95 = 1.96 * np.sqrt(performance * (1 - performance) / total)
            else:
                performance = np.nan
                sem = np.nan
                ci95 = np.nan
            performance_dict[laser_state_key][float(change_size)] = (performance, ci95)
    
    return performance_dict



def calculate_performance(group):
    hits = group['outcomes'].eq('Hit').sum()
    total = group['outcomes'].isin(['Hit', 'Miss']).sum()
    performance = hits / total if total > 0 else 0
    se = np.sqrt((performance * (1 - performance)) / total) if total > 0 else 0
    ci95 = 1.96 * se
    return pd.Series({
        'perf': performance
        ,
        # 'stderr': se,
        'ci95': ci95
    })




def filter_and_pad_data(df, change_sizes, threshold):
    """
    Filters and pads the data according to the specified change sizes and performance threshold.

    Parameters:
    - df: The DataFrame containing the data.
    - change_sizes: A list of desired change sizes to consider.
    - threshold: The performance threshold to determine passing sessions.

    Returns:
    - A DataFrame with the filtered and padded data.
    """
    subsetted_sessions = {}
    max_passing_sessions = 0

    # Determine the maximum number of passing sessions for any subject
    for subject in df.columns:
        if subject.endswith('_performance'):
            for change_size in change_sizes:
                passing_sessions_count = sum(
                    performance_data.get(change_size, (0, 0))[0] > threshold
                    for performance_data in df[subject] if performance_data is not None
                )
                max_passing_sessions = max(max_passing_sessions, passing_sessions_count)

    # Collect passing sessions and apply padding
    for subject in df.columns:
        if not subject.endswith('_performance'):
            continue

        subject_name = subject.replace('_performance', '')
        for change_size in change_sizes:
            selected_sessions = [
                df[subject_name][i] for i, performance_data in enumerate(df[subject])
                if performance_data is not None and performance_data.get(change_size, (0, 0))[0] > threshold
            ]

            padding_needed = max_passing_sessions - len(selected_sessions)
            selected_sessions.extend([None] * padding_needed)
            subsetted_sessions[f"{subject_name}"] = selected_sessions

    return pd.DataFrame.from_dict(subsetted_sessions, orient='index').transpose()

##Example of usage:
# desired_change_sizes = [4.0]  # Can be a list of multiple sizes
# performance_threshold = 0.5
# padded_df = filter_and_pad_data(all_data_df, desired_change_sizes, performance_threshold)




def z_score(values):
    return (values - np.mean(values)) / np.std(values)

def compute_zscores(session_df):
    baseline_means_df = pd.DataFrame(session_df['baseline_means'].tolist())
    zscores_df = baseline_means_df.apply(z_score)
    zscores_df['outcome'] = session_df['outcomes']
    return zscores_df