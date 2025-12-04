"""Quick learning-summary plots across sessions.

Produces a multi-panel figure with:
- Behavioral metrics per session (hit_rate, fa_rate, miss_rate, median_rt)
- Count of lick-responsive units per session (from group_meeting responsiveness outputs)
- Peak population lick mean (if FIGURES/pop_lick_responsive_across_sessions_means.csv exists)
- Scatter comparing hit_rate vs responsive-unit counts

Writes: FIGURES/learning_summary_across_sessions.png
       FIGURES/learning_summary_across_sessions_metrics.csv

Run:
    python scripts/analysis/plot_learning_summary_across_sessions.py --out FIGURES

This script is intended to be fast: it uses the manifest and precomputed responsiveness CSVs.
"""
from __future__ import annotations
from pathlib import Path
import sys
import argparse
import csv

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESP_DIR = REPO / 'group_meeting_27112025' / 'responsiveness_output'
MANIFEST = REPO / 'data' / 'BG_046_sessions_manifest.csv'
POP_MEANS = REPO / 'FIGURES' / 'pop_lick_responsive_across_sessions_means.csv'


def load_manifest():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    df = pd.read_csv(MANIFEST)
    # Construct session name strings consistent with other scripts
    # manifest session_name column appears like '01092025' etc; prepend subject
    if 'session_name' in df.columns and 'subject' in df.columns:
        df['session'] = df['subject'].astype(str) + '_' + df['session_name'].astype(str)
    elif 'session' in df.columns:
        pass
    else:
        raise RuntimeError('Manifest missing required columns')
    return df


def count_lick_responsive():
    files = sorted(RESP_DIR.glob('*_unit_responsiveness.csv'))
    counts = {}
    for f in files:
        sess = f.stem.replace('_unit_responsiveness', '')
        try:
            d = pd.read_csv(f)
            if 'lick_responsive' in d.columns:
                ids = d.loc[d['lick_responsive'] == True, 'unit_id'].unique()
            elif 'lick_pval' in d.columns:
                ids = d.loc[d['lick_pval'] < 0.05, 'unit_id'].unique()
            else:
                ids = []
            counts[sess] = len(ids)
        except Exception:
            counts[sess] = 0
    return counts


def load_pop_peaks():
    # returns dictionary session -> peak post-event mean (time > 0)
    if not POP_MEANS.exists():
        return {}
    df = pd.read_csv(POP_MEANS, index_col=0)
    peaks = {}
    # df rows are sessions, columns t_x
    for sess, row in df.iterrows():
        # convert to numpy, ignore NaNs
        vals = row.values.astype(float)
        # find column names that are t_... to map time
        times = [float(c.replace('t_', '')) for c in df.columns]
        times = np.array(times)
        vals = np.array(vals)
        # consider times > 0
        mask = times > 0
        if mask.sum() == 0:
            peaks[sess] = np.nan
        else:
            peaks[sess] = np.nanmax(vals[mask])
    return peaks


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='FIGURES')
    ap.add_argument('--cmap', default='coolwarm', help='colormap for session ordering')
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    # sessions present in manifest
    manifest_sessions = manifest['session'].tolist()
    # sessions present in responsiveness outputs
    resp_files = sorted((REPO / 'group_meeting_27112025' / 'responsiveness_output').glob('*_unit_responsiveness.csv'))
    resp_sessions = [f.stem.replace('_unit_responsiveness', '') for f in resp_files]
    # sessions present in pop means CSV (if exists)
    pop_sessions = []
    if POP_MEANS.exists():
        try:
            pop_df = pd.read_csv(POP_MEANS, index_col=0)
            pop_sessions = list(pop_df.index.astype(str))
        except Exception:
            pop_sessions = []

    # union of sessions, preserving manifest order first, then remaining
    sessions = list(dict.fromkeys(manifest_sessions + resp_sessions + pop_sessions))

    # Attempt to sort sessions chronologically. Prefer manifest 'date' if present,
    # otherwise parse the trailing DDMMYYYY from the session name (e.g. BG_046_01072025).
    session_dates = {}
    # if manifest has date column, try to use it
    if 'date' in manifest.columns:
        # manifest may have NaT or empty strings; normalize to datetime
        try:
            manifest_dates = manifest.set_index('session')['date']
        except Exception:
            manifest_dates = pd.Series(dtype='datetime64[ns]')
        for s in sessions:
            if s in manifest_dates.index:
                try:
                    session_dates[s] = pd.to_datetime(manifest_dates.loc[s], errors='coerce')
                except Exception:
                    session_dates[s] = pd.NaT
            else:
                session_dates[s] = pd.NaT
    else:
        for s in sessions:
            session_dates[s] = pd.NaT

    # fallback: parse date from session string if still NaT
    import re
    for s in sessions:
        if pd.isna(session_dates.get(s)):
            m = re.search(r'(\d{8})$', s)
            if m:
                try:
                    session_dates[s] = pd.to_datetime(m.group(1), format='%d%m%Y', errors='coerce')
                except Exception:
                    session_dates[s] = pd.NaT

    # Build a pandas Series and sort, placing unknown dates at the end
    sd = pd.Series(session_dates)
    sd = pd.to_datetime(sd, errors='coerce')
    sessions_sorted = sd.sort_values(na_position='last').index.tolist()
    # only keep sessions that are in our original list (defensive)
    sessions = [s for s in sessions_sorted if s in sessions]

    counts = count_lick_responsive()
    peaks = load_pop_peaks()

    # assemble metrics per session
    metrics = []
    for _, row in manifest.iterrows():
        sess = row['session']
        hit_rate = row.get('hit_rate', np.nan)
        fa_rate = row.get('fa_rate', np.nan)
        miss_rate = row.get('miss_rate', np.nan)
        median_rt = row.get('median_rt', np.nan)
        n_trials = row.get('n_trials', np.nan)
        n_hits = row.get('n_hits', np.nan)
        n_miss = row.get('n_miss', np.nan)
        counts_val = counts.get(sess, 0)
        peak_val = peaks.get(sess, np.nan)
        metrics.append({'session': sess, 'n_trials': n_trials, 'n_hits': n_hits, 'n_miss': n_miss,
                        'hit_rate': hit_rate, 'fa_rate': fa_rate, 'miss_rate': miss_rate,
                        'median_rt': median_rt, 'lick_resp_count': counts_val, 'pop_peak': peak_val})
    mdf = pd.DataFrame(metrics)
    # ensure rows follow the full sessions list (manifest + responsiveness + pop)
    mdf = mdf.set_index('session').reindex(sessions).reset_index()
    mdf.to_csv(out / 'learning_summary_across_sessions_metrics.csv', index=False)

    # plotting
    cmap = plt.get_cmap(args.cmap)
    n = len(sessions)
    colors = cmap(np.linspace(0, 1, max(1, n)))

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    x = np.arange(n)
    # Panel 1: hit/fa/miss rates
    axs[0].plot(x, mdf['hit_rate'], label='hit_rate', color='tab:blue')
    axs[0].plot(x, mdf['fa_rate'], label='fa_rate', color='tab:orange')
    axs[0].plot(x, mdf['miss_rate'], label='miss_rate', color='tab:green')
    axs[0].set_ylabel('rate')
    axs[0].legend()
    axs[0].set_title('Behavioral rates across sessions')

    # Panel 2: median RT and n_trials as twin axis
    ax2 = axs[1]
    ax2.plot(x, mdf['median_rt'], label='median_rt', color='tab:purple')
    ax2.set_ylabel('median RT (s)')
    ax2b = ax2.twinx()
    ax2b.bar(x, mdf['n_trials'], alpha=0.2, color='gray', label='n_trials')
    ax2b.set_ylabel('# trials')
    ax2.set_title('Reaction time and trials')

    # Panel 3: lick-responsive counts and population peak
    ax3 = axs[2]
    ax3.bar(x - 0.2, mdf['lick_resp_count'], width=0.4, color='steelblue', label='lick_resp_count')
    ax3.set_ylabel('# lick-responsive units')
    ax3b = ax3.twinx()
    ax3b.plot(x + 0.2, mdf['pop_peak'], color='crimson', marker='o', label='pop_peak')
    ax3b.set_ylabel('pop mean peak (z)')
    ax3.set_title('Neural responsiveness across sessions')

    # Additional figure: heatmap of population mean traces across sessions (if available)
    if POP_MEANS.exists():
        try:
            pop_df = pd.read_csv(POP_MEANS, index_col=0)
            # ensure we have rows in the order of `sessions`
            pop_df = pop_df.reindex(sessions)
            # convert to numeric and build matrix
            times = [float(c.replace('t_', '')) for c in pop_df.columns]
            mat = pop_df.values.astype(float)
            # replace NaN with 0 for plotting
            mat = np.nan_to_num(mat, nan=0.0)
            fig2, axh = plt.subplots(figsize=(8, max(2, 0.25 * len(sessions))))
            im = axh.imshow(mat, aspect='auto', cmap='RdYlBu_r', extent=[min(times), max(times), 0, len(sessions)])
            axh.axvline(0, color='k', linestyle='--', lw=0.8)
            axh.set_yticks(np.arange(len(sessions)) + 0.5)
            axh.set_yticklabels(sessions, fontsize=6)
            axh.set_xlabel('time (s)')
            axh.set_title('Population mean traces (sessions x time)')
            cbar = fig2.colorbar(im, ax=axh)
            cbar.set_label('z-score')
            heat_png = out / 'learning_summary_pop_traces_heatmap.png'
            fig2.tight_layout()
            fig2.savefig(heat_png, dpi=150)
            plt.close(fig2)
            print(f'Wrote heatmap: {heat_png}')
        except Exception as e:
            print('Could not create heatmap:', e)

    # x labels
    axs[-1].set_xticks(x)
    label_names = [s for s in sessions]
    axs[-1].set_xticklabels(label_names, rotation=90, fontsize=8)

    plt.tight_layout()
    out_png = out / 'learning_summary_across_sessions.png'
    plt.savefig(out_png, dpi=150)
    plt.close()

    # Scatter: hit_rate vs lick_resp_count with Pearson r
    try:
        from scipy.stats import pearsonr
        valid = (~np.isnan(mdf['hit_rate'].astype(float))) & (~np.isnan(mdf['lick_resp_count'].astype(float)))
        if valid.sum() > 1:
            xvals = mdf.loc[valid, 'hit_rate'].astype(float)
            yvals = mdf.loc[valid, 'lick_resp_count'].astype(float)
            r, p = pearsonr(xvals, yvals)
            fig3, ax3s = plt.subplots(figsize=(6, 4))
            ax3s.scatter(xvals, yvals)
            ax3s.set_xlabel('hit_rate')
            ax3s.set_ylabel('# lick-responsive units')
            ax3s.set_title(f'hit_rate vs lick-responsive count (r={r:.2f}, p={p:.3f})')
            sc_png = out / 'learning_summary_hit_vs_resp_count.png'
            fig3.tight_layout()
            fig3.savefig(sc_png, dpi=150)
            plt.close(fig3)
            print(f'Wrote scatter: {sc_png}')
    except Exception as e:
        print('Could not compute scatter corr:', e)

    print(f'Wrote figure: {out_png}')
    print(f'Wrote CSV: {out / "learning_summary_across_sessions_metrics.csv"}')


if __name__ == '__main__':
    main()
