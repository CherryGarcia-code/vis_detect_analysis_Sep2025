"""Comprehensive validation pipeline for BG_046 session preparation.

Runs all sanity checks and generates HTML report with pass/fail status,
timing plots, and actionable warnings.

Usage:
    # Validate single session
    python scripts/run_validation_suite.py --session data/BG_046_15082025.pkl
    
    # Validate all BG_046 sessions
    python scripts/run_validation_suite.py --subject BG_046
    
    # Generate full HTML report
    python scripts/run_validation_suite.py --subject BG_046 --html-report
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional
from datetime import datetime
import traceback

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
# Add scripts dir to path for importing validation modules
scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts_dir))

from src.session_io import load_session

# Import validation modules
try:
    import validate_metadata_duration as duration_validator
    import validate_photodiode_sync as sync_validator
except ImportError as e:
    logging.error(f"Failed to import validation modules: {e}")
    sys.exit(1)


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None):
    """Configure logging with optional file output."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers
    )


def validate_session_completeness(session) -> Dict[str, any]:
    """Check that session has all required fields populated."""
    result = {
        'check': 'completeness',
        'passed': True,
        'errors': [],
        'warnings': []
    }
    
    # Required fields
    if not getattr(session, 'subject', None):
        result['errors'].append("Missing subject field")
        result['passed'] = False
    
    if not getattr(session, 'session_name', None):
        result['errors'].append("Missing session_name field")
        result['passed'] = False
    
    trials = getattr(session, 'trials', None)
    if not trials or len(trials) == 0:
        result['errors'].append("No trials found")
        result['passed'] = False
    
    clusters = getattr(session, 'clusters', None)
    if not clusters or len(clusters) == 0:
        result['warnings'].append("No clusters found")
    
    ni_events = getattr(session, 'ni_events', None)
    if not ni_events:
        result['errors'].append("Missing ni_events")
        result['passed'] = False
    else:
        # Check critical events
        if 'Baseline_ON' not in ni_events:
            result['warnings'].append("Missing Baseline_ON event")
        if 'Change_ON' not in ni_events:
            result['warnings'].append("Missing Change_ON event")
        if 'Laser' not in ni_events:
            result['warnings'].append("Missing Laser event (optotagging)")
    
    good_ids = getattr(session, 'good_cluster_ids', None)
    if not good_ids:
        result['warnings'].append("No good_cluster_ids defined")
    
    return result


def validate_trial_integrity(session) -> Dict[str, any]:
    """Check trial data for consistency and completeness."""
    result = {
        'check': 'trial_integrity',
        'passed': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    trials = getattr(session, 'trials', [])
    if not trials:
        result['errors'].append("No trials to validate")
        result['passed'] = False
        return result
    
    # Count outcomes
    outcomes = [getattr(t, 'trialoutcome', None) for t in trials]
    outcome_counts = pd.Series(outcomes).value_counts()
    result['stats']['outcome_counts'] = outcome_counts.to_dict()
    result['stats']['total_trials'] = len(trials)
    
    # Check for None/NaN outcomes
    none_count = outcomes.count(None)
    if none_count > 0:
        result['warnings'].append(f"{none_count} trials with None outcome")
    
    # Check change_size distribution
    change_sizes = [getattr(t, 'change_size', None) for t in trials]
    valid_sizes = [s for s in change_sizes if s is not None]
    if len(valid_sizes) > 0:
        result['stats']['unique_change_sizes'] = len(set(valid_sizes))
        result['stats']['change_size_range'] = (min(valid_sizes), max(valid_sizes))
    else:
        result['warnings'].append("No valid change_size values found")
    
    # Check ITI distribution
    itis = [getattr(t, 'ITI', None) for t in trials]
    valid_itis = [x for x in itis if x is not None and not np.isnan(x)]
    if len(valid_itis) > 0:
        result['stats']['mean_ITI'] = float(np.mean(valid_itis))
        result['stats']['std_ITI'] = float(np.std(valid_itis))
    else:
        result['warnings'].append("No valid ITI values found")
    
    return result


def validate_spike_data(session) -> Dict[str, any]:
    """Check spike timing and cluster quality."""
    result = {
        'check': 'spike_data',
        'passed': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    clusters = getattr(session, 'clusters', [])
    if not clusters:
        result['errors'].append("No clusters found")
        result['passed'] = False
        return result
    
    result['stats']['n_clusters'] = len(clusters)
    
    # Check spike counts
    spike_counts = []
    empty_clusters = []
    
    for cluster in clusters:
        cid = getattr(cluster, 'cluster_id', -1)
        spike_times = getattr(cluster, 'spike_times', np.array([]))
        
        if len(spike_times) == 0:
            empty_clusters.append(cid)
        else:
            spike_counts.append(len(spike_times))
    
    if empty_clusters:
        result['warnings'].append(f"{len(empty_clusters)} clusters with no spikes")
    
    if spike_counts:
        result['stats']['mean_spike_count'] = float(np.mean(spike_counts))
        result['stats']['median_spike_count'] = float(np.median(spike_counts))
        result['stats']['min_spike_count'] = int(np.min(spike_counts))
        result['stats']['max_spike_count'] = int(np.max(spike_counts))
    
    # Check good cluster IDs
    good_ids = getattr(session, 'good_cluster_ids', None)
    if good_ids:
        result['stats']['n_good_clusters'] = len(good_ids)
        all_ids = [getattr(c, 'cluster_id', -1) for c in clusters]
        missing_good = [gid for gid in good_ids if gid not in all_ids]
        if missing_good:
            result['warnings'].append(
                f"{len(missing_good)} good_cluster_ids not found in clusters"
            )
    
    return result


def run_full_validation(pkl_path: Path, raw_data_root: Optional[str] = None) -> Dict[str, any]:
    """Run all validation checks on a single session.
    
    Args:
        pkl_path: Path to session .pkl file
        raw_data_root: Optional path to raw data for .meta file validation
    """
    session_result = {
        'session_file': pkl_path.name,
        'subject': None,
        'session_name': None,
        'timestamp': datetime.now().isoformat(),
        'overall_passed': False,
        'checks': {}
    }
    
    try:
        # Load session
        session = load_session(str(pkl_path))
        session_result['subject'] = getattr(session, 'subject', 'unknown')
        session_result['session_name'] = getattr(session, 'session_name', 'unknown')
        
        # Run all checks
        checks = []
        
        # 1. Completeness
        checks.append(validate_session_completeness(session))
        
        # 2. Trial integrity
        checks.append(validate_trial_integrity(session))
        
        # 3. Spike data
        checks.append(validate_spike_data(session))
        
        # 4. Duration validation
        dur_result = duration_validator.validate_session_duration(
            session, 
            threshold=0.5,
            raw_data_root=raw_data_root
        )
        checks.append({
            'check': 'duration_consistency',
            'passed': dur_result['passed'],
            'errors': [] if dur_result['passed'] else ['Duration mismatch'],
            'warnings': dur_result['notes'],
            'stats': {
                'ni_duration': dur_result['ni_duration'],
                'imec_duration': dur_result['imec_duration'],
                'deviation': dur_result['deviation']
            }
        })
        
        # 5. Photodiode sync (optional - may not have data)
        sync_result = sync_validator.validate_session_sync(session)
        checks.append({
            'check': 'photodiode_sync',
            'passed': sync_result['passed'],
            'errors': [],
            'warnings': sync_result['notes'],
            'stats': {
                'has_photodiode': sync_result['has_photodiode'],
                'has_fsm': sync_result['has_fsm'],
                'sync_quality': sync_result['sync_quality']
            }
        })
        
        # Store all check results
        for check in checks:
            session_result['checks'][check['check']] = check
        
        # Overall pass if all critical checks pass
        critical_checks = ['completeness', 'trial_integrity', 'duration_consistency']
        session_result['overall_passed'] = all(
            session_result['checks'][c]['passed'] 
            for c in critical_checks
            if c in session_result['checks']
        )
        
    except Exception as e:
        error_msg = f"Failed to load/validate session: {str(e)}"
        logging.error(f"  {error_msg}")
        logging.debug(f"  Traceback: {traceback.format_exc()}")
        
        session_result['checks']['loading'] = {
            'check': 'loading',
            'passed': False,
            'errors': [error_msg],
            'warnings': [],
            'stats': {}
        }
    
    return session_result


def generate_html_report(results: List[Dict[str, any]], output_path: Path):
    """Generate HTML report with validation results."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>BG_046 Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #666; margin-top: 30px; }
        .summary { background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .session { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .pass { color: #4CAF50; font-weight: bold; }
        .fail { color: #f44336; font-weight: bold; }
        .warning { color: #ff9800; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #4CAF50; color: white; }
        .check-name { font-weight: bold; }
        .stats { background: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 3px; }
        .timestamp { color: #999; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 BG_046 Session Validation Report</h1>
        <p class="timestamp">Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
"""
    
    # Summary statistics
    total = len(results)
    passed = sum(1 for r in results if r['overall_passed'])
    
    html += f"""
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Total Sessions:</strong> {total}</p>
            <p><strong>Passed:</strong> <span class="pass">{passed}</span></p>
            <p><strong>Failed:</strong> <span class="fail">{total - passed}</span></p>
            <p><strong>Pass Rate:</strong> {100*passed/total if total > 0 else 0:.1f}%</p>
        </div>
"""
    
    # Per-session details
    html += "<h2>Session Details</h2>"
    
    for result in results:
        status_class = 'pass' if result['overall_passed'] else 'fail'
        status_text = '[PASS]' if result['overall_passed'] else '[FAIL]'
        
        html += f"""
        <div class="session">
            <h3>{result['subject']}/{result['session_name']} 
                <span class="{status_class}">{status_text}</span></h3>
            <p><em>File: {result['session_file']}</em></p>
"""
        
        # Check results table
        html += """
            <table>
                <tr>
                    <th>Check</th>
                    <th>Status</th>
                    <th>Details</th>
                </tr>
"""
        
        for check_name, check in result['checks'].items():
            check_status = '[OK]' if check['passed'] else '[X]'
            check_class = 'pass' if check['passed'] else 'fail'
            
            details = []
            if check['errors']:
                details.extend([f"<span class='fail'>ERROR: {e}</span>" for e in check['errors']])
            if check['warnings']:
                details.extend([f"<span class='warning'>WARNING: {w}</span>" for w in check['warnings']])
            
            details_html = '<br>'.join(details) if details else '-'
            
            html += f"""
                <tr>
                    <td class="check-name">{check_name}</td>
                    <td class="{check_class}">{check_status}</td>
                    <td>{details_html}</td>
                </tr>
"""
        
        html += "</table>"
        
        # Stats summary
        html += "<div class='stats'><strong>Statistics:</strong><br>"
        for check_name, check in result['checks'].items():
            if check.get('stats'):
                html += f"<em>{check_name}:</em> "
                stats_str = ', '.join([f"{k}={v}" for k, v in check['stats'].items() if v is not None])
                if stats_str:
                    html += f"{stats_str}<br>"
        html += "</div>"
        
        html += "</div>"
    
    html += """
    </div>
</body>
</html>
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    logging.info(f"[SUCCESS] HTML report saved to: {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Comprehensive validation pipeline for session preparation"
    )
    parser.add_argument(
        '--session',
        type=Path,
        help='Validate single session .pkl file'
    )
    parser.add_argument(
        '--subject',
        type=str,
        default='BG_046',
        help='Subject ID for batch processing (default: BG_046)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=repo_root / 'data',
        help='Data directory (default: data/)'
    )
    parser.add_argument(
        '--raw-data-root',
        type=Path,
        help='Path to raw data directory for .meta files (e.g., "X:/public/.../BG_046/Raw data")'
    )
    parser.add_argument(
        '--html-report',
        action='store_true',
        help='Generate HTML report'
    )
    parser.add_argument(
        '--report-dir',
        type=Path,
        default=repo_root / 'table_output' / 'validation',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--log-file',
        type=Path,
        help='Save log to file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args(argv)
    
    if args.log_file:
        log_path = args.log_file
    else:
        log_path = args.report_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    setup_logging(args.verbose, log_path)
    
    # Determine files to process
    if args.session:
        pkl_files = [args.session]
    else:
        pattern = f"{args.subject}_*.pkl"
        pkl_files = sorted(args.data_dir.glob(pattern))
    
    if not pkl_files:
        logging.error("No .pkl files found")
        return 1
    
    logging.info("=" * 70)
    logging.info("BG_046 SESSION VALIDATION PIPELINE")
    logging.info("=" * 70)
    logging.info(f"Processing {len(pkl_files)} session(s)")
    if args.raw_data_root:
        logging.info(f"Using .meta files from: {args.raw_data_root}")
    logging.info("")
    
    # Run validation on each session
    all_results = []
    
    for idx, pkl_path in enumerate(pkl_files, 1):
        logging.info(f"[{idx}/{len(pkl_files)}] Validating: {pkl_path.name}")
        result = run_full_validation(
            pkl_path, 
            raw_data_root=str(args.raw_data_root) if args.raw_data_root else None
        )
        all_results.append(result)
        
        status = "[PASS]" if result['overall_passed'] else "[FAIL]"
        logging.info(f"  {status}\n")
    
    # Summary
    logging.info("=" * 70)
    logging.info("VALIDATION SUMMARY")
    logging.info("=" * 70)
    
    passed = sum(1 for r in all_results if r['overall_passed'])
    total = len(all_results)
    
    logging.info(f"Total sessions: {total}")
    logging.info(f"Passed: {passed}")
    logging.info(f"Failed: {total - passed}")
    
    if total > 0:
        logging.info(f"Pass rate: {100*passed/total:.1f}%")
    
    # Generate HTML report
    if args.html_report:
        report_path = args.report_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        generate_html_report(all_results, report_path)
    
    # Save CSV summary
    csv_path = args.report_dir / 'validation_summary.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary_rows = []
    for r in all_results:
        row = {
            'session_file': r['session_file'],
            'subject': r['subject'],
            'session_name': r['session_name'],
            'overall_passed': r['overall_passed']
        }
        # Add check results
        for check_name, check in r['checks'].items():
            row[f'{check_name}_passed'] = check['passed']
        summary_rows.append(row)
    
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    logging.info(f"\n[OK] CSV summary saved to: {csv_path}")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
