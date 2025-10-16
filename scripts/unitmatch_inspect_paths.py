"""
Simple inspector to validate Kilosort folders for UnitMatch requirements.
Run this in the `unitmatch_env` environment.
"""
import os
import sys
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'unitmatch_sessions.yml')

required_files = [
    'RawWaveforms',
    'channel_positions.npy',
    'spike_times.npy',
    'spike_clusters.npy',
]


def check_path(p):
    results = {'path': p, 'exists': False, 'has_raw_waveforms': False, 'has_channel_pos': False, 'has_spike_times': False, 'has_spike_clusters': False}
    if os.path.exists(p):
        results['exists'] = True
        # RawWaveforms folder
        rw = os.path.join(p, 'RawWaveforms')
        if os.path.isdir(rw):
            results['has_raw_waveforms'] = True
        # channel_positions.npy
        cp = os.path.join(p, 'channel_positions.npy')
        if os.path.isfile(cp):
            results['has_channel_pos'] = True
        # spike_times & spike_clusters
        st = os.path.join(p, 'spike_times.npy')
        sc = os.path.join(p, 'spike_clusters.npy')
        if os.path.isfile(st):
            results['has_spike_times'] = True
        if os.path.isfile(sc):
            results['has_spike_clusters'] = True
    return results


def main():
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)

    sessions = cfg.get('sessions', [])
    report = []
    for s in sessions:
        p = s.get('path')
        res = check_path(p)
        res['has_bombcell'] = s.get('has_bombcell', False)
        res['note'] = s.get('note', '')
        report.append(res)

    # Print summary
    for r in report:
        print('Path:', r['path'])
        print('  exists:', r['exists'])
        print('  RawWaveforms:', r['has_raw_waveforms'])
        print('  channel_positions.npy:', r['has_channel_pos'])
        print('  spike_times.npy:', r['has_spike_times'])
        print('  spike_clusters.npy:', r['has_spike_clusters'])
        print('  bombcell:', r['has_bombcell'])
        print('  note:', r['note'])
        print('')

    # Save report
    out_dir = os.path.join(os.path.dirname(__file__), '..', cfg.get('report_dir', 'table_output/unitmatch'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'inspect_report.yaml')
    with open(out_path, 'w') as f:
        yaml.safe_dump(report, f)
    print('Saved report to', out_path)


if __name__ == '__main__':
    main()
