
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from visdetect.core.session import Session, Cluster

@dataclass
class OptoMetrics:
    cluster_id: int
    is_responsive: bool
    latency_ms: float
    jitter_ms: float
    reliability: float
    p_value: float

class OptoTagger:
    def __init__(self, session: Session, opto_key: str = 'Opto_Stim', 
                 window_ms: Tuple[float, float] = (0, 10), 
                 baseline_ms: Tuple[float, float] = (-10, 0)):
        """
        Initialize OptoTagger.

        Args:
            session: loaded Session object.
            opto_key: Key in ni_events for opto pulses.
            window_ms: (start, end) window in ms relative to pulse to look for spikes.
            baseline_ms: (start, end) window in ms relative to pulse for baseline.
        """
        self.session = session
        self.opto_key = opto_key
        self.window = window_ms
        self.baseline = baseline_ms
        
        if not self.session.ni_events or self.opto_key not in self.session.ni_events:
            # Try to resolve case-insensitive or partial match
            found = False
            if self.session.ni_events:
                for k in self.session.ni_events.keys():
                    if 'opto' in k.lower() or 'laser' in k.lower() or 'pulse' in k.lower():
                        print(f"Warning: Exact key '{opto_key}' not found. Using '{k}' instead.")
                        self.opto_key = k
                        found = True
                        break
            if not found:
                raise ValueError(f"Opto key '{opto_key}' not found in session events.")

        # Get pulse times
        events = self.session.ni_events[self.opto_key]
        if isinstance(events, dict):
            if 'rise_t' in events:
                self.pulse_times = events['rise_t'].flatten()
            else:
                 raise ValueError(f"Could not find 'rise_t' in {self.opto_key} event dictionary.")
        else:
            self.pulse_times = np.array(events).flatten()

        print(f"Found {len(self.pulse_times)} opto pulses.")

    def analyze_unit(self, cluster: Cluster) -> OptoMetrics:
        """Calculate optotagging metrics for a single unit."""
        spikes = cluster.spike_times
        if len(spikes) == 0:
            return OptoMetrics(cluster.cluster_id, False, np.nan, np.nan, 0.0, 1.0)

        # Align spikes to pulses
        # Optimized for speed: iterate pulses and find relative spikes
        
        latencies = []
        hit_count = 0
        
        # Simple window search
        # Convert window to seconds
        win_start = self.window[0] / 1000.0
        win_end = self.window[1] / 1000.0
        
        for p in self.pulse_times:
            # Find spikes in [p+win_start, p+win_end]
            # Assumes spikes are sorted
            idx_start = np.searchsorted(spikes, p + win_start)
            idx_end = np.searchsorted(spikes, p + win_end)
            
            in_window = spikes[idx_start:idx_end]
            
            if len(in_window) > 0:
                first_spike = in_window[0]
                latencies.append((first_spike - p) * 1000.0) # ms
                hit_count += 1
                
        reliability = hit_count / len(self.pulse_times)
        
        if hit_count == 0:
            return OptoMetrics(cluster.cluster_id, False, np.nan, np.nan, 0.0, 1.0)

        latency_mean = np.mean(latencies)
        jitter = np.std(latencies)
        
        # Basic responsiveness check (Reliability > threshold AND latency < threshold)
        # UsingSALT or SALT-like test is better, but for now use Reliability > 0.2 and Jitter < 2ms?
        is_responsive = (reliability > 0.3) and (jitter < 2.0) and (latency_mean < 8.0)
        
        return OptoMetrics(
            cluster_id=cluster.cluster_id,
            is_responsive=is_responsive,
            latency_ms=latency_mean,
            jitter_ms=jitter,
            reliability=reliability,
            p_value=0.0 # Placeholder for SALT test
        )

    def analyze_all(self) -> List[OptoMetrics]:
        results = []
        for c in self.session.clusters:
            # Only analyze good clusters if defined
            if self.session.good_cluster_ids and c.cluster_id not in self.session.good_cluster_ids:
                continue
            res = self.analyze_unit(c)
            results.append(res)
        return results

