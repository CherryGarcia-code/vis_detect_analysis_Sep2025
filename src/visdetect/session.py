"""Session dataclasses for vis_detect_analysis

This module contains small dataclasses representing a Trial, Cluster, and
Session. These are intentionally minimal and should be extended as needed.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class Trial:
    trialoutcome: str
    reactiontimes: Dict[str, float] = field(default_factory=dict)
    change_size: Optional[int] = None
    orientation: Optional[int] = None
    ITI: Optional[float] = None
    change_time: Optional[float] = None
    baseline_values: Optional[float] = None


@dataclass
class Cluster:
    cluster_id: int
    spike_times: np.ndarray
    quality: Optional[str] = None


@dataclass
class Session:
    trials: List[Trial]
    clusters: List[Cluster]
    subject: str
    session_name: str
    good_cluster_ids: Optional[List[int]] = None
    ni_events: Optional[dict] = None
