"""visdetect package

Lightweight package for session dataclasses and IO helpers used across
the vis_detect_analysis project.
"""

from .session import Session, Trial, Cluster
from .io import load_mat_file_to_session, parse_good_cluster_ids, mat_struct_to_dict

__all__ = [
    "Session",
    "Trial",
    "Cluster",
    "load_mat_file_to_session",
    "parse_good_cluster_ids",
    "mat_struct_to_dict",
]
