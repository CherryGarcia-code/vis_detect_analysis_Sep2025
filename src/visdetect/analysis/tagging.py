"""Pure logic for the unified video tagger (no GUI). Testable in isolation."""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from visdetect.analysis.align import get_event_times_by_trial
from visdetect.analysis.constants import BIG_CHANGE_SIZES


@dataclass
class ChangeTarget:
    trial_index: int
    change_on_s: float
    change_size: float
    outcome: str


def build_change_queue(sess) -> List[ChangeTarget]:
    """Ordered change-anchor targets: big changes (size-4 first, then size-2),
    hit/miss go-trials only, trial order within a size. Uses the trial-indexed,
    outcome-safe Change_ON getter."""
    change_on = get_event_times_by_trial(sess, "Change_ON")
    out: List[ChangeTarget] = []
    for idx, t_on in enumerate(change_on):
        if t_on is None or np.isnan(float(t_on)):
            continue
        tr = sess.trials[idx]
        cs = tr.change_size
        if cs is None or float(cs) not in BIG_CHANGE_SIZES:
            continue
        out.append(ChangeTarget(int(idx), float(t_on), float(cs),
                                str(tr.trialoutcome).lower()))
    out.sort(key=lambda t: (0 if t.change_size == 4.0 else 1, t.trial_index))
    return out
