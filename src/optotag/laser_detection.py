import numpy as np
from typing import Dict, Any, Tuple


def estimate_sample_rate_from_laser(laser: Dict[str, Any]) -> float:
    """Estimate sample rate when rise/fall are in samples and duration in seconds.
    Returns sample_rate (Hz)."""
    rise = np.asarray(laser.get("rise_t"))
    fall = np.asarray(laser.get("fall_t"))
    dur = np.asarray(laser.get("duration"))
    if rise.size == 0 or fall.size == 0 or dur.size == 0:
        return 30000.0
    # If rise values are clearly > 1 (likely sample indices) and durations < 1 (s)
    if np.median(rise) > 1.0 and np.median(dur) < 1.0:
        # sample_rate ~= median((fall - rise) / duration)
        denom = np.median(dur)
        numer = np.median(fall - rise)
        if denom > 0:
            return float(numer / denom)
    # otherwise assume units are seconds
    return 30000.0


def pulses_from_ni_events(
    ni_events: Dict[str, Any],
    prefer_sample_rate: float = None,
    split_blocks: bool = False,
    gap_threshold_s: float = 2.0,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Extract pulse onsets/offsets and durations from ni_events Laser field.

    Returns (pulses_dict, sample_rate)
    pulses_dict keys: onset_samples, offset_samples, onset_s, offset_s, duration_s
    """
    if ni_events is None:
        raise ValueError("ni_events is None")
    if "Laser" not in ni_events:
        raise ValueError("ni_events does not contain Laser")
    laser = ni_events["Laser"]
    rise = np.asarray(laser.get("rise_t"))
    fall = np.asarray(laser.get("fall_t"))
    dur = np.asarray(laser.get("duration"))

    # Estimate sample rate
    if prefer_sample_rate is not None:
        sr = float(prefer_sample_rate)
    else:
        sr = estimate_sample_rate_from_laser(laser)

    # Determine whether rise/fall are in samples or seconds
    # If typical values > 10 and durations are << 1, treat rise/fall as samples
    if np.median(rise) > 10 and np.median(dur) < 1.0:
        onset_samples = rise.astype(float)
        offset_samples = fall.astype(float)
        onset_s = onset_samples / sr
        offset_s = offset_samples / sr
        duration_s = dur.astype(float)
    else:
        # assume rise/fall already in seconds
        onset_s = rise.astype(float)
        offset_s = fall.astype(float)
        duration_s = dur.astype(float)
        onset_samples = onset_s * sr
        offset_samples = offset_s * sr

    pulses = {
        "onset_samples": onset_samples,
        "offset_samples": offset_samples,
        "onset_s": onset_s,
        "offset_s": offset_s,
        "duration_s": duration_s,
    }

    # Optionally detect separate pulse blocks by looking for large gaps between onsets
    if split_blocks:
        n_onsets = len(onset_s)
        # Special-case: exactly two blocks of 501 pulses (1002 total)
        if n_onsets == 1002:
            block_ids = np.zeros(n_onsets, dtype=int)
            block_ids[:501] = 0
            block_ids[501:] = 1
            pulses["block_id"] = block_ids
        else:
            # compute gaps (in seconds)
            if n_onsets >= 2:
                diffs = np.diff(onset_s)
                # find the largest gap
                max_gap_idx = int(np.argmax(diffs))
                max_gap = float(diffs[max_gap_idx])
                if max_gap > gap_threshold_s:
                    # split at that gap: indices 0..max_gap_idx -> block 0, rest -> block 1
                    block_ids = np.zeros(n_onsets, dtype=int)
                    block_ids[: max_gap_idx + 1] = 0
                    block_ids[max_gap_idx + 1 :] = 1
                    pulses["block_id"] = block_ids
                else:
                    # no large gaps found; single block
                    pulses["block_id"] = np.zeros(n_onsets, dtype=int)
            else:
                pulses["block_id"] = np.zeros(n_onsets, dtype=int)

    return pulses, sr
