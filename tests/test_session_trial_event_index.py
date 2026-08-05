"""The new field must not break the ~253 existing pkls, which lack it."""
import pickle

import numpy as np

from visdetect.core.session import Session


def test_field_defaults_to_none():
    s = Session()
    assert s.trial_event_index is None


def test_old_pickle_without_the_field_still_loads_and_reads_none():
    """Simulate an existing pkl: pickle a Session, strip the key, unpickle."""
    s = Session(subject="BG_046", session_name="01072025")
    raw = pickle.loads(pickle.dumps(s))
    del raw.__dict__["trial_event_index"]          # what an old pkl looks like
    revived = pickle.loads(pickle.dumps(raw))
    # A plain None default makes this safe; default_factory would AttributeError.
    assert getattr(revived, "trial_event_index", None) is None


def test_field_round_trips_an_array():
    s = Session()
    s.trial_event_index = np.array([-1, -1, 0, 1, 2], dtype=int)
    back = pickle.loads(pickle.dumps(s))
    assert np.array_equal(back.trial_event_index, np.array([-1, -1, 0, 1, 2]))
