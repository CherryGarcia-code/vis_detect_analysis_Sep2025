import pickle
import sys
from pathlib import Path


from visdetect.core.session import Session, Trial

def inspect_pkl(pkl_path):
    print(f"Loading {pkl_path}")
    with open(pkl_path, 'rb') as f:
        session = pickle.load(f)
    
    print("Session Keys:", session.__dict__.keys())
    
    ni = session.ni_events
    if ni:
        print("\nNI Events Keys:", ni.keys())
        for k, v in ni.items():
            print(f"  {k}: {type(v)}")
            if hasattr(v, 'shape'):
                print(f"    shape: {v.shape}")
            elif isinstance(v, list):
                print(f"    len: {len(v)}")
            elif isinstance(v, dict):
                 print(f"    keys: {v.keys()}")
    else:
        print("ni_events is None or empty")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_pkl(sys.argv[1])
    else:
        print("Usage: python inspect_ni_events.py <pkl_path>")
