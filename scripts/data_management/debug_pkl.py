import pickle
import sys
from pathlib import Path


from visdetect.core.session import Session

pkl_path = "pkls/BG_046/BG_046_01072025.pkl"

print(f"Inspecting {pkl_path}...")

with open(pkl_path, 'rb') as f:
    try:
        obj = pickle.load(f)
        print(f"Type of loaded object: {type(obj)}")
        print(f"Is instance of Session? {isinstance(obj, Session)}")
        
        if hasattr(obj, '__dict__'):
            print("Object keys:", obj.__dict__.keys())
        elif isinstance(obj, dict):
            print("Dictionary keys:", obj.keys())
        else:
            print("Object representation:", str(obj)[:200])
            
    except Exception as e:
        print(f"Pickle load failed: {e}")
