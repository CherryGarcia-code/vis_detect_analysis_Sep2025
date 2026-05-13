"""
Interactive GUI to manually sync NI-DAQ Baseline_ON events to Video frames.
Bypasses algorithmic detection by letting the user scrub frames and label explicitly.
"""
import os
import sys
import cv2
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import linregress
from PIL import Image, ImageTk
import tkinter as tk

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "analysis_suite"))

from loader import load_session
from visdetect.analysis.config import VIDEO_SYNC_DIR, ROOT

def load_coarse_offset(session_name):
    # Try to load known coarse offset
    json_path = os.path.join(VIDEO_SYNC_DIR, "coarse_offsets.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            if session_name in data:
                return data[session_name]
    return 10.0 # Default guess

def get_video_path(session_name):
    # Map raw session names to actual video file names
    # e.g. 27062025 -> BG_046_270625_Eye_cam.mp4
    date_str = session_name[:4] + session_name[-2:]
    vid_path = os.path.join(ROOT, "data", "videos", f"BG_046_{date_str}_Eye_cam.mp4")
    
    if not os.path.exists(vid_path):
        vid_path = os.path.join(ROOT, "data", "videos", f"{session_name}.mp4")
    if not os.path.exists(vid_path):
        vid_path = f"Z:\\Data\\VisDetect\\video\\{session_name}.mp4"
    return vid_path

def manual_sync_gui(session_name, num_trials=12):
    print(f"Loading session {session_name}...")
    sess = load_session(session_name)
    baseline_on_times = sess.ni_events.get("Baseline_ON", [])
    if len(baseline_on_times) == 0:
        print("ERROR: No Baseline_ON events found in session.")
        return

    vid_path = get_video_path(session_name)
    if not os.path.exists(vid_path):
        print(f"ERROR: Cannot find video at {vid_path}")
        return

    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 50.0

    coarse_offset = load_coarse_offset(session_name)
    
    # Pick evenly spaced trials starting from index 20 (to avoid spurious fast/early trials)
    indices = np.linspace(20, len(baseline_on_times) - 20, num_trials).astype(int)
    
    ni_points = []
    cam_points = []

    print("\nControls:")
    print("  [d] / [Right Arrow] : Forward 1 frame")
    print("  [a] / [Left Arrow]  : Backward 1 frame")
    print("  [w] / [Up Arrow]    : Forward 10 frames")
    print("  [s] / [Down Arrow]  : Backward 10 frames")
    print("  [Enter] / [Space]   : MARK AS ONSET")
    print("  [x] / [Backspace]   : Skip this trial")
    print("  [q] / [Esc]         : Save & Quit immediately\n")

    root = tk.Tk()
    root.title(f"Manual Sync Annotator - {session_name}")
    
    label_info = tk.Label(root, text="Loading...", font=("Consolas", 14), bg="black", fg="white")
    label_info.pack(fill=tk.X)
    
    canvas = tk.Canvas(root, bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)

    state = {
        'trial_idx': 0,
        'current_frame': 0,
        'ni_times': baseline_on_times,
        'coarse_offset': coarse_offset,
        'indices': indices,
        'saved_ni': [],
        'saved_cam': [],
        'done': False,
        'photo': None
    }
    
    def update_frame():
        cap.set(cv2.CAP_PROP_POS_FRAMES, state['current_frame'])
        ret, frame = cap.read()
        if not ret: return
        
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        
        # Crosshair
        cv2.line(frame_rgb, (w//2, h//2 - 40), (w//2, h//2 + 40), (255, 0, 0), 1)
        cv2.line(frame_rgb, (w//2 - 40, h//2), (w//2 + 40, h//2), (255, 0, 0), 1)
        
        # Convert to PhotoImage
        img = Image.fromarray(frame_rgb)
        state['photo'] = ImageTk.PhotoImage(image=img)
        canvas.config(width=w, height=h)
        canvas.create_image(0, 0, image=state['photo'], anchor=tk.NW)
        
        idx = state['indices'][state['trial_idx']]
        ni_time = state['ni_times'][idx]
        msg = (f"Trial {state['trial_idx']+1}/{num_trials} (NI: {ni_time:.2f}s) | "
               f"Frame: {state['current_frame']} ({state['current_frame']/fps:.2f}s)\n"
               f"[A/D]=1f, [W/S]=10f | [Enter]=Mark | [X]=Skip | [Q]=Quit")
        label_info.config(text=msg)

    def init_trial():
        if state['trial_idx'] >= len(state['indices']):
            state['done'] = True
            root.destroy()
            return
            
        idx = state['indices'][state['trial_idx']]
        ni_time = state['ni_times'][idx]
        predicted_time = ni_time + state['coarse_offset']
        state['current_frame'] = max(0, int((predicted_time - 1.0) * fps))
        update_frame()
        
    def on_key(event):
        if state['done']: return
        
        key = event.keysym.lower()
        if key in ['right', 'd']:
            state['current_frame'] += 1
            update_frame()
        elif key in ['left', 'a']:
            state['current_frame'] = max(0, state['current_frame'] - 1)
            update_frame()
        elif key in ['up', 'w']:
            state['current_frame'] += 10
            update_frame()
        elif key in ['down', 's']:
            state['current_frame'] = max(0, state['current_frame'] - 10)
            update_frame()
        elif key in ['return', 'space']:
            idx = state['indices'][state['trial_idx']]
            ni_time = state['ni_times'][idx]
            cam_time = state['current_frame'] / fps
            print(f"Recorded Trial {state['trial_idx']+1}: NI={ni_time:.3f}s -> Frame={state['current_frame']}")
            state['saved_ni'].append(ni_time)
            state['saved_cam'].append(cam_time)
            state['trial_idx'] += 1
            init_trial()
        elif key in ['x', 'backspace']:
            print(f"Skipped trial {state['trial_idx']+1}.")
            state['trial_idx'] += 1
            init_trial()
        elif key in ['q', 'escape']:
            print("Quitting early...")
            state['done'] = True
            root.destroy()

    root.bind('<Key>', on_key)
    
    # Needs to be called via after so the window fully initializes
    root.after(100, init_trial)
    root.mainloop()

    cap.release()
    save_model(session_name, state['saved_ni'], state['saved_cam'])

def save_model(session_name, ni_times, cam_times):
    if len(ni_times) < 2:
        print("ERROR: Need at least 2 points to fit a linear regression! Not saved.")
        return
        
    slope, intercept, r_value, p_value, std_err = linregress(ni_times, cam_times)
    
    print("\n=== CLOCK MODEL RESULTS ===")
    print(f"Points used : {len(ni_times)}")
    print(f"Slope       : {slope:.8f}")
    print(f"Offset      : {intercept:.3f} s")
    print(f"R-squared   : {r_value**2:.8f} (should be 0.99999+)")
    
    # Calculate RMSE
    predictions = np.array(ni_times) * slope + intercept
    rmse = np.sqrt(np.mean((np.array(cam_times) - predictions)**2))
    print(f"RMSE        : {rmse*1000:.1f} ms")
    
    out_dir = VIDEO_SYNC_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{session_name}_corneal_sync.json")
    
    data = {
        "rmse_ms": rmse * 1000,
        "slope": slope,
        "offset": intercept,
        "n_inliers": len(ni_times),
        "total_trials": len(ni_times),
        "quality": "excellent" if rmse < 0.05 else "manual",
        "roi": "MANUAL_GUI",
        "metric_name": "manual_label",
        "method": "manual",
        "raw_ni_times": ni_times,
        "raw_cam_times": cam_times
    }
    
    with open(out_file, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"\nSaved master sync definition to:\n{out_file}")
    print("Downstream scripts will automatically load this!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Manual Video Sync Annotator")
    parser.add_argument("--session", type=str, required=True, help="Session name (e.g., 27062025)")
    parser.add_argument("--trials", type=int, default=12, help="Number of trials to sample")
    
    args = parser.parse_args()
    manual_sync_gui(args.session, args.trials)
