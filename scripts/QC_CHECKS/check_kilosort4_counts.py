import os
import sys

base = r"X:\public\projects\BeJG_20230130_VisDetect\wEPhys\BG_046\Processed data"
if not os.path.isdir(base):
    print("Directory not found:", base)
    sys.exit(0)

print("Session\tItems\tStatus\tPath")
for session in sorted(os.listdir(base)):
    sess_path = os.path.join(base, session)
    if not os.path.isdir(sess_path):
        continue
    ks4 = os.path.join(sess_path, 'Kilosort&Phy', f"{session}_g0_imec0", 'kilosort4')
    if not os.path.isdir(ks4):
        print(f"{session}\tMISSING\tMISSING\t{ks4}")
        continue
    try:
        # count visible items (files + directories)
        count = len([name for name in os.listdir(ks4)])
    except Exception as e:
        print(f"{session}\tERROR\t{e}\t{ks4}")
        continue
    status = 'COMPLETE' if count >= 30 else 'INCOMPLETE'
    print(f"{session}\t{count}\t{status}\t{ks4}")
