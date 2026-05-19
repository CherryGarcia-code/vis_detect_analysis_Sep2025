
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print("Listing scripts directory to verify UnitMatch scripts...")
scripts_dir = os.path.join(repo_root, 'scripts')
for f in os.listdir(scripts_dir):
    if os.path.isfile(os.path.join(scripts_dir, f)):
        print(f"  {f}")
