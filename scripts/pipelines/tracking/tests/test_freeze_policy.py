import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from train_deepum_common import add_deepum_to_path
from train_deepum_clip import build_model, apply_freeze

model, _, _ = build_model("cpu")
apply_freeze(model, "fcblock")
assert all(("FcBlock" in n) == p.requires_grad for n, p in model.named_parameters())
apply_freeze(model, "none")
assert all(p.requires_grad for p in model.parameters())
print("test_freeze_policy PASS")
