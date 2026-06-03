import os, sys, glob, tempfile, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.dirname(HERE)
sys.path.insert(0, TRACK)
from train_deepum_common import (add_deepum_to_path, write_synthetic_cache,
                                 load_finetuned_encoder)


def test_clip_one_epoch_smoke():
    cache = tempfile.mkdtemp()
    out = tempfile.mkdtemp()
    write_synthetic_cache(cache, n_sessions=2, units_per_session=6, seed=2)
    cmd = [
        "python", os.path.join(TRACK, "train_deepum_clip.py"),
        "--train-root", cache, "--out-dir", out,
        "--init", "shipped", "--freeze", "fcblock",
        "--epochs", "1", "--batch", "4", "--save-freq", "1", "--device", "cpu",
    ]
    subprocess.run(cmd, check=True)
    exports = glob.glob(os.path.join(out, "export_epoch_*.pt"))
    assert exports, "no export checkpoint written"
    # the export reloads into an encoder and runs inference to finite output
    import numpy as np, torch
    add_deepum_to_path()
    from DeepUnitMatch.utils.losses import clip_sim
    model = load_finetuned_encoder(exports[0], device="cpu")
    x = torch.from_numpy(np.zeros((3, 60, 30), dtype=np.float64))
    sim = clip_sim(model(x), model(x)).detach().numpy()
    assert np.isfinite(sim).all()
    print("test_clip_one_epoch_smoke PASS")


if __name__ == "__main__":
    test_clip_one_epoch_smoke()
    print("ALL SMOKE TESTS PASSED")
