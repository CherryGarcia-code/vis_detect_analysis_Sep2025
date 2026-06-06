import os, sys, glob, tempfile, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.dirname(HERE)
sys.path.insert(0, TRACK)
from train_deepum_common import write_synthetic_cache, add_deepum_to_path


def test_ae_one_epoch_then_clip_init():
    cache = tempfile.mkdtemp(); out = tempfile.mkdtemp(); clip_out = tempfile.mkdtemp()
    write_synthetic_cache(cache, n_sessions=2, units_per_session=6, seed=3)
    subprocess.run(["python", os.path.join(TRACK, "train_deepum_ae.py"),
                    "--train-root", cache, "--out-dir", out,
                    "--epochs", "1", "--batch", "4", "--save-freq", "1",
                    "--device", "cpu"], check=True)
    ae_ckpts = sorted(glob.glob(os.path.join(out, "ae_epoch_*.pt")))
    assert ae_ckpts, "no AE checkpoint written"
    # the AE encoder must initialise the CLIP stage (key 'encoder' loads into the CNN)
    subprocess.run(["python", os.path.join(TRACK, "train_deepum_clip.py"),
                    "--train-root", cache, "--out-dir", clip_out,
                    "--init", ae_ckpts[-1], "--freeze", "fcblock",
                    "--epochs", "1", "--batch", "4", "--save-freq", "1",
                    "--device", "cpu"], check=True)
    assert glob.glob(os.path.join(clip_out, "export_epoch_*.pt"))
    print("test_ae_one_epoch_then_clip_init PASS")


if __name__ == "__main__":
    test_ae_one_epoch_then_clip_init()
    print("ALL AE SMOKE TESTS PASSED")
