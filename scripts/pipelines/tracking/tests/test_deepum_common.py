import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_deepum_common import AverageMeter, augment_channel_roll


def test_average_meter():
    m = AverageMeter()
    m.update(2.0, n=1); m.update(4.0, n=3)
    assert m.count == 4
    assert abs(m.avg - (2.0 + 4.0 * 3) / 4) < 1e-9
    print("test_average_meter PASS")


def test_augment_none_is_identity():
    data = np.arange(60 * 30, dtype=float).reshape(60, 30)
    out = augment_channel_roll(data, choice="none")
    assert out.shape == data.shape
    assert np.array_equal(out, data)
    assert out is not data  # must not mutate caller's array
    print("test_augment_none_is_identity PASS")


def test_augment_roll_preserves_shape_and_changes_values():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((60, 30))
    before = data.copy()
    for choice in ("roll_up", "roll_down"):
        out = augment_channel_roll(data, choice=choice)
        assert out.shape == (60, 30)
        assert not np.array_equal(out, data)  # a non-constant input must change
    # original must be untouched by the (non-mutating) augmentation
    assert np.array_equal(data, before)
    print("test_augment_roll_preserves_shape_and_changes_values PASS")


def test_add_deepum_to_path_imports_model():
    from train_deepum_common import add_deepum_to_path
    repo = add_deepum_to_path()
    assert (repo / "DeepUnitMatch").is_dir()
    from DeepUnitMatch.utils.mymodel import SpatioTemporalCNN_V2  # noqa: F401
    print("test_add_deepum_to_path_imports_model PASS")


def test_export_and_load_roundtrip(tmp_path_str=None):
    import tempfile, os
    import torch
    from train_deepum_common import (add_deepum_to_path, build_export_checkpoint,
                                      load_finetuned_encoder)
    add_deepum_to_path()
    from DeepUnitMatch.utils.mymodel import SpatioTemporalCNN_V2
    from DeepUnitMatch.utils.losses import CustomClipLoss, Projector
    model = SpatioTemporalCNN_V2(30, 60, 256).double()
    clip_loss = CustomClipLoss().double()
    projector = Projector(256, 128, 128, 1, 0.1).double()
    export = build_export_checkpoint(model, clip_loss, projector)
    assert set(export.keys()) == {"model", "clip_loss", "projector"}
    d = tmp_path_str or tempfile.mkdtemp()
    p = os.path.join(d, "export_epoch_0.pt")
    torch.save(export, p)
    reloaded = load_finetuned_encoder(p, device="cpu")
    # weights identical to the originals
    for k, v in model.state_dict().items():
        assert torch.allclose(reloaded.state_dict()[k], v)
    print("test_export_and_load_roundtrip PASS")


if __name__ == "__main__":
    test_average_meter()
    test_augment_none_is_identity()
    test_augment_roll_preserves_shape_and_changes_values()
    test_add_deepum_to_path_imports_model()
    test_export_and_load_roundtrip()
    print("ALL COMMON-BASIC TESTS PASSED")
