"""NYU 로더 시험. 실제 데이터를 내려받지 않고 작은 가짜 npz로 확인한다."""
import numpy as np

from depthfill.data.nyu import NYUDataset, _resize_raw


def test_resize_raw_ignores_missing_values():
    raw = np.zeros((4, 4), np.float32)
    raw[:, :2] = 2.0           # 왼쪽 절반만 측정됨
    raw[0, 2] = 4.0            # 오른쪽 위 2x2 칸 중 1칸만 측정됨(절반 미만)
    out = _resize_raw(raw, (2, 2))
    assert np.allclose(out[:, 0], 2.0), "0(측정 없음)을 섞어 평균내면 값이 작아진다"
    assert out[0, 1] == 0.0, "절반 넘게 빈 칸은 측정 없음으로 둔다"


def test_dataset_reads_split_and_sensor_masks(tmp_path):
    n, h, w = 5, 6, 8
    rng = np.random.default_rng(0)
    raw = rng.uniform(1, 3, (n, h, w)).astype(np.float16)
    raw[:, 0, :] = 0  # 맨 윗줄은 센서가 못 잼
    path = tmp_path / "nyu.npz"
    np.savez(path, rgb=rng.integers(0, 255, (n, h, w, 3), dtype=np.uint8), depth=raw.astype(np.float16),
             raw=raw, train=np.array([0, 1, 2]), test=np.array([3, 4]))
    ds = NYUDataset(path, split="test")
    assert len(ds) == 2
    s = ds[0]
    assert s.name == "nyu_0003"
    assert s.rgb.max() <= 1.0 and s.depth.dtype == np.float32
    assert not s.raw_valid[0].any() and s.raw_valid[1:].all()
    assert ds.sensor_masks().shape == (2, h, w)
