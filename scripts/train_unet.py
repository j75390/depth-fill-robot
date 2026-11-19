"""U-Net 학습: 측정된 값 일부를 일부러 가리고, 가린 값을 맞히게 한다(이슈 #5, #6).

실제 로봇 데이터에는 '구멍 속 정답'이 없다. 그래서 정답은 센서가 실제로 잰 픽셀만 쓴다:
손실 = |예측 - 깊이| 를 (우리가 가린 곳 ∩ 센서가 실제로 잰 곳)에서만 평균. 원래 비어 있던 곳(알고리즘으로
채운 값)은 학습 정답으로 쓰지 않는다.

예) python scripts/train_unet.py --model unet_depth
    python scripts/train_unet.py --model unet_rgbd
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from depthfill.config import ROOT, load_config
from depthfill.data.nyu import NYUDataset
from depthfill.holes.generators import make_holes
from depthfill.methods.unet import SmallUNet, make_input, telea_prefill
from depthfill.seed import set_seed

MODELS = {"unet_depth": False, "unet_rgbd": True}  # 이름 → 컬러 사용 여부


def make_batch(ds, idx, kinds, ratio, rng, pool, max_depth, use_rgb, augment, prefill):
    xs, ts, ms, ps = [], [], [], []
    for i in idx:
        s = ds[i]
        rgb, depth, raw_valid = s.rgb, s.depth, s.raw_valid
        if augment and rng.random() < 0.5:  # 좌우 뒤집기(깊이 의미가 바뀌지 않는 유일한 단순 변형)
            rgb, depth, raw_valid = rgb[:, ::-1], depth[:, ::-1], raw_valid[:, ::-1]
        kind = kinds[int(rng.integers(len(kinds)))]
        hole = make_holes(kind, rgb, depth, rng, ratio, sensor_pool=pool, index=int(i))
        valid = raw_valid & ~hole
        pre = telea_prefill(np.where(valid, depth, 0.0).astype(np.float32), valid) if prefill == "telea" else None
        xs.append(make_input(depth, valid, rgb, max_depth, use_rgb, pre))
        ps.append(pre / max_depth if pre is not None else np.zeros_like(depth))
        ts.append(depth / max_depth)
        ms.append(hole & raw_valid)  # 손실을 재는 곳: 가렸지만 실제로 잰 값이 있는 곳
    to = lambda a: torch.from_numpy(np.ascontiguousarray(np.stack(a))).float()
    return to(xs), to(ts)[:, None], to(ms)[:, None], to(ps)[:, None]


def masked_l1(pred, target, mask):
    return (pred - target).abs().mul(mask).sum() / mask.sum().clamp(min=1)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--epochs", type=int, default=cfg["train"]["epochs"])
    args = ap.parse_args()
    use_rgb = MODELS[args.model]
    tc, dc, hc = cfg["train"], cfg["data"], cfg["holes"]
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full = NYUDataset(ROOT / dc["nyu_npz"], split="train")
    n_val = int(len(full) * tc["val_fraction"])
    order = np.random.default_rng(cfg["seed"]).permutation(len(full))
    val_idx, train_idx = order[:n_val], order[n_val:]
    pool = full.sensor_masks()  # 학습 분할의 실제 누락 모양(시험 분할 것은 쓰지 않는다)

    model = SmallUNet(5 if use_rgb else 2, tc["base_channels"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # 검증 배치는 매 에폭 같은 구멍으로 고정한다(에폭끼리 비교 가능하게)
    prefill = tc.get("prefill") or None
    val_batch = make_batch(full, val_idx, hc["kinds"], hc["ratio"], np.random.default_rng([cfg["seed"], 999]),
                           pool, dc["max_depth"], use_rgb, augment=False, prefill=prefill)

    history, best, t0 = [], float("inf"), time.time()
    ckpt_path = ROOT / "checkpoints" / f"{args.model}.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng([cfg["seed"], epoch])
        perm = rng.permutation(train_idx)
        losses = []
        for b in range(0, len(perm), tc["batch_size"]):
            x, t, m, p = (a.to(device) for a in make_batch(full, perm[b:b + tc["batch_size"]], hc["kinds"], hc["ratio"],
                                                          rng, pool, dc["max_depth"], use_rgb, augment=True, prefill=prefill))
            loss = masked_l1(model(x) + p, t, m)  # 1차는 p가 0이라 그대로, 2차는 Telea 결과 + 보정값
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            x, t, m, p = (a.to(device) for a in val_batch)
            val = masked_l1(model(x) + p, t, m).item()
        train_mae, val_mae = float(np.mean(losses)) * dc["max_depth"], val * dc["max_depth"]
        history.append({"epoch": epoch, "train_mae_m": train_mae, "val_mae_m": val_mae})
        if val < best:
            best = val
            torch.save({"state": model.state_dict(), "in_ch": 5 if use_rgb else 2, "base": tc["base_channels"],
                        "max_depth": dc["max_depth"], "epoch": epoch, "val_mae_m": val_mae, "seed": cfg["seed"],
                        "prefill": prefill}, ckpt_path)
        print(f"[{args.model}] epoch {epoch:3d}  train MAE {train_mae:.4f}m  val MAE {val_mae:.4f}m"
              f"{'  *' if val == best else ''}  ({time.time() - t0:.0f}s)", flush=True)

    out = ROOT / "results" / f"train_{args.model}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "best_val_mae_m": best * dc["max_depth"], "history": history,
                               "train_images": len(train_idx), "val_images": len(val_idx), "device": device},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"가장 좋은 검증 MAE {best * dc['max_depth']:.4f}m → {ckpt_path}")


if __name__ == "__main__":
    main()
