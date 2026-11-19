"""작은 U-Net 보완 모델 (이슈 #5 깊이만, #6 컬러 함께).

- torch는 이 파일 안에서만 가져온다(다른 모듈은 torch 없이도 돈다).
- 입력(1차): 값이 있는 곳만 남긴 깊이(÷max_depth), 값이 있는지 표시(valid), [컬러 3채널].
- 입력(2차, prefill="telea"): 깊이 칸에 OpenCV Telea로 먼저 채운 깊이를 넣고, 모델은 그 위에 더할 보정값(잔차)만 낸다.
  1차 모델이 가장 강한 기준선(Telea)을 넘지 못해서 바꿨다(판단 기록 2026-11-19).
- 출력: 전체 깊이. 값이 있던 곳은 입력을 그대로 쓰고 구멍만 모델 값으로 채운다.
- 학습은 scripts/train_unet.py. 이 파일에는 모델 정의와 추론만 둔다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from depthfill.config import ROOT


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """3번 줄이고 3번 키우는 U-Net. 120x160 입력에서 가장 작은 칸이 15x20이 된다."""

    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([_block(in_ch, c[0]), _block(c[0], c[1]), _block(c[1], c[2])])
        self.mid = _block(c[2], c[3])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[3], c[2], 2, 2), nn.ConvTranspose2d(c[2], c[1], 2, 2),
                                 nn.ConvTranspose2d(c[1], c[0], 2, 2)])
        self.dec = nn.ModuleList([_block(c[2] * 2, c[2]), _block(c[1] * 2, c[1]), _block(c[0] * 2, c[0])])
        self.head = nn.Conv2d(c[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.mid(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def telea_prefill(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from depthfill.methods.baselines import OpenCVInpaint

    return OpenCVInpaint().fill(depth, valid)


def make_input(depth: np.ndarray, valid: np.ndarray, rgb: np.ndarray | None,
               max_depth: float, use_rgb: bool, prefill: np.ndarray | None = None) -> np.ndarray:
    """(C, H, W) 입력. 학습과 추론이 같은 함수를 써서 입력이 어긋나지 않게 한다."""
    base = prefill if prefill is not None else np.where(valid, depth, 0.0)
    chans = [base / max_depth, valid.astype(np.float32)]
    if use_rgb:
        if rgb is None:
            raise ValueError("컬러 모델에는 rgb가 필요하다")
        chans += [rgb[..., k] for k in range(3)]
    return np.stack(chans).astype(np.float32)


class UNetFill:
    """평가 도구용 Method. 학습된 체크포인트를 읽어 구멍을 채운다."""

    def __init__(self, name: str, use_rgb: bool, checkpoint: str | Path | None = None):
        self.name = name
        self.use_rgb = use_rgb
        path = Path(checkpoint) if checkpoint else ROOT / "checkpoints" / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path}가 없습니다. 먼저 scripts/train_unet.py --model {name}으로 학습하세요.")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.max_depth = float(ckpt["max_depth"])
        self.prefill = ckpt.get("prefill")  # None(1차) 또는 "telea"(2차)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SmallUNet(ckpt["in_ch"], ckpt["base"]).to(self.device).eval()
        self.model.load_state_dict(ckpt["state"])

    @torch.no_grad()
    def fill(self, depth, valid, rgb=None):
        pre = telea_prefill(depth, valid) if self.prefill == "telea" else None
        x = torch.from_numpy(make_input(depth, valid, rgb, self.max_depth, self.use_rgb, pre))[None].to(self.device)
        out = self.model(x)[0, 0].cpu().numpy()
        if pre is not None:
            out = out + pre / self.max_depth  # 2차: 모델은 Telea 결과에 더할 보정값을 낸다
        pred = np.clip(out, 0, None) * self.max_depth
        return np.where(valid, depth, pred).astype(np.float32)


def unet_depth() -> UNetFill:
    return UNetFill("unet_depth", use_rgb=False)


def unet_rgbd() -> UNetFill:
    return UNetFill("unet_rgbd", use_rgb=True)
