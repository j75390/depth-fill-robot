"""모델 서비스 API (선택 과제 #9). 웹 프레임워크(Flask)는 이 폴더에서만 쓴다.

POST /v1/fill
  요청(JSON): {"depth_png": <16비트 PNG, 밀리미터, 0=값 없음, base64>,
              "rgb_png": <8비트 컬러 PNG, base64, 선택>, "method": "unet_rgbd"(선택)}
  응답(JSON): {"depth_png": <채운 16비트 PNG(mm), base64>, "method": ..., "filled_pixels": n,
              "height": h, "width": w, "elapsed_ms": t}
GET /v1/health → {"status": "ok", "methods": [...]}

입력 크기는 학습과 같은 120x160만 받는다(다른 크기는 400). 깊이 단위를 밀리미터 정수로 정한 이유는
OpenCV와 대부분의 깊이 카메라 SDK가 같은 형식을 쓰기 때문이다.
"""
from __future__ import annotations

import base64
import io
import time

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from depthfill.methods.registry import get_methods

SIZE = (120, 160)
MAX_MM = 10_000


class BadRequest(ValueError):
    pass


def decode_png(b64: str, mode: str) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64, validate=True)))
    except Exception as e:  # 잘못된 base64·PNG는 모두 400으로
        raise BadRequest(f"PNG를 읽을 수 없어요: {e}") from e
    arr = np.array(img.convert(mode) if mode == "RGB" else img)
    if arr.shape[:2] != SIZE:
        raise BadRequest(f"크기는 {SIZE[0]}x{SIZE[1]}여야 해요(받은 크기 {arr.shape[0]}x{arr.shape[1]})")
    return arr


def encode_depth_png(depth_m: np.ndarray) -> str:
    mm = np.clip(np.round(depth_m * 1000), 0, 65535).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(mm).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def create_app(methods: list[str] | None = None, default: str | None = None) -> Flask:
    names = methods or ["opencv_telea", "unet_depth", "unet_rgbd"]
    loaded = {}
    for name in names:
        try:
            loaded[name] = get_methods([name])[0]
        except (FileNotFoundError, ImportError):  # 체크포인트·torch가 없으면 그 방법만 뺀다
            continue
    if not loaded:
        raise RuntimeError("쓸 수 있는 방법이 없어요")
    fallback = default if default in loaded else next(reversed(loaded))
    app = Flask(__name__)
    app.json.ensure_ascii = False

    @app.get("/v1/health")
    def health():
        return jsonify(status="ok", methods=list(loaded), default=fallback)

    @app.post("/v1/fill")
    def fill():
        t0 = time.perf_counter()
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or "depth_png" not in body:
            return jsonify(error="JSON 본문에 depth_png(16비트 PNG, base64)가 필요해요"), 400
        name = body.get("method", fallback)
        if name not in loaded:
            return jsonify(error=f"알 수 없는 방법: {name}", methods=list(loaded)), 400
        try:
            mm = decode_png(body["depth_png"], "I;16")
            if mm.dtype != np.uint16 and mm.max() > MAX_MM:
                raise BadRequest("깊이는 밀리미터 단위 16비트 정수여야 해요")
            rgb = decode_png(body["rgb_png"], "RGB").astype(np.float32) / 255 if body.get("rgb_png") else None
        except BadRequest as e:
            return jsonify(error=str(e)), 400
        method = loaded[name]
        if rgb is None and getattr(method, "use_rgb", False):
            return jsonify(error=f"{name}에는 rgb_png가 필요해요"), 400
        depth = mm.astype(np.float32) / 1000.0
        valid = mm > 0
        if not valid.any():
            return jsonify(error="측정된 깊이가 하나도 없어요"), 400
        out = method.fill(np.where(valid, depth, 0.0).astype(np.float32), valid, rgb)
        return jsonify(depth_png=encode_depth_png(out), method=name, filled_pixels=int((~valid).sum()),
                       height=SIZE[0], width=SIZE[1], elapsed_ms=round((time.perf_counter() - t0) * 1000, 1))

    return app


if __name__ == "__main__":
    create_app().run(port=8000)
