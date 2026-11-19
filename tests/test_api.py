"""API 응답 형식 시험. 모델 체크포인트 없이 기준선(OpenCV Telea)으로 돈다."""
import base64
import io

import numpy as np
from PIL import Image

from depthfill.api.app import SIZE, create_app


def _png(arr) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _client():
    return create_app(["opencv_telea"]).test_client()


def _scene():
    mm = np.full(SIZE, 2000, np.uint16)
    mm[40:60, 50:90] = 0  # 구멍
    return mm


def test_health_lists_methods():
    r = _client().get("/v1/health")
    assert r.status_code == 200 and r.json["methods"] == ["opencv_telea"]


def test_fill_returns_full_depth_in_the_same_format():
    r = _client().post("/v1/fill", json={"depth_png": _png(_scene())})
    assert r.status_code == 200
    body = r.json
    assert set(body) == {"depth_png", "method", "filled_pixels", "height", "width", "elapsed_ms"}
    out = np.array(Image.open(io.BytesIO(base64.b64decode(body["depth_png"]))))
    assert out.shape == SIZE and out.dtype == np.uint16
    assert (out > 0).all(), "구멍이 남으면 안 된다"
    assert abs(int(out[50, 70]) - 2000) < 50, "평평한 면의 구멍은 주변 깊이로 채워져야 한다"
    assert body["filled_pixels"] == 20 * 40


def test_bad_requests_get_400_with_a_message():
    c = _client()
    assert c.post("/v1/fill", json={}).status_code == 400
    assert c.post("/v1/fill", json={"depth_png": "not-base64!"}).status_code == 400
    small = _png(np.full((10, 10), 1000, np.uint16))
    r = c.post("/v1/fill", json={"depth_png": small})
    assert r.status_code == 400 and "크기" in r.json["error"]
    assert c.post("/v1/fill", json={"depth_png": _png(np.zeros(SIZE, np.uint16))}).status_code == 400
    assert c.post("/v1/fill", json={"depth_png": _png(_scene()), "method": "nope"}).status_code == 400
