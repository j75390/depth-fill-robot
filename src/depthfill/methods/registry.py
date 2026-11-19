"""이름으로 방법을 꺼낸다. 새 방법은 여기에 한 줄 더하면 평가 도구가 바로 쓴다."""
from __future__ import annotations

from .baselines import DiffusionFill, MeanFill, NoFill, OpenCVInpaint


def _unet(kind: str):
    # torch는 모델을 실제로 쓸 때만 가져온다(기준선만 쓰는 시험·평가는 torch 없이 돈다)
    def factory():
        from . import unet

        return getattr(unet, kind)()

    return factory


_FACTORIES = {
    "nofill": NoFill,
    "meanfill": MeanFill,
    "diffusion": DiffusionFill,
    "opencv_telea": OpenCVInpaint,
    "unet_depth": _unet("unet_depth"),   # 이슈 #5: 깊이만
    "unet_rgbd": _unet("unet_rgbd"),     # 이슈 #6: 컬러 함께
}


def get_methods(names: list[str]):
    methods = []
    for name in names:
        if name not in _FACTORIES:
            raise KeyError(f"알 수 없는 방법: {name} (가능: {sorted(_FACTORIES)})")
        methods.append(_FACTORIES[name]())
    return methods
