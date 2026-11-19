"""이름으로 방법을 꺼낸다. 새 방법은 여기에 한 줄 더하면 평가 도구가 바로 쓴다."""
from __future__ import annotations

from .baselines import DiffusionFill, MeanFill, NoFill, OpenCVInpaint

_FACTORIES = {
    "nofill": NoFill,
    "meanfill": MeanFill,
    "diffusion": DiffusionFill,
    "opencv_telea": OpenCVInpaint,
    # TODO(이슈 #5): "unet_depth": UNetDepth
    # TODO(이슈 #6): "unet_rgbd": UNetRGBD
}


def get_methods(names: list[str]):
    methods = []
    for name in names:
        if name not in _FACTORIES:
            raise KeyError(f"알 수 없는 방법: {name} (가능: {sorted(_FACTORIES)})")
        methods.append(_FACTORIES[name]())
    return methods
