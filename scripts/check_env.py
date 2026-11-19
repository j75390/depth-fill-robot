"""환경 점검: 파이썬, 패키지, GPU(PyTorch CUDA)를 한 번에 보여 준다. 이슈 #1의 완료 기준."""
import importlib
import platform
import sys

print(f"Python {sys.version.split()[0]} / {platform.platform()}")
for name in ("numpy", "PIL", "matplotlib", "cv2", "torch"):
    try:
        mod = importlib.import_module(name)
        print(f"  {name:<11} {getattr(mod, '__version__', '설치됨')}")
    except ImportError:
        print(f"  {name:<11} 없음")

try:
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, VRAM {props.total_memory / 1024**3:.1f}GB  <- README 환경 항목에 적는다")
        x = torch.randn(256, 256, device="cuda")
        print("GPU 계산 확인:", float((x @ x).sum().item()) != 0.0)
    else:
        print("GPU: CUDA를 쓸 수 없음 (CPU로 학습하면 느리다. docs/HOME_START.md 3단계 확인)")
except ImportError:
    print("torch가 없어서 GPU 확인을 건너뜀 (docs/HOME_START.md 3단계)")
