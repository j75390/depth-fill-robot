# 집에서 바로 시작하기

목표: **90분 안에** 환경을 만들고, 뼈대가 도는 것을 확인하고, 첫 이슈를 시작한다.

## 0. 시작 전에
- 이 문서와 [구조 규칙](STRUCTURE.md), [판단 기록](decision-log.md)을 먼저 읽는다(10분).
- 이 저장소는 **비공개**다. 받으려면 GitHub 로그인이 필요하다.

## 1. 받기 (5분)
```powershell
git clone https://github.com/j75390/depth-patch.git
cd depth-patch
```

## 2. 환경 만들기 (10분)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```
가상환경(.venv)을 만들고 패키지를 설치한 뒤 `scripts/check_env.py`로 환경을 점검한다.
이후 모든 명령은 `.\.venv\Scripts\python`으로 실행한다.

## 3. PyTorch(GPU) 설치 (15~30분, 가장 막히기 쉬운 단계)
1. `nvidia-smi`로 그래픽카드와 드라이버를 확인한다(안 되면 NVIDIA 드라이버 먼저 설치).
2. PyTorch 공식 설치 안내 페이지(pytorch.org의 "Get Started")에서 **Windows / Pip / CUDA 버전**을 고르면 설치 명령이 나온다. 그 명령을 `.venv`에서 그대로 실행한다. (CUDA 버전은 `nvidia-smi` 오른쪽 위에 나온다. 정확한 명령은 페이지가 알려 주는 것을 쓴다.)
3. 확인: `.\.venv\Scripts\python scripts\check_env.py` 에 **GPU 이름과 VRAM**이 나오고 "GPU 계산 확인: True"이면 성공. GPU 이름과 VRAM을 README의 환경 항목에 적는다(이슈 #1).
- GPU가 안 잡히면 CPU로도 일단 진행할 수 있다(작은 이미지, 작은 모델). 시간을 쓰지 말고 이슈 #3, #4부터 시작한다.

## 4. 뼈대가 도는지 확인 (5분)
```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python scripts\run_eval.py --synthetic --n 20
```
- 시험이 모두 통과하고 비교표(유형/방법별 MAE, RMSE)가 나오면 정상이다.
- 이 표는 **합성 장면**이라서 숫자 자체에는 의미가 없다. 틀이 돈다는 확인용이다.

## 5. 이슈 순서 (그대로 따라 하면 된다)
| 순서 | 이슈 | 이미 있는 것 | 해야 할 것 |
|---|---|---|---|
| 1 | #1 환경 | setup.ps1, check_env.py | 3단계 완료, GPU·VRAM을 README에 적기 |
| 2 | #2 데이터 | 합성 데이터셋, `NYUDataset` 자리 | 공개 데이터 선택·내려받기·로더, 사용 범위를 판단 기록에 |
| 3 | #3 구멍 생성기 | 구멍 3종 초안과 시험 | **실제 이미지에 적용해 눈으로 확인하고 파라미터 조정**, 샘플 이미지 저장 |
| 4 | #4 기준선·평가 | 지표, 평가 도구, 기준선 3종 | 실제 데이터로 돌리기, 실패 사례 이미지 저장, OpenCV 기준선 추가 |
| 5 | #5 모델 A | 자리(`methods/unet.py`) | 작은 U-Net, 가려 맞히기 학습, 학습 곡선 |
| 6 | #6 모델 B | | 컬러 결합, 유형별 비교, 개선 1~2회 |
| 7 | #7 뷰어 | | 전/후 슬라이더, 3D 점 |
| 8 | #8 정리 | | README, 판단 기록, 포트폴리오 |
| 9 | #9(선택) | | 11/26까지 핵심이 끝났을 때만 |

> **합성 데이터로 만든 뼈대는 이슈 #3·#4의 절반 정도를 이미 해 둔 것**이다. 실제 데이터에서 눈으로 확인하고 고치는 일이 남아 있다.

## 6. 일하는 방법
- 이슈마다 브랜치를 만들고(`feature/3-hole-generator`), 작게 끝내 PR을 올리고 병합한다.
- 판단할 때마다 `docs/decision-log.md`에 적는다(왜 이 크기, 왜 이 손실, 뭐가 안 됐는지).
- 커밋 전에 `pytest`, `ruff check .`를 돌린다.

## 7. 막혔을 때
| 증상 | 확인 |
|---|---|
| `pip install`이 느리거나 실패 | 인터넷 확인 후 재실행. 그래도 안 되면 `pip` 오류 메시지를 이슈에 적는다 |
| torch는 설치됐는데 GPU가 안 잡힘 | 드라이버 버전, 3단계에서 고른 CUDA 버전이 맞는지 확인 |
| 학습 중 메모리 부족 | `configs/default.toml`의 `image_size`를 줄이고 배치 크기를 줄인다. 줄인 이유를 판단 기록에 |
| 학습이 너무 느림 | 이미지를 더 줄이거나 학습 장수를 줄이고, 이유를 기록 |
| 결과가 기준선보다 나쁨 | 정상이다. 숫자를 만들지 말고 원인(구멍 유형별로 어디서 지는지)을 분석해 기록 |
