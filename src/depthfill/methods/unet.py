"""TODO(이슈 #5, #6): 작은 U-Net 보완 모델.

만들 때 지킬 것
- torch는 이 파일 안에서만 가져온다(다른 모듈은 torch 없이도 돌아야 한다).
- Method 형태(base.py)를 따른다: fill(depth, valid, rgb) -> 채운 깊이.
- 학습 코드는 scripts/train_*.py에 두고, 이 파일에는 모델과 추론만 둔다.
- 입력 크기·모델 크기를 고른 이유는 docs/decision-log.md에 적는다.
"""
