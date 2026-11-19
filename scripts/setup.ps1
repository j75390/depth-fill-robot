# 처음 한 번 실행: 가상환경 만들기 + 패키지 설치 + 환경 점검
# 사용: powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path ".venv")) { python -m venv .venv }
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python scripts\check_env.py
Write-Host "`n다음: docs\HOME_START.md 3단계(PyTorch 설치)"
