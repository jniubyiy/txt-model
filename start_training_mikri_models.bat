@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python training_mikri_models.py
pause