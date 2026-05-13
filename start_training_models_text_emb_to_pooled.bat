@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python training_models_text_emb_to_pooled.py
pause