@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python generate_fixed_byte_emb.py

pause