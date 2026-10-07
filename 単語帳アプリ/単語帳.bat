@echo off
rem 論文単語帳を起動してブラウザで開く。このウィンドウを閉じると終了する。
title 論文単語帳
cd /d "%~dp0.."
uv run python "%~dp0app.py"
if errorlevel 1 pause
