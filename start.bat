@echo off
title AppaltoAI Server
echo ============================================
echo   AppaltoAI - AI Proprietaria Gare d'Appalto
echo ============================================
echo.

:: Controlla Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Python non trovato. Installalo da https://python.org
    pause
    exit /b 1
)

:: Installa dipendenze se mancano
echo [1/2] Verifica dipendenze...
pip install scikit-learn pdfplumber fastapi uvicorn python-multipart >nul 2>&1

echo [2/2] Avvio server...
echo.
echo ══════════════════════════════════════════════
echo   Apri il browser su: http://localhost:8000
echo ══════════════════════════════════════════════
echo.

cd /d "%~dp0"
python server.py

pause
