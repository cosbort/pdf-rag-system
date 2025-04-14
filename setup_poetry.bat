@echo off
echo ===================================
echo Inizializzazione PDF RAG con Poetry
echo ===================================
echo.

REM Verifica se Poetry è installato
where poetry >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Poetry non è installato. Per favore installalo seguendo le istruzioni su:
    echo https://python-poetry.org/docs/#installation
    echo.
    echo Puoi installarlo facilmente con:
    echo (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content ^| python -
    exit /b 1
)

echo Poetry è installato. Procedo con l'inizializzazione...
echo.

REM Esegui lo script Python di setup
python setup_poetry.py

if %ERRORLEVEL% NEQ 0 (
    echo Si è verificato un errore durante l'inizializzazione.
    exit /b 1
)

echo.
echo Per avviare l'applicazione, esegui:
echo poetry run streamlit run streamlit_app/app.py
echo.
echo Buon lavoro!
