"""
Script per avviare l'applicazione Streamlit.
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    """Avvia l'applicazione Streamlit."""
    # Ottieni il percorso assoluto della directory del progetto
    project_dir = Path(__file__).parent.absolute()
    
    # Percorso dell'applicazione Streamlit
    streamlit_app_path = os.path.join(project_dir, "streamlit_app", "app.py")
    
    # Verifica che il file esista
    if not os.path.exists(streamlit_app_path):
        print(f"Errore: Il file {streamlit_app_path} non esiste.")
        return
    
    # Avvia l'applicazione Streamlit
    print("Avvio dell'applicazione Streamlit...")
    print(f"Applicazione: {streamlit_app_path}")
    print("Premi Ctrl+C per terminare l'applicazione.")
    
    # Comando per avviare Streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", streamlit_app_path, "--server.port=8501"]
    
    try:
        # Esegui il comando
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nApplicazione terminata dall'utente.")
    except Exception as e:
        print(f"Errore durante l'avvio dell'applicazione: {str(e)}")

if __name__ == "__main__":
    main()