#!/usr/bin/env python
"""
Script per l'inizializzazione dell'ambiente Poetry.
"""
import os
import subprocess
import sys
from pathlib import Path

def main():
    """Inizializza l'ambiente Poetry."""
    print("🚀 Inizializzazione dell'ambiente Poetry per PDF RAG System...")
    
    # Verifica se Poetry è installato
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
        print("✅ Poetry è già installato nel sistema.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Poetry non è installato. Installalo seguendo le istruzioni su https://python-poetry.org/docs/#installation")
        sys.exit(1)
    
    # Crea l'ambiente virtuale
    print("\n📦 Creazione dell'ambiente virtuale con Poetry...")
    subprocess.run(["poetry", "install"], check=True)
    
    # Verifica se il file .env esiste, altrimenti lo crea
    env_file = Path(".env")
    if not env_file.exists():
        print("\n🔑 Creazione del file .env...")
        with open(env_file, "w") as f:
            f.write("# Inserisci qui la tua chiave API OpenAI\n")
            f.write("OPENAI_API_KEY=\n\n")
            f.write("# Configurazioni opzionali\n")
            f.write("EMBEDDING_MODEL=text-embedding-3-small\n")
            f.write("LLM_MODEL=gpt-3.5-turbo\n")
        print("✅ File .env creato. Ricordati di inserire la tua chiave API OpenAI.")
    else:
        print("✅ File .env già esistente.")
    
    # Crea le directory necessarie
    print("\n📁 Creazione delle directory necessarie...")
    Path("documents").mkdir(exist_ok=True)
    Path("vector_db").mkdir(exist_ok=True)
    Path("cache").mkdir(exist_ok=True)
    print("✅ Directory create con successo.")
    
    print("\n🎉 Inizializzazione completata! Ora puoi eseguire l'applicazione con:")
    print("   poetry run streamlit run streamlit_app/app.py")

if __name__ == "__main__":
    main()
