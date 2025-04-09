"""
Utilità per l'applicazione Streamlit.
"""
import streamlit as st
import os
import json
from pathlib import Path

def set_page_config():
    """Configura le impostazioni della pagina Streamlit."""
    st.set_page_config(
        page_title="PDF RAG System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Aggiungi CSS personalizzato
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stApp {
            background-color: #f8f9fa;
        }
        .stSidebar .sidebar-content {
            background-color: #f1f3f5;
        }
        .stButton>button {
            background-color: #4F8BF9;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #3670d6;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #e6f3ff;
            border-left: 5px solid #4F8BF9;
        }
        .assistant-message {
            background-color: #f0f0f0;
            border-left: 5px solid #6c757d;
        }
        .message-content {
            margin-top: 0.5rem;
        }
        .document-card {
            padding: 1rem;
            border-radius: 5px;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            margin-bottom: 0.5rem;
        }
        .document-title {
            font-weight: bold;
            color: #4F8BF9;
        }
        .document-info {
            font-size: 0.8em;
            color: #6c757d;
        }
        .document-actions {
            display: flex;
            justify-content: flex-end;
            margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def initialize_session_state():
    """Inizializza lo stato della sessione Streamlit."""
    # Inizializza la cronologia dei messaggi
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Inizializza l'ID della sessione
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    
    # Inizializza la lista dei documenti
    if "documents" not in st.session_state:
        st.session_state.documents = []
    
    # Inizializza il modello selezionato
    if "model" not in st.session_state:
        st.session_state.model = "gpt-4o"
    
    # Inizializza il percorso della directory dei documenti
    if "pdf_dir" not in st.session_state:
        st.session_state.pdf_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
        # Crea la directory se non esiste
        os.makedirs(st.session_state.pdf_dir, exist_ok=True)
    
    # Inizializza il percorso della directory del database vettoriale
    if "vector_db_dir" not in st.session_state:
        st.session_state.vector_db_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_db")
        # Crea la directory se non esiste
        os.makedirs(st.session_state.vector_db_dir, exist_ok=True)
    
    # Inizializza le impostazioni del retriever
    if "retriever_settings" not in st.session_state:
        st.session_state.retriever_settings = {
            "k": 4,
            "use_multi_query": False,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "index_type": "faiss"
        }
    
    # Inizializza lo stato dell'indice
    if "index_status" not in st.session_state:
        st.session_state.index_status = check_index_status()

def check_index_status():
    """Controlla se l'indice vettoriale esiste."""
    vector_db_dir = st.session_state.vector_db_dir
    
    # Controlla se la directory esiste e contiene file
    if not os.path.exists(vector_db_dir):
        return False
    
    # Per FAISS, controlla se esistono i file dell'indice
    faiss_files = [f for f in os.listdir(vector_db_dir) if f.endswith('.faiss') or f.endswith('.pkl')]
    if faiss_files:
        return True
    
    # Per Chroma, controlla se esistono le directory di Chroma
    chroma_dirs = [d for d in os.listdir(vector_db_dir) if os.path.isdir(os.path.join(vector_db_dir, d))]
    if chroma_dirs:
        return True
    
    return False

def save_retriever_settings():
    """Salva le impostazioni del retriever in un file JSON."""
    settings_path = os.path.join(st.session_state.vector_db_dir, "settings.json")
    
    with open(settings_path, "w") as f:
        json.dump(st.session_state.retriever_settings, f)

def load_retriever_settings():
    """Carica le impostazioni del retriever da un file JSON."""
    settings_path = os.path.join(st.session_state.vector_db_dir, "settings.json")
    
    if os.path.exists(settings_path):
        with open(settings_path, "r") as f:
            st.session_state.retriever_settings = json.load(f)

def get_document_list():
    """Ottiene la lista dei documenti PDF nella directory specificata."""
    pdf_dir = st.session_state.pdf_dir
    
    if not os.path.exists(pdf_dir):
        return []
    
    pdf_files = []
    for file in os.listdir(pdf_dir):
        if file.lower().endswith('.pdf'):
            file_path = os.path.join(pdf_dir, file)
            file_stats = os.stat(file_path)
            
            pdf_files.append({
                "filename": file,
                "path": file_path,
                "size": file_stats.st_size,
                "modified": file_stats.st_mtime,
                "id": file  # Usiamo il nome del file come ID
            })
    
    # Ordina per data di modifica (più recente prima)
    pdf_files.sort(key=lambda x: x["modified"], reverse=True)
    
    return pdf_files

def format_file_size(size_bytes):
    """Formatta la dimensione del file in un formato leggibile."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def format_timestamp(timestamp):
    """Formatta il timestamp in una data leggibile."""
    import datetime
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime("%d/%m/%Y %H:%M")