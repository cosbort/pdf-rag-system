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
        </style>
        """,
        unsafe_allow_html=True,
    )

def initialize_session_state():
    """Inizializza le variabili di sessione."""
    # Directory per i documenti PDF
    if "pdf_dir" not in st.session_state:
        st.session_state.pdf_dir = os.path.join(os.getcwd(), "documents")
        # Crea la directory se non esiste
        os.makedirs(st.session_state.pdf_dir, exist_ok=True)
    
    # Directory per il database vettoriale
    if "vector_db_dir" not in st.session_state:
        st.session_state.vector_db_dir = os.path.join(os.getcwd(), "vector_db")
        # Crea la directory se non esiste
        os.makedirs(st.session_state.vector_db_dir, exist_ok=True)
    
    # Modello LLM predefinito
    if "model" not in st.session_state:
        st.session_state.model = "gpt-3.5-turbo"
    
    # Impostazioni del retriever
    if "retriever_settings" not in st.session_state:
        st.session_state.retriever_settings = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "k": 4,
            "use_multi_query": False
        }
    
    # Stato dell'indice
    if "index_status" not in st.session_state:
        st.session_state.index_status = check_index_status()
    
    # Cronologia dei messaggi
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Lista dei documenti
    if "documents" not in st.session_state:
        st.session_state.documents = get_document_list()
    
    # Stato dell'indicizzazione
    if "indexing_in_progress" not in st.session_state:
        st.session_state.indexing_in_progress = False

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
    documents = []
    
    if os.path.exists(pdf_dir):
        for filename in os.listdir(pdf_dir):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(pdf_dir, filename)
                file_stats = os.stat(file_path)
                
                documents.append({
                    'name': filename,
                    'filename': filename,  # Per retrocompatibilità
                    'path': file_path,
                    'size': file_stats.st_size,
                    'modified': file_stats.st_mtime
                })
    
    return documents

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