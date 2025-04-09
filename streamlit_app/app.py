"""
Applicazione Streamlit per il sistema RAG di analisi di documenti PDF.
"""
import streamlit as st
import os
import sys
import time
from pathlib import Path

# Aggiungi la directory principale al path per importare i moduli
sys.path.append(str(Path(__file__).parent.parent))

# Importa i componenti dell'interfaccia
from streamlit_app.sidebar import display_sidebar
from streamlit_app.chat_interface import display_chat_interface
from streamlit_app.utils import set_page_config, initialize_session_state

def main():
    """Funzione principale dell'applicazione Streamlit."""
    # Configura la pagina
    set_page_config()
    
    # Inizializza lo stato della sessione
    initialize_session_state()
    
    # Titolo dell'applicazione con stile personalizzato
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1 style='color: #4F8BF9;'>📚 PDF RAG System</h1>
            <p style='font-size: 1.2em; color: #666;'>Sistema avanzato di Retrieval Augmented Generation per l'analisi di documenti PDF</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Mostra la sidebar
    display_sidebar()
    
    # Mostra l'interfaccia di chat
    display_chat_interface()
    
    # Aggiungi footer
    st.markdown(
        """
        <div style='text-align: center; margin-top: 30px; padding: 10px; color: #888; font-size: 0.8em;'>
            Sviluppato con ❤️ utilizzando LangChain e Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()