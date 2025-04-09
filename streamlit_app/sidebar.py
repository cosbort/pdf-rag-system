"""
Componente sidebar per l'applicazione Streamlit.
"""
import streamlit as st
import os
import sys
import time
from pathlib import Path
import subprocess
import threading

# Aggiungi la directory principale al path per importare i moduli
sys.path.append(str(Path(__file__).parent.parent))

from pdf_loader import PDFProcessor
from vector_store import VectorStoreManager
from streamlit_app.utils import (
    get_document_list, 
    format_file_size, 
    format_timestamp, 
    save_retriever_settings,
    check_index_status
)

def run_indexing_process():
    """Esegue il processo di indicizzazione in background."""
    try:
        # Ottieni i parametri dalla sessione
        pdf_dir = st.session_state.pdf_dir
        vector_db_dir = st.session_state.vector_db_dir
        settings = st.session_state.retriever_settings
        
        # Inizializza il processore PDF
        pdf_processor = PDFProcessor(pdf_dir)
        
        # Carica i documenti PDF
        with st.session_state.get("indexing_status_placeholder"):
            st.write("Caricamento dei documenti PDF...")
        documents = pdf_processor.load_documents()
        
        if not documents:
            with st.session_state.get("indexing_status_placeholder"):
                st.error("Nessun documento PDF trovato nella directory specificata.")
            st.session_state.indexing_in_progress = False
            return
        
        # Dividi i documenti in chunks
        with st.session_state.get("indexing_status_placeholder"):
            st.write(f"Divisione dei documenti in chunks (dimensione: {settings['chunk_size']}, sovrapposizione: {settings['chunk_overlap']})...")
        chunks = pdf_processor.split_documents(
            chunk_size=settings['chunk_size'],
            chunk_overlap=settings['chunk_overlap']
        )
        
        # Inizializza il gestore del database vettoriale
        if settings['index_type'] == "chroma":
            with st.session_state.get("indexing_status_placeholder"):
                st.write("Creazione del database Chroma...")
            vector_store_manager = VectorStoreManager(
                persist_directory=vector_db_dir
            )
            # Crea il database Chroma
            vector_store = vector_store_manager.create_chroma_db(chunks)
        else:  # faiss
            with st.session_state.get("indexing_status_placeholder"):
                st.write("Creazione dell'indice FAISS...")
            vector_store_manager = VectorStoreManager()
            # Crea l'indice FAISS
            vector_store = vector_store_manager.create_faiss_index(
                chunks, 
                save_path=vector_db_dir
            )
        
        # Salva le impostazioni
        save_retriever_settings()
        
        # Aggiorna lo stato dell'indice
        st.session_state.index_status = True
        
        with st.session_state.get("indexing_status_placeholder"):
            st.success(f"Indicizzazione completata con successo! {len(documents)} documenti indicizzati in {len(chunks)} chunks.")
        
    except Exception as e:
        with st.session_state.get("indexing_status_placeholder"):
            st.error(f"Errore durante l'indicizzazione: {str(e)}")
    
    finally:
        # Imposta lo stato di indicizzazione come completato
        st.session_state.indexing_in_progress = False

def display_sidebar():
    """Visualizza la sidebar dell'applicazione."""
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/cosbort/pdf-rag-system/master/docs/logo.png", width=100, use_column_width=False)
        
        # Sezione modello
        st.subheader("🤖 Modello LLM")
        model_options = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        selected_model = st.selectbox(
            "Seleziona il modello da utilizzare:",
            options=model_options,
            index=model_options.index(st.session_state.model) if st.session_state.model in model_options else 0,
            key="model_selector"
        )
        st.session_state.model = selected_model
        
        # Separatore
        st.markdown("---")
        
        # Sezione gestione documenti
        st.subheader("📄 Gestione Documenti")
        
        # Upload di documenti
        uploaded_file = st.file_uploader("Carica un documento PDF", type=["pdf"])
        if uploaded_file:
            # Salva il file nella directory dei documenti
            save_path = os.path.join(st.session_state.pdf_dir, uploaded_file.name)
            
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"File '{uploaded_file.name}' caricato con successo!")
            
            # Aggiorna la lista dei documenti
            st.session_state.documents = get_document_list()
            
            # Resetta lo stato dell'indice
            st.session_state.index_status = check_index_status()
        
        # Visualizza la lista dei documenti
        st.subheader("📚 Documenti Caricati")
        
        # Pulsante per aggiornare la lista
        if st.button("🔄 Aggiorna lista"):
            st.session_state.documents = get_document_list()
        
        # Ottieni la lista dei documenti
        documents = get_document_list()
        st.session_state.documents = documents
        
        if not documents:
            st.info("Nessun documento caricato. Carica un documento PDF per iniziare.")
        else:
            for doc in documents:
                with st.container():
                    st.markdown(
                        f"""
                        <div class='document-card'>
                            <div class='document-title'>{doc['filename']}</div>
                            <div class='document-info'>
                                Dimensione: {format_file_size(doc['size'])} | 
                                Modificato: {format_timestamp(doc['modified'])}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Selezione documento da eliminare
            if len(documents) > 0:
                selected_doc = st.selectbox(
                    "Seleziona un documento da eliminare:",
                    options=[doc['filename'] for doc in documents],
                    key="doc_to_delete"
                )
                
                if st.button("🗑️ Elimina documento"):
                    # Trova il documento selezionato
                    doc_to_delete = next((doc for doc in documents if doc['filename'] == selected_doc), None)
                    
                    if doc_to_delete:
                        # Elimina il file
                        try:
                            os.remove(doc_to_delete['path'])
                            st.success(f"Documento '{doc_to_delete['filename']}' eliminato con successo!")
                            
                            # Aggiorna la lista dei documenti
                            st.session_state.documents = get_document_list()
                            
                            # Resetta lo stato dell'indice
                            st.session_state.index_status = check_index_status()
                            
                        except Exception as e:
                            st.error(f"Errore durante l'eliminazione del documento: {str(e)}")
        
        # Separatore
        st.markdown("---")
        
        # Sezione indicizzazione
        st.subheader("🔍 Indicizzazione")
        
        # Impostazioni del retriever
        with st.expander("⚙️ Impostazioni Avanzate"):
            # Tipo di indice
            index_type = st.radio(
                "Tipo di indice:",
                options=["faiss", "chroma"],
                index=0 if st.session_state.retriever_settings['index_type'] == "faiss" else 1,
                key="index_type"
            )
            st.session_state.retriever_settings['index_type'] = index_type
            
            # Dimensione dei chunk
            chunk_size = st.slider(
                "Dimensione dei chunk (caratteri):",
                min_value=100,
                max_value=2000,
                value=st.session_state.retriever_settings['chunk_size'],
                step=100,
                key="chunk_size"
            )
            st.session_state.retriever_settings['chunk_size'] = chunk_size
            
            # Sovrapposizione dei chunk
            chunk_overlap = st.slider(
                "Sovrapposizione dei chunk (caratteri):",
                min_value=0,
                max_value=500,
                value=st.session_state.retriever_settings['chunk_overlap'],
                step=50,
                key="chunk_overlap"
            )
            st.session_state.retriever_settings['chunk_overlap'] = chunk_overlap
            
            # Numero di documenti da recuperare
            k_value = st.slider(
                "Numero di documenti da recuperare (k):",
                min_value=1,
                max_value=10,
                value=st.session_state.retriever_settings['k'],
                step=1,
                key="k_value"
            )
            st.session_state.retriever_settings['k'] = k_value
            
            # Utilizzo di multi-query
            use_multi_query = st.checkbox(
                "Utilizza multi-query retrieval",
                value=st.session_state.retriever_settings['use_multi_query'],
                key="use_multi_query"
            )
            st.session_state.retriever_settings['use_multi_query'] = use_multi_query
        
        # Stato dell'indice
        if st.session_state.index_status:
            st.success("✅ Indice creato")
        else:
            st.warning("⚠️ Indice non creato")
        
        # Pulsante per indicizzare i documenti
        if "indexing_in_progress" not in st.session_state:
            st.session_state.indexing_in_progress = False
        
        if st.session_state.indexing_in_progress:
            st.info("Indicizzazione in corso...")
            
            # Placeholder per lo stato dell'indicizzazione
            if "indexing_status_placeholder" not in st.session_state:
                st.session_state.indexing_status_placeholder = st.empty()
        else:
            if st.button("🔍 Indicizza documenti"):
                if not documents:
                    st.error("Nessun documento da indicizzare. Carica almeno un documento PDF.")
                else:
                    st.session_state.indexing_in_progress = True
                    st.session_state.indexing_status_placeholder = st.empty()
                    
                    # Avvia il processo di indicizzazione in un thread separato
                    threading.Thread(target=run_indexing_process).start()
                    st.experimental_rerun()
        
        # Separatore
        st.markdown("---")
        
        # Informazioni sull'applicazione
        with st.expander("ℹ️ Informazioni"):
            st.markdown("""
            **PDF RAG System** è un sistema avanzato di Retrieval Augmented Generation per l'analisi di documenti PDF.
            
            Sviluppato con:
            - LangChain
            - OpenAI
            - FAISS/Chroma
            - Streamlit
            
            [GitHub Repository](https://github.com/cosbort/pdf-rag-system)
            """)