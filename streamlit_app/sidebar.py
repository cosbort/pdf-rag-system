"""
Componente sidebar per l'applicazione Streamlit.
"""
import streamlit as st
import os
import sys
import time
from pathlib import Path
import subprocess

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

def display_sidebar():
    """Visualizza la sidebar dell'applicazione."""
    with st.sidebar:
        st.title("📚 PDF RAG System")
        
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
            
            st.success(f"File {uploaded_file.name} caricato con successo!")
            
            # Aggiorna la lista dei documenti
            st.session_state.documents = get_document_list()
            
            # Resetta lo stato dell'indice
            st.session_state.index_status = check_index_status()
        
        # Lista dei documenti caricati
        documents = st.session_state.documents
        
        if documents:
            st.subheader("Documenti Caricati")
            
            for doc in documents:
                st.write(f"{doc['name']}")
                st.write(f"Dimensione: {format_file_size(doc['size'])} | Modificato: {format_timestamp(doc['modified'])}")
                
                # Selezione del documento da eliminare
                if st.button("🗑️ Elimina documento", key=f"delete_{doc['name']}"):
                    try:
                        # Elimina il file
                        os.remove(os.path.join(st.session_state.pdf_dir, doc['name']))
                        st.success(f"Documento {doc['name']} eliminato con successo!")
                        
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
        
        # Crea un container per visualizzare lo stato dell'indicizzazione
        indexing_status_container = st.empty()
        
        # Pulsante per indicizzare i documenti
        if st.button("🔍 Indicizza documenti"):
            if not documents:
                st.error("Nessun documento da indicizzare. Carica almeno un documento PDF.")
            else:
                # Esegui il processo di indicizzazione direttamente (senza thread)
                try:
                    # Ottieni i parametri dalla sessione
                    pdf_dir = st.session_state.pdf_dir
                    vector_db_dir = st.session_state.vector_db_dir
                    settings = st.session_state.retriever_settings
                    
                    # Log iniziale
                    print("DEBUG: Avvio processo di indicizzazione")
                    indexing_status_container.info("Avvio processo di indicizzazione...")
                    
                    # Inizializza il processore PDF
                    print("DEBUG: Inizializzazione processore PDF")
                    indexing_status_container.info("Caricamento dei documenti PDF...")
                    start_time = time.time()
                    
                    pdf_processor = PDFProcessor(pdf_dir)
                    print(f"DEBUG: Caricamento documenti da {pdf_dir}")
                    documents = pdf_processor.load_documents()
                    
                    loading_time = time.time() - start_time
                    print(f"DEBUG: Caricamento completato in {loading_time:.2f} secondi. Trovati {len(documents)} documenti")
                    indexing_status_container.info(f"Caricamento completato in {loading_time:.2f} secondi. Trovati {len(documents)} documenti")
                    
                    if not documents:
                        print("DEBUG: Nessun documento trovato")
                        indexing_status_container.error("Nessun documento PDF trovato nella directory specificata.")
                        return
                    
                    # Dividi i documenti in chunks
                    print("DEBUG: Inizio divisione in chunks")
                    indexing_status_container.info(f"Divisione dei documenti in chunks (dimensione: {settings['chunk_size']}, sovrapposizione: {settings['chunk_overlap']})...")
                    start_time = time.time()
                    
                    chunks = pdf_processor.split_documents(
                        chunk_size=settings['chunk_size'],
                        chunk_overlap=settings['chunk_overlap']
                    )
                    
                    chunking_time = time.time() - start_time
                    print(f"DEBUG: Divisione completata in {chunking_time:.2f} secondi. Generati {len(chunks)} chunks")
                    indexing_status_container.info(f"Divisione in chunks completata in {chunking_time:.2f} secondi. Generati {len(chunks)} chunks.")
                    
                    # Inizializza il gestore del database vettoriale
                    # Utilizziamo solo FAISS per semplicità
                    print("DEBUG: Inizio creazione indice FAISS")
                    indexing_status_container.info("Creazione dell'indice FAISS (generazione degli embedding e indicizzazione)...")
                    indexing_status_container.info("Questa operazione può richiedere tempo, specialmente per documenti grandi.")
                    indexing_status_container.info("La maggior parte del tempo è spesa nella chiamata all'API OpenAI per generare gli embedding.")
                    start_time = time.time()
                    
                    vector_store_manager = VectorStoreManager()
                    # Crea l'indice FAISS
                    print("DEBUG: Chiamata a create_faiss_index")
                    vector_store = vector_store_manager.create_faiss_index(
                        chunks, 
                        save_path=vector_db_dir
                    )
                    
                    indexing_time = time.time() - start_time
                    print(f"DEBUG: Creazione indice completata in {indexing_time:.2f} secondi")
                    indexing_status_container.info(f"Creazione dell'indice FAISS completata in {indexing_time:.2f} secondi.")
                    
                    # Salva le impostazioni
                    print("DEBUG: Salvataggio impostazioni")
                    save_retriever_settings()
                    
                    # Aggiorna lo stato dell'indice
                    st.session_state.index_status = True
                    
                    # Calcola il tempo totale
                    total_time = loading_time + chunking_time + indexing_time
                    
                    print(f"DEBUG: Indicizzazione completata in {total_time:.2f} secondi")
                    indexing_status_container.success(f"Indicizzazione completata con successo in {total_time:.2f} secondi!")
                    indexing_status_container.info(f"Dettagli: {len(documents)} documenti indicizzati in {len(chunks)} chunks.")
                    indexing_status_container.info(f"Tempo di caricamento: {loading_time:.2f}s | Tempo di chunking: {chunking_time:.2f}s | Tempo di embedding/indicizzazione: {indexing_time:.2f}s")
                    
                except Exception as e:
                    print(f"DEBUG: ERRORE durante l'indicizzazione: {str(e)}")
                    indexing_status_container.error(f"Errore durante l'indicizzazione: {str(e)}")
        
        # Separatore
        st.markdown("---")
        
        # Informazioni sull'applicazione
        with st.expander("ℹ️ Informazioni"):
            st.markdown("""
            **PDF RAG System** è un sistema avanzato di Retrieval Augmented Generation per l'analisi di documenti PDF.
            
            Sviluppato con:
            - LangChain
            - OpenAI
            - FAISS
            - Streamlit
            
            [GitHub Repository](https://github.com/cosbort/pdf-rag-system)
            """)