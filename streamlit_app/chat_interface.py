"""
Componente di interfaccia chat per l'applicazione Streamlit.
"""
import streamlit as st
import os
import sys
import time
import uuid
from pathlib import Path
import threading

# Aggiungi la directory principale al path per importare i moduli
sys.path.append(str(Path(__file__).parent.parent))

from vector_store import VectorStoreManager
from rag_generator import RAGGenerator
from advanced_retrieval import QueryTransformer, MultiQueryRetriever
from cache_manager import QueryCache

def process_query(question):
    """
    Processa una query dell'utente e genera una risposta.
    
    Args:
        question: La domanda dell'utente
        
    Returns:
        Dizionario con la risposta e i documenti recuperati
    """
    try:
        # Verifica che l'indice esista
        if not st.session_state.index_status:
            return {
                "answer": "⚠️ L'indice non è stato creato. Per favore, carica dei documenti e crea l'indice prima di porre domande.",
                "documents": [],
                "error": True
            }
        
        # Inizializza il gestore della cache
        cache = QueryCache()
        
        # Controlla se la risposta è nella cache
        cached_docs, cached_answer = cache.get_from_cache(question)
        
        if cached_answer:
            return {
                "answer": cached_answer,
                "documents": cached_docs,
                "from_cache": True,
                "error": False
            }
        
        # Ottieni le impostazioni del retriever
        settings = st.session_state.retriever_settings
        vector_db_dir = st.session_state.vector_db_dir
        
        # Inizializza il gestore del database vettoriale - usiamo solo FAISS
        vector_store_manager = VectorStoreManager()
        # Carica l'indice FAISS
        vector_store = vector_store_manager.load_faiss_index(vector_db_dir)
        
        # Ottieni il retriever
        base_retriever = vector_store_manager.get_retriever(k=settings['k'])
        
        # Utilizza il retriever multi-query se richiesto
        if settings['use_multi_query']:
            query_transformer = QueryTransformer()
            multi_query_retriever = MultiQueryRetriever(base_retriever, query_transformer)
            retriever = multi_query_retriever
            retrieved_docs = multi_query_retriever.get_relevant_documents(question)
        else:
            retriever = base_retriever
            retrieved_docs = base_retriever.get_relevant_documents(question)
        
        # Inizializza il generatore RAG
        rag_generator = RAGGenerator(llm_model=st.session_state.model)
        
        # Genera la risposta
        answer = rag_generator.answer_question(retriever, question)
        
        # Salva nella cache
        cache.save_to_cache(question, retrieved_docs, answer)
        
        return {
            "answer": answer,
            "documents": retrieved_docs,
            "from_cache": False,
            "error": False
        }
    
    except Exception as e:
        return {
            "answer": f"Si è verificato un errore durante l'elaborazione della domanda: {str(e)}",
            "documents": [],
            "error": True
        }

def display_chat_interface():
    """Visualizza l'interfaccia di chat."""
    # Contenitore principale
    chat_container = st.container()
    
    with chat_container:
        # Aggiungiamo l'immagine da Unsplash
        st.image(
            "https://images.unsplash.com/photo-1485827404703-89b55fcc595e",
            caption="Sistema RAG per l'analisi di documenti PDF",
            use_container_width=True
        )
        
        # Visualizza la cronologia dei messaggi
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Mostra i documenti recuperati se presenti
                if message["role"] == "assistant" and "documents" in message and message["documents"]:
                    with st.expander("📄 Documenti di riferimento"):
                        for i, doc in enumerate(message["documents"]):
                            source = doc.metadata.get("source", "Documento")
                            page = doc.metadata.get("page", "")
                            page_info = f" (Pagina {page})" if page else ""
                            
                            st.markdown(f"**Documento {i+1}:** {source}{page_info}")
                            st.text_area(
                                f"Contenuto {i+1}",
                                value=doc.page_content,
                                height=150,
                                key=f"doc_{message.get('id', i)}_{i}"
                            )
        
        # Gestisci l'input dell'utente
        if prompt := st.chat_input("Fai una domanda sui tuoi documenti..."):
            # Aggiungi il messaggio dell'utente alla cronologia
            message_id = str(uuid.uuid4())
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "id": message_id
            })
            
            # Visualizza il messaggio dell'utente
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Processa la query in un thread separato
            if "processing" not in st.session_state:
                st.session_state.processing = False
            
            if not st.session_state.processing:
                st.session_state.processing = True
                
                # Visualizza il messaggio dell'assistente con un indicatore di caricamento
                with st.chat_message("assistant"):
                    with st.spinner("Elaborazione della risposta..."):
                        # Processa la query
                        result = process_query(prompt)
                        
                        if result["error"]:
                            st.error(result["answer"])
                        else:
                            st.markdown(result["answer"])
                            
                            # Mostra i documenti recuperati
                            if "documents" in result and result["documents"]:
                                with st.expander("📄 Documenti di riferimento"):
                                    for i, doc in enumerate(result["documents"]):
                                        source = doc.metadata.get("source", "Documento")
                                        page = doc.metadata.get("page", "")
                                        page_info = f" (Pagina {page})" if page else ""
                                        
                                        st.markdown(f"**Documento {i+1}:** {source}{page_info}")
                                        st.text_area(
                                            f"Contenuto {i+1}",
                                            value=doc.page_content,
                                            height=150,
                                            key=f"doc_{message_id}_{i}"
                                        )
                            
                            # Mostra un badge se la risposta proviene dalla cache
                            if result.get("from_cache", False):
                                st.info("⚡ Risposta recuperata dalla cache")
                
                # Aggiungi la risposta alla cronologia
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "documents": result.get("documents", []),
                    "from_cache": result.get("from_cache", False),
                    "id": str(uuid.uuid4())
                })
                
                st.session_state.processing = False
    
    # Pulsanti di azione sotto l'interfaccia di chat
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Cancella Cronologia", key="clear_history"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("💾 Esporta Cronologia", key="export_history"):
            # Crea una versione esportabile della cronologia
            export_data = ""
            for msg in st.session_state.messages:
                role = "Tu" if msg["role"] == "user" else "Assistente"
                export_data += f"{role}: {msg['content']}\n\n"
            
            # Crea un download link
            st.download_button(
                label="📥 Scarica",
                data=export_data,
                file_name=f"chat_export_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_chat"
            )