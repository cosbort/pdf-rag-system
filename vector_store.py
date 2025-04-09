"""
Modulo per la gestione del database vettoriale per il sistema RAG.
"""
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

class VectorStoreManager:
    """
    Classe per gestire il database vettoriale per il sistema RAG.
    """
    
    def __init__(self, 
                embedding_model: Optional[str] = None,
                persist_directory: Optional[str] = None):
        """
        Inizializza il gestore del database vettoriale.
        
        Args:
            embedding_model: Nome del modello di embedding da utilizzare
            persist_directory: Directory dove persistere il database (solo per Chroma)
        """
        # Utilizza il modello specificato o quello predefinito dall'env
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vector_store = None
        
    def create_faiss_index(self, documents: List[Document], save_path: Optional[str] = None) -> FAISS:
        """
        Crea un indice FAISS dai documenti.
        
        Args:
            documents: Lista di documenti da indicizzare
            save_path: Percorso dove salvare l'indice FAISS
            
        Returns:
            Istanza di FAISS
        """
        if not documents:
            raise ValueError("Nessun documento fornito per la creazione dell'indice.")
        
        # Stampa informazioni sui documenti per debug
        print(f"DEBUG FAISS: Creazione indice FAISS per {len(documents)} documenti")
        
        try:
            # Crea l'indice FAISS
            print("DEBUG FAISS: Inizio generazione embedding")
            self.vector_store = FAISS.from_documents(
                documents, 
                self.embeddings
            )
            print("DEBUG FAISS: Embedding generati con successo")
            
            # Salva l'indice se è specificato un percorso
            if save_path:
                print(f"DEBUG FAISS: Salvataggio indice in {save_path}")
                self.vector_store.save_local(save_path)
                print(f"DEBUG FAISS: Indice FAISS salvato in {save_path}")
                
            return self.vector_store
        except Exception as e:
            print(f"DEBUG FAISS: ERRORE durante la creazione dell'indice: {str(e)}")
            raise e
    
    def create_chroma_db(self, documents: List[Document]) -> Chroma:
        """
        Crea un database Chroma dai documenti.
        
        Args:
            documents: Lista di documenti da indicizzare
            
        Returns:
            Istanza di Chroma
        """
        if not documents:
            raise ValueError("Nessun documento fornito per la creazione del database.")
        
        if not self.persist_directory:
            raise ValueError("È necessario specificare una directory di persistenza per Chroma.")
        
        # Crea il database Chroma
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Persiste il database
        self.vector_store.persist()
        print(f"Database Chroma creato e persistito in {self.persist_directory}")
        
        return self.vector_store
    
    def load_faiss_index(self, load_path: str) -> FAISS:
        """
        Carica un indice FAISS esistente.
        
        Args:
            load_path: Percorso da cui caricare l'indice
            
        Returns:
            Istanza di FAISS
        """
        self.vector_store = FAISS.load_local(load_path, self.embeddings)
        print(f"Indice FAISS caricato da {load_path}")
        return self.vector_store
    
    def load_chroma_db(self) -> Chroma:
        """
        Carica un database Chroma esistente.
        
        Returns:
            Istanza di Chroma
        """
        if not self.persist_directory:
            raise ValueError("È necessario specificare una directory di persistenza per caricare Chroma.")
        
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print(f"Database Chroma caricato da {self.persist_directory}")
        return self.vector_store
    
    def get_retriever(self, search_type: str = "similarity", 
                     k: int = 4, 
                     score_threshold: Optional[float] = None) -> VectorStoreRetriever:
        """
        Ottiene un retriever dal vector store.
        
        Args:
            search_type: Tipo di ricerca ("similarity", "mmr", o "similarity_score_threshold")
            k: Numero di documenti da recuperare
            score_threshold: Soglia di punteggio per similarity_score_threshold
            
        Returns:
            Istanza di VectorStoreRetriever
        """
        if not self.vector_store:
            raise ValueError("Nessun vector store disponibile. Creane o caricane uno prima.")
        
        search_kwargs = {"k": k}
        if search_type == "similarity_score_threshold" and score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )