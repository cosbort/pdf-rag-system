"""
Modulo per la gestione della cache e l'ottimizzazione delle prestazioni del sistema RAG.
"""
import os
import json
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
from langchain_core.documents import Document
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

class QueryCache:
    """
    Classe per la gestione della cache delle query.
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Inizializza il gestore della cache.
        
        Args:
            cache_dir: Directory dove salvare i file di cache
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Carica la cache esistente
        self.cache_index_path = os.path.join(cache_dir, "cache_index.json")
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Carica l'indice della cache.
        
        Returns:
            Dizionario con l'indice della cache
        """
        if os.path.exists(self.cache_index_path):
            try:
                with open(self.cache_index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Errore durante il caricamento dell'indice della cache: {e}")
                return {}
        else:
            return {}
    
    def _save_cache_index(self):
        """
        Salva l'indice della cache.
        """
        try:
            with open(self.cache_index_path, "w", encoding="utf-8") as f:
                json.dump(self.cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Errore durante il salvataggio dell'indice della cache: {e}")
    
    def _get_query_hash(self, query: str) -> str:
        """
        Calcola l'hash di una query.
        
        Args:
            query: Query da hashare
            
        Returns:
            Hash della query
        """
        return hashlib.md5(query.encode("utf-8")).hexdigest()
    
    def _serialize_document(self, doc: Document) -> Dict[str, Any]:
        """
        Serializza un documento per il salvataggio nella cache.
        
        Args:
            doc: Documento da serializzare
            
        Returns:
            Dizionario con i dati serializzati
        """
        return {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
    
    def _deserialize_document(self, data: Dict[str, Any]) -> Document:
        """
        Deserializza un documento dalla cache.
        
        Args:
            data: Dati serializzati
            
        Returns:
            Documento deserializzato
        """
        return Document(
            page_content=data["page_content"],
            metadata=data["metadata"]
        )
    
    def get_from_cache(self, query: str) -> Tuple[Optional[List[Document]], Optional[str]]:
        """
        Recupera documenti e risposta dalla cache.
        
        Args:
            query: Query da cercare nella cache
            
        Returns:
            Tupla con documenti e risposta, o None se non trovati
        """
        query_hash = self._get_query_hash(query)
        
        if query_hash in self.cache_index:
            cache_entry = self.cache_index[query_hash]
            
            # Verifica se la cache è scaduta (7 giorni)
            if time.time() - cache_entry["timestamp"] > 7 * 24 * 60 * 60:
                return None, None
            
            # Carica i documenti dalla cache
            cache_file_path = os.path.join(self.cache_dir, f"{query_hash}.json")
            
            try:
                with open(cache_file_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                
                # Deserializza i documenti
                docs = [self._deserialize_document(doc_data) for doc_data in cache_data["documents"]]
                answer = cache_data["answer"]
                
                return docs, answer
            except Exception as e:
                print(f"Errore durante il caricamento della cache: {e}")
                return None, None
        else:
            return None, None
    
    def save_to_cache(self, query: str, documents: List[Document], answer: str):
        """
        Salva documenti e risposta nella cache.
        
        Args:
            query: Query da salvare
            documents: Documenti da salvare
            answer: Risposta da salvare
        """
        query_hash = self._get_query_hash(query)
        
        # Aggiorna l'indice della cache
        self.cache_index[query_hash] = {
            "query": query,
            "timestamp": time.time()
        }
        
        # Serializza i documenti
        serialized_docs = [self._serialize_document(doc) for doc in documents]
        
        # Prepara i dati da salvare
        cache_data = {
            "query": query,
            "documents": serialized_docs,
            "answer": answer,
            "timestamp": time.time()
        }
        
        # Salva i dati nella cache
        cache_file_path = os.path.join(self.cache_dir, f"{query_hash}.json")
        
        try:
            with open(cache_file_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            # Aggiorna l'indice della cache
            self._save_cache_index()
        except Exception as e:
            print(f"Errore durante il salvataggio nella cache: {e}")
    
    def clear_cache(self):
        """
        Cancella tutta la cache.
        """
        # Rimuovi tutti i file di cache
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Errore durante la rimozione del file {file_path}: {e}")
        
        # Resetta l'indice della cache
        self.cache_index = {}
        self._save_cache_index()
        
        print("Cache cancellata con successo.")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Ottiene statistiche sulla cache.
        
        Returns:
            Dizionario con le statistiche
        """
        total_entries = len(self.cache_index)
        total_size = 0
        oldest_timestamp = time.time()
        newest_timestamp = 0
        
        # Calcola la dimensione totale e le timestamp
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.cache_dir, filename)
                total_size += os.path.getsize(file_path)
                
                # Controlla se è un file di cache (non l'indice)
                if filename != "cache_index.json":
                    query_hash = filename.split(".")[0]
                    if query_hash in self.cache_index:
                        timestamp = self.cache_index[query_hash]["timestamp"]
                        oldest_timestamp = min(oldest_timestamp, timestamp)
                        newest_timestamp = max(newest_timestamp, timestamp)
        
        # Converti i timestamp in date leggibili
        oldest_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(oldest_timestamp)) if total_entries > 0 else "N/A"
        newest_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest_timestamp)) if total_entries > 0 else "N/A"
        
        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_entry": oldest_date,
            "newest_entry": newest_date
        }