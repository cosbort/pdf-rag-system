"""
Modulo per tecniche avanzate di recupero di documenti nel sistema RAG.
"""
from typing import List, Dict, Any, Optional, Callable
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

class QueryTransformer:
    """
    Classe per la trasformazione delle query per migliorare il recupero dei documenti.
    """
    
    def __init__(self, 
                llm_model: Optional[str] = None,
                temperature: float = 0.0):
        """
        Inizializza il trasformatore di query.
        
        Args:
            llm_model: Nome del modello LLM da utilizzare
            temperature: Temperatura per la generazione
        """
        self.llm_model = llm_model or "gpt-4o"
        self.temperature = temperature
        
        # Inizializza il modello LLM
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature
        )
        
        # Template per l'espansione della query
        self.expansion_template = """
Sei un assistente esperto nella ricerca di informazioni. 
La tua attività è espandere una query di ricerca per migliorare il recupero di documenti pertinenti.

Query originale: {query}

Genera una versione espansa della query che:
1. Includa sinonimi dei termini chiave
2. Specifichi meglio il contesto o il dominio
3. Consideri possibili formulazioni alternative

Restituisci solo la query espansa, senza spiegazioni o introduzioni.
"""
        
        # Template per la generazione di query multiple
        self.multi_query_template = """
Sei un assistente esperto nella ricerca di informazioni.
La tua attività è generare query alternative per migliorare il recupero di documenti pertinenti.

Query originale: {query}

Genera 3 query alternative che:
1. Riformulino la domanda originale in modi diversi
2. Esplorino aspetti diversi della stessa domanda
3. Utilizzino terminologia diversa ma mantengano lo stesso significato

Restituisci solo le 3 query alternative, una per riga, senza numerazione, spiegazioni o introduzioni.
"""
        
        # Inizializza i prompt
        self.expansion_prompt = ChatPromptTemplate.from_template(self.expansion_template)
        self.multi_query_prompt = ChatPromptTemplate.from_template(self.multi_query_template)
    
    def expand_query(self, query: str) -> str:
        """
        Espande una query per migliorare il recupero dei documenti.
        
        Args:
            query: Query originale
            
        Returns:
            Query espansa
        """
        # Crea la catena di espansione
        expansion_chain = (
            self.expansion_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Espandi la query
        expanded_query = expansion_chain.invoke({"query": query})
        
        return expanded_query
    
    def generate_multi_queries(self, query: str) -> List[str]:
        """
        Genera query multiple per migliorare il recupero dei documenti.
        
        Args:
            query: Query originale
            
        Returns:
            Lista di query alternative
        """
        # Crea la catena di generazione di query multiple
        multi_query_chain = (
            self.multi_query_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Genera query multiple
        result = multi_query_chain.invoke({"query": query})
        
        # Dividi il risultato in righe e rimuovi eventuali righe vuote
        queries = [q.strip() for q in result.split("\n") if q.strip()]
        
        # Aggiungi la query originale
        all_queries = [query] + queries
        
        return all_queries


class EnsembleRetriever(BaseRetriever):
    """
    Retriever che combina i risultati di più retriever.
    """
    
    def __init__(self, 
                retrievers: List[BaseRetriever],
                weights: Optional[List[float]] = None):
        """
        Inizializza il retriever ensemble.
        
        Args:
            retrievers: Lista di retriever da combinare
            weights: Pesi da assegnare a ciascun retriever (opzionale)
        """
        super().__init__()
        self.retrievers = retrievers
        
        # Se i pesi non sono specificati, assegna pesi uguali
        if weights is None:
            self.weights = [1.0] * len(retrievers)
        else:
            if len(weights) != len(retrievers):
                raise ValueError("Il numero di pesi deve essere uguale al numero di retriever")
            self.weights = weights
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Recupera documenti pertinenti combinando i risultati di più retriever.
        
        Args:
            query: Query di ricerca
            run_manager: Manager per i callback
            
        Returns:
            Lista di documenti pertinenti
        """
        # Dizionario per tenere traccia dei documenti unici e dei loro punteggi
        doc_dict = {}
        
        # Recupera documenti da ciascun retriever
        for i, retriever in enumerate(self.retrievers):
            docs = retriever.get_relevant_documents(query, callbacks=run_manager.get_child())
            weight = self.weights[i]
            
            for doc in docs:
                # Crea una chiave unica per il documento
                key = f"{doc.page_content}_{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}"
                
                # Se il documento esiste già, aggiorna il punteggio
                if key in doc_dict:
                    doc_dict[key]["score"] += weight
                else:
                    doc_dict[key] = {
                        "doc": doc,
                        "score": weight
                    }
        
        # Ordina i documenti per punteggio in ordine decrescente
        sorted_docs = sorted(doc_dict.values(), key=lambda x: x["score"], reverse=True)
        
        # Estrai solo i documenti
        result_docs = [item["doc"] for item in sorted_docs]
        
        return result_docs


class MultiQueryRetriever:
    """
    Classe per il recupero di documenti utilizzando query multiple.
    """
    
    def __init__(self, 
                base_retriever: BaseRetriever,
                query_transformer: QueryTransformer):
        """
        Inizializza il retriever multi-query.
        
        Args:
            base_retriever: Retriever di base da utilizzare
            query_transformer: Trasformatore di query
        """
        self.base_retriever = base_retriever
        self.query_transformer = query_transformer
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Recupera documenti pertinenti utilizzando query multiple.
        
        Args:
            query: Query originale
            
        Returns:
            Lista di documenti pertinenti
        """
        # Genera query multiple
        queries = self.query_transformer.generate_multi_queries(query)
        
        # Dizionario per tenere traccia dei documenti unici
        unique_docs = {}
        
        # Recupera documenti per ciascuna query
        for q in queries:
            docs = self.base_retriever.get_relevant_documents(q)
            
            for doc in docs:
                # Crea una chiave unica per il documento
                key = f"{doc.page_content}_{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}"
                
                # Aggiungi il documento se non esiste già
                if key not in unique_docs:
                    unique_docs[key] = doc
        
        # Converti il dizionario in una lista
        result_docs = list(unique_docs.values())
        
        return result_docs