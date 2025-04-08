"""
Modulo per la valutazione delle prestazioni del sistema RAG.
"""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

class RAGEvaluator:
    """
    Classe per valutare le prestazioni del sistema RAG.
    """
    
    def __init__(self, 
                llm_model: Optional[str] = None,
                temperature: float = 0.0):
        """
        Inizializza il valutatore RAG.
        
        Args:
            llm_model: Nome del modello LLM da utilizzare per la valutazione
            temperature: Temperatura per la generazione
        """
        self.llm_model = llm_model or "gpt-4o"
        self.temperature = temperature
        
        # Inizializza il modello LLM per la valutazione
        self.eval_llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature
        )
        
        # Template per la valutazione della pertinenza
        self.relevance_template = """
Sei un valutatore esperto di sistemi RAG (Retrieval Augmented Generation).
Valuta la pertinenza dei documenti recuperati rispetto alla domanda dell'utente.

Domanda: {question}

Documenti recuperati:
{retrieved_docs}

Valuta quanto i documenti recuperati sono pertinenti alla domanda su una scala da 1 a 5, dove:
1 = Per nulla pertinenti
2 = Poco pertinenti
3 = Moderatamente pertinenti
4 = Molto pertinenti
5 = Estremamente pertinenti

Fornisci un punteggio numerico e una breve spiegazione della tua valutazione.
"""
        
        # Template per la valutazione della risposta
        self.answer_template = """
Sei un valutatore esperto di sistemi RAG (Retrieval Augmented Generation).
Valuta la qualità della risposta generata rispetto alla domanda e ai documenti recuperati.

Domanda: {question}

Documenti recuperati:
{retrieved_docs}

Risposta generata:
{answer}

Valuta la risposta su una scala da 1 a 5 per ciascuno dei seguenti criteri:

1. Accuratezza (la risposta è fattualmente corretta secondo i documenti)
2. Completezza (la risposta copre tutti gli aspetti rilevanti della domanda)
3. Concisione (la risposta è concisa e va al punto)
4. Citazione delle fonti (la risposta cita correttamente le fonti quando necessario)

Per ogni criterio, fornisci un punteggio numerico e una breve spiegazione.
Infine, fornisci un punteggio complessivo da 1 a 5 e un breve riassunto della tua valutazione.
"""
        
        # Inizializza i prompt
        self.relevance_prompt = ChatPromptTemplate.from_template(self.relevance_template)
        self.answer_prompt = ChatPromptTemplate.from_template(self.answer_template)
    
    def _format_docs(self, docs: List[Document]) -> str:
        """
        Formatta i documenti per la valutazione.
        
        Args:
            docs: Lista di documenti da formattare
            
        Returns:
            Stringa formattata con i contenuti dei documenti
        """
        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", f"Documento {i+1}")
            page = doc.metadata.get("page", "")
            page_info = f" (Pagina {page})" if page else ""
            
            formatted_doc = f"--- Documento {i+1}: {source}{page_info} ---\n{doc.page_content}\n"
            formatted_docs.append(formatted_doc)
            
        return "\n".join(formatted_docs)
    
    def evaluate_retrieval(self, question: str, retrieved_docs: List[Document]) -> str:
        """
        Valuta la pertinenza dei documenti recuperati.
        
        Args:
            question: Domanda dell'utente
            retrieved_docs: Documenti recuperati
            
        Returns:
            Valutazione della pertinenza
        """
        # Formatta i documenti
        formatted_docs = self._format_docs(retrieved_docs)
        
        # Crea la catena di valutazione
        eval_chain = (
            self.relevance_prompt 
            | self.eval_llm 
            | StrOutputParser()
        )
        
        # Esegui la valutazione
        evaluation = eval_chain.invoke({
            "question": question,
            "retrieved_docs": formatted_docs
        })
        
        return evaluation
    
    def evaluate_answer(self, question: str, retrieved_docs: List[Document], answer: str) -> str:
        """
        Valuta la qualità della risposta generata.
        
        Args:
            question: Domanda dell'utente
            retrieved_docs: Documenti recuperati
            answer: Risposta generata
            
        Returns:
            Valutazione della risposta
        """
        # Formatta i documenti
        formatted_docs = self._format_docs(retrieved_docs)
        
        # Crea la catena di valutazione
        eval_chain = (
            self.answer_prompt 
            | self.eval_llm 
            | StrOutputParser()
        )
        
        # Esegui la valutazione
        evaluation = eval_chain.invoke({
            "question": question,
            "retrieved_docs": formatted_docs,
            "answer": answer
        })
        
        return evaluation