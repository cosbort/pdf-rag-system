"""
Modulo per la generazione di risposte utilizzando il sistema RAG.
"""
import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

class RAGGenerator:
    """
    Classe per generare risposte utilizzando il sistema RAG.
    """
    
    def __init__(self, 
                llm_model: Optional[str] = None,
                temperature: float = 0.0,
                streaming: bool = False):
        """
        Inizializza il generatore RAG.
        
        Args:
            llm_model: Nome del modello LLM da utilizzare
            temperature: Temperatura per la generazione (0.0 = più deterministica)
            streaming: Se abilitare lo streaming delle risposte
        """
        # Utilizza il modello specificato o quello predefinito dall'env
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "gpt-4o")
        self.temperature = temperature
        self.streaming = streaming
        
        # Inizializza il modello LLM
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            streaming=self.streaming
        )
        
        # Template di prompt predefinito per RAG in italiano
        self.default_system_template = """Sei un assistente AI esperto che risponde a domande basandosi esclusivamente sui documenti forniti.
Utilizza solo le informazioni presenti nei documenti per rispondere.
Se l'informazione non è presente nei documenti, rispondi onestamente che non puoi rispondere basandoti sui documenti forniti.
Cita le fonti specifiche (nomi dei documenti) quando possibile.
Mantieni le risposte concise, accurate e informative.
"""
        
        self.default_human_template = """
Documenti:
{context}

Domanda: {question}

Risposta (basata solo sui documenti forniti):
"""
        
        # Inizializza il prompt predefinito
        self.set_default_prompt()
        
    def set_default_prompt(self):
        """
        Imposta il prompt predefinito per il sistema RAG.
        """
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.default_system_template),
            ("human", self.default_human_template)
        ])
        
    def set_custom_prompt(self, system_template: str, human_template: str):
        """
        Imposta un prompt personalizzato per il sistema RAG.
        
        Args:
            system_template: Template per il messaggio di sistema
            human_template: Template per il messaggio umano
        """
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
    
    def _format_docs(self, docs: List[Document]) -> str:
        """
        Formatta i documenti per l'inclusione nel prompt.
        
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
            
            formatted_doc = f"--- Documento: {source}{page_info} ---\n{doc.page_content}\n"
            formatted_docs.append(formatted_doc)
            
        return "\n".join(formatted_docs)
    
    def create_rag_chain(self, retriever):
        """
        Crea una catena RAG completa.
        
        Args:
            retriever: Retriever per recuperare i documenti
            
        Returns:
            Catena RAG configurata
        """
        # Definiamo la catena RAG
        rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def answer_question(self, retriever, question: str) -> str:
        """
        Risponde a una domanda utilizzando il sistema RAG.
        
        Args:
            retriever: Retriever per recuperare i documenti
            question: Domanda da rispondere
            
        Returns:
            Risposta generata
        """
        # Crea la catena RAG
        rag_chain = self.create_rag_chain(retriever)
        
        # Genera la risposta
        response = rag_chain.invoke(question)
        
        return response