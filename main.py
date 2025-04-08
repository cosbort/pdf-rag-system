"""
Sistema RAG per l'analisi di documenti PDF utilizzando LangChain.
"""
import os
import argparse
from typing import Optional
from dotenv import load_dotenv
from pdf_loader import PDFProcessor
from vector_store import VectorStoreManager
from rag_generator import RAGGenerator

# Carica le variabili d'ambiente
load_dotenv()

def setup_argparse():
    """Configura l'analisi degli argomenti da linea di comando."""
    parser = argparse.ArgumentParser(
        description="Sistema RAG per l'analisi di documenti PDF"
    )
    
    parser.add_argument(
        "--pdf_dir", 
        type=str, 
        required=True,
        help="Directory contenente i documenti PDF da analizzare"
    )
    
    parser.add_argument(
        "--index_type", 
        type=str, 
        choices=["faiss", "chroma"], 
        default="faiss",
        help="Tipo di indice vettoriale da utilizzare (faiss o chroma)"
    )
    
    parser.add_argument(
        "--persist_dir", 
        type=str, 
        default="./vector_db",
        help="Directory dove salvare l'indice vettoriale"
    )
    
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=1000,
        help="Dimensione dei chunk in caratteri"
    )
    
    parser.add_argument(
        "--chunk_overlap", 
        type=int, 
        default=200,
        help="Sovrapposizione tra i chunk in caratteri"
    )
    
    parser.add_argument(
        "--k", 
        type=int, 
        default=4,
        help="Numero di documenti da recuperare per ogni query"
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Avvia la modalità interattiva per porre domande"
    )
    
    return parser.parse_args()

def main():
    """Funzione principale del sistema RAG."""
    # Analizza gli argomenti da linea di comando
    args = setup_argparse()
    
    # Verifica che la directory dei PDF esista
    if not os.path.isdir(args.pdf_dir):
        print(f"Errore: La directory {args.pdf_dir} non esiste.")
        return
    
    # Crea la directory di persistenza se non esiste
    os.makedirs(args.persist_dir, exist_ok=True)
    
    print(f"Inizializzazione del sistema RAG per l'analisi di documenti PDF in {args.pdf_dir}...")
    
    # Inizializza il processore PDF
    pdf_processor = PDFProcessor(args.pdf_dir)
    
    # Carica i documenti PDF
    documents = pdf_processor.load_documents()
    if not documents:
        print("Nessun documento PDF trovato nella directory specificata.")
        return
    
    # Dividi i documenti in chunks
    chunks = pdf_processor.split_documents(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Inizializza il gestore del database vettoriale
    if args.index_type == "chroma":
        vector_store_manager = VectorStoreManager(
            persist_directory=args.persist_dir
        )
        # Crea il database Chroma
        vector_store = vector_store_manager.create_chroma_db(chunks)
    else:  # faiss
        vector_store_manager = VectorStoreManager()
        # Crea l'indice FAISS
        vector_store = vector_store_manager.create_faiss_index(
            chunks, 
            save_path=args.persist_dir
        )
    
    # Ottieni il retriever
    retriever = vector_store_manager.get_retriever(k=args.k)
    
    # Inizializza il generatore RAG
    rag_generator = RAGGenerator()
    
    print("\nSistema RAG inizializzato con successo!")
    print(f"- {len(documents)} documenti caricati")
    print(f"- {len(chunks)} chunks creati")
    print(f"- Indice vettoriale {args.index_type} creato in {args.persist_dir}")
    
    # Modalità interattiva
    if args.interactive:
        print("\n=== Modalità Interattiva ===")
        print("Digita 'exit' o 'quit' per uscire.")
        
        while True:
            question = input("\nInserisci la tua domanda: ")
            
            if question.lower() in ["exit", "quit"]:
                break
            
            if not question.strip():
                continue
            
            print("\nElaborazione della domanda...")
            answer = rag_generator.answer_question(retriever, question)
            print("\nRisposta:")
            print(answer)
    
    print("\nGrazie per aver utilizzato il sistema RAG!")

if __name__ == "__main__":
    main()