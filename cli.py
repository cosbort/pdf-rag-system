"""
Interfaccia a riga di comando per il sistema RAG.
"""
import os
import sys
import argparse
import time
from typing import List, Optional
from dotenv import load_dotenv
from pdf_loader import PDFProcessor
from vector_store import VectorStoreManager
from rag_generator import RAGGenerator
from evaluation import RAGEvaluator
from advanced_retrieval import QueryTransformer, MultiQueryRetriever
from cache_manager import QueryCache

# Carica le variabili d'ambiente
load_dotenv()

def setup_argparse():
    """Configura l'analisi degli argomenti da linea di comando."""
    parser = argparse.ArgumentParser(
        description="Sistema RAG avanzato per l'analisi di documenti PDF"
    )
    
    # Comandi principali
    subparsers = parser.add_subparsers(dest="command", help="Comando da eseguire")
    
    # Comando per indicizzare i documenti
    index_parser = subparsers.add_parser("index", help="Indicizza i documenti PDF")
    index_parser.add_argument(
        "--pdf_dir", 
        type=str, 
        required=True,
        help="Directory contenente i documenti PDF da analizzare"
    )
    index_parser.add_argument(
        "--index_type", 
        type=str, 
        choices=["faiss", "chroma"], 
        default="faiss",
        help="Tipo di indice vettoriale da utilizzare (faiss o chroma)"
    )
    index_parser.add_argument(
        "--persist_dir", 
        type=str, 
        default="./vector_db",
        help="Directory dove salvare l'indice vettoriale"
    )
    index_parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=1000,
        help="Dimensione dei chunk in caratteri"
    )
    index_parser.add_argument(
        "--chunk_overlap", 
        type=int, 
        default=200,
        help="Sovrapposizione tra i chunk in caratteri"
    )
    
    # Comando per interrogare il sistema
    query_parser = subparsers.add_parser("query", help="Interroga il sistema RAG")
    query_parser.add_argument(
        "--question", 
        type=str,
        help="Domanda da porre al sistema"
    )
    query_parser.add_argument(
        "--index_type", 
        type=str, 
        choices=["faiss", "chroma"], 
        default="faiss",
        help="Tipo di indice vettoriale da utilizzare (faiss o chroma)"
    )
    query_parser.add_argument(
        "--persist_dir", 
        type=str, 
        default="./vector_db",
        help="Directory dove caricare l'indice vettoriale"
    )
    query_parser.add_argument(
        "--k", 
        type=int, 
        default=4,
        help="Numero di documenti da recuperare per ogni query"
    )
    query_parser.add_argument(
        "--use_cache", 
        action="store_true",
        help="Utilizza la cache per le query"
    )
    query_parser.add_argument(
        "--use_multi_query", 
        action="store_true",
        help="Utilizza la tecnica multi-query per migliorare i risultati"
    )
    query_parser.add_argument(
        "--show_docs", 
        action="store_true",
        help="Mostra i documenti recuperati"
    )
    query_parser.add_argument(
        "--evaluate", 
        action="store_true",
        help="Valuta la qualità della risposta"
    )
    
    # Comando per la modalità interattiva
    interactive_parser = subparsers.add_parser("interactive", help="Avvia la modalità interattiva")
    interactive_parser.add_argument(
        "--index_type", 
        type=str, 
        choices=["faiss", "chroma"], 
        default="faiss",
        help="Tipo di indice vettoriale da utilizzare (faiss o chroma)"
    )
    interactive_parser.add_argument(
        "--persist_dir", 
        type=str, 
        default="./vector_db",
        help="Directory dove caricare l'indice vettoriale"
    )
    interactive_parser.add_argument(
        "--k", 
        type=int, 
        default=4,
        help="Numero di documenti da recuperare per ogni query"
    )
    interactive_parser.add_argument(
        "--use_cache", 
        action="store_true",
        help="Utilizza la cache per le query"
    )
    interactive_parser.add_argument(
        "--use_multi_query", 
        action="store_true",
        help="Utilizza la tecnica multi-query per migliorare i risultati"
    )
    interactive_parser.add_argument(
        "--show_docs", 
        action="store_true",
        help="Mostra i documenti recuperati"
    )
    
    # Comando per gestire la cache
    cache_parser = subparsers.add_parser("cache", help="Gestisci la cache")
    cache_parser.add_argument(
        "--clear", 
        action="store_true",
        help="Cancella la cache"
    )
    cache_parser.add_argument(
        "--stats", 
        action="store_true",
        help="Mostra statistiche sulla cache"
    )
    
    return parser.parse_args()

def index_documents(args):
    """Indicizza i documenti PDF."""
    print(f"Indicizzazione dei documenti PDF in {args.pdf_dir}...")
    
    # Verifica che la directory dei PDF esista
    if not os.path.isdir(args.pdf_dir):
        print(f"Errore: La directory {args.pdf_dir} non esiste.")
        return
    
    # Crea la directory di persistenza se non esiste
    os.makedirs(args.persist_dir, exist_ok=True)
    
    # Inizializza il processore PDF
    pdf_processor = PDFProcessor(args.pdf_dir)
    
    # Carica i documenti PDF
    start_time = time.time()
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
    
    end_time = time.time()
    
    print("\nIndicizzazione completata con successo!")
    print(f"- {len(documents)} documenti caricati")
    print(f"- {len(chunks)} chunks creati")
    print(f"- Indice vettoriale {args.index_type} creato in {args.persist_dir}")
    print(f"- Tempo di elaborazione: {end_time - start_time:.2f} secondi")

def query_system(args):
    """Interroga il sistema RAG."""
    # Verifica che l'indice esista
    if not os.path.exists(args.persist_dir):
        print(f"Errore: L'indice {args.persist_dir} non esiste.")
        return
    
    # Inizializza il gestore della cache se richiesto
    cache = None
    if args.use_cache:
        cache = QueryCache()
    
    # Ottieni la domanda
    question = args.question
    if not question:
        question = input("Inserisci la tua domanda: ")
    
    print("\nElaborazione della domanda...")
    
    # Controlla se la risposta è nella cache
    cached_docs = None
    cached_answer = None
    if cache:
        cached_docs, cached_answer = cache.get_from_cache(question)
    
    if cached_answer:
        print("\nRisposta (dalla cache):")
        print(cached_answer)
        
        if args.show_docs and cached_docs:
            print("\nDocumenti recuperati:")
            for i, doc in enumerate(cached_docs):
                source = doc.metadata.get("source", f"Documento {i+1}")
                page = doc.metadata.get("page", "")
                page_info = f" (Pagina {page})" if page else ""
                print(f"\n--- Documento {i+1}: {source}{page_info} ---")
                print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
        
        return
    
    # Inizializza il gestore del database vettoriale
    if args.index_type == "chroma":
        vector_store_manager = VectorStoreManager(
            persist_directory=args.persist_dir
        )
        # Carica il database Chroma
        vector_store = vector_store_manager.load_chroma_db()
    else:  # faiss
        vector_store_manager = VectorStoreManager()
        # Carica l'indice FAISS
        vector_store = vector_store_manager.load_faiss_index(args.persist_dir)
    
    # Ottieni il retriever
    base_retriever = vector_store_manager.get_retriever(k=args.k)
    
    # Utilizza il retriever multi-query se richiesto
    if args.use_multi_query:
        query_transformer = QueryTransformer()
        multi_query_retriever = MultiQueryRetriever(base_retriever, query_transformer)
        retriever = multi_query_retriever
        retrieved_docs = multi_query_retriever.get_relevant_documents(question)
    else:
        retriever = base_retriever
        retrieved_docs = base_retriever.get_relevant_documents(question)
    
    # Inizializza il generatore RAG
    rag_generator = RAGGenerator()
    
    # Genera la risposta
    start_time = time.time()
    answer = rag_generator.answer_question(retriever, question)
    end_time = time.time()
    
    print("\nRisposta:")
    print(answer)
    print(f"\nTempo di risposta: {end_time - start_time:.2f} secondi")
    
    # Salva nella cache se richiesto
    if cache:
        cache.save_to_cache(question, retrieved_docs, answer)
    
    # Mostra i documenti recuperati se richiesto
    if args.show_docs:
        print("\nDocumenti recuperati:")
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get("source", f"Documento {i+1}")
            page = doc.metadata.get("page", "")
            page_info = f" (Pagina {page})" if page else ""
            print(f"\n--- Documento {i+1}: {source}{page_info} ---")
            print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
    
    # Valuta la risposta se richiesto
    if args.evaluate:
        print("\nValutazione in corso...")
        evaluator = RAGEvaluator()
        
        # Valuta la pertinenza dei documenti
        relevance_eval = evaluator.evaluate_retrieval(question, retrieved_docs)
        print("\nValutazione della pertinenza dei documenti:")
        print(relevance_eval)
        
        # Valuta la qualità della risposta
        answer_eval = evaluator.evaluate_answer(question, retrieved_docs, answer)
        print("\nValutazione della risposta:")
        print(answer_eval)

def interactive_mode(args):
    """Avvia la modalità interattiva."""
    # Verifica che l'indice esista
    if not os.path.exists(args.persist_dir):
        print(f"Errore: L'indice {args.persist_dir} non esiste.")
        return
    
    # Inizializza il gestore della cache se richiesto
    cache = None
    if args.use_cache:
        cache = QueryCache()
    
    # Inizializza il gestore del database vettoriale
    if args.index_type == "chroma":
        vector_store_manager = VectorStoreManager(
            persist_directory=args.persist_dir
        )
        # Carica il database Chroma
        vector_store = vector_store_manager.load_chroma_db()
    else:  # faiss
        vector_store_manager = VectorStoreManager()
        # Carica l'indice FAISS
        vector_store = vector_store_manager.load_faiss_index(args.persist_dir)
    
    # Ottieni il retriever
    base_retriever = vector_store_manager.get_retriever(k=args.k)
    
    # Inizializza il query transformer se richiesto
    query_transformer = None
    if args.use_multi_query:
        query_transformer = QueryTransformer()
        multi_query_retriever = MultiQueryRetriever(base_retriever, query_transformer)
    
    # Inizializza il generatore RAG
    rag_generator = RAGGenerator()
    
    print("\n=== Modalità Interattiva ===")
    print("Digita 'exit' o 'quit' per uscire.")
    print("Digita 'help' per visualizzare i comandi disponibili.")
    
    while True:
        try:
            question = input("\nInserisci la tua domanda: ")
            
            if question.lower() in ["exit", "quit"]:
                break
            
            if question.lower() == "help":
                print("\nComandi disponibili:")
                print("- exit, quit: Esci dalla modalità interattiva")
                print("- help: Visualizza questo messaggio di aiuto")
                print("- clear: Pulisci lo schermo")
                print("- cache stats: Visualizza statistiche sulla cache")
                print("- cache clear: Cancella la cache")
                continue
            
            if question.lower() == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                continue
            
            if question.lower() == "cache stats" and cache:
                stats = cache.get_cache_stats()
                print("\nStatistiche sulla cache:")
                print(f"- Totale voci: {stats['total_entries']}")
                print(f"- Dimensione totale: {stats['total_size_mb']:.2f} MB")
                print(f"- Voce più vecchia: {stats['oldest_entry']}")
                print(f"- Voce più recente: {stats['newest_entry']}")
                continue
            
            if question.lower() == "cache clear" and cache:
                cache.clear_cache()
                continue
            
            if not question.strip():
                continue
            
            print("\nElaborazione della domanda...")
            
            # Controlla se la risposta è nella cache
            cached_docs = None
            cached_answer = None
            if cache:
                cached_docs, cached_answer = cache.get_from_cache(question)
            
            if cached_answer:
                print("\nRisposta (dalla cache):")
                print(cached_answer)
                
                if args.show_docs and cached_docs:
                    print("\nDocumenti recuperati:")
                    for i, doc in enumerate(cached_docs):
                        source = doc.metadata.get("source", f"Documento {i+1}")
                        page = doc.metadata.get("page", "")
                        page_info = f" (Pagina {page})" if page else ""
                        print(f"\n--- Documento {i+1}: {source}{page_info} ---")
                        print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                
                continue
            
            # Utilizza il retriever multi-query se richiesto
            if args.use_multi_query:
                print("Generazione di query multiple...")
                retrieved_docs = multi_query_retriever.get_relevant_documents(question)
            else:
                retrieved_docs = base_retriever.get_relevant_documents(question)
            
            # Genera la risposta
            start_time = time.time()
            answer = rag_generator.answer_question(base_retriever, question)
            end_time = time.time()
            
            print("\nRisposta:")
            print(answer)
            print(f"\nTempo di risposta: {end_time - start_time:.2f} secondi")
            
            # Salva nella cache se richiesto
            if cache:
                cache.save_to_cache(question, retrieved_docs, answer)
            
            # Mostra i documenti recuperati se richiesto
            if args.show_docs:
                print("\nDocumenti recuperati:")
                for i, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get("source", f"Documento {i+1}")
                    page = doc.metadata.get("page", "")
                    page_info = f" (Pagina {page})" if page else ""
                    print(f"\n--- Documento {i+1}: {source}{page_info} ---")
                    print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
        
        except KeyboardInterrupt:
            print("\nOperazione interrotta dall'utente.")
            break
        
        except Exception as e:
            print(f"\nErrore durante l'elaborazione: {e}")

def manage_cache(args):
    """Gestisce la cache."""
    cache = QueryCache()
    
    if args.clear:
        cache.clear_cache()
    
    if args.stats:
        stats = cache.get_cache_stats()
        print("\nStatistiche sulla cache:")
        print(f"- Totale voci: {stats['total_entries']}")
        print(f"- Dimensione totale: {stats['total_size_mb']:.2f} MB")
        print(f"- Voce più vecchia: {stats['oldest_entry']}")
        print(f"- Voce più recente: {stats['newest_entry']}")

def main():
    """Funzione principale."""
    args = setup_argparse()
    
    if args.command == "index":
        index_documents(args)
    elif args.command == "query":
        query_system(args)
    elif args.command == "interactive":
        interactive_mode(args)
    elif args.command == "cache":
        manage_cache(args)
    else:
        print("Comando non riconosciuto. Usa --help per visualizzare i comandi disponibili.")

if __name__ == "__main__":
    main()