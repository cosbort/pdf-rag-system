"""
Esempio di utilizzo del sistema RAG per l'analisi di documenti PDF.
"""
import os
from dotenv import load_dotenv
from pdf_loader import PDFProcessor
from vector_store import VectorStoreManager
from rag_generator import RAGGenerator
from advanced_retrieval import QueryTransformer, MultiQueryRetriever

# Carica le variabili d'ambiente
load_dotenv()

def main():
    """Funzione principale dell'esempio."""
    # Definisci il percorso della directory contenente i PDF
    pdf_dir = "./docs"
    
    # Verifica che la directory esista
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"Creata la directory {pdf_dir}. Inserisci i tuoi PDF qui prima di eseguire questo script.")
        return
    
    # Verifica che ci siano PDF nella directory
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"Nessun file PDF trovato in {pdf_dir}. Inserisci alcuni PDF prima di eseguire questo script.")
        return
    
    print(f"Trovati {len(pdf_files)} file PDF: {', '.join(pdf_files)}")
    
    # Crea la directory per l'indice vettoriale
    vector_db_dir = "./vector_db"
    os.makedirs(vector_db_dir, exist_ok=True)
    
    # Passo 1: Carica e processa i documenti PDF
    print("\n1. Caricamento e processamento dei documenti PDF...")
    pdf_processor = PDFProcessor(pdf_dir)
    documents = pdf_processor.load_documents()
    
    # Passo 2: Dividi i documenti in chunks
    print("\n2. Divisione dei documenti in chunks...")
    chunks = pdf_processor.split_documents(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Passo 3: Crea l'indice vettoriale
    print("\n3. Creazione dell'indice vettoriale FAISS...")
    vector_store_manager = VectorStoreManager()
    vector_store = vector_store_manager.create_faiss_index(
        chunks,
        save_path=vector_db_dir
    )
    
    # Passo 4: Ottieni il retriever base
    print("\n4. Configurazione del retriever...")
    base_retriever = vector_store_manager.get_retriever(k=4)
    
    # Passo 5: Configura il retriever avanzato con multi-query
    print("\n5. Configurazione del retriever avanzato con multi-query...")
    query_transformer = QueryTransformer()
    multi_query_retriever = MultiQueryRetriever(base_retriever, query_transformer)
    
    # Passo 6: Inizializza il generatore RAG
    print("\n6. Inizializzazione del generatore RAG...")
    rag_generator = RAGGenerator()
    
    # Passo 7: Esempio di domanda e risposta
    print("\n7. Esempio di domanda e risposta:")
    
    # Domanda di esempio
    question = "Quali sono i concetti principali discussi nei documenti?"
    print(f"\nDomanda: {question}")
    
    # Genera query multiple
    print("\nGenerazione di query multiple...")
    queries = query_transformer.generate_multi_queries(question)
    print(f"Query generate: {queries}")
    
    # Recupera documenti pertinenti
    print("\nRecupero dei documenti pertinenti...")
    retrieved_docs = multi_query_retriever.get_relevant_documents(question)
    print(f"Recuperati {len(retrieved_docs)} documenti pertinenti")
    
    # Genera la risposta
    print("\nGenerazione della risposta...")
    answer = rag_generator.answer_question(base_retriever, question)
    
    print("\nRisposta:")
    print(answer)
    
    # Mostra i documenti recuperati
    print("\nDocumenti recuperati:")
    for i, doc in enumerate(retrieved_docs[:2]):  # Mostra solo i primi 2 per brevità
        source = doc.metadata.get("source", f"Documento {i+1}")
        page = doc.metadata.get("page", "")
        page_info = f" (Pagina {page})" if page else ""
        print(f"\n--- Documento {i+1}: {source}{page_info} ---")
        print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
    
    print("\nEsempio completato con successo!")

if __name__ == "__main__":
    main()