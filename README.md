# Sistema RAG per l'Analisi di Documenti PDF

Un sistema avanzato di Retrieval Augmented Generation (RAG) per l'analisi di documenti PDF utilizzando LangChain e le migliori tecniche implementative.

## Caratteristiche

- **Caricamento e Processamento PDF**: Supporto per il caricamento di documenti PDF da una cartella, incluse sottocartelle
- **Chunking Intelligente**: Divisione dei documenti in chunks ottimali con sovrapposizione configurabile
- **Database Vettoriale**: Supporto per FAISS (locale, veloce) e Chroma (persistente)
- **Tecniche Avanzate di Retrieval**:
  - Multi-query retrieval per migliorare la qualità dei risultati
  - Query expansion per arricchire le query dell'utente
  - Ensemble retrieval per combinare diversi approcci
- **Caching**: Sistema di cache per le query per migliorare le prestazioni
- **Valutazione**: Strumenti per valutare la qualità delle risposte e la pertinenza dei documenti
- **Interfaccia CLI**: Interfaccia a riga di comando completa e flessibile

## Installazione

1. Clona il repository o copia i file in una directory locale
2. Crea un ambiente virtuale Python:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Installa le dipendenze:
   ```
   pip install -r requirements.txt
   ```
4. Configura le tue chiavi API nel file `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Utilizzo

### Indicizzazione dei Documenti

Per indicizzare una cartella di documenti PDF:

```
python cli.py index --pdf_dir "percorso/ai/tuoi/pdf" --index_type faiss --persist_dir "./vector_db"
```

Opzioni:
- `--pdf_dir`: Directory contenente i documenti PDF (obbligatorio)
- `--index_type`: Tipo di indice vettoriale (`faiss` o `chroma`, default: `faiss`)
- `--persist_dir`: Directory dove salvare l'indice (default: `./vector_db`)
- `--chunk_size`: Dimensione dei chunk in caratteri (default: 1000)
- `--chunk_overlap`: Sovrapposizione tra i chunk in caratteri (default: 200)

### Interrogazione del Sistema

Per porre una singola domanda al sistema:

```
python cli.py query --question "La tua domanda qui" --index_type faiss --persist_dir "./vector_db"
```

Opzioni:
- `--question`: Domanda da porre al sistema (se omessa, verrà richiesta)
- `--index_type`: Tipo di indice vettoriale (`faiss` o `chroma`, default: `faiss`)
- `--persist_dir`: Directory da cui caricare l'indice (default: `./vector_db`)
- `--k`: Numero di documenti da recuperare (default: 4)
- `--use_cache`: Utilizza la cache per le query
- `--use_multi_query`: Utilizza la tecnica multi-query per migliorare i risultati
- `--show_docs`: Mostra i documenti recuperati
- `--evaluate`: Valuta la qualità della risposta

### Modalità Interattiva

Per avviare la modalità interattiva:

```
python cli.py interactive --index_type faiss --persist_dir "./vector_db"
```

Opzioni:
- `--index_type`: Tipo di indice vettoriale (`faiss` o `chroma`, default: `faiss`)
- `--persist_dir`: Directory da cui caricare l'indice (default: `./vector_db`)
- `--k`: Numero di documenti da recuperare (default: 4)
- `--use_cache`: Utilizza la cache per le query
- `--use_multi_query`: Utilizza la tecnica multi-query per migliorare i risultati
- `--show_docs`: Mostra i documenti recuperati

### Gestione della Cache

Per gestire la cache:

```
python cli.py cache --stats
python cli.py cache --clear
```

Opzioni:
- `--stats`: Mostra statistiche sulla cache
- `--clear`: Cancella la cache

## Struttura del Progetto

- `main.py`: Script principale per l'utilizzo rapido del sistema
- `cli.py`: Interfaccia a riga di comando avanzata
- `pdf_loader.py`: Modulo per il caricamento e l'elaborazione dei documenti PDF
- `vector_store.py`: Modulo per la gestione del database vettoriale
- `rag_generator.py`: Modulo per la generazione di risposte
- `evaluation.py`: Modulo per la valutazione delle prestazioni
- `advanced_retrieval.py`: Modulo per tecniche avanzate di retrieval
- `cache_manager.py`: Modulo per la gestione della cache

## Esempio di Utilizzo

1. Indicizza i tuoi documenti PDF:
   ```
   python cli.py index --pdf_dir "C:\Documenti\Manuali" --index_type faiss
   ```

2. Avvia la modalità interattiva:
   ```
   python cli.py interactive --use_cache --show_docs
   ```

3. Poni domande sui tuoi documenti:
   ```
   Inserisci la tua domanda: Quali sono le procedure di sicurezza descritte nei manuali?
   ```

## Consigli per le Prestazioni

- Utilizza `faiss` per velocità e `chroma` per persistenza
- Abilita la cache per migliorare i tempi di risposta per domande ripetute
- Utilizza la tecnica multi-query per domande complesse
- Sperimenta con diverse dimensioni di chunk e sovrapposizione per ottimizzare i risultati
- Per documenti molto grandi, considera di aumentare il valore di `k` per recuperare più contesto

## Requisiti

- Python 3.8+
- OpenAI API Key
- Dipendenze elencate in `requirements.txt`

## Licenza

MIT