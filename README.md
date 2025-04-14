# PDF RAG System

Sistema avanzato di Retrieval Augmented Generation per l'analisi di documenti PDF.

## Descrizione

Questo sistema utilizza tecniche di RAG (Retrieval Augmented Generation) per analizzare documenti PDF e rispondere a domande basate sul loro contenuto. Combina modelli di linguaggio di grandi dimensioni con tecniche di recupero di informazioni per fornire risposte precise e contestuali.

## Caratteristiche

- **Caricamento e analisi di documenti PDF**
- **Indicizzazione vettoriale con FAISS**
- **Interfaccia grafica con Streamlit**
- **Risposte contestuali basate sui documenti**
- **Cache per migliorare i tempi di risposta**
- **Tecniche avanzate di retrieval (multi-query)**
- **Valutazione delle prestazioni**

## Installazione

1. Clona il repository:
   ```
   git clone https://github.com/cosbort/pdf-rag-system.git
   cd pdf-rag-system
   ```

2. Installa le dipendenze:
   ```
   # Usando pip
   pip install -r requirements.txt
   
   # Oppure usando Poetry (consigliato)
   poetry install
   ```

3. Configura la tua chiave API OpenAI:
   ```
   # Crea un file .env nella directory principale
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

4. Crea le cartelle necessarie:
   ```
   # Crea le cartelle per i documenti e il database vettoriale
   mkdir documents
   mkdir vector_db
   ```

## Utilizzo

### Interfaccia Grafica Streamlit

Per avviare l'interfaccia grafica Streamlit:

```
# Usando Python direttamente
python run_streamlit_app.py
```

oppure:

```
# Usando Python direttamente
streamlit run streamlit_app/app.py

# Usando Poetry (consigliato)
poetry run streamlit run streamlit_app/app.py
```

L'interfaccia Streamlit ti permetterà di:
- Caricare documenti PDF tramite l'interfaccia web
- Indicizzare i documenti con parametri personalizzabili
- Fare domande sui tuoi documenti
- Visualizzare le risposte con i documenti di riferimento
- Esportare la cronologia delle chat

### Interfaccia a Riga di Comando

#### Indicizzazione dei documenti

Per indicizzare i tuoi documenti PDF:

```
# Usando Python direttamente
python cli.py index --pdf_dir "C:\Documenti\Manuali" --index_type faiss

# Usando Poetry (consigliato)
poetry run python cli.py index --pdf_dir "C:\Documenti\Manuali" --index_type faiss
```

Opzioni:
- `--pdf_dir`: Directory contenente i documenti PDF
- `--index_type`: Tipo di indice vettoriale (`faiss` o `chroma`, default: `faiss`)
- `--persist_dir`: Directory dove salvare l'indice (default: `./vector_db`)
- `--chunk_size`: Dimensione dei chunk in caratteri (default: 1000)
- `--chunk_overlap`: Sovrapposizione dei chunk in caratteri (default: 200)

#### Modalità Interattiva

Per avviare la modalità interattiva:

```
# Usando Python direttamente
python cli.py interactive --index_type faiss --persist_dir "./vector_db"

# Usando Poetry (consigliato)
poetry run python cli.py interactive --index_type faiss --persist_dir "./vector_db"
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
# Usando Python direttamente
python cli.py cache --stats
python cli.py cache --clear

# Usando Poetry (consigliato)
poetry run python cli.py cache --stats
poetry run python cli.py cache --clear
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
- `streamlit_app/`: Directory contenente l'interfaccia grafica Streamlit
  - `app.py`: File principale dell'applicazione Streamlit
  - `sidebar.py`: Componente per la gestione dei documenti e delle impostazioni
  - `chat_interface.py`: Componente per l'interazione con l'utente
  - `utils.py`: Funzioni di utilità per l'applicazione Streamlit

## Consigli per le Prestazioni

- Utilizza `faiss` per velocità e `chroma` per persistenza
- Abilita la cache per migliorare i tempi di risposta per domande ripetute
- Utilizza la tecnica multi-query per domande complesse
- Sperimenta con diverse dimensioni di chunk e sovrapposizione per ottimizzare i risultati
- Per documenti molto grandi, considera di aumentare il valore di `k` per recuperare più contesto

## Requisiti

- Python 3.8+
- OpenAI API Key
- Dipendenze gestite tramite Poetry o elencate in `requirements.txt`

## Licenza

MIT