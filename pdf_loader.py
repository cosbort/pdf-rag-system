"""
Modulo per il caricamento e l'elaborazione dei documenti PDF.
"""
import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:
    """
    Classe per caricare e processare documenti PDF da una cartella.
    """
    
    def __init__(self, directory_path: str):
        """
        Inizializza il processore PDF con il percorso della directory.
        
        Args:
            directory_path: Percorso della directory contenente i file PDF
        """
        self.directory_path = directory_path
        self.documents = []
        self.chunks = []
        
    def load_documents(self) -> List[Document]:
        """
        Carica tutti i documenti PDF dalla directory specificata.
        
        Returns:
            Lista di documenti caricati
        """
        try:
            # Utilizziamo DirectoryLoader per caricare tutti i PDF nella directory
            loader = DirectoryLoader(
                self.directory_path,
                glob="**/*.pdf",  # Carica tutti i file PDF, anche nelle sottocartelle
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            
            self.documents = loader.load()
            print(f"Caricati {len(self.documents)} documenti dalla directory {self.directory_path}")
            return self.documents
            
        except Exception as e:
            print(f"Errore durante il caricamento dei documenti: {e}")
            return []
    
    def split_documents(self, 
                       chunk_size: int = 1000, 
                       chunk_overlap: int = 200,
                       add_start_index: bool = True) -> List[Document]:
        """
        Divide i documenti in chunks più piccoli per l'elaborazione.
        
        Args:
            chunk_size: Dimensione di ogni chunk in caratteri
            chunk_overlap: Sovrapposizione tra chunks in caratteri
            add_start_index: Se aggiungere l'indice di inizio nel documento originale
            
        Returns:
            Lista di chunks di documento
        """
        if not self.documents:
            print("Nessun documento da dividere. Carica prima i documenti.")
            return []
        
        # Utilizziamo RecursiveCharacterTextSplitter per dividere i documenti
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"Documenti divisi in {len(self.chunks)} chunks")
        
        return self.chunks
    
    def get_document_metadata(self) -> List[dict]:
        """
        Restituisce i metadati di tutti i documenti caricati.
        
        Returns:
            Lista di metadati dei documenti
        """
        if not self.documents:
            print("Nessun documento caricato.")
            return []
        
        metadata_list = []
        for doc in self.documents:
            metadata_list.append(doc.metadata)
            
        return metadata_list