import os
import re
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from src.utils.logger import log_info, log_error, log_progress

# Initialize ChromaDB client
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_client.delete_collection("statistics_knowledge")
    collection = chroma_client.get_or_create_collection(
        "statistics_knowledge",
        embedding_function=SentenceTransformerEmbeddingFunction("BAAI/bge-large-en")
    )
    log_info("ChromaDB initialized successfully.")
except Exception as e:
    log_error(f"Error initializing ChromaDB: {e}")

def extract_text_from_tex(file_path):
    """Extracts text from LaTeX (.tex) files, removing LaTeX commands."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove LaTeX commands using regex
        content = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", content)
        content = re.sub(r"\\begin\{.*?\}|\\end\{.*?\}", "", content) 
        content = re.sub(r"%.*", "", content)
        log_info(f"Extracted text from: {file_path}")
        return content.strip()
    except Exception as e:
        log_error(f"Error processing LaTeX file {file_path}: {e}")
        return ""

def process_data_files(data_folder):
    """Processes .tex and .csv files and loads them into ChromaDB."""
    log_progress("Starting data processing", f"Scanning folder: {data_folder}")

    for root, _, files in os.walk(data_folder):
        for file in files:
            file_path = os.path.join(root, file)

            try:
                if file.endswith(".tex"):
                    log_info(f"Processing LaTeX file: {file}")
                    text = extract_text_from_tex(file_path)
                    collection.add(ids=[file], documents=[text])
                    log_progress("LaTeX file added to ChromaDB", file)

                elif file.endswith(".csv"):
                    log_info(f"Processing CSV file: {file}")
                    df = pd.read_csv(file_path)
                    for idx, row in df.iterrows():
                        text = " ".join(map(str, row.values))
                        collection.add(ids=[f"{file}_{idx}"], documents=[text])
                    log_progress("CSV file added to ChromaDB", file)

            except Exception as e:
                log_error(f"Error processing file {file_path}: {e}")

    log_progress("Data processing complete", "All files added to ChromaDB")

if __name__ == "__main__":
    process_data_files("src/data/raw/data")
    process_data_files("src/data/raw/tex")