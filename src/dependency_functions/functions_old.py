import chromadb
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.utils.logger import log_info, log_error, log_progress
from huggingface_hub import login
import os
from llama_cpp import Llama
from accelerate import infer_auto_device_map
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Login to Hugging Face
login(os.environ['HUGGINGFACEHUB_API_TOKEN'])

# Load ChromaDB collection
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection("statistics_knowledge")
    log_info("ChromaDB collection loaded successfully.")
except Exception as e:
    log_error(f"Error loading ChromaDB collection: {e}")

# Load LLM (Hugging Face)
try:
    model_id = "Qwen/Qwen2-0.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.half() 

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    log_info("LLM model loaded successfully.")
except Exception as e:
    log_error(f"Error loading LLM: {e}")

def retrieve_context(query):
    """Retrieves the most relevant context from ChromaDB."""
    embedding_model = SentenceTransformer("BAAI/bge-large-en")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    try:
        log_progress("Retrieving context", query)
        query_embedding = embedding_model.encode(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=10)

        all_docs = [doc for doc in results["documents"][0]]
        tokenized_docs = [doc.split() for doc in all_docs]

        # BM25 Re-Ranking
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(query.split())
        sorted_docs = [doc for _, doc in sorted(zip(bm25_scores, all_docs), reverse=True)]

        # LLM Re-Ranking
        query_doc_pairs = [(query, doc) for doc in sorted_docs]
        scores = reranker.predict(query_doc_pairs)
        ranked_docs = [doc for _, doc in sorted(zip(scores, sorted_docs), reverse=True)]

        context = " ".join(ranked_docs[:3]) 
        log_info(f"Retrieved context for query: {query}")
        return context
    except Exception as e:
        log_error(f"Error retrieving context: {e}")
        return "Error fetching context."

def ask_llm(query):
    """Fetches relevant context and generates a response from the LLM."""
    try:
        context = retrieve_context(query)
        prompt = f"Using this context: {context}\nAnswer the question: {query}"

        log_info(f"Querying LLM with context: {context[:50]}...") 
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        log_progress("LLM Response Generated", response)

        return response
    except Exception as e:
        log_error(f"Error querying LLM: {e}")
        return "Error generating response."

if __name__ == "__main__":
    output_format = """#Python Program
                       ##Python code which can be directly executed.
                       #-------------------------------------------
                    """
    query = f"For x(independent variable) = [1, 2, 3] and y(dependent variable) = [2, 4, 6] Give me the formula to calculate linear regression using python?"#  Strictly follow the following format: {output_format}."
    print(ask_llm(query))