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
import requests
import json

def retrieve_context(query: str) -> str:
    """Retrieves the most relevant context from ChromaDB."""
    # Load ChromaDB collection
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection("statistics_knowledge")
        log_info("ChromaDB collection loaded successfully.")
    except Exception as e:
        log_error(f"Error loading ChromaDB collection: {e}")

    # Load LLM (Hugging Face)
    try:
        model_id = "BAAI/bge-large-en"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        log_info("LLM model loaded successfully.")
    except Exception as e:
        log_error(f"Error loading LLM: {e}")

    embedding_model = SentenceTransformer("BAAI/bge-large-en")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    try:
        log_progress("Retrieving context", query)
        query_embedding = embedding_model.encode(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=10)

        all_docs = [
            tokenizer.decode([int(token) for token in doc.split()], skip_special_tokens=True)
            if isinstance(doc, str) and all(token.isdigit() for token in doc.split()) 
            else doc
            for doc in results["documents"][0]
        ]
        log_info(all_docs)
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

def ask_llm(query: str) -> str:
    """Fetches relevant context and generates a response from the LLM."""
    # Login to Hugging Face
    login(os.getenv('HUGGINGFACEHUB_API_TOKEN'))

    try:
        context = retrieve_context(query)
        output_format = """```python
                           #Code Block
                           ```
                        """

        prompt = f"""<persona> You are an assitant helping the user with writing python program for the given question. </persona>
            <question> {query} </question>
            <context> {context} </context>
            <output_format> {output_format} </output_format>
            <instruction>
            - Carefully analyze the context given between these lines <context> and </context> and answer the question between <question> and </question>.
            - Write down the python code only (make sure that this part is directly executable).
            - MAKE SURE THAT THE RESPONE YOU PROVIDE STRICTLY FOLLOWs ONLY THE OUTPUT FORMAT GIVEN BETWEEN <output_format> AND </output_format> LINES IN A VALID JSON STRUCTURE.
            </instruction>"""

        log_info(f"Querying LLM with context: {context[:50]}...")
        response = requests.post(
            url='https://openrouter.ai/api/v1/chat/completions',
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json",
                },
            data=json.dumps({
                    "model": "meta-llama/llama-3.3-70b-instruct:free",
                    "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            })
        )
        result_json = response.json()
        log_info(result_json)
        # print(response["choices"][0]["message"]["content"])
        result = result_json["choices"][0]["message"]["content"]
        log_progress("LLM Response Generated", result)
        return result
    except Exception as e:
        log_error(f"Error querying LLM: {e}")
        return "Error generating response."