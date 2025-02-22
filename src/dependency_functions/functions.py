import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from src.utils.logger import log_info, log_error

# Load Models
log_info("Loading Sentence Transformer and CLIP")
text_model = SentenceTransformer('all-MiniLM-L6-v2')
vision_model = SentenceTransformer("clip-vit-base-patch32")
llm = Llama(model_path="Llama-Model.bin")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="stats_knowledge")

def retrieve_context(query, top_k=5):
    """Retrieve relevant text & images for the query."""
    try:
        log_info(f"Retrieving context for: {query}")
        query_embedding = text_model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        
        context = [res["text"] for res in results["metadatas"][0] if "text" in res]
        images = [
            {"file": res["image"], "caption": res.get("caption", ""), "label": res.get("label", "")} 
            for res in results["metadatas"][0] if "image" in res
        ]

        log_info(f"Context Retrieved: {context[:1]}... Images: {images}")
        return context, images
    except Exception as e:
        log_error(f"Error retrieving context: {e}")
        return [], []

def ask_llm(query):
    """Fetch relevant context & use Llama for response."""
    try:
        context, images = retrieve_context(query)
        context_text = " ".join(context)
        image_text = "\n".join([f"[Image: {img['file']} | Caption: {img['caption']}]" for img in images])
        
        full_prompt = f"Context: {context_text}\n{image_text}\nAnswer this: {query}"
        log_info(f"Sending prompt to Llama model")
        response = llm(full_prompt)
        log_info(f"Response received from Llama")
        return response, images
    except Exception as e:
        log_error(f"Error in LLM call: {e}")
        return "An error occurred.", []

if __name__ == "__main__":
    question = "What is the probability distribution of trousers?"
    answer, related_images = ask_llm(question)
    log_info(f"Question: {question} | Answer: {answer[:50]}... | Images: {related_images}")