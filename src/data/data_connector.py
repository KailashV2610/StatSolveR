import os
import chromadb
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from PIL import Image
from torchvision import transforms
from src.utils.logger import log_info, log_error, log_progress

# Load Embedding Models
log_info("Loading Models...")
text_model = SentenceTransformer('all-MiniLM-L6-v2')
vision_model = SentenceTransformer("clip-vit-base-patch32")

# Connect to ChromaDB
log_info("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="stats_knowledge")

# Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def convert_eps_to_png(eps_file):
    """Convert EPS images to PNG format."""
    png_file = eps_file.replace(".eps", ".png")
    try:
        img = Image.open(eps_file)
        img.save(png_file, format="PNG")
        log_info(f"Converted {eps_file} to {png_file}")
        return png_file
    except Exception as e:
        log_error(f"Error converting {eps_file}: {e}")
        return None

def get_image_embedding(image_path):
    """Extract image embedding using CLIP Vision Transformer."""
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            embedding = vision_model.encode(image).tolist()
        return embedding
    except Exception as e:
        log_error(f"Error extracting embedding for {image_path}: {e}")
        return None

def store_data():
    """Reads & stores text/images in ChromaDB."""
    base_path = "src/data/raw/"

    try:
        log_progress("Starting data processing...")

        # Store CSV data
        csv_folder = os.path.join(base_path, "data")
        for csv_file in os.listdir(csv_folder):
            if csv_file.endswith(".csv"):
                log_progress(f"Processing CSV: {csv_file}")
                df = pd.read_csv(os.path.join(csv_folder, csv_file))
                for i, row in df.iterrows():
                    text = " ".join(map(str, row.to_dict().values()))
                    embedding = text_model.encode(text).tolist()
                    collection.add(ids=[f"{csv_file}_{i}"], embeddings=[embedding], metadatas=[{"text": text}])

        log_progress("CSV Data Processed Successfully!")

        # Store images with references
        image_folder = os.path.join(base_path, "image")
        reference_file = os.path.join(base_path, "image_references.csv")  # Store LaTeX references here

        if os.path.exists(reference_file):
            ref_df = pd.read_csv(reference_file)  # Expected columns: ['filename', 'caption', 'label']
        else:
            ref_df = pd.DataFrame(columns=["filename", "caption", "label"])

        for file in os.listdir(image_folder):
            file_path = os.path.join(image_folder, file)
            
            if file.endswith(".eps"):
                log_progress(f"Converting EPS: {file}")
                file_path = convert_eps_to_png(file_path)
                if not file_path:
                    continue  # Skip if conversion failed

            if file.endswith(".png"):
                log_progress(f"Processing Image: {file}")
                embedding = get_image_embedding(file_path)
                
                # Find associated caption/label
                metadata = ref_df[ref_df["filename"] == file]
                caption = metadata["caption"].values[0] if not metadata.empty else "No caption available"
                label = metadata["label"].values[0] if not metadata.empty else ""

                if embedding:
                    collection.add(
                        ids=[file],
                        embeddings=[embedding],
                        metadatas=[{"image": file, "caption": caption, "label": label}]
                    )

        log_progress("Images Processed Successfully!")

    except Exception as e:
        log_error(f"Error while processing data: {e}")

if __name__ == "__main__":
    store_data()
    log_info("Data stored successfully.")