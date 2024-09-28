import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import faiss
from PIL import Image
import requests
from io import BytesIO
from datasets import load_from_disk
import os
import logging
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from PIL.Image import Resampling
    RESAMPLE_FILTER = Resampling.LANCZOS
except ImportError:
    # For Pillow < 10, fall back to ANTIALIAS
    RESAMPLE_FILTER = Image.LANCZOS  # Image.ANTIALIAS is an alias for Image.LANCZOS in older versions
    logger.warning("PIL.Image.Resampling not found. Falling back to Image.LANCZOS.")

FAISS_INDEX_PATH = 'image_index.faiss'
IMAGE_IDS_PATH = 'image_ids.npy'
DATASET_PATH = 'dataset-app'
CHERRY_IMAGE_PATH = 'cherry.png'

# Function to load FAISS index and image IDs
def load_faiss_index(index_path, ids_path):
    try:
        logger.info(f"Loading FAISS index from: {index_path}")
        index = faiss.read_index(index_path)
        logger.info("FAISS index loaded successfully.")
        logger.info(f"Loading image IDs from: {ids_path}")
        image_ids = np.load(ids_path)
        logger.info("Image IDs loaded successfully.")
        return index, image_ids
    except Exception as e:
        logger.error(f"Error loading FAISS index or image IDs: {e}")
        raise

# Function to load the dataset
def load_dataset_from_disk(dataset_path):
    try:
        logger.info(f"Loading dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        logger.info("Dataset loaded successfully.")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

# Function to load the cherry image
def load_cherry_image(image_path, max_width=150):
    try:
        logger.info(f"Loading Cherry image from: {image_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Cherry image not found at path: {image_path}")
        cherry_image = Image.open(image_path).convert('RGB')
        # Resize the image to have a maximum width while maintaining aspect ratio
        width_percent = (max_width / float(cherry_image.size[0]))
        new_height = int((float(cherry_image.size[1]) * float(width_percent)))
        cherry_image = cherry_image.resize((max_width, new_height), RESAMPLE_FILTER)

        logger.info("Cherry image loaded and resized successfully.")
        return cherry_image
    except Exception as e:
        logger.error(f"Error loading cherry image: {e}")
        raise

# Load resources
index, image_ids = load_faiss_index(FAISS_INDEX_PATH, IMAGE_IDS_PATH)
dataset = load_dataset_from_disk(DATASET_PATH)
cherry_image = load_cherry_image(CHERRY_IMAGE_PATH, max_width=150)

model_name = 'openai/clip-vit-base-patch32'
processor = CLIPProcessor.from_pretrained(model_name)

# Define a custom model that adds a classifier on top of CLIP's image embeddings
class CLIPImageClassifier(nn.Module):
    def __init__(self, model_name, num_classes=135):
        super(CLIPImageClassifier, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        # Define a classifier on top of image embeddings
        self.classifier = nn.Linear(self.clip_model.config.projection_dim, num_classes)

    def forward(self, pixel_values, return_embeddings=False):
        # Get image embeddings
        image_embeddings = self.clip_model.get_image_features(pixel_values=pixel_values)
        logits = self.classifier(image_embeddings)
        if return_embeddings:
            return logits, image_embeddings
        else:
            return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CLIPImageClassifier(model_name, num_classes=135)
PATH = 'clip_finetuned.pth'
model.load_state_dict(torch.load(PATH, weights_only=True, map_location=device))

model.to(device)

def get_similar_images(uploaded_image, image_url):
    try:
        logger.info("Received request to find similar images.")
        # Determine the input source
        if uploaded_image is not None:
            input_image = uploaded_image
            logger.info("Using uploaded image for similarity search.")
        elif image_url.strip() != '':
            logger.info(f"Fetching image from URL: {image_url}")
            response = requests.get(image_url)
            response.raise_for_status()
            input_image = Image.open(BytesIO(response.content)).convert('RGB')
            logger.info("Image fetched from URL successfully.")
        else:
            logger.warning("No image provided by the user.")
            return ["Please provide an image or image URL."]
        
        # Preprocess the input image
        inputs = processor(images=input_image, return_tensors="pt").to(device)

        # Compute image embeddings
        with torch.no_grad():
            image_features = model.clip_model.get_image_features(**inputs)

        # Normalize embeddings
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy().astype('float32')

        # Search for similar images
        logger.info("Performing FAISS similarity search.")
        D, I = index.search(image_features, k=5)
        logger.info(f"FAISS search completed. Top indices: {I[0]}")

        # Retrieve similar images from the dataset
        similar_images = []
        for idx in I[0]:
            try:
                image = dataset[int(idx)]['image']
                if isinstance(image, Image.Image):
                    similar_images.append(image)
                else:
                    # If image is not a PIL Image, convert it
                    similar_images.append(Image.fromarray(image))
            except Exception as e:
                logger.error(f"Error retrieving image at index {idx}: {e}")
                continue

        logger.info("Similar images retrieved successfully.")
        return similar_images

    except Exception as e:
        logger.error(f"Error in get_similar_images: {e}")
        return [f"An error occurred: {e}"]


# Create Gradio interface
with gr.Blocks(css="""
    .container { max-width: 850px; margin: auto; padding: 20px; }
    .header { display: flex; align-items: center; margin-bottom: 20px; }
    .logo { width: 50px; margin-right: 20px; }
    h1 { margin: 0; }
    .input-section { margin-bottom: 20px; }
    .output-section { margin-top: 20px; }
    .footer { margin-top: 30px; text-align: center; font-size: 0.8em; color: #666; }
""") as demo:
    with gr.Column(elem_classes="container"):
        # Header
        with gr.Row(elem_classes="header"):
            gr.Image(value=cherry_image, elem_classes="logo", show_label=False)
            gr.Markdown("# Cherry Apparel Image Similarity Search")
        
        gr.Markdown("Upload an apparel image or provide an image URL to find similar apparel items.")
        
        # Input Section
        with gr.Row(elem_classes="input-section"):
            with gr.Column(scale=1):
                uploaded_image = gr.Image(type='pil', label="Upload Image", elem_id="image-upload")
            with gr.Column(scale=1):
                image_url = gr.Textbox(label="Or Enter Image URL", placeholder="https://example.com/image.jpg")
        
        search_button = gr.Button("Search for Similar Apparel", variant="primary")
        
        # Output Section
        with gr.Column(elem_classes="output-section"):
            gallery = gr.Gallery(label="Top 5 Similar Apparel Items", columns=5, height="auto", elem_id="results-gallery")
        
        # Footer
        gr.Markdown("### How it works", elem_classes="footer")
        gr.Markdown("""
        1. Upload an image or provide an image URL of an apparel item.
        2. Click the 'Search' button to find similar items.
        3. View the top 5 most similar apparel items in our database.
        """, elem_classes="footer")
    
    # Define the interaction
    search_button.click(
        fn=get_similar_images,
        inputs=[uploaded_image, image_url],
        outputs=gallery
    )

if __name__ == "__main__":
    demo.launch(share=True)
