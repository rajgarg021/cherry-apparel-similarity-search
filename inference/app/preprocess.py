import os
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
import faiss
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load the finetuned CLIP model
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

    # Load the dataset
    dataset = load_dataset('ashraq/fashion-product-images-small', split='train')

    # Prepare lists to store embeddings and image IDs
    embeddings = []
    image_ids = []

    # Process images and compute embeddings
    for idx, item in enumerate(dataset):
        image = item['image']

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Compute image embeddings
        with torch.no_grad():
            image_features = model.clip_model.get_image_features(**inputs)

        # Normalize embeddings
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Convert to NumPy array and store
        embeddings.append(image_features.cpu().numpy())
        image_ids.append(idx)

        if idx % 100 == 0:
            logger.info(f'Processed {idx} images')

    # Convert list of embeddings to a NumPy array
    embeddings = np.vstack(embeddings).astype('float32')
    logger.info("All embeddings computed.")

    # Build a FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Using Inner Product (cosine similarity)
    index.add(embeddings)
    logger.info(f"FAISS index built with {index.ntotal} vectors.")

    # Save the FAISS index and image IDs
    faiss.write_index(index, 'image_index.faiss')
    np.save('image_ids.npy', np.array(image_ids))
    logger.info("FAISS index and image IDs saved successfully.")

    # Save the dataset for later retrieval
    dataset.save_to_disk('dataset-app')

    logger.info("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()