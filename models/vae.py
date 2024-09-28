import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
from diffusers import AutoencoderKL
from ..evaluation.evaluation import compute_performance_metrics

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the dataset
dataset_path = "../data/dataset"
dataset = load_from_disk(dataset_path)

# Load pre-trained VAE model from diffusers
vae_model_name = 'stabilityai/sd-vae-ft-mse'
vae = AutoencoderKL.from_pretrained(vae_model_name)
vae = vae.to(device)
vae.train()  # Set VAE to training mode to allow fine-tuning

# Define a custom model that adds a classifier on top of VAE's image embeddings
class VAEImageClassifier(nn.Module):
    def __init__(self, vae, num_classes):
        super(VAEImageClassifier, self).__init__()
        self.vae = vae
        # Define a pooling layer to reduce latent dimensions
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # Get latent dimensions from VAE config
        latent_channels = self.vae.config.latent_channels
        self.classifier = nn.Linear(latent_channels, num_classes)

    def forward(self, pixel_values):
        # Encode images to latent representations
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215  # Scaling factor as per Stable Diffusion
        # Pool the latents
        pooled = self.pool(latents)
        # Flatten to obtain embeddings
        embeddings = self.flatten(pooled)
        # Pass through classifier
        logits = self.classifier(embeddings)
        return logits

    def get_embeddings(self, pixel_values):
        # Extract embeddings from images
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215
        pooled = self.pool(latents)
        embeddings = self.flatten(pooled)
        return embeddings

# Initialize the custom classifier
model = VAEImageClassifier(vae, 135)
model = model.to(device)

# Define image transformations
# VAE expects images normalized to [-1, 1]
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Prepare the dataset with batching
def transform_examples(examples):
    images = [img.convert('RGB') for img in examples['image']]
    transformed_images = [transform(img) for img in images]
    # Stack into a tensor
    pixel_values = torch.stack(transformed_images)
    examples['pixel_values'] = pixel_values
    return examples

dataset = dataset.map(transform_examples, batched=True)

# Set the format for PyTorch
dataset.set_format(type='torch', columns=['pixel_values', 'label'])

# Split the dataset into training and testing sets
split_ratio = 0.2  # 20% for testing
train_test_split = dataset.train_test_split(test_size=split_ratio, stratify_by_column='label', seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, pin_memory=True)

# Define loss function
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW([
    {'params': model.vae.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
])

# Define a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    if scheduler:
        scheduler.step()

    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Save the model
torch.save(model.state_dict(), '../weights/vae_finetuned.pt')
print("Model saved as 'vae_finetuned.pt'.")

# Extract features for similarity search
def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            inputs = batch['pixel_values'].to(device, non_blocking=True)
            batch_labels = batch['label'].cpu().numpy()
            labels.extend(batch_labels)
            # Get embeddings
            embeddings = model.get_embeddings(inputs)
            features.append(embeddings.cpu())
    features = torch.cat(features)
    labels = np.array(labels)
    return features, labels

train_features, train_labels = extract_features(model, train_loader)
test_features, test_labels = extract_features(model, test_loader)

# Compute performance metrics for top-k retreival
k = 5
mean_precision, mean_recall, retrieval_accuracy = compute_performance_metrics(train_features, train_labels, test_features, test_labels, k)

print(f'Mean Precision@{k}: {mean_precision:.4f}')
print(f'Mean Recall@{k}: {mean_recall:.4f}')
print(f'Retrieval Accuracy@{k}: {retrieval_accuracy:.4f}')

# Calculating metrics: 100%|██████████| 8814/8814 [00:02<00:00, 3404.89it/s]
# Evaluation Metrics:
# Mean Precision@5: 0.4895
# Mean Recall@5: 0.0026
# Retrieval Accuracy@5: 0.7474