import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
from ..evaluation.evaluation import compute_performance_metrics

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the dataset
dataset_path = "../data/dataset"
dataset = load_from_disk(dataset_path)

# Load pre-trained CLIP model and processor
model_name = 'openai/clip-vit-base-patch32'
processor = CLIPProcessor.from_pretrained(model_name)

# Define a custom model that adds a classifier on top of CLIP's image embeddings
class CLIPImageClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
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

model = CLIPImageClassifier(model_name, 135)
model = model.to(device)

# Prepare the dataset with batching
def transform_examples(examples):
    images = [img.convert('RGB') for img in examples['image']]
    # Apply processor
    inputs = processor(images=images, return_tensors='pt', padding=True)
    examples['pixel_values'] = inputs['pixel_values']
    return examples

dataset = dataset.map(transform_examples, batched=True, batch_size=32)

# Set the format for PyTorch
dataset.set_format(type='torch', columns=['pixel_values', 'label'])

# Split the dataset
split_ratio = 0.2  # 20% for testing
train_test_split = dataset.train_test_split(test_size=split_ratio, stratify_by_column='label', seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, num_workers=4, pin_memory=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

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

    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Save the model
torch.save(model.state_dict(), '../weights/clip_finetuned.pth')

# Extract features for similarity search
def extract_features(dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            inputs = batch['pixel_values'].to(device)
            batch_labels = batch['label'].cpu().numpy()
            labels.extend(batch_labels)
            # Get logits and embeddings
            logits, image_embeddings = model(inputs, return_embeddings=True)
            features.append(image_embeddings.cpu())
    features = torch.cat(features)
    labels = np.array(labels)
    return features, labels

train_features, train_labels = extract_features(train_loader)
test_features, test_labels = extract_features(test_loader)

# Compute performance metrics for top-k retreival
k = 5
mean_precision, mean_recall, retrieval_accuracy = compute_performance_metrics(train_features, train_labels, test_features, test_labels, k)

print(f'Mean Precision@{k}: {mean_precision:.4f}')
print(f'Mean Recall@{k}: {mean_recall:.4f}')
print(f'Retrieval Accuracy@{k}: {retrieval_accuracy:.4f}')

# Calculating metrics: 100%|██████████| 8814/8814 [00:13<00:00, 638.35it/s]
# Mean Precision@5: 0.8858
# Mean Recall@5: 0.0113
# Retrieval Accuracy@5: 0.9344

