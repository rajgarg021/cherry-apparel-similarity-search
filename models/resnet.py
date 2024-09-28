import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification
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

# Load pre-trained model and image processor
model_name = 'microsoft/resnet-50'
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = ResNetForImageClassification.from_pretrained(model_name)

# Modify the model to match the number of classes
model.num_labels = 135

# Access the last Linear layer in the classifier
if isinstance(model.classifier, nn.Sequential):
    in_features = model.classifier[-1].in_features
    # Replace the last layer with a new Linear layer
    model.classifier[-1] = nn.Linear(in_features, 135)
elif isinstance(model.classifier, nn.Linear):
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 135)
else:
    raise ValueError("Unexpected classifier type")

model = model.to(device)

# Prepare the dataset with batching
def transform_examples(examples):
    images = examples['image']
    # Apply image_processor
    inputs = image_processor([i.convert('RGB') for i in images], return_tensors='pt') # Ensure all images have 3 channels (RGB)
    examples['pixel_values'] = [pv for pv in inputs['pixel_values']]
    return examples

dataset = dataset.map(transform_examples, batched=True)

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

        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Save the model
torch.save(model.state_dict(), '../weights/resnet_finetuned.pth')

# Extract features for similarity search
def extract_features(dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            inputs = batch['pixel_values'].to(device)
            outputs = model.resnet(inputs)
            # Get the features before the classifier
            feature = outputs.pooler_output  # Shape: (batch_size, hidden_size)
            features.append(feature.cpu())
            labels.extend(batch['label'].cpu().numpy())
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

# Calculating metrics: 100%|██████████| 8814/8814 [00:13<00:00, 635.27it/s]
# Mean Precision@5: 0.8508
# Mean Recall@5: 0.0104
# Retrieval Accuracy@5: 0.9268
