import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
from transformers import AutoImageProcessor
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import random
from ..evaluation.evaluation import compute_performance_metrics

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the dataset
dataset_path = "../data/dataset"
dataset = load_from_disk(dataset_path)

# Define the image processor (from transformers)
model_name = 'microsoft/resnet-50'
image_processor = AutoImageProcessor.from_pretrained(model_name)

# Define transformations
def transform_examples(example):
    # Ensure all images have 3 channels (RGB)
    example['image'] = example['image'].convert('RGB')
    # Apply image_processor
    inputs = image_processor(example['image'], return_tensors='pt')
    example['pixel_values'] = inputs['pixel_values'].squeeze(0)
    return example

dataset = dataset.map(transform_examples, batched=False)

# Set the format for PyTorch
dataset.set_format(type='torch', columns=['pixel_values', 'label'])

# Split the dataset
split_ratio = 0.2  # 20% for testing
train_test_split = dataset.train_test_split(test_size=split_ratio, stratify_by_column='label', seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Define a custom Siamese Dataset
class SiameseDataset(Dataset):
    def __init__(self, dataset, transform=None, positive_ratio=0.5):
        """
        Args:
            dataset (Dataset): HuggingFace dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            positive_ratio (float): Ratio of positive pairs in the dataset.
        """
        self.dataset = dataset
        self.transform = transform
        self.positive_ratio = positive_ratio
        self.labels = self.dataset['label']
        self.label_to_indices = self._create_label_to_indices()

        # Prepare list of indices for easier access
        self.indices = list(range(len(self.dataset)))

        #Get a list of unique labels from the dataset
        self.unique_labels = sorted(set(self.labels.tolist())) # Convert tensor to list

    def _create_label_to_indices(self):
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label.item() in label_to_indices: # Use label.item() to get the value
                label_to_indices[label.item()].append(idx)
            else:
                label_to_indices[label.item()] = [idx]
        return label_to_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the first image and its label
        img1 = self.dataset[idx]['pixel_values']
        label1 = self.labels[idx].item() # Use label.item() to get the value

        # Decide whether to create a positive pair or a negative pair
        if random.random() < self.positive_ratio:
            # Positive pair: same class
            label = 1
            # Ensure that there are at least two samples in this class
            while True:
                idx2 = random.choice(self.label_to_indices[label1])
                if idx2 != idx:
                    break
            img2 = self.dataset[idx2]['pixel_values']
        else:
            # Negative pair: different class
            label = 0
            # Choose a different label
            label2 = label1
            while label2 == label1:
                #Use unique labels from the SiameseDataset
                label2 = random.choice(self.unique_labels) # Use unique_labels list
            idx2 = random.choice(self.label_to_indices[label2])
            img2 = self.dataset[idx2]['pixel_values']

        return (img1, img2), torch.tensor(label, dtype=torch.float32)

# Create Siamese datasets
train_siamese_dataset = SiameseDataset(train_dataset, positive_ratio=0.5)
test_siamese_dataset = SiameseDataset(test_dataset, positive_ratio=0.0)  # Not used for training

# Create DataLoaders
batch_size = 128
train_loader = DataLoader(train_siamese_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
# Test loader is used for feature extraction, same as original
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
train_feature_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Define the Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, model_name='microsoft/resnet-50', embedding_dim=512):
        super(SiameseNetwork, self).__init__()
        # Load a pre-trained ResNet model
        self.backbone = models.resnet50(pretrained=True)
        # Replace the final fully connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, embedding_dim)
        # Optionally, add a normalization layer
        self.embedding_dim = embedding_dim

    def forward(self, x):
        embedding = self.backbone(x)
        # Normalize the embeddings
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

# Initialize the Siamese Network
embedding_dim = 512
siamese_net = SiameseNetwork(model_name=model_name, embedding_dim=embedding_dim)
siamese_net = siamese_net.to(device)

# Define the Contrastive Loss
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)
        # Contrastive loss
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss

# Initialize the loss function and optimizer
criterion = ContrastiveLoss(margin=1.0)
optimizer = torch.optim.AdamW(siamese_net.parameters(), lr=1e-4)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    siamese_net.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        (img1, img2), labels = batch
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        output1 = siamese_net(img1)
        output2 = siamese_net(img2)

        # Compute loss
        loss = criterion(output1, output2, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img1.size(0)

    epoch_loss = running_loss / len(train_siamese_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Save the model
torch.save(siamese_net.state_dict(), '../weights/siamese_network_finetuned.pth')
print("Model saved as 'siamese_network_finetuned.pth'.")

# Function to extract features from a DataLoader
def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            inputs = batch['pixel_values'].to(device)
            embeddings = model(inputs)
            features.append(embeddings.cpu())
            labels.extend(batch['label'].cpu().numpy())
    features = torch.cat(features)
    labels = np.array(labels)
    return features, labels

# Extract features for training and testing sets
train_features, train_labels = extract_features(siamese_net, train_feature_loader)
test_features, test_labels = extract_features(siamese_net, test_loader)

# Compute performance metrics for top-k retreival
k = 5
mean_precision, mean_recall, retrieval_accuracy = compute_performance_metrics(train_features, train_labels, test_features, test_labels, k)

print(f'Mean Precision@{k}: {mean_precision:.4f}')
print(f'Mean Recall@{k}: {mean_recall:.4f}')
print(f'Retrieval Accuracy@{k}: {retrieval_accuracy:.4f}')

# Calculating metrics: 100%|██████████| 8814/8814 [00:13<00:00, 633.95it/s]
# Mean Precision@5: 0.6074
# Mean Recall@5: 0.0051
# Retrieval Accuracy@5: 0.8592