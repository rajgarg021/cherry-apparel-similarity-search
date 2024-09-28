import torch
from tqdm import tqdm
import numpy as np
import torch.functional as F

# Function to compute top-k retrievals
def retrieve_top_k(test_feature, train_features, k=5):
    similarities = torch.matmul(train_features, test_feature.unsqueeze(1)).squeeze(1)
    topk_similarities, topk_indices = similarities.topk(k)
    return topk_indices, topk_similarities

# Compute performance metrics
total_precisions = []
total_recalls = []
total_accuracies = []

def compute_performance_metrics(train_features, train_labels, test_features, test_labels, k = 5):
    """
    This function calculates precision, recall, and accuracy metrics for top-k image retrieval
    using the provided feature vectors and labels for both training and test sets.
    It computes these metrics by comparing the top-k retrieved items from the training set
    for each item in the test set.

    Args:
        train_features (torch.Tensor): Feature vectors of the training set.
        train_labels (numpy.ndarray): Labels of the training set.
        test_features (torch.Tensor): Feature vectors of the test set.
        test_labels (numpy.ndarray): Labels of the test set.
        k (int, optional): The number of top items to retrieve. Defaults to 5.

    Returns:
        Return mean precision, mean recall, and retrieval accuracy for the top-k retrievals.
    """

    # Normalize features
    train_features = F.normalize(train_features, p=2, dim=1)
    test_features = F.normalize(test_features, p=2, dim=1)

    for i in tqdm(range(len(test_features)), desc="Calculating metrics"):
        test_feature = test_features[i]
        test_label = test_labels[i]
        topk_indices, _ = retrieve_top_k(test_feature, train_features, k)
        retrieved_labels = train_labels[topk_indices.numpy()]
        relevant = (retrieved_labels == test_label).astype(int)
        precision = relevant.sum() / k
        recall = relevant.sum() / (train_labels == test_label).sum()
        accuracy = 1 if relevant.sum() > 0 else 0  # 1 if at least one relevant item retrieved
        total_precisions.append(precision)
        total_recalls.append(recall)
        total_accuracies.append(accuracy)

    mean_precision = np.mean(total_precisions)
    mean_recall = np.mean(total_recalls)
    retrieval_accuracy = np.mean(total_accuracies)

    return mean_precision, mean_recall, retrieval_accuracy
