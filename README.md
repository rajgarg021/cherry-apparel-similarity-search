# Cherry Apparel Image Similarity Search Project Overview

This project implements an image similarity search system for apparel items using deep learning techniques. Here's an overview of how it works:

## 1. Data Preparation - data_preparation.py

- Loads a fashion product image dataset ('ashraq/fashion-product-images-small')
- Filters out classes with only one sample
- Creates a label mapping for article types
- Saves the processed dataset for later use

## 2. Model Training and Feature Extraction

The project implements and compares four different models:

### a. VAE (Variational Autoencoder) - vae.py

- Uses a pre-trained VAE from the 'stabilityai/sd-vae-ft-mse' model
- Adds a classifier on top of the VAE's image embeddings
- Fine-tunes the model on the fashion dataset

### b. CLIP (Contrastive Language-Image Pre-training) - clip.py

- Uses the 'openai/clip-vit-base-patch32' model
- Adds a classifier on top of CLIP's image embeddings
- Fine-tunes the model on the fashion dataset

### c. ResNet - resnet.py

- Uses a pre-trained ResNet-50 model
- Modifies the classifier to match the number of classes in the fashion dataset
- Fine-tunes the model on the fashion dataset

### d. Siamese Network - siamese.py

- Implements a Siamese network using ResNet-50 as the backbone
- Uses contrastive loss for training
- Creates positive and negative pairs for training

Each model script includes:
- Data loading and preprocessing
- Model definition and training
- Feature extraction for similarity search
- Performance evaluation using precision, recall, and retrieval accuracy metrics

## 3. Evaluation - evaluation.py

- Implements functions to compute top-k retrievals
- Calculates precision, recall, and accuracy metrics for image retrieval

## 5. Inference Application - app.py

- A Gradio-based web application for user interaction.
- Users can upload an image or provide an image URL.
- The application uses the finetuned CLIP-based model to extract features from the input image.
- It then uses FAISS (Facebook AI Similarity Search) to quickly find similar images based on the extracted features.
- The top 5 most similar apparel items are displayed to the user.

## Main Workflow for a User Query

1. User provides an image.
2. The image is processed using the CLIP-based model to extract features.
3. These features are compared against the pre-computed FAISS index.
4. The most similar images are retrieved and displayed.

The project demonstrates a complete pipeline for building an image similarity search system, from data preparation and model training to deployment as a web application. It also showcases the comparison of different deep learning architectures for this task, allowing for performance evaluation and selection of the best model for the specific use case.

# Detailed Comparison Report

This report compares four different approaches to building an image similarity search system:

1. CLIP (Contrastive Language-Image Pre-training)
2. ResNet
3. Siamese Network
4. VAE (Variational Autoencoder)

Each approach was fine-tuned and tested on the 'ashraq/fashion-product-images-small' dataset, with performance metrics calculated for precision, recall, and retrieval accuracy.

## Performance Metrics

Here's a summary of the performance metrics for each approach:

| Method   | Precision@5 | Recall@5 | Retrieval Accuracy@5 |
|----------|-------------|----------|----------------------|
| CLIP     | 0.8858      | 0.0113   | 0.9344               |
| ResNet   | 0.8508      | 0.0104   | 0.9268               |
| Siamese  | 0.6074      | 0.0051   | 0.8592               |
| VAE      | 0.4895      | 0.0026   | 0.7474               |

## Analysis and Insights

### CLIP (Contrastive Language-Image Pre-training)

- **Performance**: CLIP  demonstrates the best overall performance with the highest precision, recall, and retrieval accuracy
- **Strengths**: 
  - Versatile for various image types and domains
  - Pre-trained on a diverse dataset, making it robust for transfer learning
- **Computational Efficiency and Scalability**:
  - Scales well with GPU acceleration but may be costly for large-scale deployments
  - CLIP embeddings are fast to compute and can be efficiently stored in FAISS for scalable retrieval
- **Use cases**: 
  - Ideal for applications requiring high accuracy and the ability to handle diverse image types
- **Limitations**:
  - Initial embedding computation is expensive in terms of time and computational resources

### ResNet

- **Performance**: ResNet shows the second-best performance, close to CLIP's results
- **Strengths**: 
  - Good balance between accuracy and computational efficiency
  - Well-established architecture with proven performance in various computer vision tasks
- **Computational Efficiency and Scalability**: 
  - Computation of embeddings is faster than CLIP
  - Suitable for real-time applications with proper optimizations
  - Scales well and can handle large datasets efficiently
- **Use cases**: 
  - Suitable for large-scale image similarity search where a balance between accuracy and speed is required
  - Good for applications with specific image domains (e.g., fashion, as in this case)
- **Limitations**: 
  - Struggles with highly varied datasets. Less efficient in handling out-of-domain images compared to CLIP

### Siamese Network

- **Performance**: Siamese Network shows moderate performance, lower than CLIP and ResNet but higher than VAE
- **Strengths**: 
  - Specifically designed for similarity comparisons
  - Can be trained on pairs of similar and dissimilar images, potentially capturing fine-grained differences
- **Computational Efficiency and Scalability**:
  - Can be optimized for real-time usage, especially with smaller backbone networks
  - Scales reasonably well but may face challenges with very large datasets due to the pairwise nature of training
- **Use cases**: 
  - Ideal for highly specific similarity tasks, where precise matches are needed, such as biometric identification or fine-grained visual recognition
- **Limitations** 
  - Does not scale well to large datasets or real-time applications

### VAE (Variational Autoencoder)

- **Performance**: VAE shows the lowest performance among the four methods
- **Strengths**: 
  - Capable of generating new images, which can be useful for data augmentation or creative applications
  - Learns a compact latent representation of images
- **Computational Efficiency and Scalability**:
  - Moderate to high computational requirements, especially during training due to the complexity of the model and training process
  - Since VAE embeddings are compact, they can be efficiently stored and retrieved. However, their lower accuracy limits their use in many real-time scenarios
- **Use cases**:
  - Best for cases where dimensionality reduction and fast retrieval are more important than accuracy, such as multimedia retrieval systems that focus on speed
  - Suitable for applications where image generation or reconstruction is also required
- **Limitations**: 
  - Lower accuracy for similarity search compared to the other methods

## Conclusion

1. CLIP performs best when accuracy and generalization are prioritized. For high-accuracy, general-purpose image similarity search, use CLIP, especially if text-based queries are also important.
2. For a balance between accuracy and speed in domain-specific applications, use ResNet with fine-tuning.
3. For specific pairwise similarity tasks or when fine-grained distinctions are crucial, consider Siamese Networks. Though keep in mind that they suffer from scalability issues.
4. VAE is useful for cases where speed and compact embeddings are crucial, but its lower accuracy limits its use for critical similarity tasks.
