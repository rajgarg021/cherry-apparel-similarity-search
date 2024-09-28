from datasets import load_dataset, ClassLabel
from collections import Counter

# Load the dataset
dataset = load_dataset('ashraq/fashion-product-images-small', split='train')

# Remove classes with only one sample before creating label mapping
label_counts = Counter(dataset['articleType'])
classes_to_keep = [label for label, count in label_counts.items() if count > 1]
dataset = dataset.filter(lambda example: example['articleType'] in classes_to_keep)

# Get unique article types and create label mapping
article_types = dataset['articleType']
unique_article_types = sorted(set(article_types))
article_type_to_idx = {article_type: idx for idx, article_type in enumerate(unique_article_types)}

def get_label(example):
    example['label'] = article_type_to_idx[example['articleType']]
    return example

dataset = dataset.map(get_label)

# Cast the 'label' column to ClassLabel
num_classes = len(unique_article_types)
dataset = dataset.cast_column('label', ClassLabel(num_classes=num_classes, names=list(unique_article_types)))

# Save the dataset for later retrieval
dataset.save_to_disk('dataset')
