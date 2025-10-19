import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
import json

print(f"PyTorch version: {torch.__version__}")
print("✅ Libraries loaded!")



# Load the graph we created
# NEW (working):
data = torch.load('data/processed/fraud_graph_pyg.pt', weights_only=False)


print("✅ Graph loaded!")
print(f"\n{data}")
print(f"\nNodes: {data.num_nodes}")
print(f"Edges: {data.num_edges}")
print(f"Features: {data.num_node_features}")
print(f"\nFraud cases: {(data.y == 1).sum().item()}")
print(f"Normal cases: {(data.y == 0).sum().item()}")


# Create boolean masks for train/val/test splits
# We'll use: 60% train, 20% val, 20% test

num_nodes = data.num_nodes

# Get indices
indices = torch.arange(num_nodes)
labels = data.y.numpy()

# First split: 80% train+val, 20% test
train_val_idx, test_idx = train_test_split(
    indices, 
    test_size=0.2, 
    stratify=labels,  # Keep class distribution
    random_state=42
)

# Second split: 60% train, 20% val (from train+val)
train_labels = labels[train_val_idx]
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.25,  # 0.25 of 80% = 20% of total
    stratify=train_labels,
    random_state=42
)

# Create boolean masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

# Add masks to data object
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

print("✅ Train/Val/Test splits created!")
print(f"\nTrain nodes: {train_mask.sum().item()} ({train_mask.sum().item()/num_nodes*100:.1f}%)")
print(f"Val nodes: {val_mask.sum().item()} ({val_mask.sum().item()/num_nodes*100:.1f}%)")
print(f"Test nodes: {test_mask.sum().item()} ({test_mask.sum().item()/num_nodes*100:.1f}%)")



# Verify class balance in each split
print("📊 CLASS DISTRIBUTION ACROSS SPLITS:")
print("="*60)

splits = {
    'Train': train_mask,
    'Val': val_mask,
    'Test': test_mask
}

for split_name, mask in splits.items():
    split_labels = data.y[mask]
    fraud_count = (split_labels == 1).sum().item()
    normal_count = (split_labels == 0).sum().item()
    total = mask.sum().item()
    
    print(f"\n{split_name} Set:")
    print(f"  Total: {total}")
    print(f"  Fraud: {fraud_count} ({fraud_count/total*100:.2f}%)")
    print(f"  Normal: {normal_count} ({normal_count/total*100:.2f}%)")
    print(f"  Ratio (Normal:Fraud): {normal_count/fraud_count:.1f}:1")

print("="*60)



# Apply SMOTE only to training set to balance classes
print("Applying SMOTE to training set...")

# Extract training data
X_train = data.x[train_mask].numpy()
y_train = data.y[train_mask].numpy()

print(f"\nBefore SMOTE:")
print(f"  Fraud: {(y_train == 1).sum()}")
print(f"  Normal: {(y_train == 0).sum()}")

# Apply SMOTE with 1:2 ratio (more conservative than 1:1)
# This gives fraud more representation without extreme oversampling
smote = SMOTE(
    sampling_strategy=0.5,  # Make fraud 50% of majority class
    random_state=42,
    k_neighbors=5
)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"  Fraud: {(y_train_resampled == 1).sum()}")
print(f"  Normal: {(y_train_resampled == 0).sum()}")
print(f"  New ratio: {(y_train_resampled == 0).sum() / (y_train_resampled == 1).sum():.1f}:1")

print("\n✅ SMOTE applied!")
print(f"Training set size increased from {len(y_train)} to {len(y_train_resampled)}")



# Create a new data object with balanced training data
# Note: We keep val/test with natural imbalance for realistic evaluation

# Convert resampled data to tensors
X_train_balanced = torch.tensor(X_train_resampled, dtype=torch.float)
y_train_balanced = torch.tensor(y_train_resampled, dtype=torch.long)

# For balanced training, we'll store this separately
# We'll rebuild edges for the balanced training set using KNN

from sklearn.neighbors import NearestNeighbors

K_NEIGHBORS = 5

print("Rebuilding edges for balanced training set...")
knn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric='cosine')
knn.fit(X_train_balanced.numpy())
distances, indices = knn.kneighbors(X_train_balanced.numpy())

# Create edge list
edge_list_train = []
for i in range(len(X_train_balanced)):
    for j in range(1, K_NEIGHBORS + 1):
        neighbor = indices[i][j]
        edge_list_train.append([i, neighbor])
        edge_list_train.append([neighbor, i])

# Remove duplicates
edge_list_train = list(set(map(tuple, edge_list_train)))

edge_index_train = torch.tensor(edge_list_train, dtype=torch.long).t().contiguous()

# Create balanced training data object
train_data = Data(
    x=X_train_balanced,
    edge_index=edge_index_train,
    y=y_train_balanced
)

print(f"\n✅ Balanced training data created!")
print(f"{train_data}")



# Compute class weights for loss function
# Even with SMOTE, we can use weights for better learning

from sklearn.utils.class_weight import compute_class_weight

# Compute weights based on ORIGINAL training distribution (before SMOTE)
classes = np.array([0, 1])
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

print("📊 CLASS WEIGHTS:")
print(f"  Normal (Class 0): {class_weights[0]:.4f}")
print(f"  Fraud (Class 1): {class_weights[1]:.4f}")
print(f"\nFraud weight is {class_weights[1]/class_weights[0]:.1f}x higher")

# Save class weights
torch.save(class_weights_tensor, 'data/processed/class_weights.pt')
print("\n✅ Class weights saved!")



# Save the main data object with masks
torch.save(data, 'data/processed/fraud_graph_with_splits.pt')

# Save balanced training data
torch.save(train_data, 'data/processed/train_data_balanced.pt')

print("✅ Data saved!")
print("\nSaved files:")
print("  1. fraud_graph_with_splits.pt (full graph with train/val/test masks)")
print("  2. train_data_balanced.pt (balanced training graph with SMOTE)")
print("  3. class_weights.pt (for weighted loss)")



# Create comprehensive summary
# Create comprehensive summary (with type conversion for JSON)
summary = {
    'full_graph': {
        'num_nodes': int(data.num_nodes),
        'num_edges': int(data.num_edges),
        'num_features': int(data.num_node_features),
        'num_fraud': int((data.y == 1).sum().item()),
        'num_normal': int((data.y == 0).sum().item()),
    },
    'splits': {
        'train_nodes': int(train_mask.sum().item()),
        'val_nodes': int(val_mask.sum().item()),
        'test_nodes': int(test_mask.sum().item()),
    },
    'train_original': {
        'total': int(len(y_train)),
        'fraud': int((y_train == 1).sum()),
        'normal': int((y_train == 0).sum()),
    },
    'train_balanced': {
        'total': int(len(y_train_resampled)),
        'fraud': int((y_train_resampled == 1).sum()),
        'normal': int((y_train_resampled == 0).sum()),
    },
    'class_weights': {
        'normal': float(class_weights[0]),
        'fraud': float(class_weights[1]),
    }
}

# Save summary
with open('data/processed/data_preparation_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print("📊 FINAL SUMMARY:")
print("="*60)
print(json.dumps(summary, indent=2))
print("="*60)
print("\n✅ Summary saved to data_preparation_summary.json")
print("\n🎉 Data preparation complete! Ready for model training!")

