import torch
import numpy as np
import networkx as nx
import pickle
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import json

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("✅ Libraries loaded!")



# Load the graph we built
import pickle

# Load the graph we built (using pickle directly)
with open('data/processed/fraud_graph.gpickle', 'rb') as f:
    G = pickle.load(f)

print(f"Loaded graph:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")

# Load metadata
with open('data/processed/graph_metadata.json', 'r') as f:
    metadata = json.load(f)
print(f"\nMetadata: {metadata}")




# Get node features and labels
num_nodes = G.number_of_nodes()

# Initialize arrays
node_features = []
node_labels = []

# Extract features from each node (must be in order!)
for node_id in range(num_nodes):
    node_data = G.nodes[node_id]
    node_features.append(node_data['features'])
    node_labels.append(node_data['label'])

# Convert to tensors
x = torch.tensor(node_features, dtype=torch.float)
y = torch.tensor(node_labels, dtype=torch.long)

print(f"Node feature matrix shape: {x.shape}")
print(f"  -> {x.shape[0]} nodes, {x.shape[1]} features per node")
print(f"Label tensor shape: {y.shape}")
print(f"  -> Fraud cases: {y.sum().item()}")
print(f"  -> Normal cases: {(y == 0).sum().item()}")



# Convert edges to PyTorch Geometric format
# edge_index: [2, num_edges] tensor with source and target nodes

edge_list = list(G.edges())
edge_weights = []

# Get source and target nodes
source_nodes = []
target_nodes = []

for src, tgt in edge_list:
    source_nodes.append(src)
    target_nodes.append(tgt)
    
    # Get edge weight (similarity score)
    weight = G[src][tgt].get('weight', 1.0)
    edge_weights.append(weight)
    
    # Add reverse edge (undirected graph)
    source_nodes.append(tgt)
    target_nodes.append(src)
    edge_weights.append(weight)

# Convert to tensor
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

print(f"Edge index shape: {edge_index.shape}")
print(f"  -> {edge_index.shape[1]} directed edges (includes reverse)")
print(f"Edge attributes shape: {edge_attr.shape}")
print(f"  -> Each edge has {edge_attr.shape[1]} attribute(s)")



# Split nodes into train/val/test sets
# Strategy: 60% train, 20% val, 20% test
# IMPORTANT: Stratified split to maintain fraud ratio

# Get all node indices
all_indices = np.arange(num_nodes)
labels_np = y.numpy()

# First split: 80% train+val, 20% test
train_val_idx, test_idx = train_test_split(
    all_indices, 
    test_size=0.2, 
    stratify=labels_np,
    random_state=42
)

# Second split: 75% train (of 80%), 25% val (of 80%) → 60% train, 20% val overall
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.25,  # 0.25 * 0.8 = 0.2
    stratify=labels_np[train_val_idx],
    random_state=42
)

# Create boolean masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

# Print split statistics
print("Dataset Split:")
print("="*50)
print(f"Train: {train_mask.sum().item()} nodes ({train_mask.sum().item()/num_nodes*100:.1f}%)")
print(f"  - Frauds: {y[train_mask].sum().item()}")
print(f"  - Normal: {(y[train_mask] == 0).sum().item()}")

print(f"\nVal: {val_mask.sum().item()} nodes ({val_mask.sum().item()/num_nodes*100:.1f}%)")
print(f"  - Frauds: {y[val_mask].sum().item()}")
print(f"  - Normal: {(y[val_mask] == 0).sum().item()}")

print(f"\nTest: {test_mask.sum().item()} nodes ({test_mask.sum().item()/num_nodes*100:.1f}%)")
print(f"  - Frauds: {y[test_mask].sum().item()}")
print(f"  - Normal: {(y[test_mask] == 0).sum().item()}")
print("="*50)



# Create the Data object
data = Data(
    x=x,                    # Node features [num_nodes, num_features]
    edge_index=edge_index,  # Graph connectivity [2, num_edges]
    edge_attr=edge_attr,    # Edge features [num_edges, edge_features]
    y=y,                    # Node labels [num_nodes]
    train_mask=train_mask,  # Training nodes
    val_mask=val_mask,      # Validation nodes
    test_mask=test_mask     # Test nodes
)

print("PyTorch Geometric Data Object Created!")
print("="*50)
print(data)
print("="*50)

# Validate the data
print("\nData Validation:")
print(f"✅ Has node features: {data.x is not None}")
print(f"✅ Has edge index: {data.edge_index is not None}")
print(f"✅ Has labels: {data.y is not None}")
print(f"✅ Has train/val/test masks: {data.train_mask is not None}")
print(f"✅ Number of nodes: {data.num_nodes}")
print(f"✅ Number of edges: {data.num_edges}")
print(f"✅ Number of features: {data.num_features}")



# Save the data object
torch.save(data, 'data/processed/fraud_graph_pyg.pt')
print("✅ PyTorch Geometric data saved to: data/processed/fraud_graph_pyg.pt")

# Save split indices for reference
split_info = {
    'train_indices': train_idx.tolist(),
    'val_indices': val_idx.tolist(),
    'test_indices': test_idx.tolist(),
    'num_train': int(train_mask.sum().item()),
    'num_val': int(val_mask.sum().item()),
    'num_test': int(test_mask.sum().item()),
    'train_fraud': int(y[train_mask].sum().item()),
    'val_fraud': int(y[val_mask].sum().item()),
    'test_fraud': int(y[test_mask].sum().item())
}

with open('data/processed/split_info.json', 'w') as f:
    json.dump(split_info, f, indent=4)

print("✅ Split info saved to: data/processed/split_info.json")




# Visualize data structure
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Dataset split
splits = ['Train', 'Val', 'Test']
fraud_counts = [
    y[train_mask].sum().item(),
    y[val_mask].sum().item(),
    y[test_mask].sum().item()
]
normal_counts = [
    (y[train_mask] == 0).sum().item(),
    (y[val_mask] == 0).sum().item(),
    (y[test_mask] == 0).sum().item()
]

x_pos = np.arange(len(splits))
axes[0].bar(x_pos - 0.2, fraud_counts, 0.4, label='Fraud', color='red', alpha=0.7)
axes[0].bar(x_pos + 0.2, normal_counts, 0.4, label='Normal', color='green', alpha=0.7)
axes[0].set_xlabel('Split')
axes[0].set_ylabel('Count')
axes[0].set_title('Train/Val/Test Split Distribution')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(splits)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Feature distribution sample (first feature)
axes[1].hist(x[:, 0].numpy(), bins=50, alpha=0.7, color='blue')
axes[1].set_xlabel('Feature Value (First Feature)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Sample Feature Distribution')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Data ready for GNN training!")
