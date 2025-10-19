import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("✅ Libraries loaded!")



# Load dataset
df = pd.read_csv('data/raw/creditcard.csv')

# Use subset for faster prototyping (50k transactions)
df_subset = df.iloc[:50000].copy()

print(f"Working with {len(df_subset)} transactions")
print(f"Fraud cases: {df_subset['Class'].sum()}")
print(f"Normal cases: {(df_subset['Class']==0).sum()}")

# Separate features and labels
X = df_subset.drop('Class', axis=1).values
y = df_subset['Class'].values

print(f"\n✅ Data loaded!")
print(f"Feature shape: {X.shape}")
print(f"Label distribution: {np.bincount(y)}")



# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Features standardized!")
print(f"Mean: {X_scaled.mean():.4f}")
print(f"Std: {X_scaled.std():.4f}")

# Save scaler
with open('data/processed/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved!")



# Build graph using K-Nearest Neighbors
# Each transaction connects to its K most similar transactions

K_NEIGHBORS = 5  # Connect each node to 5 nearest neighbors

print(f"Building KNN graph with k={K_NEIGHBORS}...")
print("This creates edges based on feature similarity")

# Use KNN to find neighbors
knn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric='cosine')
knn.fit(X_scaled)
distances, indices = knn.kneighbors(X_scaled)

# Create edge list
edge_list = []

for i in tqdm(range(len(X_scaled)), desc="Creating edges"):
    for j in range(1, K_NEIGHBORS + 1):  # Skip first (self)
        neighbor = indices[i][j]
        # Add both directions for undirected graph
        edge_list.append([i, neighbor])
        edge_list.append([neighbor, i])

# Remove duplicates
edge_list = list(set(map(tuple, edge_list)))

print(f"\n✅ Created {len(edge_list)} edges (with duplicates removed)")



# Convert to PyTorch tensors

# Node features (transactions)
x = torch.tensor(X_scaled, dtype=torch.float)

# Edge index (connections between transactions)
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# Labels (0=normal, 1=fraud)
y = torch.tensor(y, dtype=torch.long)

# Additional node attributes
amount = torch.tensor(df_subset['Amount'].values, dtype=torch.float)
time = torch.tensor(df_subset['Time'].values, dtype=torch.float)

print("✅ Converted to PyTorch tensors!")
print(f"Node features (x): {x.shape}")
print(f"Edge index: {edge_index.shape}")
print(f"Labels (y): {y.shape}")



# Create PyTorch Geometric Data object
data = Data(
    x=x,                    # Node features [num_nodes, num_features]
    edge_index=edge_index,  # Graph connectivity [2, num_edges]
    y=y,                    # Node labels [num_nodes]
    amount=amount,          # Transaction amounts [num_nodes]
    time=time              # Transaction times [num_nodes]
)

# Validate the data object
data.validate(raise_on_error=True)

print("✅ PyTorch Geometric Data object created!")
print(f"\n{data}")
print(f"\nNumber of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Number of features: {data.num_node_features}")
print(f"Has isolated nodes: {data.has_isolated_nodes()}")
print(f"Has self-loops: {data.has_self_loops()}")
print(f"Is undirected: {data.is_undirected()}")



# Analyze graph structure
print("📊 GRAPH STATISTICS:")
print("="*60)

# Basic stats
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")

# Fraud distribution
fraud_mask = (data.y == 1)
normal_mask = (data.y == 0)

print(f"\nFraud nodes: {fraud_mask.sum().item()}")
print(f"Normal nodes: {normal_mask.sum().item()}")
print(f"Fraud percentage: {(fraud_mask.sum().item()/data.num_nodes)*100:.3f}%")

# Degree distribution
from torch_geometric.utils import degree

node_degrees = degree(data.edge_index[0], data.num_nodes)

print(f"\nDegree statistics:")
print(f"  Min degree: {node_degrees.min().item()}")
print(f"  Max degree: {node_degrees.max().item()}")
print(f"  Mean degree: {node_degrees.mean().item():.2f}")
print(f"  Median degree: {node_degrees.median().item():.2f}")

# Fraud nodes degree
fraud_degrees = node_degrees[fraud_mask]
print(f"\nFraud nodes degree:")
print(f"  Mean: {fraud_degrees.mean().item():.2f}")
print(f"  Median: {fraud_degrees.median().item():.2f}")

print("="*60)



import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Degree distribution
axes[0].hist(node_degrees.numpy(), bins=50, color='skyblue', edgecolor='black')
axes[0].set_xlabel('Node Degree')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Degree Distribution')
axes[0].set_yscale('log')

# Fraud vs Normal degree comparison
axes[1].hist([node_degrees[normal_mask].numpy(), 
              node_degrees[fraud_mask].numpy()], 
             bins=30, label=['Normal', 'Fraud'], 
             color=['green', 'red'], alpha=0.7)
axes[1].set_xlabel('Node Degree')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Degree: Fraud vs Normal')
axes[1].legend()

# Amount distribution
axes[2].scatter(range(len(amount[:1000])), amount[:1000], 
               c=y[:1000], cmap='RdYlGn_r', s=10, alpha=0.6)
axes[2].set_xlabel('Transaction Index')
axes[2].set_ylabel('Amount')
axes[2].set_title('Transaction Amounts (First 1000)')
axes[2].set_yscale('log')

plt.tight_layout()
plt.show()

print("✅ Visualization complete!")



# Save the PyG data object
torch.save(data, 'data/processed/fraud_graph_pyg.pt')

print("✅ PyG Data object saved to: data/processed/fraud_graph_pyg.pt")

# Save metadata
metadata = {
    'num_nodes': data.num_nodes,
    'num_edges': data.num_edges,
    'num_features': data.num_node_features,
    'num_fraud': fraud_mask.sum().item(),
    'num_normal': normal_mask.sum().item(),
    'k_neighbors': K_NEIGHBORS,
    'graph_construction': 'KNN',
    'metric': 'cosine'
}

import json
with open('data/processed/graph_metadata_pyg.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("✅ Metadata saved!")
print("\n🎉 Graph construction complete!")
