import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries loaded!")


# Load dataset
df = pd.read_csv('data/raw/creditcard.csv')

# For faster experimentation, let's start with a subset
# Use first 50,000 transactions (contains ~85 fraud cases)
df_subset = df.iloc[:50000].copy()

print(f"Working with {len(df_subset)} transactions")
print(f"Fraud cases: {df_subset['Class'].sum()}")
print(f"Normal cases: {(df_subset['Class']==0).sum()}")

# Separate features and labels
features = df_subset.drop('Class', axis=1).values
labels = df_subset['Class'].values

print(f"\n✅ Data prepared!")
print(f"Feature shape: {features.shape}")



# Standardize features for better similarity computation
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

print("✅ Features normalized!")
print(f"Scaled feature shape: {features_scaled.shape}")

# Save scaler for later use
with open('data/processed/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved!")




# Create graph
G = nx.Graph()

# Add nodes (each transaction is a node)
print("Adding nodes...")
for i in range(len(df_subset)):
    G.add_node(i, 
               features=features_scaled[i],
               label=int(labels[i]),
               amount=df_subset.iloc[i]['Amount'],
               time=df_subset.iloc[i]['Time'])

print(f"✅ Added {G.number_of_nodes()} nodes")




# Add edges based on:
# 1. Temporal proximity (within time window)
# 2. Feature similarity (cosine similarity > threshold)

print("Building edges...")
print("This may take a few minutes...")

TIME_WINDOW = 3600  # 1 hour in seconds
SIMILARITY_THRESHOLD = 0.9
MAX_NEIGHBORS = 10  # Limit edges per node to avoid dense graph

edge_count = 0

for i in tqdm(range(len(df_subset))):
    current_time = df_subset.iloc[i]['Time']
    
    # Find candidates within time window
    time_mask = (df_subset['Time'] >= current_time) & \
                (df_subset['Time'] <= current_time + TIME_WINDOW) & \
                (df_subset.index > i)  # Only connect to future transactions
    
    candidates = df_subset[time_mask].index.tolist()
    
    if len(candidates) == 0:
        continue
    
    # Compute similarity with candidates
    candidate_features = features_scaled[candidates]
    current_feature = features_scaled[i].reshape(1, -1)
    
    similarities = cosine_similarity(current_feature, candidate_features)[0]
    
    # Connect to top similar neighbors above threshold
    similar_indices = np.where(similarities > SIMILARITY_THRESHOLD)[0]
    
    # Limit connections
    if len(similar_indices) > MAX_NEIGHBORS:
        top_indices = np.argsort(similarities)[-MAX_NEIGHBORS:]
        similar_indices = [idx for idx in top_indices if similarities[idx] > SIMILARITY_THRESHOLD]
    
    # Add edges
    for idx in similar_indices:
        neighbor_id = candidates[idx]
        G.add_edge(i, neighbor_id, weight=float(similarities[idx]))
        edge_count += 1

print(f"\n✅ Graph construction complete!")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")





# Analyze graph structure
print("📊 GRAPH STATISTICS:")
print("="*50)

# Basic stats
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.6f}")

# Connected components
num_components = nx.number_connected_components(G)
print(f"Connected components: {num_components}")

# Largest component
largest_cc = max(nx.connected_components(G), key=len)
print(f"Largest component size: {len(largest_cc)}")

# Degree distribution
degrees = [G.degree(n) for n in G.nodes()]
print(f"Average degree: {np.mean(degrees):.2f}")
print(f"Max degree: {np.max(degrees)}")
print(f"Min degree: {np.min(degrees)}")

# Fraud nodes statistics
fraud_nodes = [n for n in G.nodes() if G.nodes[n]['label'] == 1]
print(f"\nFraud nodes: {len(fraud_nodes)}")
print(f"Average degree of fraud nodes: {np.mean([G.degree(n) for n in fraud_nodes]):.2f}")

print("="*50)



# Save the graph
print("Saving graph...")
nx.write_gpickle(G, 'data/processed/fraud_graph.gpickle')
print("✅ Graph saved to: data/processed/fraud_graph.gpickle")

# Save metadata
graph_metadata = {
    'num_nodes': G.number_of_nodes(),
    'num_edges': G.number_of_edges(),
    'num_fraud': len(fraud_nodes),
    'time_window': TIME_WINDOW,
    'similarity_threshold': SIMILARITY_THRESHOLD,
    'max_neighbors': MAX_NEIGHBORS
}

import json
with open('data/processed/graph_metadata.json', 'w') as f:
    json.dump(graph_metadata, f, indent=4)

print("✅ Metadata saved!")
