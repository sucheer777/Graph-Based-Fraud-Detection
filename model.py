import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import GRU

class TSSGC(nn.Module):
    """
    Temporal-Spatial-Semantic Graph Convolution
    Full implementation based on FraudGNN-RL paper
    
    Components:
    1. Temporal: GRU + time-aware attention
    2. Spatial: GAT for neighbor aggregation  
    3. Semantic: Node type embeddings
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, num_layers=3, 
                 dropout=0.5, temporal_dim=32, num_node_types=2):
        super(TSSGC, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # ============== TEMPORAL COMPONENT ==============
        # GRU for sequential temporal modeling
        self.temporal_gru = GRU(
            input_size=input_dim + 2,  # features + time + amount
            hidden_size=temporal_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Temporal attention weights
        self.temporal_attention = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim),
            nn.Tanh(),
            nn.Linear(temporal_dim, 1)
        )
        
        # ============== SPATIAL COMPONENT ==============
        # Multi-layer GAT for spatial neighbor aggregation
        self.spatial_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.spatial_convs.append(
            GATConv(input_dim, hidden_dim, heads=4, dropout=dropout, concat=True)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 4))
        
        # Middle layers
        for i in range(num_layers - 2):
            self.spatial_convs.append(
                GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout, concat=True)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 4))
        
        # Last layer (single head output)
        self.spatial_convs.append(
            GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=dropout, concat=False)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # ============== SEMANTIC COMPONENT ==============
        # Node type embeddings (normal vs fraud-prone accounts)
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        
        # Semantic feature transformation
        self.semantic_transform = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ============== FUSION LAYER ==============
        # Combine all three components
        fusion_input_dim = temporal_dim + hidden_dim + hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # ============== CLASSIFIER ==============
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def temporal_component(self, x, time, amount):
        """
        Temporal processing with GRU and time-aware attention
        """
        batch_size = x.size(0)
        
        # Normalize temporal features
        time_norm = (time - time.mean()) / (time.std() + 1e-6)
        amount_norm = (amount - amount.mean()) / (amount.std() + 1e-6)
        
        # Combine features with temporal info
        temporal_features = torch.cat([
            x, 
            time_norm.unsqueeze(1), 
            amount_norm.unsqueeze(1)
        ], dim=1)
        
        # Reshape for GRU (add sequence dimension)
        temporal_features = temporal_features.unsqueeze(1)  # [batch, 1, features]
        
        # Process through GRU
        gru_out, _ = self.temporal_gru(temporal_features)  # [batch, 1, temporal_dim]
        gru_out = gru_out.squeeze(1)  # [batch, temporal_dim]
        
        # Apply temporal attention
        attention_weights = self.temporal_attention(gru_out)  # [batch, 1]
        attention_weights = torch.sigmoid(attention_weights)
        
        # Weighted temporal representation
        h_temporal = gru_out * attention_weights
        
        return h_temporal
    
    def spatial_component(self, x, edge_index):
        """
        Spatial processing with multi-layer GAT
        """
        h = x
        
        # Pass through GAT layers
        for i, (conv, bn) in enumerate(zip(self.spatial_convs, self.batch_norms)):
            h = conv(h, edge_index)
            h = bn(h)
            
            if i < len(self.spatial_convs) - 1:  # Not last layer
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def semantic_component(self, x, node_types=None):
        """
        Semantic processing with node type embeddings
        """
        batch_size = x.size(0)
        
        # If no node types provided, assume all same type
        if node_types is None:
            node_types = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # Get node type embeddings
        type_embed = self.node_type_embedding(node_types)
        
        # Combine with node features
        semantic_features = torch.cat([x, type_embed], dim=1)
        h_semantic = self.semantic_transform(semantic_features)
        
        return h_semantic
    
    def forward(self, x, edge_index, time=None, amount=None, node_types=None):
        """
        Full forward pass combining all three components
        """
        # 1. TEMPORAL COMPONENT
        if time is not None and amount is not None:
            h_temporal = self.temporal_component(x, time, amount)
        else:
            # Fallback: use zeros if temporal info not available
            h_temporal = torch.zeros(x.size(0), 32, device=x.device)
        
        # 2. SPATIAL COMPONENT
        h_spatial = self.spatial_component(x, edge_index)
        
        # 3. SEMANTIC COMPONENT
        h_semantic = self.semantic_component(x, node_types)
        
        # 4. FUSION
        h_combined = torch.cat([h_temporal, h_spatial, h_semantic], dim=1)
        h_fused = self.fusion_layer(h_combined)
        
        # 5. CLASSIFICATION
        out = self.classifier(h_fused)
        
        return F.log_softmax(out, dim=1)


# Create TSSGC model
model_tssgc = TSSGC(
    input_dim=32,
    hidden_dim=64,
    output_dim=2,
    num_layers=3,
    dropout=0.5,
    temporal_dim=32,
    num_node_types=2
)

print("✅ TSSGC Model Created!")
print(f"\nModel architecture:\n{model_tssgc}")
print(f"\nTotal parameters: {sum(p.numel() for p in model_tssgc.parameters()):,}")



print("📊 TSSGC ARCHITECTURE BREAKDOWN:")
print("="*80)

component_params = {
    'Temporal (GRU + Attention)': sum(p.numel() for n, p in model_tssgc.named_parameters() if 'temporal' in n),
    'Spatial (GAT layers)': sum(p.numel() for n, p in model_tssgc.named_parameters() if 'spatial' in n),
    'Semantic (Embeddings)': sum(p.numel() for n, p in model_tssgc.named_parameters() if 'semantic' in n or 'node_type' in n),
    'Fusion Layer': sum(p.numel() for n, p in model_tssgc.named_parameters() if 'fusion' in n),
    'Classifier': sum(p.numel() for n, p in model_tssgc.named_parameters() if 'classifier' in n),
}

total_params = sum(component_params.values())

for component, params in component_params.items():
    percentage = (params / total_params) * 100
    print(f"{component:30s} {params:>10,} params ({percentage:>5.1f}%)")

print("="*80)
print(f"{'TOTAL':30s} {total_params:>10,} params")
print("="*80)
