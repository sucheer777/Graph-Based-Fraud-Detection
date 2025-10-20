import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import GRU

class TSSGC(nn.Module):
    """
    Temporal-Spatial-Semantic Graph Convolution
    Full implementation for fraud detection
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, num_layers=3, 
                 dropout=0.5, temporal_dim=32, num_node_types=2):
        super(TSSGC, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # ============== TEMPORAL COMPONENT ==============
        self.temporal_gru = GRU(
            input_size=input_dim + 2,
            hidden_size=temporal_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.temporal_attention = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim),
            nn.Tanh(),
            nn.Linear(temporal_dim, 1)
        )
        
        # ============== SPATIAL COMPONENT ==============
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
        
        # Last layer
        self.spatial_convs.append(
            GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=dropout, concat=False)
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # ============== SEMANTIC COMPONENT ==============
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        
        self.semantic_transform = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ============== FUSION LAYER ==============
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
        """Temporal processing with GRU and attention"""
        time_norm = (time - time.mean()) / (time.std() + 1e-6)
        amount_norm = (amount - amount.mean()) / (amount.std() + 1e-6)
        
        temporal_features = torch.cat([
            x, 
            time_norm.unsqueeze(1), 
            amount_norm.unsqueeze(1)
        ], dim=1)
        
        temporal_features = temporal_features.unsqueeze(1)
        gru_out, _ = self.temporal_gru(temporal_features)
        gru_out = gru_out.squeeze(1)
        
        attention_weights = self.temporal_attention(gru_out)
        attention_weights = torch.sigmoid(attention_weights)
        
        h_temporal = gru_out * attention_weights
        return h_temporal
    
    def spatial_component(self, x, edge_index):
        """Spatial processing with GAT layers"""
        h = x
        
        for i, (conv, bn) in enumerate(zip(self.spatial_convs, self.batch_norms)):
            h = conv(h, edge_index)
            h = bn(h)
            
            if i < len(self.spatial_convs) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def semantic_component(self, x, node_types=None):
        """Semantic processing with node embeddings"""
        batch_size = x.size(0)
        
        if node_types is None:
            node_types = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        type_embed = self.node_type_embedding(node_types)
        semantic_features = torch.cat([x, type_embed], dim=1)
        h_semantic = self.semantic_transform(semantic_features)
        
        return h_semantic
    
    def forward(self, x, edge_index, time=None, amount=None, node_types=None):
        """Full forward pass"""
        # 1. TEMPORAL
        if time is not None and amount is not None:
            h_temporal = self.temporal_component(x, time, amount)
        else:
            h_temporal = torch.zeros(x.size(0), 32, device=x.device)
        
        # 2. SPATIAL
        h_spatial = self.spatial_component(x, edge_index)
        
        # 3. SEMANTIC
        h_semantic = self.semantic_component(x, node_types)
        
        # 4. FUSION
        h_combined = torch.cat([h_temporal, h_spatial, h_semantic], dim=1)
        h_fused = self.fusion_layer(h_combined)
        
        # 5. CLASSIFICATION
        out = self.classifier(h_fused)
        
        return F.log_softmax(out, dim=1)


print("✅ TSSGC model defined!")

