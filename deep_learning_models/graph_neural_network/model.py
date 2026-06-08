import torch
from torch import nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)
    
    def forward(self, node_feats, adjacency_matrix):
        num_neighbors = adjacency_matrix.sum(dim=-1, keepdim=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adjacency_matrix, node_feats) / num_neighbors
        return node_feats
    
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=1, concat_heads=True, alpha=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.projection = nn.Linear(in_features, out_features * num_heads)
        self.attention = nn.Parameter(torch.Tensor(num_heads, out_features * 2))
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, node_feats, adjacency_matrix, print_attn_probs=False):
        batch_size, num_nodes, _ = node_feats.size()
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        edges = adjacency_matrix.nonzero(as_tuple=False)
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:,0] * num_nodes + edges[:,1]
        edge_indices_col = edges[:,0] * num_nodes + edges[:,2]

        a_input = torch.cat([
            torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
            torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
        ], dim=-1)

        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.attention)
        attn_logits = self.leakyrelu(attn_logits)

        attn_matrix = attn_logits.new_zeros(adjacency_matrix.shape+(self.num_heads,)).fill_(-9e15)
        attn_matrix[adjacency_matrix[...,None].repeat(1,1,1,self.num_heads) == 1] = attn_logits.reshape(-1)

        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)

        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats

gnn_layer_by_name = {
    "GCN": GCNLayer,
    "GAT": GATLayer
}

class GNNModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2, layer_name = "GCN", dp_rate=0.1, **kwargs):
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]
        layers = []

        in_channels, out_channels = in_features, hidden_features

        for i in range(num_layers - 1):
            layers += [
                gnn_layer(
                    in_channels, out_channels, **kwargs
                ), 
                nn.ReLU(),
                nn.Dropout(dp_rate)
            ]
            in_channels = hidden_features
        layers += [gnn_layer(in_channels, out_features, **kwargs)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, edge_index):
        for l in self.layers:
            if isinstance(l, (GCNLayer, GATLayer)):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x