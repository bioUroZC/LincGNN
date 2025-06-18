import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

# 1. Read CSV
node_feat_df = pd.read_csv('node_feat.csv')
edges_df = pd.read_csv('edges.csv')

# 2. Split node DataFrame by node type
lnc_df = node_feat_df[node_feat_df['node_type']=='lncRNA'].reset_index(drop=True)
mi_df  = node_feat_df[node_feat_df['node_type']=='miRNA'].reset_index(drop=True)
mrna_df= node_feat_df[node_feat_df['node_type']=='mRNA'].reset_index(drop=True)

# Build id -> index mapping
lnc_map = {v:i for i,v in enumerate(lnc_df['id'])}
mi_map  = {v:i for i,v in enumerate(mi_df['id'])}
mrna_map= {v:i for i,v in enumerate(mrna_df['id'])}

# 3. Build HeteroData
data = HeteroData()
# Node feature columns
feat_cols = [c for c in node_feat_df.columns if c.startswith('feat_')]
# lncRNA
data['lncRNA'].x = torch.tensor(lnc_df[feat_cols].values, dtype=torch.float)
if 'label' in lnc_df.columns:
    data['lncRNA'].y = torch.tensor(lnc_df['label'].values, dtype=torch.long)
# miRNA
data['miRNA'].x = torch.tensor(mi_df[feat_cols].values, dtype=torch.float)
# mRNA
data['mRNA'].x  = torch.tensor(mrna_df[feat_cols].values, dtype=torch.float)

# Edge mapping function
def map_edges(df_edges, src_map, dst_map):
    src = df_edges['source'].map(src_map)
    dst = df_edges['target'].map(dst_map)
    mask = src.notna() & dst.notna()
    src = src[mask].astype(int).tolist()
    dst = dst[mask].astype(int).tolist()
    if not src:
        return torch.empty((2,0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)

# lncRNA -> miRNA
df_lnc_mi = edges_df[
    (edges_df['relation']=='interacts') &
    (edges_df['source_type']=='lncRNA') &
    (edges_df['target_type']=='miRNA')
]
e1 = map_edges(df_lnc_mi, lnc_map, mi_map)
data[('lncRNA','interacts','miRNA')].edge_index = e1
data[('miRNA','rev_interacts','lncRNA')].edge_index = e1.flip(0)

# miRNA -> mRNA
df_mi_mrna = edges_df[
    (edges_df['relation']=='regulates') &
    (edges_df['source_type']=='miRNA') &
    (edges_df['target_type']=='mRNA')
]
e2 = map_edges(df_mi_mrna, mi_map, mrna_map)
data[('miRNA','regulates','mRNA')].edge_index = e2
data[('mRNA','rev_regulates','miRNA')].edge_index = e2.flip(0)

# 4. Validate structure
print(data)
for ntype in data.node_types:
    print(f"{ntype}: num_nodes={data[ntype].num_nodes}, feat_dim={data[ntype].num_features}")
for et in data.edge_types:
    print(f"{et}: num_edges={data[et].edge_index.size(1)}")

# 5. Define Heterogeneous GNN and train
class HeteroGNN(nn.Module):
    def __init__(self, data: HeteroData, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.project = nn.ModuleDict({
            'lncRNA': Linear(data['lncRNA'].num_features, hidden_channels),
            'miRNA': Linear(data['miRNA'].num_features, hidden_channels),
            'mRNA':  Linear(data['mRNA'].num_features, hidden_channels),
        })
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('lncRNA','interacts','miRNA'):     SAGEConv((-1, -1), hidden_channels),
                ('miRNA','rev_interacts','lncRNA'): SAGEConv((-1, -1), hidden_channels),
                ('miRNA','regulates','mRNA'):      SAGEConv((-1, -1), hidden_channels),
                ('mRNA','rev_regulates','miRNA'):  SAGEConv((-1, -1), hidden_channels),
            }, aggr='mean')
            self.convs.append(conv)
        self.dropout = dropout
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x = {tp: self.project[tp](x_dict[tp]) for tp in x_dict}
        for conv in self.convs:
            x = conv(x, edge_index_dict)
            x = {tp: F.relu(x_tp) for tp, x_tp in x.items()}
            x = {tp: F.dropout(x_tp, p=self.dropout, training=self.training)
                 for tp, x_tp in x.items()}
        return self.classifier(x['lncRNA'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Split train/val/test (for lncRNA only)
num_lnc_nodes = data['lncRNA'].num_nodes
perm = torch.randperm(num_lnc_nodes)
n_train = int(0.6 * num_lnc_nodes)
n_val   = int(0.2 * num_lnc_nodes)
train_idx = perm[:n_train].to(device)
val_idx   = perm[n_train:n_train+n_val].to(device)
test_idx  = perm[n_train+n_val:].to(device)

num_classes = int(data['lncRNA'].y.max().item()) + 1
model = HeteroGNN(
    data,
    hidden_channels=64,
    out_channels=num_classes,
    num_layers=2,
    dropout=0.5
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

best_val = 0.0
patience = 5
patience_cnt = 0
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = F.cross_entropy(out[train_idx], data['lncRNA'].y[train_idx])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        pred = out.argmax(dim=-1)
        train_acc = (pred[train_idx] == data['lncRNA'].y[train_idx]).float().mean().item()
        val_acc   = (pred[val_idx]   == data['lncRNA'].y[val_idx]).float().mean().item()
        test_acc  = (pred[test_idx]  == data['lncRNA'].y[test_idx]).float().mean().item()
    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

    if val_acc > best_val:
        best_val = val_acc
        best_state = model.state_dict()
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= patience:
            print("Early stopping.")
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    out = model(data.x_dict, data.edge_index_dict)
    pred = out.argmax(dim=-1)
    final_train = (pred[train_idx] == data['lncRNA'].y[train_idx]).float().mean().item()
    final_val   = (pred[val_idx]   == data['lncRNA'].y[val_idx]).float().mean().item()
    final_test  = (pred[test_idx]  == data['lncRNA'].y[test_idx]).float().mean().item()
print(f"Final Accuracies | Train: {final_train:.4f} | Val: {final_val:.4f} | Test: {final_test:.4f}")
