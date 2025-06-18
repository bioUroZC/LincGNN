import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec

# 1. Construct or read node lists and labels (example uses random generation)
np.random.seed(42)
torch.manual_seed(42)

num_lnc = 300
num_mi = 400
num_mrna = 500

lnc_ids = [f'lnc{idx}' for idx in range(num_lnc)]
lnc_labels = np.random.randint(0, 2, size=num_lnc)
lnc_df = pd.DataFrame({'id': lnc_ids, 'node_type': 'lncRNA', 'label': lnc_labels})

mi_ids = [f'mi{idx}' for idx in range(num_mi)]
mi_df = pd.DataFrame({'id': mi_ids, 'node_type': 'miRNA'})

mrna_ids = [f'mrna{idx}' for idx in range(num_mrna)]
mrna_df = pd.DataFrame({'id': mrna_ids, 'node_type': 'mRNA'})

# Merge is only for unified view, you can also process separately without merging
# node_df = pd.concat([lnc_df, mi_df, mrna_df], ignore_index=True)

# 2. Construct edge DataFrame
p_lnc_mi = 0.05
p_mi_mrna = 0.05
edges = []
for lid in lnc_ids:
    mask = np.random.rand(num_mi) < p_lnc_mi
    for j, flag in enumerate(mask):
        if flag:
            edges.append({
                'source': lid,
                'target': mi_ids[j],
                'source_type': 'lncRNA',
                'target_type': 'miRNA',
                'relation': 'interacts'
            })
for mid in mi_ids:
    mask = np.random.rand(num_mrna) < p_mi_mrna
    for j, flag in enumerate(mask):
        if flag:
            edges.append({
                'source': mid,
                'target': mrna_ids[j],
                'source_type': 'miRNA',
                'target_type': 'mRNA',
                'relation': 'regulates'
            })
edges_df = pd.DataFrame(edges)
print(edges_df.head())

# 3. Extract embeddings using Node2Vec
def build_global_edge_index(edges_df, lnc_ids, mi_ids, mrna_ids):
    num_lnc, num_mi, num_mrna = len(lnc_ids), len(mi_ids), len(mrna_ids)
    lnc_map = {v:i for i,v in enumerate(lnc_ids)}
    mi_map  = {v:(num_lnc + i) for i,v in enumerate(mi_ids)}
    mrna_map= {v:(num_lnc + num_mi + i) for i,v in enumerate(mrna_ids)}
    edge_list = []
    for _, row in edges_df.iterrows():
        src, dst = row['source'], row['target']
        if row['source_type']=='lncRNA' and row['target_type']=='miRNA':
            if src in lnc_map and dst in mi_map:
                u, v = lnc_map[src], mi_map[dst]
                edge_list.append([u,v]); edge_list.append([v,u])
        elif row['source_type']=='miRNA' and row['target_type']=='mRNA':
            if src in mi_map and dst in mrna_map:
                u, v = mi_map[src], mrna_map[dst]
                edge_list.append([u,v]); edge_list.append([v,u])
    if not edge_list:
        return torch.empty((2,0), dtype=torch.long)
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

edge_index_global = build_global_edge_index(edges_df, lnc_ids, mi_ids, mrna_ids)

from torch_geometric.nn import Node2Vec
embedding_dim = 64
node2vec = Node2Vec(
    edge_index_global,
    embedding_dim,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    p=1.0, q=1.0,
    sparse=False
)
loader = node2vec.loader(batch_size=128, shuffle=True, num_workers=0)
optimizer_n2v = torch.optim.Adam(node2vec.parameters(), lr=0.01)
device = torch.device('cpu')
node2vec = node2vec.to(device)
node2vec.train()
for epoch in range(1, 6):
    total_loss = 0.0
    for pos_rw, neg_rw in loader:
        pos_rw = pos_rw.to(device); neg_rw = neg_rw.to(device)
        loss = node2vec.loss(pos_rw, neg_rw)
        optimizer_n2v.zero_grad()
        loss.backward()
        optimizer_n2v.step()
        total_loss += loss.item()
    print(f"Node2Vec Epoch {epoch}, avg loss {total_loss/len(loader):.4f}")
node2vec.eval()
with torch.no_grad():
    embedding_global = node2vec.embedding.weight.data.cpu().numpy()
total_nodes = num_lnc + num_mi + num_mrna
assert embedding_global.shape == (total_nodes, embedding_dim)

# 4. Split embeddings by node type and generate node feature DataFrames
emb_lnc = embedding_global[0:len(lnc_ids)]
emb_mi = embedding_global[len(lnc_ids):len(lnc_ids)+len(mi_ids)]
emb_mrna = embedding_global[len(lnc_ids)+len(mi_ids):]

def make_feat_df(ids, emb_array, node_type, label_series=None):
    D = emb_array.shape[1]
    df = pd.DataFrame(emb_array, columns=[f'feat_{i}' for i in range(D)])
    df.insert(0, 'id', ids)
    df.insert(1, 'node_type', node_type)
    if label_series is not None:
        df['label'] = label_series
    return df

lnc_feat_df = make_feat_df(lnc_ids, emb_lnc, 'lncRNA', label_series=lnc_labels)
mi_feat_df = make_feat_df(mi_ids, emb_mi, 'miRNA')
mrna_feat_df = make_feat_df(mrna_ids, emb_mrna, 'mRNA')
node_feat_df = pd.concat([lnc_feat_df, mi_feat_df, mrna_feat_df], ignore_index=True)

# 5. Save as CSV
node_feat_df.to_csv('node_feat.csv', index=False)
edges_df.to_csv('edges.csv', index=False)
print("Saved node_feat.csv and edges.csv")
