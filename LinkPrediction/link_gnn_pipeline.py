import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, Node2Vec
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import random
import logging

# 1. Load network and label data
def load_data(net_path, label_path):
    net_df = pd.read_csv(net_path)
    label_df = pd.read_excel(label_path)
    return net_df, label_df

# 2. Build node mappings and edge index
def build_graph(net_df):
    all_nodes = pd.concat([net_df['Regulator'], net_df['Target']]).unique()
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    edge_index = torch.tensor([
        [node_to_idx[r] for r in net_df['Regulator']],
        [node_to_idx[t] for t in net_df['Target']]
    ], dtype=torch.long)
    node_types = {}
    for _, row in net_df.iterrows():
        node_types[node_to_idx[row['Regulator']]] = row['RegulatorType']
        node_types[node_to_idx[row['Target']]] = row['TargetType']
    return edge_index, node_to_idx, idx_to_node, node_types

# 3. Generate Node2Vec embeddings
def generate_node2vec(edge_index, num_nodes, embedding_dim=128, device='cpu'):
    print("generating embeddins")
    model = Node2Vec(
        edge_index,
        embedding_dim=embedding_dim,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1, q=1, sparse=True
    ).to(device)
    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    model.eval()
    z = model(torch.arange(num_nodes, device=device)).cpu().detach().numpy()
    print("Done generating embeddins")
    return z

# 4. Custom Dataset for Link Prediction
class LinkPredictionDataset(Dataset):
    def __init__(self, net_df, node2vec_emb, node_to_idx, node_types, transform=None):
        super().__init__(transform=transform)
        self.net_df = net_df
        self.node_to_idx = node_to_idx
        self.idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        self.node_types = node_types
        self.num_nodes = len(node_to_idx)
        self.x = torch.tensor(node2vec_emb, dtype=torch.float)
        regulators = [self.node_to_idx[reg] for reg in net_df['Regulator']]
        targets = [self.node_to_idx[target] for target in net_df['Target']]
        self.edge_index = torch.tensor([regulators, targets], dtype=torch.long)
        self.edge_labels = torch.ones(len(regulators), dtype=torch.float)
    def len(self):
        return 1
    def get(self, idx):
        data = Data(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_labels,
            num_nodes=self.num_nodes
        )
        data.node_types = self.node_types
        data.idx_to_node = self.idx_to_node
        data.node_to_idx = self.node_to_idx
        return data

def sample_lncrna_negatives(edge_index, num_nodes, node_types, num_samples):
    existing = set((u, v) for u, v in edge_index.t().tolist())
    negatives = []
    tries = 0
    max_tries = num_samples * 10
    lncrna_nodes = [idx for idx, t in node_types.items() if t == 'lncRNA']
    all_nodes = list(range(num_nodes))
    while len(negatives) < num_samples and tries < max_tries:
        src = random.choice(lncrna_nodes)
        dst = random.choice(all_nodes)
        if src == dst:
            tries += 1
            continue
        if (src, dst) not in existing and (dst, src) not in existing:
            negatives.append([src, dst])
        tries += 1
    return torch.tensor(negatives).t()

def split_edges(data, val_ratio=0.1, test_ratio=0.1):
    num_edges = data.edge_index.size(1)
    perm = torch.randperm(num_edges)
    num_val = int(val_ratio * num_edges)
    num_test = int(test_ratio * num_edges)
    val_idx = perm[:num_val]
    test_idx = perm[num_val:num_val+num_test]
    train_idx = perm[num_val+num_test:]
    train_pos = data.edge_index[:, train_idx]
    val_pos = data.edge_index[:, val_idx]
    test_pos = data.edge_index[:, test_idx]
    train_neg = sample_lncrna_negatives(data.edge_index, data.num_nodes, data.node_types, train_pos.size(1))
    val_neg = sample_lncrna_negatives(data.edge_index, data.num_nodes, data.node_types, val_pos.size(1))
    test_neg = sample_lncrna_negatives(data.edge_index, data.num_nodes, data.node_types, test_pos.size(1))
    def make_split(pos, neg):
        return Data(
            x=data.x,
            edge_index=data.edge_index,
            edge_label_index=torch.cat([pos, neg], dim=1),
            edge_label=torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))])
        )
    return make_split(train_pos, train_neg), make_split(val_pos, val_neg), make_split(test_pos, test_neg)

class GCNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=1)
    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        logits = self.decode(z, edge_label_index)
        return logits

def train(model, optimizer, loss_fn, train_data, device):
    model.train()
    optimizer.zero_grad()
    out = model(
        train_data.x.to(device),
        train_data.edge_index.to(device),
        train_data.edge_label_index.to(device)
    )
    loss = loss_fn(out, train_data.edge_label.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, device):
    model.eval()
    with torch.no_grad():
        logits = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_label_index.to(device)
        )
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = data.edge_label.cpu().numpy()
        auc_score = roc_auc_score(labels, probs)
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        precision, recall, _ = precision_recall_curve(labels, probs)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(labels, preds)
    return auc_score, acc, f1, pr_auc, cm, probs, labels

def plot_curves(probs, labels):
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()

def plot_confusion_matrix(cm):
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def print_node_embeddings(node2vec_emb, idx_to_node):
    import pandas as pd
    emb_df = pd.DataFrame(node2vec_emb, index=[idx_to_node[i] for i in range(len(idx_to_node))])
    print("\nNode2Vec Embeddings (head 30):")
    print(emb_df.head(30))
    print("\nNode2Vec Embeddings (tail 30):")
    print(emb_df.tail(30))
    return emb_df

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_path = 'data/Net_final.csv'
    label_path = 'data/Label.xlsx'
    logging.info('Loading data...')
    net_df, label_df = load_data(net_path, label_path)
    logging.info('Filtering for lncRNA-related edges...')
    net_df = net_df[(net_df['RegulatorType'] == 'lncRNA') | (net_df['TargetType'] == 'lncRNA')].reset_index(drop=True)
    logging.info('Building graph...')
    edge_index, node_to_idx, idx_to_node, node_types = build_graph(net_df)
    logging.info('Generating Node2Vec embeddings...')
    node2vec_emb = generate_node2vec(edge_index, len(node_to_idx), embedding_dim=128, device=device)
    print_node_embeddings(node2vec_emb, idx_to_node)
    logging.info('Preparing dataset...')
    dataset = LinkPredictionDataset(net_df, node2vec_emb, node_to_idx, node_types)
    data = dataset[0]
    logging.info('Splitting edges into train/val/test...')
    train_data, val_data, test_data = split_edges(data)
    logging.info('Initializing model...')
    model = GCNLinkPredictor(in_channels=train_data.x.size(1), hidden_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    epochs = 100
    logging.info('Starting training...')
    for epoch in range(1, epochs+1):
        loss = train(model, optimizer, loss_fn, train_data, device)
        if epoch % 10 == 0 or epoch == 1:
            val_auc, val_acc, val_f1, val_pr_auc, val_cm, _, _ = evaluate(model, val_data, device)
            logging.info(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
    logging.info('Evaluating on test set...')
    test_auc, test_acc, test_f1, test_pr_auc, test_cm, probs, labels = evaluate(model, test_data, device)
    print(f"\nTest AUC: {test_auc:.4f} | Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f} | Test PR AUC: {test_pr_auc:.4f}")
    print("Test Confusion Matrix:\n", test_cm)
    logging.info('Plotting ROC and PR curves...')
    plot_curves(probs, labels)
    logging.info('Plotting confusion matrix...')
    plot_confusion_matrix(test_cm)

if __name__ == '__main__':
    main()
