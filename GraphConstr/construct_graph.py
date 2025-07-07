# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 18:30:32 2025

@author: Huibo, Minfei
"""

"""
Construction of Simplified Heterogeneous Biological Regulatory Graph

This script constructs a simplified heterogeneous graph integrating regulatory interactions 
between lncRNAs, miRNAs, and genes, based on two input datasets. The graph is suitable for 
graph neural network (GNN) modeling using PyTorch Geometric and is also visualized as an 
interactive network using PyVis.

========================
Input Data Sources:
------------------------
1. Table_1.xlsx
   - Columns:
     - 'Regulator': source entity (lncRNA)
     - 'Target': target entity (miRNA or gene)
     - 'TargetType': type of the target entity ('gene', 'miRNA', or variant)
     - 'regulatory_Mechanism': regulation type

2. protein_interactions.csv
   - Columns:
     - '#node1', 'node2': interacting genes (proteins)

========================
Node Types:
------------------------
- 'lncRNA'
- 'miRNA'
- 'gene'

========================
Edge Types:
------------------------
- ('lncRNA', 'regulates_gene', 'gene')
- ('lncRNA', 'regulates_miRNA', 'miRNA')
- ('gene', 'interacts', 'gene')

========================
Outputs:
------------------------
1. heterogeneous_graph.pt
2. heterogeneous_graph.html
"""

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
from pyvis.network import Network

table_df_path = "Table_1.xlsx"
ppi_df_path = "protein_interactions.xlsx"
graph_dir = "heterogeneous_graph.pt"

class GraphConstructor():
    def __init__(self, table_df_path, ppi_df_path, graph_dir):
        self.node_maps = defaultdict(dict)
        self.node_counts = defaultdict(int)
        self.edge_list = defaultdict(list)
        self.edge_attributes = defaultdict(list)
        self.table_df = pd.read_excel(table_df_path)
        self.ppi_df = pd.read_excel(ppi_df_path)
        self.df = self.load_data(table_df_path, ppi_df_path)
        self.graph_dir = graph_dir
        self.type_alias = {
                'gene': 'gene',
                'pcg': 'gene',
                'protein-coding gene': 'gene',
                'mirna': 'miRNA',
                'microrna': 'miRNA'
            }
        self.gene_gene_edges = []
    def set_gene_gene_edges(self, gene_gene_edges):
        self.gene_gene_edges = gene_gene_edges
        
        # ========== Step 1: Load Data ==========
    def load_data(self, table_df_path, ppi_df_path):
        table_df = pd.read_excel(table_df_path)
        ppi_df = pd.read_excel(ppi_df_path)

        for df in [table_df, ppi_df]:
            df.columns = [col.strip() for col in df.columns]
        return df

    def get_or_add_node(self, entity_type, name):
        if name not in self.node_maps[entity_type]:
            self.node_maps[entity_type][name] = self.node_counts[entity_type]
            self.node_counts[entity_type] += 1
        return self.node_maps[entity_type][name]

    # ========== Step 3: Process lncRNA → miRNA/gene edges ==========
    def process_lncRNA_to_miRNA_gene_edges(self):

        for _, row in self.table_df.iterrows():
            src = str(row['Regulator']).strip()
            tgt = str(row['Target']).strip()
            raw_type = str(row['TargetType']).strip().lower()
            tgt_type = self.type_alias.get(raw_type, None)
            mechanism = str(row.get('regulatory_Mechanism', '')).strip()

            if src and tgt and tgt_type in ['gene', 'miRNA']:
                src_idx = self.get_or_add_node('lncRNA', src)
                tgt_idx = self.get_or_add_node(tgt_type, tgt)
                edge_type = f"regulates_{tgt_type}"
                self.edge_list[('lncRNA', edge_type, tgt_type)].append((src_idx, tgt_idx))
                self.edge_attributes[('lncRNA', edge_type, tgt_type)].append(mechanism or "regulation")

        print(f"lncRNA → gene edges: {len(self.edge_list.get(('lncRNA', 'regulates_gene', 'gene'), []))}")
        print(f"lncRNA → miRNA edges: {len(self.edge_list.get(('lncRNA', 'regulates_miRNA', 'miRNA'), []))}")

    # ========== Step 4: Process gene-gene interactions ==========
    def process_gene_gene_interactions(self):
        gene_gene_edges = []
        for _, row in self.ppi_df.iterrows():
            g1 = str(row['node1']).strip()
            g2 = str(row['node2']).strip()
            if g1 and g2:
                g1_idx = self.get_or_add_node('gene', g1)
                g2_idx = self.get_or_add_node('gene', g2)
                gene_gene_edges.append((g1_idx, g2_idx))

        self.set_gene_gene_edges(gene_gene_edges=gene_gene_edges)
        print(f"gene ↔ gene edges: {len(gene_gene_edges)}")

    # ========== Step 5: Build HeteroData ==========
    def build_data(self):
        data = HeteroData()

        for node_type, mapping in self.node_maps.items():
            data[node_type].x = torch.ones((len(mapping), 1))

        for (src_type, rel_type, tgt_type), edges in self.edge_list.items():
            data[(src_type, rel_type, tgt_type)].edge_index = torch.tensor(edges, dtype=torch.long).T

        if self.gene_gene_edges:
            data[('gene', 'interacts', 'gene')].edge_index = torch.tensor(self.gene_gene_edges, dtype=torch.long).T

        torch.save(data, self.graph_dir)
        print(data)

    # ========== Step 6: Visualization ==========
    def visualize(self):
        net = Network(height="800px", width="100%", directed=True)

        node_colors = {
            'lncRNA': 'skyblue',
            'miRNA': 'plum',
            'gene': 'lightgreen'
        }

        # Add all nodes
        for node_type, mapping in self.node_maps.items():
            for name, idx in mapping.items():
                net.add_node(
                    f"{node_type}_{idx}",
                    label=name,
                    color=node_colors.get(node_type, 'gray'),
                    title=node_type
                )

        # Add lncRNA → gene/miRNA edges with mechanism label
        for (src_type, rel_type, tgt_type), edges in self.edge_list.items():
            mechanisms = self.edge_attributes.get((src_type, rel_type, tgt_type), [])
            for i, (src_idx, tgt_idx) in enumerate(edges):
                label = mechanisms[i] if i < len(mechanisms) else rel_type.replace("regulates_", "")
                net.add_edge(
                    f"{src_type}_{src_idx}",
                    f"{tgt_type}_{tgt_idx}",
                    label=label,
                    title=f"{label} ({src_type} → {tgt_type})"
                )

        # Add gene-gene interaction edges
        for src_idx, tgt_idx in self.gene_gene_edges:
            net.add_edge(
                f"gene_{src_idx}",
                f"gene_{tgt_idx}",
                title="gene-gene interaction"
            )

        net.force_atlas_2based()
        net.write_html("heterogeneous_graph.html")

    def run_graph_construction_pipeline(self):
        self.process_lncRNA_to_miRNA_gene_edges()
        self.process_gene_gene_interactions()
        self.build_data()
        self.visualize()

table_df_path = "Table_1.xlsx"
ppi_df_path = "protein_interactions.xlsx"
graph_dir = "heterogeneous_graph.pt"

constructor = GraphConstructor(table_df_path, ppi_df_path, graph_dir)
constructor.run_graph_construction_pipeline()

