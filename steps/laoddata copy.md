import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
import torch
import dgl

# Define feature lists
common_features = ['pid', 'source']
node_features = [
    'uid', 'egid', 'exe', 'gid', 'euid', 'name',
    'seen time', 'ppid', 'command line', 'start time',
    'path', 'subtype', 'permissions', 'epoch', 'version',
    'remote port', 'protocol', 'remote address', 'tgid', 'fd'
]
edge_features = ['operation', 'time', 'event id', 'flags', 'mode']

def process_provenance_data(csv_path, output_path, is_phase2=False):
    """
    Process provenance data from CSV and save as JSON
    """
    print(f"Processing {csv_path}...")
    
    # Read CSV file
    df = pd.read_csv(csv_path, low_memory=False)
    
    # First pass: Collect all nodes and their features
    print("Collecting nodes...")
    nodes = {}
    node_labels = {}
    
    for idx, row in df.iterrows():
        if pd.notna(row['id']):  # Node row
            node_id = row['id']
            
            # Extract node features
            node_features_dict = {}
            
            # Process common features
            for feat in common_features:
                if feat in row and pd.notna(row[feat]):
                    node_features_dict[feat] = row[feat]
            
            # Process node-specific features
            for feat in node_features:
                if feat in row and pd.notna(row[feat]):
                    node_features_dict[feat] = row[feat]
            
            nodes[node_id] = node_features_dict
            
            # Store label if Phase2
            if is_phase2 and 'label' in row and pd.notna(row['label']):
                node_labels[node_id] = int(row['label'])
    
    # Second pass: Collect edges and create graphs
    print("Creating graphs...")
    graphs = []
    current_nodes = {}
    current_edges = []
    current_label = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        if pd.isna(row['id']):  # Edge row
            if pd.notna(row['from']) and pd.notna(row['to']):
                src_id = row['from']
                dst_id = row['to']
                
                # Skip if either source or destination node is missing
                if src_id not in nodes or dst_id not in nodes:
                    continue
                
                # Add nodes if not in current graph
                if src_id not in current_nodes:
                    current_nodes[src_id] = nodes[src_id]
                if dst_id not in current_nodes:
                    current_nodes[dst_id] = nodes[dst_id]
                
                # Extract edge features
                edge_features_dict = {}
                for feat in edge_features:
                    if feat in row and pd.notna(row[feat]):
                        edge_features_dict[feat] = row[feat]
                
                # Create edge
                edge = {
                    'from': src_id,
                    'to': dst_id,
                    'features': edge_features_dict
                }
                current_edges.append(edge)
                
                # Update label for Phase2
                if is_phase2:
                    for node_id in [src_id, dst_id]:
                        if node_id in node_labels and node_labels[node_id] == 1:
                            current_label = 1
                
                # Create graph when edge is processed
                graph = {
                    'nodes': current_nodes.copy(),
                    'edges': current_edges.copy(),
                    'label': current_label
                }
                graphs.append(graph)
                
                # Reset for next graph
                current_nodes = {}
                current_edges = []
                current_label = 0
    
    # Save processed data
    print(f"Saving {len(graphs)} graphs to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'graphs': graphs}, f)
    
    print(f"Processing complete. Total graphs: {len(graphs)}")

if __name__ == "__main__":
    # Process Phase1 (benign only)
    process_provenance_data(
        csv_path='./data/Phase1_Provenance.csv',
        output_path='./data/processed/phase1_graphs.json',
        is_phase2=False
    )
    
    # Process Phase2 (benign + malicious)
    process_provenance_data(
        csv_path='./data/Phase2_Provenance.csv',
        output_path='./data/processed/phase2_graphs.json',
        is_phase2=True
    )