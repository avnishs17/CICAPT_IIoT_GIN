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
    Process provenance data from CSV and save as structured JSON
    """
    print(f"Processing {csv_path}...")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize data structure
    data = {
        'metadata': {
            'num_nodes': 0,
            'num_edges': 0,
            'is_phase2': is_phase2,
            'features': {
                'common_features': common_features,
                'node_features': node_features,
                'edge_features': edge_features
            }
        },
        'nodes': {},
        'edges': []
    }
    
    # Process nodes
    print("Processing nodes...")
    for idx, row in tqdm(df.iterrows(), desc="Processing nodes"):
        if pd.notna(row['id']):
            node_id = row['id']
            node_features_dict = {}
            
            # Add common features
            for feat in common_features:
                if feat in row and pd.notna(row[feat]):
                    node_features_dict[feat] = row[feat]
            
            # Add node-specific features
            for feat in node_features:
                if feat in row and pd.notna(row[feat]):
                    node_features_dict[feat] = row[feat]
            
            # Add label for Phase2
            if is_phase2 and 'label' in row and pd.notna(row['label']):
                node_features_dict['label'] = int(row['label'])
            
            data['nodes'][node_id] = node_features_dict
    
    # Process edges
    print("Processing edges...")
    for idx, row in tqdm(df.iterrows(), desc="Processing edges"):
        if pd.isna(row['id']) and pd.notna(row['from']) and pd.notna(row['to']):
            edge_features_dict = {}
            
            # Add edge features
            for feat in edge_features:
                if feat in row and pd.notna(row[feat]):
                    edge_features_dict[feat] = row[feat]
            
            edge = {
                'source': row['from'],
                'target': row['to'],
                'features': edge_features_dict
            }
            
            data['edges'].append(edge)
    
    # Update metadata
    data['metadata']['num_nodes'] = len(data['nodes'])
    data['metadata']['num_edges'] = len(data['edges'])
    
    # Save processed data
    print(f"Saving processed data to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Processing complete. Nodes: {data['metadata']['num_nodes']}, Edges: {data['metadata']['num_edges']}")

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