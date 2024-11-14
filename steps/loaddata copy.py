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
    
    # Initialize list to store all graphs
    graphs = []
    
    # First pass: Collect all nodes and their features
    print("Collecting nodes...")
    current_graph = {
        'metadata': {
            'num_nodes': 0,
            'num_edges': 0,
            'label': 0,  # Default label (benign)
            'graph_id': 0  # Will be updated for each graph
        },
        'nodes': {},
        'edges': []
    }
    
    graph_id = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        if pd.notna(row['id']):  # Node row
            node_id = row['id']
            
            # Create node entry with structured features
            node_entry = {
                'id': node_id,
                'common_features': {},
                'features': {}
            }
            
            # Process common features
            for feat in common_features:
                if feat in row and pd.notna(row[feat]):
                    node_entry['common_features'][feat] = row[feat]
            
            # Process node-specific features
            for feat in node_features:
                if feat in row and pd.notna(row[feat]):
                    node_entry['features'][feat] = row[feat]
            
            current_graph['nodes'][node_id] = node_entry
            
            # Update label if Phase2
            if is_phase2 and 'label' in row and pd.notna(row['label']) and int(row['label']) == 1:
                current_graph['metadata']['label'] = 1
                
        else:  # Edge row
            if pd.notna(row['from']) and pd.notna(row['to']):
                src_id = row['from']
                dst_id = row['to']
                
                # Skip if either source or destination node is missing
                if src_id not in current_graph['nodes'] or dst_id not in current_graph['nodes']:
                    continue
                
                # Create edge entry with structured features
                edge_entry = {
                    'source': src_id,
                    'target': dst_id,
                    'features': {}
                }
                
                # Process edge features
                for feat in edge_features:
                    if feat in row and pd.notna(row[feat]):
                        edge_entry['features'][feat] = row[feat]
                
                current_graph['edges'].append(edge_entry)
                
                # Update metadata
                current_graph['metadata']['num_edges'] += 1
                
                # When edge is processed, finalize the current graph and start a new one
                if len(current_graph['edges']) > 0:
                    # Update final metadata
                    current_graph['metadata']['num_nodes'] = len(current_graph['nodes'])
                    current_graph['metadata']['graph_id'] = graph_id
                    
                    # Add to graphs list
                    graphs.append(current_graph)
                    
                    # Start new graph
                    graph_id += 1
                    current_graph = {
                        'metadata': {
                            'num_nodes': 0,
                            'num_edges': 0,
                            'label': 0,
                            'graph_id': graph_id
                        },
                        'nodes': {},
                        'edges': []
                    }
    
    # Save processed data with structured format
    print(f"Saving {len(graphs)} graphs to {output_path}")
    output_data = {
        'dataset_metadata': {
            'total_graphs': len(graphs),
            'is_phase2': is_phase2,
            'feature_lists': {
                'common_features': common_features,
                'node_features': node_features,
                'edge_features': edge_features
            }
        },
        'graphs': graphs
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
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