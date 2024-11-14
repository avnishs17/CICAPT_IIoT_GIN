import pandas as pd
import json
from context_preprocess import ContextualFeatureProcessor
from pathlib import Path
import networkx as nx
from typing import Dict, List, Tuple
import glob
import os
import numpy as np

class ProvenanceGraphProcessor:
    def __init__(self):
        self.feature_processor = ContextualFeatureProcessor()
        
    def process_single_csv(self, 
                         csv_path: str, 
                         output_dir: str,
                         phase: str):
        """Process a single CSV file into a provenance graph format."""
        
        file_name = Path(csv_path).stem
        print(f"Processing {file_name}...")
        
        try:
            # Read CSV file with low_memory=False to handle mixed types
            df = pd.read_csv(csv_path, low_memory=False)
            
            # Fill NaN values appropriately
            df = df.fillna({
                'pid': -1,
                'ppid': -1,
                'uid': -1,
                'gid': -1,
                'euid': -1,
                'egid': -1,
                'tgid': -1,
                'fd': -1,
                'remote port': -1
            })
            
            # Fill remaining NaN values with empty strings for string columns
            df = df.fillna('')
            
            # Convert numeric columns to appropriate types
            numeric_columns = ['pid', 'ppid', 'uid', 'gid', 'euid', 'egid', 'tgid', 'fd', 'remote port']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].astype('Int64')  # Using Int64 to handle NaN values
            
            # Define feature sets
            common_features = ['pid', 'source']
            node_features = ['uid', 'egid', 'exe', 'gid', 'euid', 'name',
                           'seen time', 'ppid', 'command line', 'start time',
                           'path', 'subtype', 'permissions', 'epoch', 'version',
                           'remote port', 'protocol', 'remote address', 'tgid', 'fd']
            edge_features = ['operation', 'time', 'event id', 'Flag', 'mode']
            
            # Process features
            processed_df = self.feature_processor.process_features(
                df, common_features, node_features, edge_features)
            
            # Create provenance graph
            graph = self._create_provenance_graph(processed_df)
            
            # Create output filename
            output_path = Path(output_dir) / f"{phase}_{file_name}_provenance.json"
            
            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Save to JSON
            self._save_graph_to_json(graph, output_path)
            print(f"Successfully processed {file_name}")
            
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            # Print more detailed error information
            import traceback
            print(traceback.format_exc())
        
    def _create_provenance_graph(self, df: pd.DataFrame) -> Dict:
        """Create a provenance graph structure from processed dataframe."""
        
        graph = nx.DiGraph()
        
        for _, row in df.iterrows():
            # Create unique node IDs
            source_node_id = self._create_node_id(row, 'source')
            target_node_id = self._create_node_id(row, 'target')
            
            # Add nodes if they don't exist
            if not graph.has_node(source_node_id):
                graph.add_node(source_node_id, **self._extract_node_attributes(row, 'source'))
            if not graph.has_node(target_node_id):
                graph.add_node(target_node_id, **self._extract_node_attributes(row, 'target'))
            
            # Add edge
            edge_attrs = self._extract_edge_attributes(row)
            graph.add_edge(source_node_id, target_node_id, **edge_attrs)
        
        # Add graph statistics
        stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'is_directed': graph.is_directed(),
            'is_connected': nx.is_weakly_connected(graph)
        }
        
        # Convert NetworkX graph to dictionary format
        graph_dict = {
            'metadata': {
                'statistics': stats
            },
            'nodes': [{'id': node, **data} for node, data in graph.nodes(data=True)],
            'edges': [{'source': u, 'target': v, **data} 
                     for u, v, data in graph.edges(data=True)]
        }
        
        return graph_dict
    
    def _create_node_id(self, row: pd.Series, node_type: str) -> str:
        """Create a unique node identifier."""
        try:
            if node_type == 'source':
                pid = str(row['pid']) if pd.notna(row['pid']) else 'unknown'
                exe_hash = row.get('exe_hierarchy_hash', 'unknown')
                return f"{pid}_{exe_hash}"
            else:
                path_hash = row.get('path_hierarchy_hash', 'unknown')
                cmd_hash = row.get('command_line_hash', 'unknown')
                return f"{path_hash}_{cmd_hash}"
        except Exception as e:
            # Return a fallback ID if there's an error
            return f"{node_type}_unknown_{hash(str(row))}"
    
    def _extract_node_attributes(self, row: pd.Series, node_type: str) -> Dict:
        """Extract relevant node attributes from a row."""
        attributes = {}
        
        try:
            # Common attributes
            for col in row.index:
                if col.endswith(('_hash', '_depth', '_type', '_is_root', '_range')):
                    # Convert numpy/pandas types to native Python types
                    val = row[col]
                    if pd.isna(val):
                        attributes[col] = None
                    elif isinstance(val, (np.integer, np.floating)):
                        attributes[col] = int(val) if val.is_integer() else float(val)
                    else:
                        attributes[col] = str(val)
                
                # Include original values for important fields
                if col in ['exe', 'path', 'command line', 'pid', 'ppid']:
                    val = row[col]
                    if pd.isna(val):
                        attributes[f"original_{col}"] = None
                    else:
                        attributes[f"original_{col}"] = str(val)
            
            # Add node type
            attributes['node_type'] = node_type
            
        except Exception as e:
            print(f"Error extracting node attributes: {str(e)}")
            attributes['error'] = str(e)
        
        return attributes
    
    def _extract_edge_attributes(self, row: pd.Series) -> Dict:
        """Extract relevant edge attributes from a row."""
        attributes = {}
        
        # Include relevant edge features
        edge_features = ['operation', 'time', 'event id', 'Flag', 'mode']
        for feature in edge_features:
            if feature in row:
                attributes[feature] = row[feature]
        
        # Add processed time features if available
        time_features = [col for col in row.index if col.startswith('time_')]
        for feature in time_features:
            attributes[feature] = row[feature]
        
        return attributes
    
    def _save_graph_to_json(self, graph: Dict, output_path: str):
        """Save the provenance graph to a JSON file."""
        output_file = Path(output_path)
        
        with open(output_file, 'w') as f:
            json.dump(graph, f, indent=2)
        
        # Print file size information
        file_size = output_file.stat().st_size / (1024 * 1024)  # Convert to MB
        print(f"Saved {output_file.name} ({file_size:.2f} MB)")

# Example usage
if __name__ == "__main__":
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    processor = ProvenanceGraphProcessor()
    
    # Define paths for the specific files
    phase1_file = os.path.join(project_root, 'data', 'Phase1_Provenance.csv')
    phase2_file = os.path.join(project_root, 'data', 'Phase2_Provenance.csv')
    output_dir = os.path.join(project_root, 'data', 'processed')
    
    # Process individual files
    processor.process_single_csv(phase1_file, output_dir, phase="phase1")
    processor.process_single_csv(phase2_file, output_dir, phase="phase2")
    
