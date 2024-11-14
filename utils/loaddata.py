import pickle as pkl
import time
import torch.nn.functional as F
import dgl
import networkx as nx
import json
from tqdm import tqdm
import os
import pandas as pd

# Define feature lists
common_features = ['pid', 'source']
node_features = [
    'uid', 'egid', 'exe', 'gid', 'euid', 'name',
    'seen time', 'ppid', 'command line', 'start time',
    'path', 'subtype', 'permissions', 'epoch', 'version',
    'remote port', 'protocol', 'remote address', 'tgid', 'fd'
]
edge_features = ['operation', 'time', 'event id', 'Flag', 'mode']

class EntityLevelDataset(dgl.data.DGLDataset):
    def __init__(self, name, csv_path, json_path, label_column=None):
        """
        Initialize the EntityLevelDataset.

        Parameters:
        - name (str): Name of the dataset.
        - csv_path (str): Path to the CSV file.
        - json_path (str): Path to save the processed JSON data.
        - label_column (str or None): Column name for labels in Phase2. None for Phase1.
        """
        super(EntityLevelDataset, self).__init__(name=name)
        self.csv_path = csv_path
        self.json_path = json_path
        self.label_column = label_column  # Column name for labels; None for Phase1
        self.graphs = []
        self.labels = []
        self.process()

    def process(self):
        print(f'Loading dataset from {self.csv_path}...')
        data = pd.read_csv(self.csv_path)

        # Initialize structures to hold nodes and edges for a single graph
        nodes = {}
        edges = []
        current_label = 0  # Default label

        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            if pd.notna(row['id']):
                # It's a node
                node_id = row['id']
                nodes[node_id] = {
                    'pid': row['pid'],
                    'source': row['source'],
                    'uid': row['uid'],
                    'egid': row['egid'],
                    'exe': row['exe'],
                    'gid': row['gid'],
                    'euid': row['euid'],
                    'name': row['name'],
                    'seen time': row['seen time'],
                    'ppid': row['ppid'],
                    'command line': row['command line'],
                    'start time': row['start time'],
                    'path': row['path'],
                    'subtype': row['subtype'],
                    'permissions': row['permissions'],
                    'epoch': row['epoch'],
                    'version': row['version'],
                    'remote port': row['remote port'],
                    'protocol': row['protocol'],
                    'remote address': row['remote address'],
                    'tgid': row['tgid'],
                    'fd': row['fd']
                }
                # If Phase2 and label_column is specified, capture the label
                if self.label_column and pd.notna(row.get(self.label_column, 0)):
                    current_label = int(row[self.label_column])
            else:
                # It's an edge
                edges.append({
                    'from': row['from'],
                    'to': row['to'],
                    'operation': row['operation'],
                    'time': row['time'],
                    'event id': row['event id'],
                    'Flag': row['Flag'],
                    'mode': row['mode']
                })
                # Assuming each edge row signifies the end of a graph
                # This condition should be adjusted based on actual data structure
                self._create_graph(nodes, edges, current_label)
                nodes = {}
                edges = []
                current_label = 0  # Reset label for next graph

        # After iterating, create the last graph if any
        if nodes or edges:
            self._create_graph(nodes, edges, current_label)

    def _create_graph(self, nodes, edges, label):
        """
        Create a DGL graph from nodes and edges and assign the label.

        Parameters:
        - nodes (dict): Node features.
        - edges (list): Edge features.
        - label (int): Label of the graph.
        """
        # Create NetworkX graph
        G = nx.DiGraph()

        # Add nodes with features
        for node_id, attrs in nodes.items():
            G.add_node(node_id, **attrs)

        # Add edges with features
        for edge in edges:
            G.add_edge(
                edge['from'], edge['to'],
                operation=edge['operation'],
                time=edge['time'],
                event_id=edge['event id'],
                Flag=edge['Flag'],
                mode=edge['mode']
            )

        # Convert to DGL graph
        dgl_graph = dgl.from_networkx(
            G,
            node_attrs=common_features + node_features,
            edge_attrs=edge_features
        )

        self.graphs.append(dgl_graph)
        self.labels.append(label)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


def load_rawdata_entity_level(phase1_csv, phase2_csv, output_dir='./data/entity_level', label_column='label'):
    """
    Process Phase1 and Phase2 CSV files and save the processed data.

    Parameters:
    - phase1_csv (str): Path to Phase1 CSV file.
    - phase2_csv (str): Path to Phase2 CSV file.
    - output_dir (str): Directory to save processed data.
    - label_column (str): Column name for labels in Phase2.

    Returns:
    - dict: Dictionary containing processed Phase1 and Phase2 datasets.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Process Phase1 (All labels are 0)
    phase1_pkl = os.path.join(output_dir, 'phase1.pkl')
    if os.path.exists(phase1_pkl):
        print('Loading processed Phase1 dataset...')
        phase1_data = pkl.load(open(phase1_pkl, 'rb'))
    else:
        phase1_data = EntityLevelDataset(
            name='entity_phase1',
            csv_path=phase1_csv,
            json_path=None,  # Not used since we're using pickle
            label_column=None  # No label column for Phase1
        )
        pkl.dump(phase1_data, open(phase1_pkl, 'wb'))
        print('Phase1 dataset processed and saved.')

    # Process Phase2 (Labels are defined in the csv)
    phase2_pkl = os.path.join(output_dir, 'phase2.pkl')
    if os.path.exists(phase2_pkl):
        print('Loading processed Phase2 dataset...')
        phase2_data = pkl.load(open(phase2_pkl, 'rb'))
    else:
        phase2_data = EntityLevelDataset(
            name='entity_phase2',
            csv_path=phase2_csv,
            json_path=None,  # Not used since we're using pickle
            label_column=label_column  # Specify the label column
        )
        pkl.dump(phase2_data, open(phase2_pkl, 'wb'))
        print('Phase2 dataset processed and saved.')

    return {'phase1': phase1_data, 'phase2': phase2_data}


def load_batch_level_entity_dataset(phase1_csv, phase2_csv, label_column='label'):
    """
    Load and combine Phase1 and Phase2 datasets, and prepare feature dimensions.

    Parameters:
    - phase1_csv (str): Path to Phase1 CSV file.
    - phase2_csv (str): Path to Phase2 CSV file.
    - label_column (str): Column name for labels in Phase2.

    Returns:
    - dict: Dictionary containing combined dataset and metadata.
    """
    dataset = load_rawdata_entity_level(phase1_csv, phase2_csv, label_column=label_column)

    # Combine datasets
    full_dataset = dataset['phase1'].graphs + dataset['phase2'].graphs
    full_labels = dataset['phase1'].labels + dataset['phase2'].labels

    # Determine feature dimensions (assuming categorical features)
    # Here, we'll compute maximum indices for one-hot encoding
    node_feature_dims = {}
    edge_feature_dims = {}

    for feature in common_features + node_features:
        max_val = 0
        for graph in full_dataset:
            if feature in graph.ndata:
                max_feature = graph.ndata[feature].max().item()
                max_val = max(max_val, max_feature)
        node_feature_dims[feature] = max_val + 1

    for feature in edge_features:
        max_val = 0
        for graph in full_dataset:
            if feature in graph.edata:
                max_feature = graph.edata[feature].max().item()
                max_val = max(max_val, max_feature)
        edge_feature_dims[feature] = max_val + 1

    # Prepare indices
    train_index = [i for i, label in enumerate(full_labels) if label == 0]
    full_index = list(range(len(full_dataset)))

    print('[n_graph, node_feat_dims, edge_feat_dims]: [{}, {}, {}]'.format(
        len(full_dataset),
        node_feature_dims,
        edge_feature_dims
    ))

    return {
        'dataset': full_dataset,
        'labels': full_labels,
        'train_index': train_index,
        'full_index': full_index,
        'node_feat_dims': node_feature_dims,
        'edge_feat_dims': edge_feature_dims
    }


def transform_graph_entity(g, node_feature_dims, edge_feature_dims):
    """
    Transform graph features using one-hot encoding.

    Parameters:
    - g (DGLGraph): The graph to transform.
    - node_feature_dims (dict): Dimensions for node features.
    - edge_feature_dims (dict): Dimensions for edge features.

    Returns:
    - DGLGraph: Transformed graph with one-hot encoded features.
    """
    new_g = g.clone()

    # One-hot encode node features
    node_attrs = {}
    for feature, dim in node_feature_dims.items():
        if feature in new_g.ndata:
            node_attrs[feature] = F.one_hot(new_g.ndata[feature].long(), num_classes=dim).float()
    new_g.ndata.update(node_attrs)

    # One-hot encode edge features
    edge_attrs = {}
    for feature, dim in edge_feature_dims.items():
        if feature in new_g.edata:
            edge_attrs[feature] = F.one_hot(new_g.edata[feature].long(), num_classes=dim).float()
    new_g.edata.update(edge_attrs)

    return new_g


def preload_entity_level_dataset(path):
    """
    Preload the entity level dataset by transforming graphs.

    Parameters:
    - path (str): Path to the dataset directory.
    """
    path = './data/' + path
    metadata_path = os.path.join(path, 'metadata.json')
    if not os.path.exists(metadata_path):
        print('Transforming entity level dataset...')
        # Implement transformation steps here if needed
        # For example, transforming all graphs with one-hot encoding
        # and saving the transformed graphs and metadata
        # This is a placeholder for actual transformation logic
        pass
    else:
        print('Metadata already exists. Skipping transformation.')


def load_metadata_entity(path):
    """
    Load metadata for the entity level dataset.

    Parameters:
    - path (str): Path to the dataset directory.

    Returns:
    - dict: Metadata dictionary.
    """
    preload_entity_level_dataset(path)
    with open('./data/' + path + '/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


def load_entity_level_dataset(path, t, n):
    """
    Load a specific part of the entity level dataset.

    Parameters:
    - path (str): Path to the dataset directory.
    - t (str): Type identifier.
    - n (str): Numeric identifier.

    Returns:
    - Any: Loaded data.
    """
    preload_entity_level_dataset(path)
    with open(f'./data/{path}/{t}{n}.pkl', 'rb') as f:
        data = pkl.load(f)
    return data