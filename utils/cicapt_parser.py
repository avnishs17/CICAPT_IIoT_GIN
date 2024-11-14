import pandas as pd
import dgl
import torch
import os
import pickle
import json
import networkx as nx
from collections import defaultdict
import re
import xxhash

class ProvenanceDataProcessor:
    def __init__(self):
        # Maps for different feature types
        self.node_type_map = {}
        self.edge_type_map = {}
        self.exe_map = {}
        self.path_map = {}
        self.address_map = {}
        self.protocol_map = {}
        
    def _hash_string(self, value):
        """Hash string values like in wget_parser"""
        hasher = xxhash.xxh64()
        hasher.update(str(value))
        return hasher.intdigest()
    
    def _encode_categorical(self, value, mapping_dict):
        if value not in mapping_dict:
            mapping_dict[value] = len(mapping_dict)
        return mapping_dict[value]
    
    def _process_timestamp(self, timestamp):
        """Process timestamps similar to trace_parser"""
        try:
            return float(timestamp)
        except (ValueError, TypeError):
            return 0.0
    
    def _process_node_features(self, node):
        features = {}
        
        # Process node type (like streamspot_parser lines 1-54)
        features['type'] = self._encode_categorical(node['type'], self.node_type_map)
        
        # Process attributes
        attrs = node.get('attributes', {})
        
        # Process numeric fields (like trace_parser lines 144-190)
        numeric_fields = ['pid', 'ppid', 'uid', 'gid', 'euid', 'egid', 'tgid', 'fd']
        for field in numeric_fields:
            if field in attrs:
                try:
                    features[field] = float(attrs[field])
                except (ValueError, TypeError):
                    features[field] = -1.0
        
        # Process timestamps (like wget_parser lines 577-597)
        time_fields = ['seen time', 'start time', 'epoch']
        for field in time_fields:
            if field in attrs:
                features[f"{field}_ts"] = self._process_timestamp(attrs[field])
        
        # Process paths and executables (like trace_parser lines 191-220)
        if 'exe' in attrs:
            features['exe_id'] = self._encode_categorical(attrs['exe'], self.exe_map)
        
        if 'path' in attrs:
            features['path_id'] = self._encode_categorical(attrs['path'], self.path_map)
        
        # Process network fields (like wget_parser lines 400-450)
        if 'remote address' in attrs:
            features['address_id'] = self._encode_categorical(attrs['remote address'], self.address_map)
            
        if 'protocol' in attrs:
            features['protocol_id'] = self._encode_categorical(attrs['protocol'], self.protocol_map)
            
        if 'remote port' in attrs:
            try:
                features['port'] = float(attrs['remote port'])
            except (ValueError, TypeError):
                features['port'] = -1.0
                
        return features

    def _process_edge_features(self, edge):
        features = {}
        
        # Process edge type (like streamspot_parser)
        features['type'] = self._encode_categorical(edge['type'], self.edge_type_map)
        
        attrs = edge.get('attributes', {})
        
        # Process timestamps (like trace_parser)
        if 'time' in attrs:
            features['timestamp'] = self._process_timestamp(attrs['time'])
            
        # Process IDs (like wget_parser)
        if 'event id' in attrs:
            try:
                features['event_id'] = float(attrs['event id'])
            except (ValueError, TypeError):
                features['event_id'] = -1.0
        
        # Process operation (like trace_parser)
        if 'operation' in attrs:
            features['operation'] = self._hash_string(attrs['operation'])
            
        return features

    def process_graph(self, nodes, edges):
        g = nx.DiGraph()
        
        # Add nodes with features
        print("Processing nodes...")
        for node in nodes:
            features = self._process_node_features(node)
            g.add_node(node['id'], **features)
            
        # Add edges with features
        print("Processing edges...")
        for edge in edges:
            features = self._process_edge_features(edge)
            g.add_edge(edge['from'], edge['to'], **features)
            
        return g

    def convert_to_dgl(self, nx_graph):
        # Convert NetworkX graph to DGL graph
        dgl_graph = dgl.from_networkx(nx_graph, 
                                    node_attrs=list(next(iter(nx_graph.nodes(data=True)))[1].keys()),
                                    edge_attrs=list(next(iter(nx_graph.edges(data=True)))[2].keys()))
        return dgl_graph