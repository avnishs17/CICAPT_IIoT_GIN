import pandas as pd
import numpy as np
from typing import List, Dict, Set
import hashlib
from collections import defaultdict
import re
from urllib.parse import urlparse
import ipaddress
from pathlib import Path

class ContextualFeatureProcessor:
    """Process features while preserving their contextual meaning and uniqueness."""
    
    def __init__(self):
        self.feature_mappings = defaultdict(dict)
        self.reverse_mappings = defaultdict(dict)
        self.embedding_dims = 64  # Default embedding dimension
        
    def process_features(self, df: pd.DataFrame, 
                        common_features: List[str],
                        node_features: List[str],
                        edge_features: List[str]) -> pd.DataFrame:
        """Process all features while preserving their context."""
        processed_df = df.copy()
        
        # Process each feature type
        for feature in common_features + node_features + edge_features:
            if feature in processed_df.columns:
                if self._is_path_feature(feature):
                    processed_df = self._process_path_feature(processed_df, feature)
                elif self._is_command_feature(feature):
                    processed_df = self._process_command_feature(processed_df, feature)
                elif self._is_network_feature(feature):
                    processed_df = self._process_network_feature(processed_df, feature)
                elif self._is_identifier_feature(feature):
                    processed_df = self._process_identifier_feature(processed_df, feature)
                elif self._is_timestamp_feature(feature):
                    processed_df = self._process_timestamp_feature(processed_df, feature)
                else:
                    processed_df = self._process_generic_feature(processed_df, feature)
                    
        return processed_df
    
    def _is_path_feature(self, feature: str) -> bool:
        """Check if feature contains file system paths."""
        return feature in ['exe', 'path', 'command line']
    
    def _is_command_feature(self, feature: str) -> bool:
        """Check if feature contains command lines."""
        return feature in ['command line']
    
    def _is_network_feature(self, feature: str) -> bool:
        """Check if feature contains network information."""
        return feature in ['remote address', 'remote port', 'protocol']
    
    def _is_identifier_feature(self, feature: str) -> bool:
        """Check if feature is an identifier."""
        return feature in ['pid', 'ppid', 'tgid', 'fd', 'uid', 'gid', 'euid', 'egid']
    
    def _is_timestamp_feature(self, feature: str) -> bool:
        """Check if feature is a timestamp."""
        return 'time' in feature.lower()
    
    def _process_path_feature(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Process file system paths while preserving their structure and context.
        """
        # Extract meaningful components from paths
        df[f'{feature}_components'] = df[feature].apply(self._extract_path_components)
        
        # Create path depth feature
        df[f'{feature}_depth'] = df[feature].str.count('/')
        
        # Extract file extension if present
        df[f'{feature}_ext'] = df[feature].apply(lambda x: Path(str(x)).suffix if pd.notna(x) else '')
        
        # Create hierarchical hash for uniqueness
        df[f'{feature}_hierarchy_hash'] = df[feature].apply(self._hierarchical_path_hash)
        
        # Identify common prefixes and create boolean features
        common_prefixes = ['/usr', '/etc', '/lib', '/bin', '/sbin', '/home']
        for prefix in common_prefixes:
            df[f'{feature}_is_{prefix.replace("/", "")}'] = df[feature].str.startswith(prefix).astype(int)
        
        return df
    
    def _process_command_feature(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Process command lines while preserving their structure and arguments.
        """
        def parse_command(cmd):
            if pd.isna(cmd):
                return {}
            
            parts = re.findall(r'(?:[^\s,"]|"(?:\\.|[^"])*")+', str(cmd))
            result = {
                'base_command': Path(parts[0]).name if parts else '',
                'arg_count': len(parts) - 1,
                'has_flags': any(arg.startswith('-') for arg in parts[1:] if arg),
                'flag_types': set(arg[1] for arg in parts if arg.startswith('-')),
                'has_path_args': any('/' in arg for arg in parts[1:] if arg)
            }
            return result
        
        cmd_features = df[feature].apply(parse_command)
        
        # Extract features from parsed commands
        df[f'{feature}_base'] = cmd_features.apply(lambda x: x.get('base_command', ''))
        df[f'{feature}_arg_count'] = cmd_features.apply(lambda x: x.get('arg_count', 0))
        df[f'{feature}_has_flags'] = cmd_features.apply(lambda x: x.get('has_flags', False)).astype(int)
        df[f'{feature}_flag_hash'] = cmd_features.apply(
            lambda x: hashlib.md5(str(sorted(x.get('flag_types', set()))).encode()).hexdigest()
        )
        
        return df
    
    def _process_network_feature(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Process network-related features while preserving their structure.
        """
        if feature == 'remote address':
            df[f'{feature}_type'] = df[feature].apply(self._get_address_type)
            df[f'{feature}_prefix'] = df[feature].apply(self._get_network_prefix)
            df[f'{feature}_is_private'] = df[feature].apply(self._is_private_address).astype(int)
            df[f'{feature}_hash'] = df[feature].apply(
                lambda x: hashlib.md5(str(x).encode()).hexdigest() if pd.notna(x) else ''
            )
        elif feature == 'protocol':
            # Create binary features for common protocols
            common_protocols = ['tcp', 'udp', 'icmp', 'http', 'https']
            for proto in common_protocols:
                df[f'{feature}_is_{proto}'] = df[feature].str.contains(
                    proto, case=False, na=False).astype(int)
        
        return df
    
    def _process_identifier_feature(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Process identifier features while preserving their relationships."""
        if pd.api.types.is_numeric_dtype(df[feature]):
            try:
                # Handle NaN values before quantile calculation
                non_null_values = df[feature].dropna()
                if len(non_null_values) > 0:
                    # Create range-based features for numeric IDs
                    df[f'{feature}_range'] = pd.qcut(
                        non_null_values, 
                        q=10, 
                        labels=False, 
                        duplicates='drop'
                    ).reindex(df.index).fillna(-1).astype(int)
                    
                    # Create binary features for special values
                    df[f'{feature}_is_root'] = (df[feature] == 0).astype(int)
                    df[f'{feature}_is_system'] = (
                        (df[feature] > 0) & (df[feature] < 1000)
                    ).fillna(False).astype(int)
                    df[f'{feature}_is_user'] = (
                        df[feature] >= 1000
                    ).fillna(False).astype(int)
                else:
                    # Handle case where all values are NaN
                    df[f'{feature}_range'] = -1
                    df[f'{feature}_is_root'] = 0
                    df[f'{feature}_is_system'] = 0
                    df[f'{feature}_is_user'] = 0
            except Exception as e:
                print(f"Error processing identifier feature {feature}: {str(e)}")
                # Set default values if processing fails
                df[f'{feature}_range'] = -1
                df[f'{feature}_is_root'] = 0
                df[f'{feature}_is_system'] = 0
                df[f'{feature}_is_user'] = 0
        
        return df
    
    def _process_timestamp_feature(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Process timestamp features while preserving temporal relationships.
        """
        if pd.api.types.is_numeric_dtype(df[feature]):
            df[f'{feature}_dt'] = pd.to_datetime(df[feature], unit='s')
            df[f'{feature}_hour'] = df[f'{feature}_dt'].dt.hour
            df[f'{feature}_day'] = df[f'{feature}_dt'].dt.day
            df[f'{feature}_weekday'] = df[f'{feature}_dt'].dt.weekday
            df[f'{feature}_is_weekend'] = df[f'{feature}_dt'].dt.weekday.isin([5, 6]).astype(int)
            
            # Create time window features
            df[f'{feature}_window'] = pd.qcut(
                df[f'{feature}_dt'].astype(np.int64), 
                q=24, labels=False, duplicates='drop'
            ).fillna(-1).astype(int)
        
        return df
    
    def _process_generic_feature(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Process generic features while preserving their uniqueness.
        """
        if df[feature].dtype == object:
            # Create a context-preserving hash
            df[f'{feature}_hash'] = df[feature].apply(
                lambda x: hashlib.md5(str(x).encode()).hexdigest() if pd.notna(x) else ''
            )
            
            # Store original values in mapping
            unique_values = df[feature].dropna().unique()
            self.feature_mappings[feature].update({
                val: hashlib.md5(str(val).encode()).hexdigest()
                for val in unique_values
            })
            
        return df
    
    def _extract_path_components(self, path: str) -> List[str]:
        """Extract meaningful components from a file system path."""
        if pd.isna(path):
            return []
        
        path = str(path)
        components = path.split('/')
        return [c for c in components if c]
    
    def _hierarchical_path_hash(self, path: str) -> str:
        """Create a hierarchical hash for a path preserving its structure."""
        if pd.isna(path):
            return ''
            
        components = self._extract_path_components(path)
        hierarchical_hash = ''
        
        for i in range(len(components)):
            subset = '/'.join(components[:i+1])
            hierarchical_hash += hashlib.md5(subset.encode()).hexdigest()[:8]
            
        return hierarchical_hash
    
    def _get_address_type(self, addr: str) -> str:
        """Determine the type of network address."""
        if pd.isna(addr):
            return 'unknown'
            
        try:
            ip = ipaddress.ip_address(addr)
            return 'ipv4' if isinstance(ip, ipaddress.IPv4Address) else 'ipv6'
        except ValueError:
            return 'hostname'
    
    def _get_network_prefix(self, addr: str) -> str:
        """Get network prefix for an IP address."""
        if pd.isna(addr):
            return ''
            
        try:
            ip = ipaddress.ip_address(addr)
            if isinstance(ip, ipaddress.IPv4Address):
                return str(ipaddress.ip_network(f"{addr}/24", strict=False))
            return str(ipaddress.ip_network(f"{addr}/64", strict=False))
        except ValueError:
            return ''
    
    def _is_private_address(self, addr: str) -> bool:
        """Check if an IP address is private."""
        if pd.isna(addr):
            return False
            
        try:
            ip = ipaddress.ip_address(addr)
            return ip.is_private
        except ValueError:
            return False

# # Example usage
# if __name__ == "__main__":
#     processor = ContextualFeatureProcessor()
    
#     # Define feature sets
#     common_features = ['pid', 'source']
#     node_features = ['uid', 'egid', 'exe', 'gid', 'euid', 'name',
#                     'seen time', 'ppid', 'command line', 'start time',
#                     'path', 'subtype', 'permissions', 'epoch', 'version',
#                     'remote port', 'protocol', 'remote address', 'tgid', 'fd']
#     edge_features = ['operation', 'time', 'event id', 'Flag', 'mode']
    
#     # Process features
#     processed_df = processor.process_features(df, common_features, node_features, edge_features)


