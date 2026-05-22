import torch
import pandas as pd
from typing import List
from pathlib import Path
from torch_geometric.data import Data
from preprocess.GraphBuilder import GraphBuilder, _parse_uid

class CommunityGraphBuilder(GraphBuilder):
    def __init__(self, data_dir: str, graph_dir: str, graph_type: str, load_graph_if_exists: bool = False, min_nodes: int = 20, min_edges: int = 50):
        super().__init__(data_dir, graph_dir, graph_type, load_graph_if_exists)
        self.min_nodes = min_nodes
        self.min_edges = min_edges

    def build_and_save(self) -> str:
        """Build and save a list of graphs, one per community."""
        output_dir = self.graph_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        pt_path = output_dir / f'graphs_{self.graph_type}_community.pt'

        if self.load_graph_if_exists and pt_path.exists():
            print(f"  [CommunityGraphBuilder] Graphs already exist, skipping: {pt_path}")
            return str(pt_path)

        print(f"  [CommunityGraphBuilder] Building community graphs: {self.graph_type} from {self.data_dir}")

        # 1. Load user features to get communities
        feat_path = self._find_file('user_features')
        if not feat_path:
            raise FileNotFoundError(f"User features not found in {self.data_dir}")

        print(f"    Loading user features from {feat_path.name}")
        df_nodes = self._load_any_format(feat_path)
        
        if 'community' not in df_nodes.columns:
            # Try to infer it if it's raw CSV without header
            comm_col = [c for c in df_nodes.columns if 'community' in str(c).lower()]
            if comm_col:
                df_nodes['community'] = df_nodes[comm_col[0]]
            else:
                raise ValueError("community column not found in user features")

        # 2. Group users by community
        community_groups = df_nodes.groupby('community')
        
        # 3. Load all edges into memory
        edge_dfs = {}
        for etype in ['mention', 'retweet', 'reply']:
            path = self._find_file(f'edges_{etype}')
            if path:
                edf = self._load_any_format(path)
                edge_dfs[etype] = edf
            else:
                edge_dfs[etype] = pd.DataFrame(columns=[0, 1])

        graphs = []
        for comm_id, group in community_groups:
            if comm_id == -1 or comm_id == -1.0 or comm_id is None:
                continue # Skip isolated or non-community users
                
            # Filter nodes
            uids = group['user_node_id'].apply(_parse_uid).values
            if len(uids) < self.min_nodes:
                continue
                
            uid_set = set(uids)
            uid_to_idx = {uid: i for i, uid in enumerate(uids)}
            num_nodes = len(uid_to_idx)
            
            # Extract features for this community
            cols_to_extract = [
                'total', 'retweets', 'replies', 'original', 'likes',
                'followers', 'following', 'verified', 'account_date',
                'n_unique_hashtags', 'n_unique_mentions'
            ]
            
            if 'total' in group.columns:
                features_raw = group[cols_to_extract].values
            else:
                indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14]
                features_raw = group.iloc[:, indices].values
                
            features = [self._build_feature_vector(row) for row in features_raw]
            node_features = torch.tensor(features, dtype=torch.float)

            # Extract edges for this community
            comm_edges = {}
            total_edges = 0
            for etype, edf in edge_dfs.items():
                # Filter: both src and dst must be in uid_set
                if len(edf) > 0:
                    src_col, dst_col = edf.columns[0], edf.columns[1]
                    mask = edf[src_col].isin(uid_set) & edf[dst_col].isin(uid_set)
                    filtered_edf = edf[mask]
                else:
                    filtered_edf = edf
                    
                src_mapped = filtered_edf.iloc[:, 0].apply(_parse_uid).map(uid_to_idx)
                dst_mapped = filtered_edf.iloc[:, 1].apply(_parse_uid).map(uid_to_idx)
                valid = src_mapped.notna() & dst_mapped.notna()
                src_list = src_mapped[valid].astype(int).values
                dst_list = dst_mapped[valid].astype(int).values
                filtered_edf = filtered_edf[valid]
                
                if len(filtered_edf) > 0:
                    feat_raw = filtered_edf.iloc[:, 2:].values
                    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
                    feat_tensor = torch.tensor(feat_raw, dtype=torch.float)
                    if feat_tensor.size(1) > 0:
                        feat_tensor[:, 0] = torch.log1p(feat_tensor[:, 0]) / 5.0
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    feat_dim = max(0, len(edf.columns) - 2)
                    feat_tensor = torch.empty((0, feat_dim), dtype=torch.float)
                
                if etype == 'mention':
                    if edge_index.size(1) > 0:
                        edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
                        is_reciprocal = torch.tensor(
                            [1.0 if (d, s) in edge_set else 0.0
                             for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist())],
                            dtype=torch.float
                        ).unsqueeze(1)
                        feat_tensor = torch.cat([feat_tensor, is_reciprocal], dim=1)
                    else:
                        feat_tensor = torch.empty((0, feat_tensor.size(1) + 1), dtype=torch.float)
                    
                comm_edges[etype] = (edge_index, feat_tensor)
                total_edges += edge_index.size(1)
                
            if total_edges < self.min_edges:
                continue

            # Build PyG Data
            if self.graph_type == 'late_fuse':
                all_edges = []
                for etype in ['retweet', 'reply', 'mention']:
                    ei, _ = comm_edges[etype]
                    if ei.size(1) > 0:
                        all_edges.append(ei)
                if all_edges:
                    all_edge_index = torch.cat(all_edges, dim=1)
                else:
                    all_edge_index = torch.empty((2, 0), dtype=torch.long)
                    
                struct_feats = self._compute_structural_features(all_edge_index, num_nodes)
                x = torch.cat([node_features, struct_feats], dim=1)
                
                data = Data(x=x, edge_index=all_edge_index, num_nodes=num_nodes, community_id=int(comm_id))
                for etype in ['retweet', 'reply', 'mention']:
                    ei, ea = comm_edges[etype]
                    setattr(data, f'edge_index_{etype}', ei)
                    setattr(data, f'edge_attr_{etype}', ea)
                    setattr(data, f'edge_dim_{etype}', ea.size(1) if ea.size(1) > 0 else 1)
            else:
                ei, ea = comm_edges.get(self.graph_type, (torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)))
                struct_feats = self._compute_structural_features(ei, num_nodes)
                x = torch.cat([node_features, struct_feats], dim=1)
                data = Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=num_nodes, community_id=int(comm_id))
                
            graphs.append(data)
            
        print(f"  [CommunityGraphBuilder] Created {len(graphs)} community graphs.")
        torch.save(graphs, pt_path)
        return str(pt_path)
