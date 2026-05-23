"""
GraphBuilder — Convert extracted CSV files to PyG Data objects.

Reads the extraction output (user_features, edges_retweet, edges_reply, edges_mention)
and builds PyTorch Geometric Data objects for link prediction.

Supports 4 graph types: mention, retweet, reply, late_fuse.
All nodes are always present regardless of edge type.
"""

import math
import os
import pickle
import sys
from pathlib import Path

import mlflow

import pandas as pd
import torch
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.Utils import Utils

def _parse_uid(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return val  # Passthrough if not an int (fallback)

class GraphBuilder:
    """Build PyG Data objects from extracted files (CSV, Parquet, or Pickle)."""

    def __init__(self, data_dir: str, graph_dir: str, graph_type: str, load_graph_if_exists: bool = False):
        """
        Args:
            data_dir: Path to the extraction output directory
            graph_dir: Path to the directory where graphs should be saved/loaded
            graph_type: One of 'mention', 'retweet', 'reply', 'late_fuse'
            load_graph_if_exists: If True, skip building if the graph .pt file already exists
        """
        self.data_dir = Path(data_dir)
        self.graph_dir = Path(graph_dir)
        self.graph_type = graph_type
        self.load_graph_if_exists = load_graph_if_exists

    def _find_file(self, base_name: str) -> Path:
        """Find a file with parquet, pkl, or csv extension."""
        for ext in ['.parquet', '.pkl', '.csv']:
            path = self.data_dir / (base_name + ext)
            if path.exists():
                return path
        return None

    @mlflow.trace(name="build_graph")
    def build_and_save(self) -> str:
        """Build and save the graph as a .pt file. Returns the path to the saved file."""
        output_dir = self.graph_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        pt_path = output_dir / f'graph_{self.graph_type}.pt'

        if self.load_graph_if_exists and pt_path.exists():
            print(f"  [GraphBuilder] Graph already exists, skipping extraction: {pt_path}")
            return str(pt_path)

        print(f"  [GraphBuilder] Building graph: {self.graph_type} from {self.data_dir}")

        # Load node features
        node_features, uid_to_idx = self._load_user_features()
        num_nodes = len(uid_to_idx)

        if self.graph_type == 'late_fuse':
            data = self._build_late_fuse(node_features, uid_to_idx, num_nodes)
        else:
            data = self._build_single_type(node_features, uid_to_idx, num_nodes, self.graph_type)

        torch.save(data, pt_path)
        print(f"  [GraphBuilder] Saved: {pt_path} | Nodes: {data.num_nodes} | "
              f"Edges: {data.edge_index.size(1)}")
        return str(pt_path)

    # ----------------------------------------------------------------
    # Universal Loader
    # ----------------------------------------------------------------
    def _load_any_format(self, file_path: Path):
        """Load a file regardless of its extension."""
        ext = file_path.suffix.lower()
        if ext == '.parquet':
            return pd.read_parquet(file_path)
        elif ext == '.feather':
            import pyarrow.feather as feather
            return feather.read_feather(file_path)
        elif ext == '.pkl':
            with open(file_path, 'rb') as f:
                data = []
                while True:
                    try:
                        data.extend(pickle.load(f))
                    except EOFError:
                        break
                return pd.DataFrame(data)
        else:
            return pd.read_csv(file_path)

    # ----------------------------------------------------------------
    # Node features
    # ----------------------------------------------------------------

    # Feature schema: (column_name, divisor)
    # Values marked with [L] are already log1p-scaled by the extraction pipeline.
    # We only divide by a constant to bring them to roughly [0, 1].
    FEATURE_SCHEMA = [
        # Activity  [L = already log1p at extraction]
        ('total',                      10.0),   # [L]
        ('retweets',                   10.0),   # [L]
        ('replies',                    10.0),   # [L]
        ('original',                   10.0),   # [L]
        # Profile
        ('likes',                      15.0),   # [L]
        ('followers',                  15.0),   # [L]
        ('following',                  12.0),   # [L]
        ('verified',                    1.0),   # binary
        ('account_date',          7.0e8),       # seconds since 2006 epoch (~630M in 2026, covers 2006-2029)
        ('listed_count',               10.0),   # [L]
        ('reputation_score',            1.0),   # already [0, 1]
        # Content
        ('n_unique_hashtags',         100.0),   # raw count
        ('n_unique_mentions',         100.0),   # raw count
        ('n_hashtags_total',           10.0),   # [L]
        ('hashtag_entropy',             4.0),   # entropy [0, ~4]
        # Automation signals
        ('activation_age',             15.0),   # [L] seconds
        ('tweet_regularity_score',     10.0),   # [L]
        ('regularity_reliable',         1.0),   # binary
        ('tweet_avg_interval_seconds', 15.0),   # [L]
        ('daily_score',                 1.0),   # [0, 1]
        ('daily_cv_log',                5.0),   # [0, ~5]
        ('internal_tweet_density',      1.0),   # [0, 1]
        # Flags
        ('profile_has_url',             1.0),   # binary
        ('geo_enabled_flag',            1.0),   # binary
        # Source ratios (all already [0, 1])
        ('sensitive_rate',              1.0),
        ('mobile_ratio',                1.0),
        ('web_ratio',                   1.0),
        ('news_manager_ratio',          1.0),
        ('bot_api_ratio',               1.0),
        ('source_entropy',              3.0),   # log2(4) ≈ 2
    ]

    def _load_user_features(self):
        """Load user features and return (feature_tensor, uid_to_idx mapping)."""
        feat_path = self._find_file('user_features')
        if not feat_path:
            raise FileNotFoundError(f"User features not found in {self.data_dir}")

        print(f"    [GraphBuilder] Loading user features from {feat_path.name}")
        df = self._load_any_format(feat_path)

        cols = [s[0] for s in self.FEATURE_SCHEMA]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"[GraphBuilder] user_features is missing columns: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        uids = df['user_node_id'].apply(_parse_uid).values
        uid_to_idx = {uid: i for i, uid in enumerate(uids)}

        features_raw = df[cols].values.tolist()
        print(f"    [GraphBuilder] {len(uids)} nodes, {len(cols)} features each")

        features = [self._build_feature_vector(row) for row in features_raw]
        node_features = torch.tensor(features, dtype=torch.float)

        return node_features, uid_to_idx

    @staticmethod
    def _build_feature_vector(raw_values: list) -> list:
        """
        Normalize a feature row. Values already log1p-scaled by extraction
        are simply divided by their constant to bring them to ~[0, 1].
        """
        SCHEMA = GraphBuilder.FEATURE_SCHEMA
        result = []
        for v, (_, div) in zip(raw_values, SCHEMA):
            fv = float(v) if v is not None and v == v else 0.0  # handle NaN
            if div == 1.0:
                result.append(fv)
            else:
                result.append(min(fv / div, 2.0))  # clip at 2.0 for safety
        return result

    # ----------------------------------------------------------------
    # Edge loading
    # ----------------------------------------------------------------
    def _load_edges(self, edge_type: str, uid_to_idx: dict):
        """Load an edge file and return (edge_index [2, E], edge_attr [E, D])."""
        base_name = f'edges_{edge_type}'
        edge_path = self._find_file(base_name)
        
        if not edge_path:
            print(f"    WARNING: Edge file {base_name} not found in {self.data_dir}")
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 1), dtype=torch.float),
            )

        print(f"    [GraphBuilder] Loading {edge_type} edges from {edge_path.name}")
        df = self._load_any_format(edge_path)

        # Columns 0 and 1 are src and dst
        src_uids = df.iloc[:, 0].apply(_parse_uid).values
        dst_uids = df.iloc[:, 1].apply(_parse_uid).values
        
        # Other columns are features
        feat_raw = df.iloc[:, 2:].values

        src_list, dst_list = [], []
        feat_list = []

        for i in range(len(df)):
            s_uid, d_uid = src_uids[i], dst_uids[i]
            if s_uid in uid_to_idx and d_uid in uid_to_idx:
                src_list.append(uid_to_idx[s_uid])
                dst_list.append(uid_to_idx[d_uid])
                feat_list.append(feat_raw[i])

        if not src_list:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 1), dtype=torch.float),
            )

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        feat_tensor = torch.tensor(feat_list, dtype=torch.float)
        
        # Normalize weight (first feature column) with log1p / 5.0
        if feat_tensor.size(1) > 0:
            feat_tensor[:, 0] = torch.log1p(feat_tensor[:, 0]) / 5.0

        # --- Structural edge feature: Is Reciprocal (mention only) ---
        # Computed here (on full graph) rather than during extraction,
        # because reciprocity requires the complete edge set to be correct.
        if edge_type == 'mention' and edge_index.size(1) > 0:
            edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
            is_reciprocal = torch.tensor(
                [1.0 if (d, s) in edge_set else 0.0
                 for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist())],
                dtype=torch.float
            ).unsqueeze(1)  # [E, 1]
            feat_tensor = torch.cat([feat_tensor, is_reciprocal], dim=1)

        return edge_index, feat_tensor

    # ----------------------------------------------------------------
    # Structural features (lightweight, scalable to 16M nodes)
    # ----------------------------------------------------------------
    @staticmethod
    def _compute_structural_features(edge_index, num_nodes):
        """Compute lightweight structural features from graph topology.

        Only O(E) features are computed here (degree-based).
        Heavy features like PageRank should be pre-computed separately.
        """
        in_deg = torch.zeros(num_nodes, dtype=torch.float)
        out_deg = torch.zeros(num_nodes, dtype=torch.float)

        if edge_index.size(1) > 0:
            in_deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1)))
            out_deg.index_add_(0, edge_index[0], torch.ones(edge_index.size(1)))

        deg_ratio = out_deg / (in_deg + 1.0)  # avoid div by zero

        return torch.stack([
            torch.log1p(in_deg) / 10.0,
            torch.log1p(out_deg) / 10.0,
            deg_ratio.clamp(max=5.0) / 5.0,  # cap and normalize
        ], dim=1)

    # ----------------------------------------------------------------
    # Graph builders
    # ----------------------------------------------------------------
    def _build_single_type(self, node_features, uid_to_idx, num_nodes, edge_type):
        """Build a single-type graph (mention, retweet, or reply)."""
        edge_index, edge_attr = self._load_edges(edge_type, uid_to_idx)

        # Compute structural features and append to node features
        struct_feats = self._compute_structural_features(edge_index, num_nodes)
        x = torch.cat([node_features, struct_feats], dim=1)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
        )
        return data

    def _build_late_fuse(self, node_features, uid_to_idx, num_nodes):
        """Build a late-fuse graph with all 3 edge types."""
        all_edges = []
        edge_data = {}

        for etype in ['retweet', 'reply', 'mention']:
            ei, ea = self._load_edges(etype, uid_to_idx)
            edge_data[etype] = (ei, ea)
            if ei.size(1) > 0:
                all_edges.append(ei)

        # Union edge_index for RandomLinkSplit
        if all_edges:
            all_edge_index = torch.cat(all_edges, dim=1)
        else:
            all_edge_index = torch.empty((2, 0), dtype=torch.long)

        # Structural features from the union graph
        struct_feats = self._compute_structural_features(all_edge_index, num_nodes)
        x = torch.cat([node_features, struct_feats], dim=1)

        data = Data(
            x=x,
            edge_index=all_edge_index,
            edge_attr=None,  # not used for MP in late-fuse (each channel has its own)
            num_nodes=num_nodes,
        )

        # Store per-channel edges
        for etype in ['retweet', 'reply', 'mention']:
            ei, ea = edge_data[etype]
            setattr(data, f'edge_index_{etype}', ei)
            setattr(data, f'edge_attr_{etype}', ea)
            setattr(data, f'edge_dim_{etype}', ea.size(1) if ea.size(1) > 0 else 1)

        return data
