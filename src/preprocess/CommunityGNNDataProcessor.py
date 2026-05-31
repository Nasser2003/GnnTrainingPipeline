import random
import torch

from torch_geometric.data import Batch
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit
from preprocess.GNNDataProcessor import GNNDataProcessor

class CommunityGNNDataProcessor(GNNDataProcessor):
    def __init__(self, data_path: str, train_ratio: float = 0.70,
                 val_ratio: float = 0.15, test_ratio: float = 0.15,
                 batch_size: int = 512, num_neighbors: list = None,
                 graph_split: bool = False, min_edges_for_split: int = 20,
                 max_communities: int = None, allow_self_loops: bool = False):
        super().__init__(data_path, train_ratio, val_ratio, test_ratio, batch_size, num_neighbors, allow_self_loops)
        self.graph_split = graph_split
        self.min_edges_for_split = min_edges_for_split
        self.max_communities = max_communities

    def prepare_data(self):
        use_graph_split = self.graph_split
        print(f"  [CommunityDataProcessor] Loading graphs from: {self.data_path}")
        graphs = torch.load(self.data_path, weights_only=False)
        
        if not self.allow_self_loops:
            from torch_geometric.utils import remove_self_loops
            for g in graphs:
                if g.edge_attr is not None:
                    g.edge_index, g.edge_attr = remove_self_loops(g.edge_index, g.edge_attr)
                else:
                    g.edge_index, _ = remove_self_loops(g.edge_index)
                    
                for etype in ['retweet', 'reply', 'mention']:
                    ei = getattr(g, f'edge_index_{etype}', None)
                    ea = getattr(g, f'edge_attr_{etype}', None)
                    if ei is not None:
                        if ea is not None and ea.size(0) == ei.size(1):
                            ei, ea = remove_self_loops(ei, ea)
                            setattr(g, f'edge_attr_{etype}', ea)
                        else:
                            ei, _ = remove_self_loops(ei)
                        setattr(g, f'edge_index_{etype}', ei)
                        
        print(f"  [CommunityDataProcessor] Loaded {len(graphs)} community graphs.")

        # Filter out graphs that are too small
        filtered_graphs = [g for g in graphs if g.edge_index.size(1) >= self.min_edges_for_split]
        print(f"  [CommunityDataProcessor] Retained {len(filtered_graphs)} graphs with >= {self.min_edges_for_split} edges.")

        if self.max_communities is not None and self.max_communities > 0:
            # Sort communities by size (number of nodes) ascending
            filtered_graphs = sorted(filtered_graphs, key=lambda g: getattr(g, 'num_nodes', g.x.size(0)))
            filtered_graphs = filtered_graphs[:self.max_communities]
            print(f"  [CommunityDataProcessor] Truncated to {len(filtered_graphs)} smallest communities (max_communities={self.max_communities}).")

        if not filtered_graphs:
            raise ValueError(f"No communities have >= {self.min_edges_for_split} edges. Cannot proceed with training.")

        train_list, val_list, test_list = [], [], []

        if use_graph_split:
            print(f"  [CommunityDataProcessor] Mode B: Graph-level split enabled.")
            random.shuffle(filtered_graphs)
            n_graphs = len(filtered_graphs)
            n_val = max(1, int(n_graphs * self.val_ratio))
            n_test = max(1, int(n_graphs * self.test_ratio))
            n_train = n_graphs - n_val - n_test

            if n_train <= 0:
                use_graph_split = False
                print(f"  [CommunityDataProcessor] WARNING: Only {n_graphs} graphs available. Not enough for Graph Split. Falling back to Edge Split (Mode A).")
            else:
                train_graphs = filtered_graphs[:n_train]
                val_graphs = filtered_graphs[n_train:n_train+n_val]
                test_graphs = filtered_graphs[n_train+n_val:]
                print(f"  [CommunityDataProcessor] Split: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test graphs.")

        if use_graph_split:
            for g in train_graphs:
                hidden = {k: g[k] for k in list(g.keys()) if ('retweet' in k or 'reply' in k or 'mention' in k)}
                for k in hidden: delattr(g, k)
                splitter = RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=False, add_negative_train_samples=False, disjoint_train_ratio=0.0)
                tr_g, _, _ = splitter(g)
                for k, v in hidden.items(): setattr(tr_g, k, v)
                train_list.append(tr_g)

            for g in val_graphs:
                hidden = {k: g[k] for k in list(g.keys()) if ('retweet' in k or 'reply' in k or 'mention' in k)}
                for k in hidden: delattr(g, k)
                splitter = RandomLinkSplit(num_val=self.val_ratio, num_test=0.0, is_undirected=False, add_negative_train_samples=False, disjoint_train_ratio=0.0)
                _, val_g, _ = splitter(g)
                for k, v in hidden.items(): setattr(val_g, k, v)
                val_list.append(val_g)

            for g in test_graphs:
                hidden = {k: g[k] for k in list(g.keys()) if ('retweet' in k or 'reply' in k or 'mention' in k)}
                for k in hidden: delattr(g, k)
                splitter = RandomLinkSplit(num_val=0, num_test=self.test_ratio, is_undirected=False, add_negative_train_samples=False, disjoint_train_ratio=0.0)
                _, _, te_g = splitter(g)
                for k, v in hidden.items(): setattr(te_g, k, v)
                test_list.append(te_g)
        else:
            print(f"  [CommunityDataProcessor] Mode A: Edge-level split per graph enabled.")
            for g in filtered_graphs:
                hidden = {k: g[k] for k in list(g.keys()) if ('retweet' in k or 'reply' in k or 'mention' in k)}
                for k in hidden: delattr(g, k)
                splitter = RandomLinkSplit(
                    num_val=self.val_ratio,
                    num_test=self.test_ratio,
                    is_undirected=False,
                    add_negative_train_samples=False,
                    disjoint_train_ratio=0.0,
                )
                tr_g, v_g, te_g = splitter(g)
                for k, v in hidden.items(): 
                    setattr(tr_g, k, v)
                    setattr(v_g, k, v)
                    setattr(te_g, k, v)
                train_list.append(tr_g)
                val_list.append(v_g)
                test_list.append(te_g)

        # Read per-channel edge dims BEFORE batching (Batch may not preserve scalar attributes)
        _ref = train_list[0] if train_list else None
        edge_dim_retweet = getattr(_ref, 'edge_dim_retweet', 7) if _ref else 7
        edge_dim_reply   = getattr(_ref, 'edge_dim_reply',   7) if _ref else 7
        edge_dim_mention = getattr(_ref, 'edge_dim_mention', 7) if _ref else 7

        # Batch all community graphs into a single disconnected graph.
        # NOTE: Batch.from_data_list() merges all community graphs into a single
        # disconnected graph. Each community subgraph remains isolated (no edges
        # cross community boundaries), so LinkNeighborLoader will never sample
        # neighbors across communities. This is correct by design.
        #
        # The global edge_index is the union of all community edge_indices, with
        # node indices remapped to be globally unique via Batch.batch attribute.
        train_data = Batch.from_data_list(train_list)
        val_data = Batch.from_data_list(val_list)
        test_data = Batch.from_data_list(test_list)

        # Remap custom edge indices to the batched node space
        for batched_data, orig_list in [(train_data, train_list), (val_data, val_list), (test_data, test_list)]:
            ptr = batched_data.ptr
            for etype in ['retweet', 'reply', 'mention']:
                mapped_eis = []
                for i, g in enumerate(orig_list):
                    ei = getattr(g, f'edge_index_{etype}', None)
                    if ei is not None and ei.size(1) > 0:
                        mapped_eis.append(ei + ptr[i])
                if mapped_eis:
                    setattr(batched_data, f'edge_index_{etype}', torch.cat(mapped_eis, dim=1))
                else:
                    setattr(batched_data, f'edge_index_{etype}', torch.empty((2, 0), dtype=torch.long))

        full_pos_edges = train_data.edge_index.clone()
        
        # Merge full_pos_edges if graph split is false (Mode A) to prevent negative sampling overlaps across splits of the same graph
        if not use_graph_split:
            full_pos_edges = torch.cat([train_data.edge_index, val_data.edge_index, test_data.edge_index], dim=1)

        in_channels = train_data.x.size(1)

        if hasattr(_ref, 'edge_dim_retweet'):
            edge_dim = 0  # late_fuse: no single edge_attr
        else:
            edge_dim = train_data.edge_attr.size(1) if train_data.edge_attr is not None else 0

        print(f"  [CommunityDataProcessor] Train edges (batched): {train_data.edge_label_index.size(1)}")
        print(f"  [CommunityDataProcessor] Val edges (batched):   {val_data.edge_label_index.size(1)}")
        print(f"  [CommunityDataProcessor] Test edges (batched):  {test_data.edge_label_index.size(1)}")

        train_loader = LinkNeighborLoader(
            data=train_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=train_data.edge_label_index,
            edge_label=train_data.edge_label,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        return train_data, train_loader, val_data, test_data, full_pos_edges, in_channels, edge_dim, edge_dim_retweet, edge_dim_reply, edge_dim_mention
