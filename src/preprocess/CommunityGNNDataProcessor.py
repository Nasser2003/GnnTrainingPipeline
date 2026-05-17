import random
import torch
import mlflow
from torch_geometric.data import Batch
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit
from preprocess.GNNDataProcessor import GNNDataProcessor

class CommunityGNNDataProcessor(GNNDataProcessor):
    def __init__(self, data_path: str, train_ratio: float = 0.70,
                 val_ratio: float = 0.15, test_ratio: float = 0.15,
                 batch_size: int = 512, num_neighbors: list = None,
                 graph_split: bool = False, min_edges_for_split: int = 50):
        super().__init__(data_path, train_ratio, val_ratio, test_ratio, batch_size, num_neighbors)
        self.graph_split = graph_split
        self.min_edges_for_split = min_edges_for_split

    @mlflow.trace(name="prepare_data_community")
    def prepare_data(self):
        use_graph_split = self.graph_split
        print(f"  [CommunityDataProcessor] Loading graphs from: {self.data_path}")
        graphs = torch.load(self.data_path, weights_only=False)
        print(f"  [CommunityDataProcessor] Loaded {len(graphs)} community graphs.")

        # Filter out graphs that are too small
        filtered_graphs = [g for g in graphs if g.edge_index.size(1) >= self.min_edges_for_split]
        print(f"  [CommunityDataProcessor] Retained {len(filtered_graphs)} graphs with >= {self.min_edges_for_split} edges.")

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
                # 100% of edges used for training message passing and supervision
                splitter = RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=False, add_negative_train_samples=False, disjoint_train_ratio=0.0)
                tr_g, _, _ = splitter(g)
                train_list.append(tr_g)

            for g in val_graphs:
                # Validation graph: 80% edges for message passing, 20% for evaluation target
                splitter = RandomLinkSplit(num_val=self.val_ratio, num_test=0.0, is_undirected=False, add_negative_train_samples=False, disjoint_train_ratio=0.0)
                _, val_g, _ = splitter(g)
                val_list.append(val_g)

            for g in test_graphs:
                # Test graph: 80% edges for message passing, 20% for test target
                splitter = RandomLinkSplit(num_val=0, num_test=self.test_ratio, is_undirected=False, add_negative_train_samples=False, disjoint_train_ratio=0.0)
                _, _, te_g = splitter(g)
                test_list.append(te_g)
        else:
            print(f"  [CommunityDataProcessor] Mode A: Edge-level split per graph enabled.")
            for g in filtered_graphs:
                splitter = RandomLinkSplit(
                    num_val=self.val_ratio,
                    num_test=self.test_ratio,
                    is_undirected=False,
                    add_negative_train_samples=False,
                    disjoint_train_ratio=0.0,
                )
                
                tr_g, v_g, te_g = splitter(g)
                train_list.append(tr_g)
                val_list.append(v_g)
                test_list.append(te_g)

        # Batching using PyTorch Geometric Batch
        # Batching using PyTorch Geometric Batch
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

        full_pos_edges = train_data.edge_index.clone()
        
        # Merge full_pos_edges if graph split is false (Mode A) to prevent negative sampling overlaps across splits of the same graph
        if not use_graph_split:
            full_pos_edges = torch.cat([train_data.edge_index, val_data.edge_index, test_data.edge_index], dim=1)

        in_channels = train_data.x.size(1)

        if hasattr(train_data, 'edge_dim_retweet'):
            edge_dim = 0
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

        return train_data, train_loader, val_data, test_data, full_pos_edges, in_channels, edge_dim
