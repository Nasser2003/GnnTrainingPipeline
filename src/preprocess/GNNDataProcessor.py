import mlflow
import torch
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit


class GNNDataProcessor:
    """
    Loads a PyG Data object and prepares it for scalable training.
    Applies RandomLinkSplit and creates a LinkNeighborLoader for mini-batching.
    """

    def __init__(self, data_path: str, train_ratio: float = 0.70,
                 val_ratio: float = 0.15, test_ratio: float = 0.15,
                 batch_size: int = 512, num_neighbors: list = None, allow_self_loops: bool = False):
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors or [10, 5]
        self.allow_self_loops = allow_self_loops

    @mlflow.trace(name="prepare_data")
    def prepare_data(self):
        """
        Loads the graph, splits edges, and returns the loaders/data.
        Returns:
            train_data: Full training Data object (for MLP baseline and reference)
            train_loader: LinkNeighborLoader for GNN training mini-batches
            val_data: Data object for validation
            test_data: Data object for testing
            full_pos_edges: Full positive edge_index for negative sampling exclusion
            in_channels: Number of node features
            edge_dim: Number of edge features (0 if None)
        """
        print(f"  [DataProcessor] Loading graph from: {self.data_path}")
        data = torch.load(self.data_path, weights_only=False)

        if not self.allow_self_loops:
            from torch_geometric.utils import remove_self_loops
            if data.edge_attr is not None:
                data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
            else:
                data.edge_index, _ = remove_self_loops(data.edge_index)
                
            for etype in ['retweet', 'reply', 'mention']:
                ei = getattr(data, f'edge_index_{etype}', None)
                ea = getattr(data, f'edge_attr_{etype}', None)
                if ei is not None:
                    if ea is not None and ea.size(0) == ei.size(1):
                        ei, ea = remove_self_loops(ei, ea)
                        setattr(data, f'edge_attr_{etype}', ea)
                    else:
                        ei, _ = remove_self_loops(ei)
                    setattr(data, f'edge_index_{etype}', ei)

        # Full positive edges for negative sampling logic during evaluation.
        # This is the COMPLETE edge set before splitting — used to ensure
        # negatives sampled at eval time are truly absent from the graph.
        full_pos_edges = data.edge_index.clone()
        
        in_channels = data.x.size(1)
        
        # Check edge_attr dimension (could be None or specific to graph type)
        if hasattr(data, 'edge_dim_retweet'):
            # Late-fuse graph: no single edge_attr, each channel has its own dim
            edge_dim = 0
        else:
            edge_dim = data.edge_attr.size(1) if data.edge_attr is not None else 0

        # Split edges into train/val/test supervision sets
        print(f"  [DataProcessor] Applying RandomLinkSplit "
              f"({self.train_ratio}/{self.val_ratio}/{self.test_ratio})")
        
        num_edges = data.edge_index.size(1)
        num_val = int(num_edges * self.val_ratio)
        num_test = int(num_edges * self.test_ratio)
        
        splitter = RandomLinkSplit(
            num_val=num_val,
            num_test=num_test,
            is_undirected=False,
            # Negatives are sampled dynamically during the training loop,
            # NOT baked into the split. This avoids double-counting if
            # add_negative_train_samples were ever changed to True.
            add_negative_train_samples=False,
            disjoint_train_ratio=0.0,
        )
        
        # Hide custom edge attributes to prevent RandomLinkSplit from crashing on them
        hidden = {k: data[k] for k in list(data.keys()) if ('retweet' in k or 'reply' in k or 'mention' in k)}
        for k in hidden: delattr(data, k)
        
        train_data, val_data, test_data = splitter(data)
        
        # Restore them
        for k, v in hidden.items():
            setattr(train_data, k, v)
            setattr(val_data, k, v)
            setattr(test_data, k, v)
        
        print(f"  [DataProcessor] Train edges: {train_data.edge_label_index.size(1)}")
        print(f"  [DataProcessor] Val edges:   {val_data.edge_label_index.size(1)}")
        print(f"  [DataProcessor] Test edges:  {test_data.edge_label_index.size(1)}")

        # Create mini-batch loader for scalable training
        print(f"  [DataProcessor] Creating LinkNeighborLoader "
              f"(batch_size={self.batch_size}, neighbors={self.num_neighbors})")
              
        train_loader = LinkNeighborLoader(
            data=train_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=train_data.edge_label_index,
            edge_label=train_data.edge_label,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to >0 if multiprocessing is desired and safe
        )

        # For the standard (non-community) processor, read edge dims from data directly
        edge_dim_retweet = getattr(data, 'edge_dim_retweet', 0)
        edge_dim_reply   = getattr(data, 'edge_dim_reply',   0)
        edge_dim_mention = getattr(data, 'edge_dim_mention', 0)

        return train_data, train_loader, val_data, test_data, full_pos_edges, in_channels, edge_dim, edge_dim_retweet, edge_dim_reply, edge_dim_mention
