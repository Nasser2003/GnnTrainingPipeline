# GNN Link Prediction Pipeline — GraphAnalysis (v2)

## Background

The extraction phase is complete. CSV outputs under `resources/<run_id>/`:
- `user_features` — ~2001 users, 13 columns
- `edges_retweet` — ~13K edges: `src, dst, weight`
- `edges_reply` — ~24K edges: `src, dst, weight`
- `edges_mention` — ~426 resolved edges: `src, dst, weight`

---

## Changes from v1 (User Feedback)

> [!IMPORTANT]
> **Variable edge dimensions per edge type.** Mention edges will have more features than retweet/reply edges in the future. Each graph type has its own `edge_dim`, and the `LateFuseEncoder` handles 3 different edge dims for its sub-encoders.

> [!IMPORTANT]
> **Scalable training with `LinkNeighborLoader`.** Instead of loading the entire graph into memory for training, we use PyG's `LinkNeighborLoader` which performs mini-batch link prediction with neighbor sampling. This scales to much larger graphs.

> [!NOTE]
> **Normalization in code, not config.** Instead of a config toggle, normalization is handled via easily-commentable code lines in GraphBuilder, since some features may already be pre-normalized and need manual verification.

> [!NOTE]
> **Multi-execution config.** `graph_type` and `encoder` are now dicts of booleans — Main.py iterates over all enabled combinations (e.g. 4 graph types × 7 encoders = 28 runs). No separate benchmark script needed for basic sweeps.

---

## Architecture Overview

```
GraphAnalysis/
├── gnn/                              ← NEW MODULE
│   ├── __init__.py
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── GraphBuilder.py           ← CSV → PyG Data
│   │   └── GNNDataProcessor.py       ← Load .pt, LinkNeighborLoader
│   ├── model/
│   │   ├── __init__.py
│   │   ├── LinkPredictor.py          ← Encoder + Decoder wrapper
│   │   ├── Decoders.py               ← MLP + DotProduct
│   │   ├── ModelFactory.py           ← Factory pattern
│   │   ├── RawMLPConcatPredictor.py  ← Baseline MLP
│   │   └── encoders/
│   │       ├── __init__.py
│   │       ├── GCNEncoder.py
│   │       ├── GATEncoder.py
│   │       ├── GATv2Encoder.py
│   │       ├── GINEEncoder.py
│   │       ├── SAGEEncoder.py
│   │       ├── TransformerEncoder.py
│   │       ├── NNConvEncoder.py
│   │       └── LateFuseEncoder.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── GNNTraining.py            ← Mini-batch with LinkNeighborLoader
│   │   └── BaselineTrainer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── BaselineScorer.py
│   │   ├── CosineBaselineScorer.py
│   │   ├── MLPBaselineScorer.py
│   │   ├── GNNScorer.py
│   │   ├── Evaluator.py              ← + PR-AUC
│   │   └── GNNEvaluator.py
│   └── utils/
│       ├── __init__.py
│       └── GraphUtils.py
├── properties/
│   └── prop.json                     ← MODIFIED
├── scripts/
│   └── run_benchmark.py              ← NEW
└── Main.py                           ← MODIFIED
```

---

## Proposed Changes

### Component 1: Configuration

#### [MODIFY] [prop.json](GraphAnalysis/properties/prop.json)

Add `gnn_training` section. `data_dir` points to the run_id folder produced by extraction:

```json
{
  "gnn_training": {
    "to_execute": false,
    "data": {
      "data_dir": "resources/4e06a07847b411f18cf79c29764a0d18",
      "graph_type": {
        "reply": true,
        "retweet": true,
        "mention": true,
        "late_fuse": true
      },
      "train_ratio": 0.70,
      "val_ratio": 0.15,
      "test_ratio": 0.15
    },
    "model": {
      "encoder": {
        "gcn": true,
        "gat": true,
        "gatv2": true,
        "gine": true,
        "sage": true,
        "transformer": true,
        "nnconv": true
      },
      "decoder": "mlp",
      "num_epochs": 50,
      "learning_rate": 0.001,
      "weight_decay": 1e-5,
      "hidden_dim": 128,
      "embed_dim": 64,
      "dropout": 0.2,
      "neg_ratio": 1.0,
      "decoder_hidden": 64,
      "decoder_dropout": 0.3,
      "scheduler": false,
      "grad_clip": null,
      "early_stopping_patience": 10,
      "seed": 777,
      "fusion": "mean",
      "batch_size": 512,
      "num_neighbors": [10, 5]
    },
    "evaluation": {
      "approaches": ["cosine", "mlp", "gnn"],
      "neg_ratios": [1, 9, 100]
    },
    "output_dir": "output/models",
    "log_dir": "output/logs",
    "print_logs": false
  }
}
```

**Key design:**
- `graph_type` is a dict of booleans — all `true` entries are executed sequentially
- `encoder` is a dict of booleans — all `true` entries are tested per graph type
- For `late_fuse` graph type, each enabled encoder is used as the base encoder for the 3 sub-encoders (no separate `fused_base_encoder` parameter)
- `batch_size` / `num_neighbors`: control `LinkNeighborLoader` for scalable training

---

### Component 2: Preprocessing

#### [NEW] [GraphBuilder.py](GraphAnalysis/gnn/preprocess/GraphBuilder.py)

**Responsibilities:**
1. Load `user_features` CSV → node feature tensor
2. Build `uid_to_idx` mapping for all users
3. Load edge CSVs per type
4. Build and save `.pt` file

**Key design — variable edge dimensions:**

Each edge type defines its own feature schema. Currently:

| Edge Type | Columns                   | Dim                 |
| --------- | ------------------------- | ------------------- |
| retweet   | `[log1p(weight)/5, 0, 0]` | 3                   |
| reply     | `[log1p(weight)/5, 0, 0]` | 3                   |
| mention   | `[log1p(weight)/5, 0, 0]` | 3 (will grow later) |

The builder stores edge_dim metadata in the Data object:

```python
# For single-type graphs (mention/retweet/reply):
data = Data(
    x=node_features,          # [N, 11]
    edge_index=edge_index,    # [2, E]
    edge_attr=edge_attr,      # [E, edge_dim]  — dim varies by type
    num_nodes=N,
)

# For late-fuse graph:
data = Data(
    x=node_features,                      # [N, 11]
    # Primary edges = union of all (for RandomLinkSplit compatibility)
    edge_index=all_edge_index,            # [2, E_total]
    edge_attr=None,                       # not used for MP in late-fuse
    # Per-channel edges (used by LateFuseEncoder)
    edge_index_retweet=ei_rt,             # [2, E_rt]
    edge_attr_retweet=ea_rt,              # [E_rt, dim_rt]
    edge_index_reply=ei_re,               # [2, E_re]
    edge_attr_reply=ea_re,                # [E_re, dim_re]
    edge_index_mention=ei_mn,             # [2, E_mn]
    edge_attr_mention=ea_mn,             # [E_mn, dim_mn]
    num_nodes=N,
)
```

**Feature normalization** — in-code with easy toggle via commenting:

```python
def build_feature_vector(raw_values: list) -> list:
    """Convert raw user stats to feature vector.
    
    To disable normalization, comment out the normalize block below
    and uncomment the raw passthrough.
    """
    # raw_values = [total, retweets, replies, original, likes, 
    #               followers, following, verified, account_age_days,
    #               n_unique_hashtags, n_unique_mentions]
    
    # --- RAW PASSTHROUGH (uncomment to disable normalization) ---
    # return [float(v) for v in raw_values]
    
    # --- NORMALIZED (comment out to use raw values) ---
    NORMS = [10, 10, 10, 10, 15, 15, 12, 1, 10, 8, 8]
    result = []
    for v, div in zip(raw_values, NORMS):
        if div == 1:  # binary feature (verified)
            result.append(float(v))
        else:
            result.append(math.log1p(max(float(v), 0)) / div)
    return result
```

#### [NEW] [GNNDataProcessor.py](GraphAnalysis/gnn/preprocess/GNNDataProcessor.py)

**Single-graph, scalable approach using `LinkNeighborLoader`:**

```python
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit

class GNNDataProcessor:
    def __init__(self, data_path, train_ratio, val_ratio, test_ratio,
                 batch_size=512, num_neighbors=[10, 5]):
        ...
    
    def prepare_data(self):
        """Load graph, split edges, return loaders."""
        data = torch.load(self.data_path, weights_only=False)
        
        # Split edges into train/val/test supervision sets
        splitter = RandomLinkSplit(
            num_val=self.val_ratio,
            num_test=self.test_ratio,
            is_undirected=False,
            add_negative_train_samples=True,
            disjoint_train_ratio=0.2,
        )
        train_data, val_data, test_data = splitter(data)
        
        # Create mini-batch loaders for scalable training
        train_loader = LinkNeighborLoader(
            data=train_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=train_data.edge_label_index,
            edge_label=train_data.edge_label,
            batch_size=self.batch_size,
            shuffle=True,
        )
        
        # Val/test: full graph evaluation (smaller, no sampling needed)
        # Or also use loaders if RAM is tight
        
        in_channels = data.x.size(1)
        edge_dim = data.edge_attr.size(1) if data.edge_attr is not None else 0
        
        return (train_loader, val_data, test_data, 
                data.edge_index.clone(), in_channels, edge_dim)
```

**Why `LinkNeighborLoader`:**
- Samples a local subgraph around supervision edges
- Each mini-batch = ~`batch_size` supervision edges + their k-hop neighborhoods
- GPU memory is bounded regardless of full graph size
- Standard approach for large-scale link prediction in PyG

---

### Component 3: Models

#### [NEW] 7 Encoders — Unified interface

**All encoders share the same signature:** `forward(x, edge_index, edge_attr=None)`

Models that don't natively use edge features (GCN, SAGE, GAT) simply ignore the parameter:

```python
# GCNEncoder, SAGEEncoder, GATEncoder
def forward(self, x, edge_index, edge_attr=None):
    # edge_attr is accepted but not used
    x = self.conv1(x, edge_index)
    ...
```

This means no `use_edge_attr` flag is needed in the factory or predictor.

#### [NEW] [LateFuseEncoder.py](GraphAnalysis/gnn/model/encoders/LateFuseEncoder.py)

Key difference from FreebaseQA's `FusedEncoder`: takes **pre-separated edge indices** with **independent edge_dims**:

```python
class LateFuseEncoder(nn.Module):
    """Late-fusion: 3 sub-encoders with potentially different edge_dims."""
    
    def __init__(self, encoder_class, in_ch, hidden, embed_dim,
                 edge_dim_retweet, edge_dim_reply, edge_dim_mention,
                 dropout=0.5, fusion='mean'):
        super().__init__()
        self.enc_retweet = encoder_class(
            in_ch, hidden, embed_dim, edge_dim_retweet, dropout=dropout)
        self.enc_reply = encoder_class(
            in_ch, hidden, embed_dim, edge_dim_reply, dropout=dropout)
        self.enc_mention = encoder_class(
            in_ch, hidden, embed_dim, edge_dim_mention, dropout=dropout)
        self.fusion = fusion
        self.embed_dim = embed_dim
    
    @property
    def out_dim(self):
        return 3 * self.embed_dim if self.fusion == 'concat' else self.embed_dim
    
    def forward(self, x, edge_index=None, edge_attr=None, **kwargs):
        """
        Expects kwargs with per-channel edge data:
          edge_index_retweet, edge_attr_retweet,
          edge_index_reply,   edge_attr_reply,
          edge_index_mention, edge_attr_mention
        """
        z_rt = self.enc_retweet(
            x, kwargs['edge_index_retweet'], kwargs.get('edge_attr_retweet'))
        z_re = self.enc_reply(
            x, kwargs['edge_index_reply'], kwargs.get('edge_attr_reply'))
        z_mn = self.enc_mention(
            x, kwargs['edge_index_mention'], kwargs.get('edge_attr_mention'))
        
        if self.fusion == 'mean':
            return torch.stack([z_rt, z_re, z_mn]).mean(0)
        return torch.cat([z_rt, z_re, z_mn], dim=-1)
```

#### [NEW] [LinkPredictor.py](GraphAnalysis/gnn/model/LinkPredictor.py)

Handles both standard and late-fuse encoding:

```python
class LinkPredictor(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._is_late_fuse = isinstance(encoder, LateFuseEncoder)
    
    def encode(self, data):
        data = data.to(self.device)
        if self._is_late_fuse:
            return self.encoder(
                data.x, data.edge_index, data.edge_attr,
                edge_index_retweet=data.edge_index_retweet,
                edge_attr_retweet=getattr(data, 'edge_attr_retweet', None),
                edge_index_reply=data.edge_index_reply,
                edge_attr_reply=getattr(data, 'edge_attr_reply', None),
                edge_index_mention=data.edge_index_mention,
                edge_attr_mention=getattr(data, 'edge_attr_mention', None),
            )
        return self.encoder(data.x, data.edge_index, 
                           getattr(data, 'edge_attr', None))
```

#### [NEW] [ModelFactory.py](GraphAnalysis/gnn/model/ModelFactory.py)

Factory creates any of the 7 encoders + late-fuse. For late-fuse, the `base_encoder_name` is passed by the Main.py loop (= the current encoder being iterated):

```python
class ModelFactory:
    @staticmethod
    def create_predictor(encoder_name, decoder_name, in_channels, hidden, 
                         embed_dim, edge_dim=0, dropout=0.3,
                         decoder_hidden=64, decoder_dropout=0.3,
                         fusion='mean',
                         edge_dim_retweet=3, edge_dim_reply=3, 
                         edge_dim_mention=3,
                         late_fuse_base_encoder=None):
        if encoder_name == 'late-fuse':
            # late_fuse_base_encoder = the current encoder from the loop
            base_cls = ModelFactory._resolve_encoder_class(late_fuse_base_encoder)
            encoder = LateFuseEncoder(
                base_cls, in_channels, hidden, embed_dim,
                edge_dim_retweet, edge_dim_reply, edge_dim_mention,
                dropout=dropout, fusion=fusion)
            dec_dim = encoder.out_dim
        else:
            encoder = ModelFactory.create_encoder(
                encoder_name, in_channels, hidden, embed_dim,
                edge_dim=edge_dim, dropout=dropout)
            dec_dim = embed_dim
        
        decoder = ModelFactory.create_decoder(
            decoder_name, dec_dim, decoder_hidden, decoder_dropout)
        return LinkPredictor(encoder, decoder)
```

---

### Component 4: Training

#### [NEW] [GNNTraining.py](GraphAnalysis/gnn/train/GNNTraining.py)

**Mini-batch training with `LinkNeighborLoader`:**

```python
class GNNTraining:
    def train(self, train_loader, val_data):
        """Mini-batch training loop."""
        model = ModelFactory.create_predictor(...)
        optimizer = torch.optim.Adam(...)
        
        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Encode the sampled subgraph
                z = model.encode(batch)
                
                # Supervision edges in the batch
                pos_mask = batch.edge_label == 1
                neg_mask = batch.edge_label == 0
                pos_ei = batch.edge_label_index[:, pos_mask]
                neg_ei = batch.edge_label_index[:, neg_mask]
                
                loss = self._link_pred_loss(z, pos_ei, neg_ei, 
                                            model.decoder, self.neg_ratio)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Full-graph validation
            val_auc = self._evaluate_auc(model, val_data, device)
            # Early stopping logic...
```

**For late-fuse with `LinkNeighborLoader`:**
The late-fuse case is slightly more complex — the `LinkNeighborLoader` samples based on the union edge_index, but the per-channel edges need to be sliced to the sampled subgraph. We handle this in the batch preparation by mapping global node indices to local batch indices.

---

### Component 5: Evaluation

#### [NEW] [Evaluator.py](GraphAnalysis/gnn/evaluation/Evaluator.py)

Same as FreebaseQA **plus PR-AUC**:

```python
def evaluate_metrics_master(self, name, data, neg_ratio, scorer, threshold=0.5):
    y, p = self._collect_scores(data, neg_ratio, scorer)
    
    acc, pr, re, f1 = self.metrics_from_scores(y, p, threshold)
    auc_roc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    
    # NEW: PR-AUC
    prec_curve, rec_curve, _ = precision_recall_curve(y, p)
    pr_auc = auc(rec_curve, prec_curve)
    
    print(
        f"[{name} | 1:{neg_ratio} | Thr={threshold:.4f}] "
        f"F1={f1:.4f} P={pr:.4f} R={re:.4f} Acc={acc:.4f} "
        f"AUC={auc_roc:.4f} AP={ap:.4f} PR-AUC={pr_auc:.4f}"
    )
```

---

### Component 6: Benchmark Script

#### [NEW] [run_benchmark.py](GraphAnalysis/scripts/run_benchmark.py)

Systematic benchmark runner:

```
Graph Types:   mention, retweet, reply, late-fuse
Encoders:      gcn, gat, gatv2, gine, sage, transformer, nnconv
Neg Ratios:    1, 9, 100
Scorers:       cosine, mlp, gnn
```

Robustness display output:
```
==================================================================================
BENCHMARK — GraphAnalysis GNN Link Prediction
==================================================================================

Graph: REPLY | Encoder: gine | Decoder: mlp
----------------------------------------------------------------------------------
| Scorer | Neg Ratio | F1     | P      | R      | Acc    | AUC    | AP     | PR-AUC |
| ------ | --------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Cosine | 1:1       | 0.7234 | 0.6891 | 0.7612 | 0.7012 | 0.7801 | 0.7456 | 0.7123 |
| MLP    | 1:1       | 0.7567 | 0.7234 | 0.7923 | 0.7345 | 0.8123 | 0.7789 | 0.7456 |
| GNN    | 1:1       | 0.8234 | 0.8012 | 0.8467 | 0.8123 | 0.8901 | 0.8567 | 0.8234 |
...
```

---

### Component 7: Main Entry Point

#### [MODIFY] [Main.py](GraphAnalysis/Main.py)

Add GNN training task. Main.py iterates over all enabled `graph_type × encoder` combinations:

```python
if do_gnn_training:
    gnn_cfg = prop["gnn_training"]
    
    # Collect enabled graph types and encoders
    enabled_graphs = [k for k, v in gnn_cfg["data"]["graph_type"].items() if v]
    enabled_encoders = [k for k, v in gnn_cfg["model"]["encoder"].items() if v]
    
    for graph_type in enabled_graphs:
        # Build .pt graph if not exists
        builder = GraphBuilder(data_dir=gnn_cfg["data"]["data_dir"], graph_type=graph_type)
        pt_path = builder.build_and_save()
        
        # Load & split
        processor = GNNDataProcessor(pt_path, ...)
        train_loader, val_data, test_data, full_pos, in_ch, edge_dim = processor.prepare_data()
        
        for encoder_name in enabled_encoders:
            print(f"\n{'='*60}")
            print(f"  Graph: {graph_type} | Encoder: {encoder_name}")
            print(f"{'='*60}")
            
            # For late_fuse graph_type:
            # each encoder becomes the base encoder for the 3 sub-encoders
            actual_encoder = 'late-fuse' if graph_type == 'late_fuse' else encoder_name
            
            # Train
            trainer = GNNTraining(
                encoder=actual_encoder,
                late_fuse_base_encoder=encoder_name if graph_type == 'late_fuse' else None,
                ...)
            model = trainer.train(train_loader, val_data)
            
            # Evaluate with all neg_ratios and scorers
            evaluator = GNNEvaluator(model=model)
            evaluator.evaluate(val_data, test_data, neg_ratios=neg_ratios)
```

CLI overrides are still available to force a single graph_type or encoder for quick testing:
```
python Main.py --graph_type reply --encoder gine
```

---

## Key Differences vs FreebaseQA

| Aspect                | FreebaseQA            | GraphAnalysis                                  |
| --------------------- | --------------------- | ---------------------------------------------- |
| Data source           | PHEME JSON threads    | MongoDB → CSV                                  |
| Graph count           | Many small graphs     | **One large graph**                            |
| Nodes                 | Only in-thread users  | **All users always**                           |
| Edge dims             | Fixed per approach    | **Variable per edge type**                     |
| Training              | Full-graph batch      | **Mini-batch (LinkNeighborLoader)**            |
| Fused approach        | Column-mask splitting | **Pre-separated edge indices**                 |
| Model edge handling   | use_edge_attr flag    | **Unified interface (all accept edge_attr)**   |
| PR-AUC                | ❌                     | ✅                                              |
| Feature normalization | Always on             | **In-code toggle (comment/uncomment)**         |
| Execution model       | Single run            | **Iterates over enabled graph_type × encoder** |

---

## Implementation Order

1. `gnn/utils/GraphUtils.py` — utility functions
2. `gnn/preprocess/GraphBuilder.py` — CSV → .pt
3. `gnn/model/encoders/*` — all 7 encoders + LateFuse
4. `gnn/model/Decoders.py`, `LinkPredictor.py`, `ModelFactory.py`
5. `gnn/model/RawMLPConcatPredictor.py` — baseline
6. `gnn/preprocess/GNNDataProcessor.py` — data loading + LinkNeighborLoader
7. `gnn/train/GNNTraining.py` + `BaselineTrainer.py`
8. `gnn/evaluation/*` — all evaluation files
9. `properties/prop.json` — add gnn_training config
10. `Main.py` — integrate GNN task
11. `scripts/run_benchmark.py` — benchmark runner

---

## Verification Plan

### Automated Tests
1. **GraphBuilder**: Build `.pt` from extracted data → verify shapes
2. **Smoke test**: Train GINE on `reply` for 5 epochs → loss decreases, AUC > 0.5
3. **Late-fuse**: Verify 3 channels with different edge dims work
4. **LinkNeighborLoader**: Verify mini-batches are correctly formed
5. **PR-AUC**: Verify metric is computed alongside existing ones

### Manual Verification
1. Run benchmark with `scripts/run_benchmark.py`
2. Verify robustness tables are formatted correctly
3. Compare GNN vs baselines
