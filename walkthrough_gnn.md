# Scalable GNN Pipeline Implementation

The Scalable GNN Pipeline for `GraphAnalysis` has been successfully implemented according to the architectural decisions. It fully transitions the graph processing from an in-memory approach to a scalable, mini-batch architecture capable of handling the 16-million-node graphs using PyTorch Geometric's `LinkNeighborLoader`.

## Achievements

1. **Scalable Data Ingestion & Preprocessing**
   - `GraphBuilder.py` was created to parse the CSV outputs of the extraction phase, supporting multi-relational graphs (`mention`, `retweet`, `reply`) and dynamic feature dimensionality. Lightweight structural features (in/out degree) are generated inline.
   - `GNNDataProcessor.py` utilizes `RandomLinkSplit` and `LinkNeighborLoader` to enable bounded-memory training batches.

2. **Late-Fuse Architecture**
   - Implemented `LateFuseEncoder.py` that delegates to 3 independent sub-encoders, one for each channel type (mention, reply, retweet), passing their unique graph structures and variable dimension edge attributes. The output is fused via the user's choice of aggregation (mean, sum, concat).

3. **Multi-Model Encoders**
   - 7 standardized encoders are available (`GCN`, `GAT`, `GATv2`, `GINE`, `SAGE`, `Transformer`, `NNConv`). The `ModelFactory` abstracts the instantiation, routing the inputs dynamically (e.g. `edge_dim` passed only where applicable).

4. **Dynamic Configuration & Execution**
   - `prop.json` contains a robust `gnn_training` block allowing combinations of Graph Types, Encoders, and hyperparameters to be switched via booleans.
   - `Main.py` loops automatically through enabled configurations to support batch processing.

5. **Robust Evaluation**
   - The Evaluation suite calculates Micro (F1, P, R, Acc, AUC, AP) metrics, including the newly requested **PR-AUC**.
   - `Cosine` and `MLP` baselines are fully supported, using a bespoke `BaselineTrainer`.

6. **Benchmarking Script**
   - Added `scripts/run_benchmark.py` to automate a full sweep (4 graph types × 7 encoders) for easy overnight evaluation.

## Verification

To verify the setup:

> [!TIP]
> Run the full evaluation benchmark matrix using the script:
> ```bash
> python scripts/run_benchmark.py
> ```
> Or, invoke the pipeline directly with arguments:
> ```bash
> python Main.py --graph_type late_fuse --encoder gine
> ```

Please review the implemented components and let me know if you would like any modifications or to deploy these changes into production.
