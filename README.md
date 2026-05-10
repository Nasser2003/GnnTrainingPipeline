# uniroma2-twitter-ml-pipeline

## project exection example
```python
python .\src\main.py model.encoder=gine data.graph_type=reply 
```

## Containers:
### Step 1: Build the images
```bash
podman build -t gnn-pipeline -f ./docker/Dockerfile .
```

### Step: 2: Run the gnn-pipeline container
local machine
```bash
podman run --gpus all \
  -v ../GraphAnalysis/resources:/app/data \
  -e RUN_ID=add1322e4c9011f1b7318ae8655e7b49 \
  -v ./outputs:/app/outputs \
  -v ./conf:/app/conf \
  gnn-pipeline
```

production
```bash
podman run --gpus all \
  -e DATA_DIR=/app/data \
  -e RUN_ID=add1322e4c9011f1b7318ae8655e7b49 \
  -v /ipazianas/pasquini/extraction/output:/app/data:ro \
  -v /ipazianas/pasquini/training/outputs:/app/outputs \
  -v ./conf:/app/conf \
  gnn-pipeline
```

custom parameters:
```bash
podman run -v ../GraphAnalysis/resources:/app/data -e RUN_ID=add1322e4c9011f1b7318ae8655e7b49 -v ./outputs:/app/outputs gnn-pipeline --multirun model.encoder='gine,gcn,gat,sage' data.graph_type='reply'
```

### Step: 3: Run the mlflow container
local machine
```bash
podman run -d \
  -p 5000:5000 \
  -v ./outputs/mlflow:/mlflow \
  ghcr.io/mlflow/mlflow \
  mlflow server --host 0.0.0.0 --allowed-hosts "*" \
  --backend-store-uri sqlite:////outputs/mlflow/mlflow.db --default-artifact-root /outputs/mlflow/artifacts
```

production
```bash
podman run -d \
 -p 5000:5000 \
 -v /ipazianas/pasquini/training/mlflow:/mlflow \
 ghcr.io/mlflow/mlflow \
 mlflow server --host 0.0.0.0 --allowed-hosts "*" \
 --backend-store-uri sqlite:////mlflow/mlflow.db --default-artifact-root /mlflow/artifacts
```

### Troubleshooting: NVIDIA GPU Error
If you get `unresolvable CDI devices nvidia.com/gpu=all`:
1.  **To run on CPU only**: Remove the `--gpus all` flag from the `podman run` command.
2.  **To fix GPU support**: Ensure NVIDIA Container Toolkit is installed and run:
    `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`
    Then try using `--device nvidia.com/gpu=all` instead of `--gpus all`.