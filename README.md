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
podman run \
  -v ../GraphAnalysis/resources:/app/data \
  -v ./outputs:/app/outputs \
  -v ./conf:/app/conf \
  -e RUN_ID=dfa1ea604ea911f1b8779c29764a0d1b \
  -e "MLFLOW_TRACKING_URI=http://host.containers.internal:5000" \
  -e "MLFLOW_TRACKING_USERNAME=admin" \
  -e "MLFLOW_TRACKING_PASSWORD=password12345" \
  gnn-pipeline \
  --multirun \
  # model.encoder='gine,gcn,gat,sage' \
  # data.graph_type='reply' \
  # hydra/launcher=joblib \
  # hydra.launcher.n_jobs=2
```

production
```bash
podman run --gpus all \
  --network host \
  -e DATA_DIR=/app/data \
  -v /ipazianas/pasquini/extraction/output:/app/data:ro \
  -v /ipazianas/pasquini/training/outputs:/app/outputs \
  -v ./conf:/app/conf \
  -e "MLFLOW_TRACKING_URI=http://localhost:5000" \
  -e "MLFLOW_TRACKING_USERNAME=admin" \
  -e "MLFLOW_TRACKING_PASSWORD=password12345" \
  gnn-pipeline
  --multirun
  # model.encoder='gine,gcn,gat,sage' \
  # data.graph_type='reply' \
  # hydra/launcher=joblib \
  # hydra.launcher.n_jobs=2
```

custom parameters:
```bash
podman run -v ../GraphAnalysis/resources:/app/data -e RUN_ID=add1322e4c9011f1b7318ae8655e7b49 -v ./outputs:/app/outputs gnn-pipeline --multirun model.encoder='gine,gcn,gat,sage' data.graph_type='reply'
```

### Step 3: Run PostgreSQL for MLflow

#### Local Machine
```bash
podman volume create mlflow-postgres-data
podman run -d \
  --name mlflow-db \
  -e POSTGRES_USER=mlflow \
  -e POSTGRES_PASSWORD=your_robust_password \
  -e POSTGRES_DB=mlflow_db \
  -v mlflow-postgres-data:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:15
```

#### Production
```bash
podman volume create mlflow-postgres-data
podman run -d \
  --name mlflow-db \
  -e POSTGRES_USER=mlflow \
  -e POSTGRES_PASSWORD=your_robust_password \
  -e POSTGRES_DB=mlflow_db \
  -v mlflow-postgres-data:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:15
```

### Step 4: Run the MLflow container (Secured)

To enable authentication, you first need to build the custom MLflow image:

```bash
# Build the image with auth dependencies
podman build -t mlflow-auth -f docker/Dockerfile.mlflow .
```

> [!IMPORTANT]
> The admin password must be at least **13 characters long**.

#### Local Machine
```bash
podman run -d \
-p 5000:5000 \
-v ./outputs/mlflow:/mlflow \
-e MLFLOW_HTTP_ALLOW_ALL_HOSTS=true \
-e MLFLOW_AUTH_CONFIG_PATH=/mlflow/auth_config.ini \
-e MLFLOW_FLASK_SERVER_SECRET_KEY="your_secret" \
mlflow-auth \
mlflow server --host 0.0.0.0 \
--allowed-hosts "*" \
--app-name basic-auth \
--backend-store-uri postgresql://mlflow:your_robust_password@host.containers.internal:5432/mlflow_db \
--default-artifact-root /mlflow/artifacts
```

#### Production
```bash
podman run -d \
 -p 5000:5000 \
 --network host \
 -v /ipazianas/pasquini/training/mlflow:/mlflow \
 -e MLFLOW_AUTH_CONFIG_PATH=/mlflow/auth_config.ini \
 -e MLFLOW_FLASK_SERVER_SECRET_KEY="your_secret" \
 mlflow-auth \
 mlflow server --host 0.0.0.0 \
 --app-name basic-auth \
 --backend-store-uri postgresql://mlflow:your_robust_password@localhost:5432/mlflow_db \
 --default-artifact-root /mlflow/artifacts
```

### Troubleshooting: NVIDIA GPU Error
If you get `unresolvable CDI devices nvidia.com/gpu=all`:
1.  **To run on CPU only**: Remove the `--gpus all` flag from the `podman run` command.
2.  **To fix GPU support**: Ensure NVIDIA Container Toolkit is installed and run:
    `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`
    Then try using `--device nvidia.com/gpu=all` instead of `--gpus all`.