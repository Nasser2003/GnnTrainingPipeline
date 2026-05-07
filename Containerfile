# Base image: PyTorch 2.6.0 + CUDA 12.4 + Python 3.11
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .

# Install PyG CUDA wheels then remaining deps
RUN pip install --no-cache-dir \
        torch-scatter==2.1.2 \
        torch-sparse==0.6.18 \
        torch-cluster==1.6.3 \
        -f https://data.pyg.org/whl/torch-2.6.0+cu124.html \
    && pip install --no-cache-dir torch-geometric \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY conf/ ./conf/

ENV PYTHONPATH=/app/src

# /data      → GraphAnalysis resources (read-only)
# /app/output → model checkpoints
# /app/mlruns → MLflow tracking store
VOLUME ["/data", "/app/output", "/app/mlruns"]

ENTRYPOINT ["python", "src/main.py"]
CMD ["data.data_dir=/data", \
     "output.output_dir=/app/output", \
     "output.mlflow_tracking_uri=sqlite:////app/mlruns/mlflow.db"]
