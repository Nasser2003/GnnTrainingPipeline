# Base image: PyTorch 2.6.0 + CUDA 12.4 + Python 3.11
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install pyg-lib -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

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
COPY .git/ ./.git/

ENV DATA_DIR=/app/data

# /app/data    → Source features (from GraphAnalysis)
# /app/outputs → Training results (models, logs, graphs)
VOLUME ["/app/data", "/app/outputs"]

ENTRYPOINT ["python", "src/main.py"]
# No default CMD needed, Hydra will read DATA_DIR from environment
CMD []
