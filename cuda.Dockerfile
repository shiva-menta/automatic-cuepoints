FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    git \
    software-properties-common \
    ffmpeg \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Use requirements-modal.txt (kept separate for Modal deployments)
COPY requirements-modal.txt ./
RUN uv pip install --system -r requirements-modal.txt

# Install natten (requires torch first)
RUN uv pip install --system https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.4/natten-0.17.4%2Btorch250cu124-cp312-cp312-linux_x86_64.whl

RUN python -c "import allin1; import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

COPY track_interface/cuepoint_engines/modal_app.py ./modal_app.py

RUN mkdir -p /mnt/local
ENV LOCAL_MOUNT_PATH=/mnt/local
ENV TORCH_HOME=/root/cache

CMD ["/bin/bash"]
