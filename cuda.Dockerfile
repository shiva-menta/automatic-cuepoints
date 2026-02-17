FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies (added ffmpeg for audio format support)
RUN apt-get update && \
    apt-get install -y \
    git \
    software-properties-common \
    ffmpeg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    python -m ensurepip --upgrade && \
    python -m pip install --upgrade pip

# Use requirements file instead of Pipfile (no pipenv needed)
COPY requirements-modal.txt ./
RUN pip install --no-cache-dir -r requirements-modal.txt

# Install natten (requires torch first)
RUN pip install --no-cache-dir https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.4/natten-0.17.4%2Btorch250cu124-cp312-cp312-linux_x86_64.whl

# Verify imports work at build time
RUN python -c "import allin1; import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Note: Modal serializes function code during deploy, so no project files are
# strictly required. We copy modal_app.py only for local testing convenience.
COPY track_interface/cuepoint_engines/modal_app.py ./modal_app.py

RUN mkdir -p /mnt/local
ENV LOCAL_MOUNT_PATH=/mnt/local
ENV TORCH_HOME=/root/cache

CMD ["/bin/bash"]
