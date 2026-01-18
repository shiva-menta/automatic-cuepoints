FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default and install pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    python -m ensurepip --upgrade && \
    python -m pip install --upgrade pip

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install dependencies from Pipfile
RUN pipenv lock --clear
RUN pipenv install --system

# Install natten (requires torch to be installed first)
RUN pip install https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.4/natten-0.17.4%2Btorch250cu124-cp312-cp312-linux_x86_64.whl

# Create mount point for local filesystem
RUN mkdir -p /mnt/local

# Copy application code
COPY . .

# Set the mount point as an environment variable for easy reference
ENV LOCAL_MOUNT_PATH=/mnt/local

CMD ["/bin/bash"]
