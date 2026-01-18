FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Set working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install dependencies from Pipfile
RUN pipenv lock --clear
RUN pipenv install --system

# Install natten (requires torch to be installed first)
RUN pip install https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.4/natten-0.17.4%2Btorch250cpu-cp312-cp312-linux_x86_64.whl

# Create mount point for local filesystem
RUN mkdir -p /mnt/local

# Copy application code
COPY . .

# Set the mount point as an environment variable for easy reference
ENV LOCAL_MOUNT_PATH=/mnt/local

CMD ["/bin/bash"]
