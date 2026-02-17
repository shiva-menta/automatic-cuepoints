FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Install natten (requires torch first)
RUN uv pip install --system https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.4/natten-0.17.4%2Btorch250cpu-cp312-cp312-linux_x86_64.whl

RUN mkdir -p /mnt/local
COPY . .

ENV LOCAL_MOUNT_PATH=/mnt/local
CMD ["/bin/bash"]
