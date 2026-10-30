# Reviewed linux/amd64 manifest, not a floating CUDA tag.
FROM nvidia/cuda:13.2.1-devel-ubuntu22.04@sha256:689268c255aad87459d7a8f561a4e264632c43aee333b3176e1ea2fb01665694
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=void \
    CUDA_VISIBLE_DEVICES=-1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/opt/release-env
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git g++ make binutils python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir uv==0.9.11
WORKDIR /opt/release
COPY pyproject.toml uv.lock ./
RUN uv python install 3.12.12 \
    && uv sync --locked --python 3.12.12 \
    && uv pip install --python /opt/release-env/bin/python cmake==4.4.4
ENV PATH=/opt/release-env/bin:$PATH
# Only a GPU-free builder. Source and output are explicit runtime mounts.
WORKDIR /src
