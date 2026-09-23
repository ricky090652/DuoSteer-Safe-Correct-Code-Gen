# Python 3.9 + PyTorch (CUDA) + CodeQL CLI for the DuoSteer pipeline.
# The host only needs an NVIDIA driver and nvidia-container-toolkit; the CUDA
# runtime ships inside the PyTorch wheel.
FROM python:3.9-slim

# PyTorch wheel index. cu121 works with host drivers >= 530; switch to cu118
# for older drivers (check `nvidia-smi` on the server).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
# Versions used in the paper (see README.md / dataset_construction/README.md).
ARG CODEQL_VERSION=2.25.2
ARG PYTHON_QUERIES_VERSION=1.8.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl unzip git ca-certificates tmux \
    && rm -rf /var/lib/apt/lists/*

# CodeQL CLI + codeql/python-queries pack (downloaded to /root/.codeql/packages).
RUN curl -fsSL -o /tmp/codeql.zip \
        https://github.com/github/codeql-cli-binaries/releases/download/v${CODEQL_VERSION}/codeql-linux64.zip \
    && unzip -q /tmp/codeql.zip -d /opt \
    && rm /tmp/codeql.zip \
    && /opt/codeql/codeql pack download codeql/python-queries@${PYTHON_QUERIES_VERSION}

ENV PATH=/opt/codeql:$PATH \
    CODEQL_BIN=/opt/codeql/codeql \
    CODEQL_QLPACK=/root/.codeql/packages/codeql/python-queries/${PYTHON_QUERIES_VERSION}/Security \
    CODEQL_SUITE_ALL=/root/.codeql/packages/codeql/python-queries/${PYTHON_QUERIES_VERSION}/codeql-suites/python-security-extended.qls

# Install torch from the CUDA index first so requirements.txt does not pull a
# CPU-only or mismatched build.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install torch --index-url ${TORCH_INDEX_URL} \
    && pip install -r /tmp/requirements.txt

# The repository is bind-mounted here by docker-compose.yml.
WORKDIR /workspace
