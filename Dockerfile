# ============================================================
# Dockerfile for
# "Harnessing BERT for Advanced Email Filtering in Cybersecurity"
#
# This image is intended for:
# - running ML/DL/BERT experiments
# - executing the test suite
# - interactive work (e.g., notebooks) if needed
#
# NOTE:
# - For GPU acceleration, run the container with NVIDIA runtime
#   (e.g., `--gpus all` with recent Docker + nvidia-container-toolkit).
# - The base image is CPU-friendly; you can swap to a CUDA image
#   if you need tighter control over GPU drivers/toolkit.
# ============================================================

FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------

# Install basic tools and build dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Optional but recommended: matplotlib for plotting utilities & notebooks
# Uncomment if you want plotting inside the container.
# RUN pip install --no-cache-dir matplotlib jupyter

# ------------------------------------------------------------
# Project source code
# ------------------------------------------------------------

# Copy the entire project into the container
# (adjust if you prefer a more selective copy)
COPY . /app

# Ensure experiments directories exist (logs, models, results, artifacts)
RUN mkdir -p experiments/logs \
    experiments/models \
    experiments/results \
    experiments/artifacts \
    experiments/bert \
    data/raw \
    data/processed \
    notebooks \
    docs \
    tests

# ------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------

# Default env vars you might want to tweak
ENV PYTHONPATH="/app:${PYTHONPATH}" \
    # Set to 1 if you ever want to run BERT tests inside the container
    RUN_BERT_TESTS=0

# ------------------------------------------------------------
# Default command
# ------------------------------------------------------------

# By default we drop into a shell; you can override the command when running.
# Example usages:
#   docker build -t bert-email-filter .
#   docker run --rm -it bert-email-filter bash
#
# Inside the container, you can then run:
#   python -m scripts.run_all_experiments
#   pytest
#
CMD ["/bin/bash"]
