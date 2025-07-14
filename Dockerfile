FROM python:3.12-slim

WORKDIR /app

# Set env variable for k2eg
# Uncomment if you need to use k2eg to communicate with EPICS
# ENV K2EG_PYTHON_CONFIGURATION_PATH_FOLDER=/app/config

# Install system dependencies, then clean up
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install --no-install-recommends -qy git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# If you need torch, it's recommended to install it without GPU
# support to keep the image size small, unless you have a specific need for it.
RUN pip install torch~=2.7.1 --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache

COPY . .