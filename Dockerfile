FROM python:3.12-slim

WORKDIR /app

# Set env variables
ENV PYTHONUNBUFFERED=1
# Uncomment if you need to use k2eg to communicate with EPICS
# ENV K2EG_PYTHON_CONFIGURATION_PATH_FOLDER=/app/config

# Set EPICS environment variables
ENV EPICS_CA_AUTO_ADDR_LIST=NO
ENV EPICS_CA_ADDR_LIST=172.24.3.10:5068

# Install system dependencies, then clean up
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install --no-install-recommends -qy git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
# Torch is a lume-model dependency. Here we are installing the CPU version to save on image size.
# If you need GPU support, change the index-url to the appropriate one for your CUDA version.
RUN pip install --upgrade pip && \
    pip install torch~=2.7.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache

COPY . .