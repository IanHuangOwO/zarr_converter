# Dockerfile for Zarr-Tiff Format Transform
FROM python:3.10-slim

# Install Package dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /workspace

COPY __main__.py .