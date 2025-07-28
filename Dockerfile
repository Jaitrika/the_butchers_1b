# Use official Python image with ML libraries support
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies needed for PyMuPDF, ML libraries, and timezone
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone explicitly (this affects datetime.now())
ENV TZ=Asia/Kolkata

# Copy requirements file first (helps with Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 🔽 Pre-download embedding and cross-encoder models into container cache
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/root/.cache'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', cache_folder='/root/.cache')"

# Copy the application files
COPY app.py .
COPY parser.py .
COPY /input/input.json .

# Copy the documents folder
COPY documents/ ./documents/

# Create output directory for results
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the main application
CMD ["python", "app.py"]
