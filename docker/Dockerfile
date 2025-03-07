# Use the slim version of Python 3.9 for a smaller image size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    libomp5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first for better caching
COPY requirements.txt .

# Upgrade pip, setuptools, and install core dependencies first
RUN pip install --no-cache-dir --upgrade pip setuptools setuptools-scm

# Install dependencies in a structured order

# Step 1: Install typing-inspect and typing-extensions first (needed for Pydantic)
RUN pip install --no-cache-dir typing-inspect==0.8.0 typing_extensions==4.5.0

# Step 2: Install Pydantic (before spaCy)
RUN pip install --no-cache-dir pydantic==1.10.11

# Step 3: Install NumPy and other core libraries
RUN pip install --no-cache-dir numpy==1.26.4 wheel cython

# Step 4: Install spaCy and its dependencies separately
RUN pip install --no-cache-dir spacy==3.5.3

# Step 5: Install the Ukrainian language model for spaCy
RUN python -m spacy download uk_core_news_md

# Step 6: Install `huggingface_hub` with a compatible version
RUN pip install --no-cache-dir huggingface_hub==0.10.1

# Step 7: Install the rest of the project dependencies
RUN pip install --no-cache-dir \
    fastapi==0.68.2 \
    uvicorn==0.34.0 \
    joblib==1.4.2 \
    sentence-transformers==3.4.1 \
    scikit-learn==1.6.1 \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

# Step 8: Install BERT-related libraries
RUN pip install --no-cache-dir \
    bertopic==0.16.4 \
    transformers==4.48.2

# Step 9: Install seqeval explicitly before pyabsa to avoid dependency issues
RUN pip install --no-cache-dir seqeval==1.2.2

# Step 10: Install pyabsa (Aspect-Based Sentiment Analysis)
RUN pip install --no-cache-dir pyabsa==2.4.1

# Copy the application source code
COPY . .

# Ensure models directory exists
RUN mkdir -p models

# Expose the FastAPI application port
EXPOSE 8000

# Command to start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
