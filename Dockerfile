# Dockerfile for NFL ML Predictions Backend
# Using Python 3.12-slim for a balance of size and compatibility
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV DATA_DIR=data/datasets
ENV SCHEDULE_PATH=data/Nfl_schedule_2025.csv
ENV MODELS_DIR=models

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure
# We copy everything to ensure the backend/ and frontend/ folders are present
# as the app expects relative paths for some artifacts.
COPY . .

# Expose the default local port. Production platforms can override PORT.
EXPOSE 8000

# Command to run the application
# Using uvicorn as specified in the Procfile/tasks
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
