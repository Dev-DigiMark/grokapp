# Use official Python image as base
FROM python:3.11-slim

# Install system dependencies required for OpenCV, Tesseract, and others
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

COPY .env .env

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501"]