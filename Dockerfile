# Use official Python image as base
FROM python:3.11-slim

# Install system dependencies required for OpenCV, Tesseract, pyzbar, and others
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    tesseract-ocr \
    libzbar0 \
    libzbar-dev \
    libtesseract5 \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Copy .env file if it exists
COPY .env .env

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501"]
