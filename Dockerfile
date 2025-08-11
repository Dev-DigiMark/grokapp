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
    libgthread-2.0-0 \
    libgtk-3-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libv4l-0 \
    libxvidcore4 \
    libx264-155 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff5 \
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
