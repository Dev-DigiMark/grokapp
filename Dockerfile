# Use official Python image as base
FROM python:3.11-slim

# Install system dependencies required for PIL, pytesseract, and basic image processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
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
