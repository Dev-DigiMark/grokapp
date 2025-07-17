# Use official Python image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

COPY .env .env

# Expose port (adjust if your app uses a different port)
EXPOSE 8000

# Set environment variables (optional, can be overridden)
# ENV VAR_NAME=value

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501"]