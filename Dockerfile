FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal system libs needed by opencv (and libGL)
RUN apt-get update

# Copy requirements and install Python deps
COPY requirements.txt ./

# IMPORTANT:
# - Prefer opencv-python-headless in requirements.txt (see notes below).
# - Pin torch to the appropriate CPU/CUDA build that you want.
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 8501


CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]