# Use an official Python image
FROM python:3.10-slim

# Install system dependencies required for dlib and face_recognition
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    make \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# Expose port (optional for local use)
EXPOSE 5000

# Run the app
CMD ["python", "server.py"]
