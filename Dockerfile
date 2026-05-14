FROM python:3.9-slim

# Install system dependencies required for pygame and building wheels
RUN apt-get update && apt-get install -y \
    gcc \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev \
    libx11-6 \
    xvfb \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command to run
CMD ["python", "--version"]
