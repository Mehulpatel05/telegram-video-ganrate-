FROM python:3.11-slim

# Install system dependencies & ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements & install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1
ENV PORT=8080

# Expose web port for Render keepalive
EXPOSE 8080

# Start bot + keepalive webserver
CMD [python, telegram_bot.py]
