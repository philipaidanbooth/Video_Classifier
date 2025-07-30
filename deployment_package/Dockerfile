# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/json_outputs /app/videos

# Expose port
EXPOSE 8080

# Set environment variable for port
ENV PORT=8080

# Make startup script executable
RUN chmod +x start.sh

# Run the application
CMD ["./start.sh"] 