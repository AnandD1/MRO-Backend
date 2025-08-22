# MRO Backend Docker Setup

This repository contains a Flask application with YOLOv8 AI processing, WebSocket support for Unity clients, and real-time image processing capabilities.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)

## Quick Start

### Using Docker Compose (Recommended)

1. **Build and run the application:**
   ```bash
   docker-compose up --build
   ```

2. **Run in background:**
   ```bash
   docker-compose up -d --build
   ```

3. **Stop the application:**
   ```bash
   docker-compose down
   ```

### Using Docker directly

1. **Build the Docker image:**
   ```bash
   docker build -t mro-backend .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name mro-backend \
     -p 5000:5000 \
     -p 8765:8765 \
     -v ./received_images:/app/received_images \
     mro-backend
   ```

## Access Points

- **Web Interface:** http://localhost:5000
- **WebSocket Endpoint:** ws://localhost:8765
- **API Status:** http://localhost:5000/api/status

## Ports

- `5000`: Flask web application and REST API
- `8765`: WebSocket server for Unity client connections

## Volumes

- `./received_images`: Persists received images from Unity clients

## Environment Variables

- `PYTHONUNBUFFERED=1`: Ensures Python output is sent straight to terminal
- `FLASK_ENV=production`: Sets Flask to production mode

## Health Check

The container includes a health check that verifies the application is responding on the status endpoint every 30 seconds.

## Logs

View application logs:
```bash
# With docker-compose
docker-compose logs -f

# With docker directly
docker logs -f mro-backend
```

## Development

For development with live code reloading, you can mount the source code:

```bash
docker run -d \
  --name mro-backend-dev \
  -p 5000:5000 \
  -p 8765:8765 \
  -v $(pwd):/app \
  -v ./received_images:/app/received_images \
  mro-backend
```

## Troubleshooting

1. **Port conflicts:** If ports 5000 or 8765 are already in use, change the port mapping:
   ```bash
   docker run -p 5001:5000 -p 8766:8765 mro-backend
   ```

2. **Permission issues:** Ensure the `received_images` directory has appropriate permissions:
   ```bash
   chmod 755 received_images
   ```

3. **Memory issues:** The YOLOv8 model requires sufficient memory. Ensure Docker has at least 2GB RAM allocated.

## Model File

Ensure the `best.pt` YOLOv8 model file is present in the project root directory before building the Docker image.
