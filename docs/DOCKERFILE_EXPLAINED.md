# Dockerfile (line-by-line)

```dockerfile
# Base image: official Python 3.13 on Debian slim variant (small footprint)
FROM python:3.13-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory for the app inside the container
WORKDIR /app

# Copy dependency list first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies without cache to keep image size minimal
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the project
COPY . /app

# Document the port the app listens on
EXPOSE 8000

# Default command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## How to pass environment variables

- One-off `docker run` with specific variables:

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e NEO4J_URL=bolt://host.docker.internal:7687 \
  -e NEO4J_USER=neo4j \
  -e NEO4J_PASSWORD=your_password \
  product-rag-api:latest
```

- Load variables from `.env` using Docker Compose (recommended):

```yaml
version: "3.9"
services:
  api:
    image: product-rag-api:latest
    env_file:
      - .env
    ports:
      - "8000:8000"
```

This avoids passing each `-e` flag manually and keeps sensitive values out of shell history.
