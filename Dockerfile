FROM continuumio/miniconda3

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1
# Defaults for container runtime; override with docker run -e or docker-compose env_file
ENV NEO4J_URL=neo4j://host.docker.internal:7687 \
  NEO4J_DATABASE=neo4j

WORKDIR /app

# Copy environment and project files
COPY environment.yml /app/environment.yml
COPY requirements.txt /app/requirements.txt
COPY . /app

# Create the conda environment and install pip deps
RUN conda env create -f /app/environment.yml && \
  /bin/bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate rag311 && pip install --no-cache-dir -r /app/requirements.txt"

# Ensure the conda env is on PATH
ENV PATH /opt/conda/envs/rag311/bin:$PATH
ENV CONDA_DEFAULT_ENV=rag311

EXPOSE 8000

# Use uvicorn from the conda env
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


