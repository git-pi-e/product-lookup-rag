FROM continuumio/miniconda3

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1
# Defaults for container runtime; override with docker run -e or docker-compose env_file
# ENV NEO4J_URL=neo4j://host.docker.internal:7687 \
# NEO4J_DATABASE=neo4j

WORKDIR /app

# Copy only dependency manifests first so install layers can be cached independently
COPY environment.yml /app/environment.yml
COPY requirements.txt /app/requirements.txt

# Install dependencies using cache mounts (BuildKit). This keeps package downloads
# cached between builds and avoids re-running expensive installs when only source
# files change. Requires DOCKER_BUILDKIT=1 when building.
RUN --mount=type=cache,target=/opt/conda/pkgs \
  --mount=type=cache,target=/root/.cache/pip \
  conda env create -n rag311 -f /app/environment.yml --yes && \
  conda run -n rag311 pip install --no-cache-dir -r /app/requirements.txt && \
  conda clean --all -y

# Copy application source after installing deps so code changes don't invalidate
# the heavy dependency layers.
COPY . /app

# Ensure the conda env is on PATH for subsequent RUN steps and as a helpful default
ENV PATH="/opt/conda/envs/rag311/bin:$PATH"
ENV CONDA_DEFAULT_ENV=rag311

EXPOSE 8000

# Run the application using 'conda run' to guarantee the environment is activated in non-login shells.
# --no-capture-output is used so uvicorn logs appear in the container stdout.
CMD ["conda", "run", "-n", "rag311", "--no-capture-output", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


