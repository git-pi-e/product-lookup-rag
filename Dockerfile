FROM continuumio/miniconda3

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

WORKDIR /app

COPY environment.yml /app/environment.yml
COPY requirements.txt /app/requirements.txt

# Keep package downloads cached between builds. Requires DOCKER_BUILDKIT=1 when building.
RUN --mount=type=cache,target=/opt/conda/pkgs \
  --mount=type=cache,target=/root/.cache/pip \
  conda env create -n rag311 -f /app/environment.yml --yes && \
  conda run -n rag311 pip install --no-cache-dir -r /app/requirements.txt && \
  conda clean --all -y

COPY . /app

# To override DB URL from passed .env file
ENV _IMAGE_NEO4J_URL=neo4j://host.docker.internal:7687

# Create a small startup script inside the image (no external files) and use it
RUN /bin/sh -lc 'cat > /usr/local/bin/docker-start.sh <<'"'SH'"'
#!/bin/sh
# Prefer image-provided override
if [ -n "${_IMAGE_NEO4J_URL:-}" ]; then
  export NEO4J_URL="${_IMAGE_NEO4J_URL}"
fi
# Strip surrounding quotes and rewrite localhost hosts to host.docker.internal
if [ -n "${NEO4J_URL:-}" ]; then
  url="${NEO4J_URL%\"}"
  url="${url#\"}"
  url=$(printf '%s' "$url" | sed -E 's#(://)(127\\.0\\.0\\.1|localhost|0\\.0\\.0\\.0)([:/])#\\1host.docker.internal\\3#')
  export NEO4J_URL="$url"
fi
# Activate conda env and start uvicorn
. /opt/conda/etc/profile.d/conda.sh
conda activate rag311
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
SH'
RUN chmod +x /usr/local/bin/docker-start.sh
ENTRYPOINT ["/usr/local/bin/docker-start.sh"]
# Ensure the conda env is on PATH for subsequent RUN steps and as a helpful default
ENV PATH="/opt/conda/envs/rag311/bin:$PATH"
ENV CONDA_DEFAULT_ENV=rag311

EXPOSE 8000

# Run the application using 'conda run' to guarantee the environment is activated in non-login shells.
# --no-capture-output is used so uvicorn logs appear in the container stdout.



