FROM continuumio/miniconda3

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1
# Defaults for container runtime; override with docker run -e or docker-compose env_file
# ENV NEO4J_URL=neo4j://host.docker.internal:7687 \
  # NEO4J_DATABASE=neo4j

WORKDIR /app

# Copy only the files needed to create the conda environment first (cache-friendly)
COPY environment.yml /app/environment.yml
COPY requirements.txt /app/requirements.txt

# Copy the rest of the project
COPY . /app

# Create the conda environment named 'rag311' and install pip deps into it.
# Use non-interactive flags and ensure we can run pip inside the created env via 'conda run'.
RUN conda env create -n rag311 -f /app/environment.yml --quiet --yes && \
  conda run -n rag311 pip install --no-cache-dir -r /app/requirements.txt

# Ensure the conda env is on PATH for subsequent RUN steps and as a helpful default
ENV PATH /opt/conda/envs/rag311/bin:$PATH
ENV CONDA_DEFAULT_ENV=rag311

EXPOSE 8000

# Run the application using 'conda run' to guarantee the environment is activated in non-login shells.
# --no-capture-output is used so uvicorn logs appear in the container stdout.
CMD ["conda", "run", "-n", "rag311", "--no-capture-output", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


