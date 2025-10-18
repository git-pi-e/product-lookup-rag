# Amazon Product Lookup RAG

RAG FastAPI service on a Neo4j product graph: ask for products in natural language; it extracts entities, queries Neo4j with embeddings, and returns grounded results plus similar items. This project targets Python 3.11 in local and container environments.

## Requirements

- Python 3.11+ (Docker uses Python 3.13)
- Neo4j 5.x reachable
- OpenAI API key

## Setup (conda recommended)

Copy `example.env` to `.env` and fill values:

```env
OPENAI_API_KEY=""
NEO4J_USER=""
NEO4J_PASSWORD=""
NEO4J_URL=""
NEO4J_DATABASE=""
EMBEDDINGS_MODEL=text-embedding-3-small
```

## Run locally (Conda)

1. Create environment and install compiled packages via conda-forge:

```bash
conda create -n rag311 -c conda-forge python=3.11 -y
conda activate rag311
conda install -c conda-forge pandas numpy scipy neo4j -y
```

2. Install pip packages (LangChain family + others):

```bash
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

3. Ingest data and build embeddings:

```bash
python scripts/ingest.py
python scripts/setup_embeddings.py
```

4. Run API:

```bash
uvicorn app.main:app --reload
```

Test:

```bash
curl -s -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Suggest some red clothes for adults"}' | jq
```

## Docker

Build and run:

```bash
docker build -t product-rag-api:latest .
# Option A: pass envs individually
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e NEO4J_URL=bolt://host.docker.internal:7687 \
  -e NEO4J_USER=neo4j \
  -e NEO4J_PASSWORD=your_password \
  product-rag-api:latest
# Option B: use compose to load .env
docker compose up --build
```

## Kubernetes

```bash
kubectl apply -f k8s/secret.example.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl port-forward svc/product-rag-api 8000:80
```

## More docs

- docs/CONCEPTS.md (RAG, design, workflow)
- docs/DOCKERFILE_EXPLAINED.md (line-by-line Dockerfile)


