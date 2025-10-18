# Concepts and Design Decisions

## What is RAG?

Retrieval Augmented Generation (RAG) supplements an LLM with retrieved, task-relevant data. The LLM is used for understanding and formatting, but answers are grounded in your data store to reduce hallucinations.

- Retrieval target here: Neo4j graph of products and entities (`category`, `color`, `brand`, etc.).
- We embed both products and entities and use cosine similarity in Cypher to filter relevant nodes.

## Why a Graph Database?

Relationships are first-class: products relate to categories, characteristics, brands, colors, and age groups. Graph queries naturally express multi-hop, "items like X that also have Y" queries.

## Agentic vs Deterministic

- Agentic approach (notebook) showed an agent that can plan tool use but may fabricate outputs.
- Deterministic approach (service):
  - LLM only extracts entities and creates embeddings.
  - Retrieval and business logic happen via Cypher templates and fixed workflows.
  - Result: reliability and debuggability, still leveraging LLM strengths.

## Workflow Overview

1. User prompt (e.g., "Suggest some red clothes for adults").
2. LLM extracts entities as strict JSON: `{ "color": "red", "category": "clothes", "age_group": "adults" }`.
3. Build Cypher using entity→relation mapping and cosine similarity conditions.
4. Execute query. If empty, fallback to product similarity using the full prompt embedding.
5. For each match, fetch similar items by shared category and common entities.

## Embeddings

- Model: OpenAI `text-embedding-3-small` (configurable via `.env`).
- Stored on `Product.embedding` and entity label nodes' `embedding` property.
- Vector indexes built using LangChain's `Neo4jVector.from_existing_graph` helper.

## Security and Reliability Considerations

- The LLM output is constrained to JSON, minimizing prompt injection risks.
- All database access is parameterized; embeddings are passed as parameters.
- No direct execution of arbitrary Cypher from the LLM.

## Extensibility

- Add new entity types in `app/core/entities.py`.
- Swap LLM provider or embeddings by updating `LLMService` and `.env`.
- Add reranking, filters, or business rules in `RetrievalService` or `Orchestrator`.

## Why each technology was chosen

- Neo4j (Graph DB): built to model relationships as first-class citizens. For product discovery, relationships (category, brand, color, shared attributes) are the core query surface. It enables pattern matching and similarity constraints directly on connected data.
- LangChain: created to standardize LLM integrations (chat, tools, memory, agents). We leverage it to build an agent interface, tools, and keep LLM integration modular.
- OpenAI embeddings: purpose-built vector representations with strong semantic performance; widely supported and easy to swap.
- FastAPI: modern, async-first Python web framework optimized for developer productivity and OpenAPI generation.
- Docker: standard runtime packaging for reproducible deployments.
- Kubernetes: orchestration for scaling and operating the service.

## Why not MCP here?

Model Context Protocol (MCP) exposes tools/data sources in a structured, server-driven way for LLMs. For this service, the LLM usage is minimal and bounded: extract entities and produce embeddings. Retrieval is deterministic via Cypher and does not require exposing arbitrary tool surfaces to the LLM. MCP could be useful if:

- You wanted the model to dynamically discover multiple data sources/tools across teams/services.
- You needed a standardized catalog and permissioned access layer for many tools.

Given our narrow scope and the desire for deterministic retrieval, MCP adds little value and complexity here.

## Improvements

- Add reranking using a cross-encoder or structured reasoning over retrieved items.
- Add caching (semantic cache) and request-level tracing/observability.
- Add auth, rate limiting, and abuse prevention.
- Batch embeddings, add background jobs for ingestion/indexing.
- Expand entity ontology and add guardrails/validation on LLM JSON outputs.


