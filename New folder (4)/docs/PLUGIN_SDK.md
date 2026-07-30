# Plugin SDK

The platform supports extending capabilities without altering core source code.

## Registering Custom AI Skills
You can add custom logic for the ResearchSociety by creating a skill script.
Place your python files in `plugins/skills/` and register them via the `UnifiedRegistry`.

## Adding Custom Vector Stores
Implement the `VectorStore` interface defined in `backend/core/vector_store.py` to add new backends (e.g., Pinecone, Milvus).

## Custom UI Widgets
React components can be registered dynamically. (Documentation for the frontend plugin SDK is coming soon in v1.1).
