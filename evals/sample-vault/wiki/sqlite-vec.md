# sqlite-vec

**Summary**: Vector search extension for SQLite.

---

## Storage

sqlite-vec stores float32 vectors in a vec0 virtual table. The `chunk_embeddings` table uses float[384].

## Limits

KNN queries reject k values above 4096 with an error.
