# Chapter 32: Vector Databases

## Why Vector Search?

Your search bar returns exact keyword matches:

```
User searches: "affordable transportation"
Keyword match finds: documents with words "affordable" and "transportation"
Misses: "cheap cars", "budget vehicles", "inexpensive bikes"

Result: 3 results when 300 semantically relevant documents exist
```

Traditional search fails because:
- **Synonyms** ("cheap" vs "affordable")
- **Related concepts** ("vehicle" vs "car" vs "bike")
- **Intent** ("affordable transportation" means budget vehicles)
- **Multiple languages** (same meaning, different words)

Vector search solves this by representing meaning as numbers, enabling semantic similarity search.

---

## Vector Embeddings: The Foundation

**The Problem:** Computers can't understand "meaning". They need numbers.

**How It Works:**

```
Text: "The cat sits on the mat"
         ↓
   Embedding Model (BERT, OpenAI, etc)
         ↓
Vector: [0.23, -0.41, 0.67, ..., 0.12]  (768 dimensions)

Similar text: "A feline rests on the rug"
         ↓
Vector: [0.25, -0.39, 0.65, ..., 0.11]  (close in space!)

Unrelated: "Quantum physics equations"
         ↓
Vector: [-0.82, 0.15, -0.34, ..., 0.91]  (far away)
```

**Embeddings map semantically similar items close together in high-dimensional space.**

```python
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions

# Generate embeddings
texts = [
    "The cat sits on the mat",
    "A feline rests on the rug",
    "Quantum physics equations"
]

embeddings = model.encode(texts)
print(embeddings.shape)  # (3, 384)
print(embeddings[0][:5])  # [0.234, -0.412, 0.671, ...]
```

**Embedding dimensions:**
- Small (384): Fast, less precise (all-MiniLM-L6-v2)
- Medium (768): Balanced (BERT, sentence-transformers)
- Large (1536): Accurate, slower (OpenAI text-embedding-3-small)
- XL (3072+): Highest quality (OpenAI text-embedding-3-large)

---

## Similarity Metrics

**The Problem:** How do we measure "closeness" in high-dimensional space?

### 1. Cosine Similarity

Measures the angle between vectors (ignores magnitude).

```
     Vector A
        ↗
       /  θ (small angle = similar)
      /
     /____→ Vector B
```

```python
import numpy as np

def cosine_similarity(vec_a, vec_b):
    """Range: [-1, 1]. 1 = identical, 0 = orthogonal, -1 = opposite"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

# Example
vec1 = np.array([1, 2, 3])
vec2 = np.array([2, 4, 6])  # Same direction, different magnitude
vec3 = np.array([-1, -2, -3])  # Opposite direction

print(cosine_similarity(vec1, vec2))  # 1.0 (identical direction)
print(cosine_similarity(vec1, vec3))  # -1.0 (opposite)
```

**When to use:** Text embeddings (magnitude doesn't matter, only meaning)
**When NOT to use:** When magnitude is significant (e.g., image features)

### 2. Euclidean Distance (L2)

Straight-line distance between points.

```python
def euclidean_distance(vec_a, vec_b):
    """Range: [0, ∞]. 0 = identical, larger = more different"""
    return np.sqrt(np.sum((vec_a - vec_b) ** 2))

# Example
vec1 = np.array([0, 0, 0])
vec2 = np.array([3, 4, 0])

print(euclidean_distance(vec1, vec2))  # 5.0
```

**When to use:** Image embeddings, spatial data
**When NOT to use:** High dimensions (curse of dimensionality)

### 3. Dot Product

Combines angle and magnitude.

```python
def dot_product_similarity(vec_a, vec_b):
    """Range: [-∞, ∞]. Higher = more similar"""
    return np.dot(vec_a, vec_b)
```

**When to use:** When vectors are pre-normalized
**When NOT to use:** Unnormalized vectors (magnitude skews results)

**Comparison:**

| Metric | Normalized Vectors | Magnitude Matters | Best For |
|--------|-------------------|-------------------|----------|
| Cosine | No | No (direction only) | Text search |
| Euclidean | Yes (helps) | Yes | Images, spatial |
| Dot Product | Required | Yes (when normalized) | Fast cosine alternative |

---

## Approximate Nearest Neighbor Search

**The Problem:** Exact search is O(n) - compare query to ALL vectors.

```
Dataset: 10 million vectors × 768 dimensions
Query: Compare to all 10M vectors
Time: ~30 seconds PER QUERY

Unusable for production!
```

**Solution:** Approximate Nearest Neighbor (ANN) algorithms trade accuracy for speed.

```
Exact: O(n) - check everything
ANN:   O(log n) - check small subset

10M vectors:
- Exact: 10,000,000 comparisons
- ANN:   ~1,000 comparisons (99.5% recall)
```

---

## HNSW: Hierarchical Navigable Small World

**The Problem:** How to quickly navigate millions of vectors?

**How It Works:**

HNSW builds a multi-layer graph structure.

```
Layer 2 (sparse):  A ─────────────────── E

Layer 1 (medium):  A ───── C ───── D ─── E
                           │       │
Layer 0 (dense):   A ─ B ─ C ─ D ─ E ─ F ─ G
                   │   │   │   │   │   │   │
                   H   I   J   K   L   M   N

Search from top layer (big jumps) → narrow down → bottom layer (precise)
```

**Search algorithm:**

```python
def hnsw_search(query_vector, num_results, entry_point, max_layer):
    current_nearest = entry_point

    # Start from top layer, work down
    for layer in range(max_layer, -1, -1):
        # Greedy search: move to nearest neighbor
        changed = True
        while changed:
            changed = False
            for neighbor in current_nearest.neighbors[layer]:
                if distance(query_vector, neighbor) < distance(query_vector, current_nearest):
                    current_nearest = neighbor
                    changed = True

    # Return k nearest from bottom layer
    return get_k_nearest(current_nearest, num_results, layer=0)
```

**Parameters:**

```python
index_params = {
    "M": 16,        # Connections per node (higher = better recall, slower build)
    "efConstruction": 200,  # Candidates during build (higher = better quality)
    "efSearch": 100  # Candidates during search (higher = better recall)
}

# Trade-offs
M=16, efSearch=50:   95% recall, 10ms
M=32, efSearch=200:  99% recall, 40ms
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Recall | 95-99% with tuning | Not exact |
| Speed | 10-50ms for 10M vectors | Build time slow |
| Memory | ~2x vector size | High memory |
| Updates | Supports insert/delete | Slower than read-only |

**When to use:** High recall needed, memory available, sub-50ms latency
**When NOT to use:** Exact results required, memory constrained

---

## IVF: Inverted File Index

**The Problem:** HNSW uses too much memory for billion-scale datasets.

**How It Works:**

Cluster vectors, search only relevant clusters.

```
1. Training phase: Cluster vectors into N groups

┌─────────────────────────────────────────────┐
│  Cluster 1    Cluster 2       Cluster 3     │
│   ●●●●●        ●●●●●●          ●●●         │
│   ●●●●         ●●●●●●          ●●●●        │
│                                             │
│  Centroid 1   Centroid 2      Centroid 3   │
│     ★             ★               ★         │
└─────────────────────────────────────────────┘

2. Query: Find nearest centroid(s), search only those clusters

Query ◄───────── Nearest centroid is #2
                 Only search Cluster 2 (1/3 of data!)
```

```python
import faiss
import numpy as np

# Create IVF index
dimension = 768
num_clusters = 100  # More clusters = faster, lower recall

# Training data (sample of your vectors)
training_data = np.random.random((10000, dimension)).astype('float32')

# Create quantizer (finds nearest cluster)
quantizer = faiss.IndexFlatL2(dimension)

# Create IVF index
index = faiss.IndexIVFFlat(quantizer, dimension, num_clusters)

# Train: learn cluster centroids
index.train(training_data)

# Add vectors
vectors = np.random.random((1000000, dimension)).astype('float32')
index.add(vectors)

# Search
query = np.random.random((1, dimension)).astype('float32')
index.nprobe = 5  # Search top 5 clusters (out of 100)
distances, indices = index.search(query, k=10)

print(f"Found {len(indices[0])} results")
```

**Parameters:**

```
num_clusters: 100 for 1M vectors, 1000 for 100M, 10000 for 1B
nprobe: Search N clusters (higher = better recall, slower)

nprobe=1:  Fast, ~70% recall
nprobe=10: Medium, ~90% recall
nprobe=50: Slow, ~98% recall
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Memory | Low (1x vector size) | Lower recall than HNSW |
| Speed | Very fast with low nprobe | Recall drops quickly |
| Scale | Billions of vectors | Requires training |
| Updates | Slow (rebuild clusters) | Not real-time friendly |

**When to use:** Billion-scale datasets, memory constrained, batch updates
**When NOT to use:** Real-time updates, high recall required (>95%)

---

## Product Quantization (PQ): Compression

**The Problem:** 1 billion vectors × 768 dims × 4 bytes = 3TB of memory!

**How It Works:**

Compress vectors by quantizing sub-vectors.

```
Original vector (768 dims × 4 bytes = 3072 bytes):
[0.23, -0.41, 0.67, ..., 0.12]

Split into 8 sub-vectors of 96 dims each:
sub1: [0.23, -0.41, ...]  96 dims
sub2: [0.15, 0.32, ...]   96 dims
...
sub8: [..., 0.12]         96 dims

For each sub-vector:
- Find nearest centroid from 256 learned centroids
- Store centroid ID (1 byte) instead of 96 floats (384 bytes)

Compressed: 8 bytes (96x smaller!)
```

```python
import faiss

dimension = 768
num_subquantizers = 8  # Split into 8 chunks
bits_per_code = 8      # 256 centroids per chunk

# Create PQ index
index = faiss.IndexPQ(dimension, num_subquantizers, bits_per_code)

# Train on sample data
training_data = np.random.random((10000, dimension)).astype('float32')
index.train(training_data)

# Add vectors (automatically compressed)
vectors = np.random.random((1000000, dimension)).astype('float32')
index.add(vectors)

print(f"Memory per vector: {num_subquantizers} bytes")
print(f"Compression: {dimension * 4 / num_subquantizers}x")

# Search (decompresses on-the-fly)
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=10)
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Memory | 96x compression (768 dims → 8 bytes) | 5-10% recall loss |
| Speed | Faster (less data to load) | Decompression overhead |
| Accuracy | Good for top-k | Distance approximation |
| Scale | Billions on single machine | Not lossless |

**Combine IVF + PQ for best of both:**

```python
# IVF for fast cluster search + PQ for compression
index = faiss.IndexIVFPQ(quantizer, dimension, num_clusters, num_subquantizers, bits_per_code)
```

**When to use:** Memory constrained, acceptable recall loss (90-95%)
**When NOT to use:** Exact distances needed, high recall critical

---

## Metadata Filtering

**The Problem:** "Find similar documents, but only from the last 7 days"

```
Without filtering:
1. Vector search finds 1000 nearest neighbors
2. Filter by date
3. Return ~10 results (990 discarded!)

With filtering:
1. Filter to documents from last 7 days
2. Vector search within filtered set
3. Return 10 results (efficient!)
```

**Pre-filtering (better):**

```python
# Qdrant example
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range

client = QdrantClient("localhost", port=6333)

results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="published_date",
                range=Range(
                    gte="2025-12-05",  # Last 7 days
                    lt="2025-12-12"
                )
            ),
            FieldCondition(
                key="category",
                match={"value": "technology"}
            )
        ]
    ),
    limit=10
)
```

**Post-filtering (fallback):**

```python
# When pre-filtering not supported
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=1000  # Get more results
)

# Filter in application
filtered_results = [
    r for r in results
    if r.payload["published_date"] >= "2025-12-05"
    and r.payload["category"] == "technology"
][:10]
```

**Index structure for filtering:**

```
Option 1: Partitioning
- Separate index per category
- Fast filtering, complex management

Option 2: Compound index
- Single index with metadata
- Flexible filtering, slower with many filters
```

---

## Hybrid Search: Vector + Keyword

**The Problem:** Vector search misses exact matches.

```
Query: "iPhone 15 Pro Max"
Vector search finds: "Latest Apple flagship phone" (semantically similar)
Misses: Exact product listing with "iPhone 15 Pro Max" in title
```

**Solution:** Combine vector (semantic) + BM25 (keyword) search.

```
┌─────────────────────────────────────────────────┐
│                  Query                           │
└─────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐      ┌─────────────────┐
│  Vector Search  │      │   BM25 Search   │
│  (semantic)     │      │   (keyword)     │
└─────────────────┘      └─────────────────┘
         │                          │
    [Doc A: 0.89]            [Doc B: 0.92]
    [Doc B: 0.85]            [Doc A: 0.78]
    [Doc C: 0.82]            [Doc D: 0.71]
         │                          │
         └──────────┬───────────────┘
                    ▼
           ┌───────────────┐
           │ Reciprocal    │
           │ Rank Fusion   │
           └───────────────┘
                    │
                    ▼
         [Doc B, Doc A, Doc C, Doc D]
```

**Reciprocal Rank Fusion (RRF):**

```python
def reciprocal_rank_fusion(vector_results, keyword_results, k=60):
    """
    Combine rankings from multiple sources.
    k=60 is empirically good default.
    """
    scores = {}

    # Score from vector search
    for rank, doc_id in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # Score from keyword search
    for rank, doc_id in enumerate(keyword_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # Sort by combined score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Example
vector_results = ["doc_a", "doc_b", "doc_c"]  # Ranked by similarity
keyword_results = ["doc_b", "doc_a", "doc_d"]  # Ranked by BM25

combined = reciprocal_rank_fusion(vector_results, keyword_results)
# Result: [("doc_b", 0.0328), ("doc_a", 0.0328), ("doc_c", 0.0164), ("doc_d", 0.0164)]
```

**Alternative: Weighted average:**

```python
def weighted_hybrid(vector_score, keyword_score, alpha=0.7):
    """
    alpha=0.7: 70% vector, 30% keyword
    Tune based on your use case
    """
    return alpha * vector_score + (1 - alpha) * keyword_score
```

**When to use hybrid:**
- Product search (exact SKU matching + similar products)
- Document search (exact phrase + semantic meaning)
- Legal/medical (terminology precision + concept similarity)

---

## Indexing Strategies

### Strategy 1: Flat Index (Exact Search)

```python
import faiss

dimension = 768
index = faiss.IndexFlatL2(dimension)  # Brute force, exact results

vectors = np.random.random((100000, dimension)).astype('float32')
index.add(vectors)

# Search: compares query to ALL vectors
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=10)
```

**Performance:**
```
Dataset size: 100K vectors
Search time:  ~50ms
Recall:       100% (exact)
Memory:       768 dims × 4 bytes × 100K = 307MB
```

**When to use:** <100K vectors, exact results required
**When NOT to use:** >1M vectors (too slow)

### Strategy 2: HNSW (High Recall)

```python
dimension = 768
index = faiss.IndexHNSWFlat(dimension, 32)  # M=32 connections
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 128

vectors = np.random.random((10000000, dimension)).astype('float32')
index.add(vectors)
```

**Performance:**
```
Dataset size: 10M vectors
Search time:  ~20ms
Recall:       98%
Memory:       768 dims × 4 bytes × 10M × 2 = 61GB
```

**When to use:** <100M vectors, high recall needed (95%+), memory available
**When NOT to use:** Memory constrained, billion-scale

### Strategy 3: IVF + PQ (Scale)

```python
dimension = 768
num_clusters = 4096
num_subquantizers = 64
bits = 8

quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(quantizer, dimension, num_clusters, num_subquantizers, bits)

# Train
training_data = np.random.random((100000, dimension)).astype('float32')
index.train(training_data)

# Add
vectors = np.random.random((1000000000, dimension)).astype('float32')
index.add(vectors)

# Search
index.nprobe = 16  # Check 16 clusters
distances, indices = index.search(query, k=10)
```

**Performance:**
```
Dataset size: 1B vectors
Search time:  ~100ms
Recall:       90%
Memory:       64 bytes × 1B = 64GB (vs 3TB uncompressed!)
```

**When to use:** Billion-scale, memory constrained, 90%+ recall acceptable
**When NOT to use:** Real-time updates, exact results required

**Comparison:**

| Strategy | Max Vectors | Search Time | Recall | Memory | Best For |
|----------|-------------|-------------|--------|--------|----------|
| Flat | 100K | 50ms | 100% | 1x | Exact search |
| HNSW | 100M | 20ms | 98% | 2x | High recall |
| IVF | 100M | 50ms | 95% | 1x | Balanced |
| IVF+PQ | 1B+ | 100ms | 90% | 0.05x | Scale |

---

## Vector Database Comparison

| Feature | Pinecone | Weaviate | Milvus | Qdrant | pgvector |
|---------|----------|----------|--------|--------|----------|
| **Deployment** | Cloud only | Self-hosted or cloud | Self-hosted | Self-hosted or cloud | Postgres extension |
| **Index Type** | Proprietary | HNSW | IVF, HNSW, more | HNSW | IVF, HNSW |
| **Max Scale** | Billions | Billions | Billions | Billions | Millions |
| **Filtering** | Pre-filter | Pre-filter | Pre-filter | Pre-filter | SQL WHERE |
| **Hybrid Search** | No | Yes (built-in) | Yes | Yes | Manual (BM25 + vector) |
| **Multi-tenancy** | Namespaces | Tenants | Partitions | Collections | Schemas |
| **Updates** | Real-time | Real-time | Real-time | Real-time | ACID transactions |
| **Cost** | $$ (pay per use) | $ (self-host) | Free (self-host) | Free (self-host) | Free (Postgres) |
| **Best For** | Managed, no ops | Production flexibility | Large scale research | Performance, filtering | Existing Postgres stack |

**Detailed comparison:**

**Pinecone:**
```python
import pinecone

pinecone.init(api_key="your-key", environment="us-west1-gcp")
index = pinecone.Index("my-index")

# Upsert vectors
index.upsert(vectors=[
    ("id1", [0.1, 0.2, ...], {"category": "tech"}),
    ("id2", [0.3, 0.4, ...], {"category": "sports"})
])

# Query with filter
results = index.query(
    vector=[0.5, 0.6, ...],
    filter={"category": {"$eq": "tech"}},
    top_k=10
)
```

Pros: Zero ops, auto-scaling, good docs
Cons: Vendor lock-in, expensive at scale, no hybrid search

**Weaviate:**
```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Create schema
schema = {
    "class": "Document",
    "vectorizer": "text2vec-openai",
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "category", "dataType": ["string"]}
    ]
}
client.schema.create_class(schema)

# Query with hybrid search
results = client.query.get("Document", ["content", "category"]) \
    .with_hybrid(query="machine learning", alpha=0.5) \
    .with_where({"path": ["category"], "operator": "Equal", "valueString": "tech"}) \
    .with_limit(10) \
    .do()
```

Pros: Hybrid search, modular vectorizers, GraphQL API
Cons: Complex setup, resource intensive

**Milvus:**
```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

connections.connect(host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100)
]
schema = CollectionSchema(fields)
collection = Collection("documents", schema)

# Create index
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 1024}
}
collection.create_index("embedding", index_params)

# Search
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 16}},
    limit=10,
    expr='category == "tech"'
)
```

Pros: Multiple index types, high performance, active development
Cons: Complex architecture, steep learning curve

**Qdrant:**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

# Upsert
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],
            payload={"category": "tech", "text": "..."}
        )
    ]
)

# Search with filter
results = client.search(
    collection_name="documents",
    query_vector=[0.5, 0.6, ...],
    query_filter=Filter(
        must=[FieldCondition(key="category", match={"value": "tech"})]
    ),
    limit=10
)
```

Pros: Fast, excellent filtering, simple API, Rust performance
Cons: Smaller community, fewer integrations

**pgvector:**
```sql
-- Enable extension
CREATE EXTENSION vector;

-- Create table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    category VARCHAR(100),
    embedding vector(768)
);

-- Create index
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Query
SELECT id, content, 1 - (embedding <=> '[0.1, 0.2, ...]') AS similarity
FROM documents
WHERE category = 'tech'
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 10;
```

Pros: Postgres transactions, SQL familiarity, no new infrastructure
Cons: Limited scale (<1M vectors), slower than specialized DBs

**When to use what:**

- **Pinecone:** Prototype quickly, no ML ops team, budget available
- **Weaviate:** Need hybrid search, modular architecture, Kubernetes
- **Milvus:** Billion-scale, research use case, performance critical
- **Qdrant:** Production app, complex filtering, want simplicity
- **pgvector:** Already on Postgres, <1M vectors, need ACID

---

## Key Concepts Checklist

- [ ] Explain what vector embeddings represent (semantic meaning as numbers)
- [ ] Compare cosine vs euclidean vs dot product similarity
- [ ] Describe HNSW algorithm and multi-layer graph structure
- [ ] Explain IVF clustering for approximate search
- [ ] Understand Product Quantization for compression
- [ ] Design hybrid search (vector + keyword) with RRF
- [ ] Choose appropriate index strategy (Flat vs HNSW vs IVF+PQ)
- [ ] Compare vector databases for your use case
- [ ] Know when to use pgvector vs dedicated vector DB

---

## Practical Insights

**Embedding model selection matters more than database:**
- Better embeddings (OpenAI 3, Cohere v3) = 10-20% recall improvement
- Better index tuning = 2-5% recall improvement
- Choose model based on domain: general (sentence-transformers), code (CodeBERT), multi-lingual (LaBSE)
- Always use same model for indexing and querying

**Index tuning is empirical:**
```python
# Measure recall vs latency trade-off
test_queries = load_test_queries()
ground_truth = exact_search(test_queries)  # Flat index

for ef_search in [16, 32, 64, 128, 256]:
    index.hnsw.efSearch = ef_search
    results = index.search(test_queries)
    recall = calculate_recall(results, ground_truth)
    latency = measure_latency(index, test_queries)
    print(f"efSearch={ef_search}: recall={recall:.2%}, latency={latency:.1f}ms")

# Pick the point where recall stops improving significantly
```

**Hybrid search weight tuning:**
```
Start with alpha=0.5 (50% vector, 50% keyword)
A/B test different values:
- Product search: alpha=0.3 (favor exact matches)
- Document search: alpha=0.7 (favor semantics)
- Code search: alpha=0.5 (balance both)

Monitor metrics:
- Click-through rate (CTR)
- Time to find result
- User satisfaction scores
```

**Filtering performance:**
```
Pre-filtering is 10-100x faster than post-filtering
But requires database support

If your filters eliminate >90% of data:
→ Create separate index for that segment
Example: Recent (7 days) vs archive (>7 days)

If filters are highly selective (<1% remaining):
→ Consider separate filtered indexes per category
```

**Memory planning:**
```
Uncompressed: dimensions × 4 bytes × num_vectors
Example: 768 × 4 × 10M = 30GB

With PQ (96x compression): 8 bytes × 10M = 80MB
Trade-off: 5-10% recall loss

Rule of thumb:
- <10M vectors: HNSW (2x memory, high recall)
- 10M-100M: IVF (1x memory, good recall)
- >100M: IVF+PQ (0.05x memory, acceptable recall)
```

**When NOT to use vector databases:**
```
❌ Exact keyword matching only → Use Elasticsearch
❌ Structured data queries → Use Postgres/MySQL
❌ Time-series data → Use InfluxDB
❌ <10K documents → Use in-memory search (numpy)
❌ Real-time updates every millisecond → Vector DBs have write latency

✅ Semantic search (meaning, not keywords)
✅ Recommendation systems (similar items)
✅ Anomaly detection (outliers in vector space)
✅ Multimodal search (text + image + audio)
```

**pgvector vs dedicated vector DB decision:**
```
Use pgvector when:
- Already on Postgres
- <1M vectors
- Need ACID transactions with vectors
- Simple deployment (no new infrastructure)
- Team familiar with SQL

Use dedicated vector DB when:
- >10M vectors
- Sub-20ms latency required
- Complex filtering with pre-filtering
- Hybrid search needed
- Horizontal scaling required
```
