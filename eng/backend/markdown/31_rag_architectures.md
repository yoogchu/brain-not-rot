# Chapter 31: RAG Architectures

## Why RAG?

```
Problem with vanilla LLMs:
- Knowledge cutoff (doesn't know recent events)
- Hallucinations (makes up facts confidently)
- No access to private data (your docs, codebase)
- Generic responses (not tailored to your domain)

RAG Solution:
Retrieve relevant context → Augment prompt → Generate answer

"What's our refund policy?"
  ↓
[Retrieve] → Find refund policy doc chunks
  ↓
[Augment] → "Given this context: {refund policy}, answer: What's our refund policy?"
  ↓
[Generate] → LLM generates grounded answer
```

---

## Basic RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     Indexing Pipeline (Offline)                  │
│                                                                   │
│  Documents → Chunk → Embed → Store in Vector DB                  │
│                                                                   │
│  ┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐   │
│  │ Docs │───►│ Chunker  │───►│ Embedder │───►│  Vector DB   │   │
│  └──────┘    └──────────┘    └──────────┘    └──────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Query Pipeline (Online)                      │
│                                                                   │
│  Query → Embed → Search → Retrieve → Augment → Generate          │
│                                                                   │
│  ┌───────┐   ┌────────┐   ┌──────────────┐   ┌───────┐          │
│  │ Query │──►│ Embed  │──►│  Vector DB   │──►│  LLM  │──► Answer│
│  └───────┘   └────────┘   │   Search     │   └───────┘          │
│                           └──────────────┘                       │
│                                 │                                │
│                          Top-K chunks                            │
└─────────────────────────────────────────────────────────────────┘
```

### Step 1: Document Chunking

```python
# Problem: Documents too long for context window
# Solution: Split into overlapping chunks

def chunk_document(text: str, chunk_size: int = 512, overlap: int = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap prevents cutting mid-sentence
    return chunks

# Better: Semantic chunking (respect document structure)
def semantic_chunk(text: str):
    # Split by paragraphs, sections, or sentences
    # Keep headers with their content
    # Respect code blocks, tables as units
    ...
```

**Chunk size trade-offs:**

| Size | Pros | Cons |
|------|------|------|
| Small (256 tokens) | Precise retrieval | May miss context |
| Medium (512 tokens) | Balanced | Common choice |
| Large (1024 tokens) | More context | Less precise, fewer chunks fit |

### Step 2: Embedding

```python
from openai import OpenAI

client = OpenAI()

def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding  # 1536-dimensional vector

# Embed all chunks
chunks = chunk_document(document)
embeddings = [embed_text(chunk) for chunk in chunks]
```

**Popular embedding models:**

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| text-embedding-3-small | 1536 | Fast | Good |
| text-embedding-3-large | 3072 | Medium | Better |
| Cohere embed-v3 | 1024 | Fast | Good |
| BGE-large | 1024 | Medium | Good (open source) |

### Step 3: Vector Storage

```python
import pinecone

# Initialize
pinecone.init(api_key="...", environment="...")
index = pinecone.Index("my-index")

# Upsert embeddings
vectors = [
    {
        "id": f"chunk-{i}",
        "values": embedding,
        "metadata": {
            "text": chunk,
            "source": "refund-policy.pdf",
            "page": 3
        }
    }
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
]
index.upsert(vectors=vectors)
```

### Step 4: Retrieval

```python
def retrieve(query: str, top_k: int = 5) -> list[str]:
    # Embed the query
    query_embedding = embed_text(query)
    
    # Search vector DB
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Extract text chunks
    chunks = [match.metadata["text"] for match in results.matches]
    return chunks
```

### Step 5: Augmented Generation

```python
def rag_answer(query: str) -> str:
    # Retrieve relevant chunks
    chunks = retrieve(query, top_k=5)
    
    # Build prompt with context
    context = "\n\n".join(chunks)
    
    prompt = f"""Answer the question based on the provided context.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {query}

Answer:"""
    
    # Generate answer
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content
```

---

## Advanced RAG Patterns

### Hybrid Search

```
Vector search alone misses exact matches:
Query: "Error code E-5012"
Vector: Finds semantically similar errors
BM25:  Finds exact "E-5012" match

Hybrid = Vector + BM25 (keyword)

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
         └──────────┬───────────────┘
                    ▼
           ┌───────────────┐
           │ Reciprocal    │
           │ Rank Fusion   │
           └───────────────┘
                    │
                    ▼
            Merged Results
```

```python
def hybrid_search(query: str, alpha: float = 0.5):
    # Vector search
    vector_results = vector_db.search(embed(query), top_k=20)
    
    # BM25 search
    bm25_results = bm25_index.search(query, top_k=20)
    
    # Reciprocal Rank Fusion
    scores = {}
    k = 60  # RRF constant
    
    for rank, doc_id in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + alpha / (k + rank)
    
    for rank, doc_id in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) / (k + rank)
    
    # Sort by combined score
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
```

### Reranking

```
Problem: Embedding similarity ≠ relevance
Solution: Cross-encoder reranker

Step 1: Retrieve top 100 with fast vector search
Step 2: Rerank with slow but accurate cross-encoder
Step 3: Return top 10

┌─────────────────────────────────────────────────┐
│         Vector Search (fast, approximate)        │
│                    Top 100                       │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│          Cross-Encoder Reranker (slow, accurate) │
│                    Top 10                        │
└─────────────────────────────────────────────────┘
```

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query: str, documents: list[str], top_k: int = 10):
    # Score each document against query
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)
    
    # Sort by score
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]
```

### Query Expansion

```
Problem: User query may not match document terminology
Solution: Expand query with variations

Original: "How to fix login issues?"

Expanded queries:
- "How to fix login issues?"
- "authentication problems troubleshooting"
- "sign in error solutions"
- "login failure debugging"

Retrieve for all, merge results
```

```python
def expand_query(query: str) -> list[str]:
    prompt = f"""Generate 3 alternative phrasings for this search query.
Return only the alternatives, one per line.

Query: {query}"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    alternatives = response.choices[0].message.content.strip().split("\n")
    return [query] + alternatives
```

### Contextual Compression

```
Problem: Retrieved chunks have irrelevant parts
Solution: Compress to relevant portions only

Retrieved chunk (500 tokens):
"The company was founded in 1995. [lots of history...]
Our refund policy allows returns within 30 days.
[more irrelevant text...]"

After compression (50 tokens):
"Our refund policy allows returns within 30 days."
```

---

## Vector Databases

### Comparison

| Database | Type | Strengths | Best For |
|----------|------|-----------|----------|
| Pinecone | Managed | Easy, scalable | Production, any scale |
| Weaviate | Open source | Hybrid search built-in | Self-hosted |
| Qdrant | Open source | Performance, filtering | Self-hosted |
| pgvector | PostgreSQL ext | Existing Postgres | Simple use cases |
| Chroma | Embedded | Easy prototyping | Development |
| Milvus | Open source | Large scale | Billions of vectors |

### Index Types

```
HNSW (Hierarchical Navigable Small World):
- Build: O(n log n)
- Search: O(log n)
- Memory: High (stores graph)
- Best for: Most use cases

IVF (Inverted File Index):
- Build: O(n)
- Search: O(sqrt(n))
- Memory: Lower
- Best for: Large datasets, memory constrained

PQ (Product Quantization):
- Compresses vectors
- Trades accuracy for memory
- Best for: Very large datasets
```

---

## LLM Serving

### Key Optimizations

**KV Cache:**
```
Problem: Transformer recomputes all token attentions

Without cache:
Generate token 100: Compute attention for tokens 1-99
Generate token 101: Compute attention for tokens 1-100 (again!)

With KV cache:
Store key/value tensors from previous tokens
Only compute attention for new token
```

**Continuous Batching:**
```
Traditional batching:
Batch waits for longest sequence to finish

[Request 1: ████████████████████░░░░░░░░░░░░░░░░░░░░░]
[Request 2: ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] (waiting)
[Request 3: ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] (waiting)

Continuous batching:
New requests join as old ones finish

[Request 1: ████████████████████]
[Request 2: ██████][Request 4: ████████████]
[Request 3: ████████████][Request 5: ██████████]
```

**Speculative Decoding:**
```
Use small model to draft, large model to verify

Draft model (fast): Generates 8 tokens quickly
Large model (slow): Verifies in parallel (1 forward pass)

If 6/8 match: Accept 6, regenerate from there
Speedup: ~4x for well-matched draft/target
```

**Quantization:**
```
FP32: 32 bits per weight → 1x memory
FP16: 16 bits per weight → 2x memory savings
INT8: 8 bits per weight  → 4x memory savings
INT4: 4 bits per weight  → 8x memory savings

Trade-off: Some quality loss at lower precision
```

### Serving Frameworks

| Framework | Strengths | Use Case |
|-----------|-----------|----------|
| vLLM | Continuous batching, PagedAttention | Production |
| TGI | Hugging Face integration | Hugging Face models |
| Triton | Multi-model, GPU optimization | Complex pipelines |
| Ollama | Local development | Testing |

---

## Evaluation

### RAG Metrics

**Retrieval Quality:**
```
Precision@K: What % of retrieved docs are relevant?
Recall@K: What % of relevant docs were retrieved?
MRR: How high is the first relevant result ranked?
```

**Generation Quality:**
```
Faithfulness: Does answer match retrieved context?
Relevance: Does answer address the question?
Completeness: Are all parts of question answered?
```

### RAGAS Framework

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Evaluate RAG pipeline
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)

# Faithfulness: 0.92 (answer grounded in context)
# Answer Relevancy: 0.88 (answer addresses question)
# Context Precision: 0.75 (retrieved context is relevant)
```

---

## Production Considerations

### Caching

```
┌─────────────────────────────────────────────────────────────┐
│                      Query                                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Semantic Cache                             │
│        (similar questions → cached answers)                  │
└─────────────────────────────────────────────────────────────┘
         │                                    │
    Cache Hit                            Cache Miss
         │                                    │
         ▼                                    ▼
   Return cached                      Run RAG pipeline
      answer                          Cache result
```

### Guardrails

```python
def safe_rag_answer(query: str) -> str:
    # Input guardrail
    if is_harmful_query(query):
        return "I can't help with that request."
    
    # Generate answer
    answer = rag_answer(query)
    
    # Output guardrail
    if contains_pii(answer):
        answer = redact_pii(answer)
    
    if is_hallucination(answer, retrieved_context):
        return "I don't have enough information to answer that."
    
    return answer
```

---

## Key Concepts Checklist

- [ ] Explain basic RAG pipeline components
- [ ] Describe chunking strategies and trade-offs
- [ ] Explain embedding models and vector search
- [ ] Know hybrid search and reranking
- [ ] Describe LLM optimization techniques (KV cache, batching)
- [ ] Discuss evaluation metrics for RAG systems

---

## Practical Insights

**Chunking strategy:**
- Start with 512 tokens, overlap 50
- Consider document structure
- Include metadata (source, page, section)
- Test different sizes with your data

**Embedding model selection:**
- Match training domain to your domain
- Benchmark on your actual queries
- Consider latency vs quality trade-off
- Fine-tune for specialized domains

**Scaling RAG:**
```
Bottlenecks:
1. Embedding API (batch, cache, self-host)
2. Vector search (index tuning, sharding)
3. LLM inference (caching, streaming)
4. Reranking (batch, prune candidates)
```

**Failure modes:**
- Wrong chunks retrieved → Better chunking, hybrid search
- Right chunks, wrong answer → Better prompt, reranking
- Slow responses → Caching, streaming, smaller models
- Hallucinations → Stricter prompts, fact verification
