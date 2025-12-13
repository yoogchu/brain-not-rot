# Chapter 13: Search Systems & Elasticsearch

## Why Search Systems Matter

Your e-commerce platform stores 50 million products in PostgreSQL. A user searches for "wireless bluetooth headphones":

```
SELECT * FROM products WHERE name LIKE '%wireless bluetooth headphones%';

Query time: 45 seconds
Result: Timeout, user left

Meanwhile, Amazon returns 10,000 relevant results in 200ms
```

The problem isn't just speed—it's relevance. Your SQL query:
- Misses "Bluetooth wireless earbuds" (different word order)
- Returns "headphones stand wireless charging" (contains all words, wrong product)
- Can't rank results by relevance
- Doesn't handle typos like "wirless" or "headphons"

Search systems solve three hard problems: **speed** (sub-second on billions of documents), **relevance** (find what users mean, not just what they type), and **scale** (distributed full-text search).

---

## The Inverted Index

The fundamental data structure behind all search engines.

### The Problem with Forward Indexes

A traditional database stores documents and scans for matches:

```
┌─────────────────────────────────────────────────────┐
│ Forward Index (Traditional DB)                      │
├─────────────────────────────────────────────────────┤
│ Doc1: "The quick brown fox jumps over lazy dog"     │
│ Doc2: "Quick brown dogs are lazy"                   │
│ Doc3: "The fox is quick and brown"                  │
├─────────────────────────────────────────────────────┤
│ Query: "quick brown"                                │
│ → Scan ALL documents, check each for match          │
│ → O(n) where n = total documents                    │
└─────────────────────────────────────────────────────┘
```

### How Inverted Indexes Work

Flip the relationship: map terms to documents containing them.

```
┌──────────────────────────────────────────────────────────────┐
│ Inverted Index                                               │
├──────────────────────────────────────────────────────────────┤
│ Term        │ Document IDs (Posting List)                    │
├─────────────┼────────────────────────────────────────────────┤
│ brown       │ [Doc1, Doc2, Doc3]                             │
│ dog         │ [Doc1, Doc2]                                   │
│ fox         │ [Doc1, Doc3]                                   │
│ jumps       │ [Doc1]                                         │
│ lazy        │ [Doc1, Doc2]                                   │
│ over        │ [Doc1]                                         │
│ quick       │ [Doc1, Doc2, Doc3]                             │
│ the         │ [Doc1, Doc3]                                   │
├─────────────┼────────────────────────────────────────────────┤
│ Query: "quick brown"                                         │
│ → Look up "quick": [Doc1, Doc2, Doc3]                        │
│ → Look up "brown": [Doc1, Doc2, Doc3]                        │
│ → Intersect: [Doc1, Doc2, Doc3]                              │
│ → O(k) where k = matching documents                          │
└──────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
from collections import defaultdict
from typing import Dict, List, Set

class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Set[int]] = defaultdict(set)
        self.documents: Dict[int, str] = {}
        self.doc_count = 0

    def add_document(self, text: str) -> int:
        doc_id = self.doc_count
        self.doc_count += 1
        self.documents[doc_id] = text

        # Tokenize and index
        terms = self._tokenize(text)
        for term in terms:
            self.index[term].add(doc_id)

        return doc_id

    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenization: lowercase, split on whitespace
        return text.lower().split()

    def search(self, query: str) -> List[int]:
        terms = self._tokenize(query)
        if not terms:
            return []

        # Start with first term's documents
        result = self.index.get(terms[0], set()).copy()

        # Intersect with remaining terms (AND query)
        for term in terms[1:]:
            result &= self.index.get(term, set())

        return list(result)

# Usage
index = InvertedIndex()
index.add_document("The quick brown fox")
index.add_document("Quick brown dogs")
index.add_document("The fox is quick")

results = index.search("quick brown")  # Returns [0, 1, 2]
```

### Posting List Enhancements

Production indexes store more than document IDs:

```
Term: "elasticsearch"
┌─────────────────────────────────────────────────────────────┐
│ Posting List Entry                                          │
├─────────────────────────────────────────────────────────────┤
│ doc_id: 42                                                  │
│ term_frequency: 5          (appears 5 times in doc)         │
│ positions: [12, 45, 89, 120, 156]  (word positions)         │
│ field: "body"              (which field contains term)      │
│ payloads: [...]            (custom metadata)                │
└─────────────────────────────────────────────────────────────┘
```

This enables:
- **TF-IDF scoring**: Term frequency matters for relevance
- **Phrase queries**: "quick brown" requires adjacent positions
- **Field boosting**: Title matches weighted higher than body

---

## Text Analysis Pipeline

Raw text must be processed before indexing.

```
┌──────────────────────────────────────────────────────────────┐
│ "The Quick Brown Fox's jumping!"                             │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ Character Filters                                            │
│ → Strip HTML, convert accents, normalize unicode             │
│ Result: "The Quick Brown Fox's jumping!"                     │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ Tokenizer                                                    │
│ → Split into tokens on whitespace/punctuation                │
│ Result: ["The", "Quick", "Brown", "Fox's", "jumping"]        │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ Token Filters                                                │
│ → Lowercase: ["the", "quick", "brown", "fox's", "jumping"]   │
│ → Possessive: ["the", "quick", "brown", "fox", "jumping"]    │
│ → Stop words: ["quick", "brown", "fox", "jumping"]           │
│ → Stemming: ["quick", "brown", "fox", "jump"]                │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
              Indexed terms: [quick, brown, fox, jump]
```

### Common Analyzers

```python
# Elasticsearch analyzer configuration
{
    "settings": {
        "analysis": {
            "analyzer": {
                "english_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "english_possessive_stemmer",
                        "english_stop",
                        "english_stemmer"
                    ]
                }
            },
            "filter": {
                "english_stop": {
                    "type": "stop",
                    "stopwords": "_english_"
                },
                "english_stemmer": {
                    "type": "stemmer",
                    "language": "english"
                },
                "english_possessive_stemmer": {
                    "type": "stemmer",
                    "language": "possessive_english"
                }
            }
        }
    }
}
```

### Analysis Trade-offs

| Technique | Benefit | Cost |
|-----------|---------|------|
| Stemming | "running" matches "run" | "universe" might match "university" |
| Stop words | Smaller index, faster search | "To be or not to be" becomes empty |
| Lowercase | Case-insensitive search | "US" matches "us" (ambiguity) |
| Synonyms | "car" matches "automobile" | Index size grows, maintenance burden |

**When to use aggressive analysis:** General search, blog posts, product descriptions
**When to use minimal analysis:** Exact matching, product SKUs, identifiers, log search

---

## Relevance Scoring

Not all matches are equal. Scoring ranks results by relevance.

### TF-IDF

**Term Frequency (TF):** How often does the term appear in this document?

```
Document: "Search search search systems"
TF("search") = 3/4 = 0.75
TF("systems") = 1/4 = 0.25
```

**Inverse Document Frequency (IDF):** How rare is this term across all documents?

```
Total documents: 1,000,000
Documents containing "the": 900,000
Documents containing "elasticsearch": 5,000

IDF("the") = log(1,000,000 / 900,000) = 0.05  (common, low weight)
IDF("elasticsearch") = log(1,000,000 / 5,000) = 2.3  (rare, high weight)
```

**TF-IDF Score:** TF × IDF

```python
import math
from collections import Counter

def compute_tf(term: str, document: List[str]) -> float:
    """Term frequency: count / total terms"""
    term_count = document.count(term)
    return term_count / len(document) if document else 0

def compute_idf(term: str, documents: List[List[str]]) -> float:
    """Inverse document frequency: log(total_docs / docs_containing_term)"""
    docs_containing = sum(1 for doc in documents if term in doc)
    if docs_containing == 0:
        return 0
    return math.log(len(documents) / docs_containing)

def compute_tfidf(term: str, document: List[str], corpus: List[List[str]]) -> float:
    return compute_tf(term, document) * compute_idf(term, corpus)
```

### BM25 (Best Match 25)

The industry standard, used by Elasticsearch and Lucene. Improves on TF-IDF:

```
                     IDF(term) × TF(term) × (k1 + 1)
BM25(term, doc) = ───────────────────────────────────────────────
                   TF(term) + k1 × (1 - b + b × |doc| / avgdl)

Where:
- k1: Term frequency saturation (default 1.2)
- b: Length normalization (default 0.75)
- |doc|: Document length
- avgdl: Average document length
```

**Key improvements over TF-IDF:**
- **Saturation:** Term appearing 100x isn't 100x more relevant than 1x
- **Length normalization:** Long documents don't unfairly dominate

```python
import math

def bm25_score(
    term: str,
    doc: List[str],
    corpus: List[List[str]],
    k1: float = 1.2,
    b: float = 0.75
) -> float:
    tf = doc.count(term)
    doc_len = len(doc)
    avg_doc_len = sum(len(d) for d in corpus) / len(corpus)

    # IDF with smoothing
    docs_containing = sum(1 for d in corpus if term in d)
    idf = math.log((len(corpus) - docs_containing + 0.5) / (docs_containing + 0.5) + 1)

    # BM25 term score
    numerator = tf * (k1 + 1)
    denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))

    return idf * (numerator / denominator)
```

### Boosting Fields

Not all fields are equal:

```json
{
    "query": {
        "multi_match": {
            "query": "elasticsearch guide",
            "fields": [
                "title^3",
                "description^2",
                "body"
            ]
        }
    }
}
```

Match in title is 3x more valuable than match in body.

---

## Elasticsearch Architecture

### Cluster Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Elasticsearch Cluster                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐     │
│  │  Master Node  │   │  Data Node 1  │   │  Data Node 2  │     │
│  │               │   │               │   │               │     │
│  │ • Cluster     │   │ • Shard P0    │   │ • Shard P1    │     │
│  │   state       │   │ • Shard R1    │   │ • Shard R0    │     │
│  │ • Index       │   │               │   │               │     │
│  │   management  │   │ • Indexing    │   │ • Indexing    │     │
│  │               │   │ • Searching   │   │ • Searching   │     │
│  └───────────────┘   └───────────────┘   └───────────────┘     │
│                                                                  │
│  ┌───────────────┐   ┌───────────────┐                         │
│  │ Coordinating  │   │  Ingest Node  │                         │
│  │    Node       │   │               │                         │
│  │               │   │ • Pipelines   │                         │
│  │ • Route       │   │ • Transform   │                         │
│  │   queries     │   │ • Enrich      │                         │
│  │ • Aggregate   │   │               │                         │
│  │   results     │   │               │                         │
│  └───────────────┘   └───────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Sharding and Replication

```
Index: "products" (50 million documents)
Shards: 5 primary, 1 replica each

┌─────────────────────────────────────────────────────────────────┐
│ Index: products                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Node 1              Node 2              Node 3                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   P0        │    │   P1        │    │   P2        │         │
│  │ (Primary)   │    │ (Primary)   │    │ (Primary)   │         │
│  │ 10M docs    │    │ 10M docs    │    │ 10M docs    │         │
│  ├─────────────┤    ├─────────────┤    ├─────────────┤         │
│  │   R1        │    │   R2        │    │   R0        │         │
│  │ (Replica)   │    │ (Replica)   │    │ (Replica)   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                  │
│  Node 4              Node 5                                     │
│  ┌─────────────┐    ┌─────────────┐                            │
│  │   P3        │    │   P4        │                            │
│  │ (Primary)   │    │ (Primary)   │                            │
│  │ 10M docs    │    │ 10M docs    │                            │
│  ├─────────────┤    ├─────────────┤                            │
│  │   R4        │    │   R3        │                            │
│  │ (Replica)   │    │ (Replica)   │                            │
│  └─────────────┘    └─────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Shard routing: hash(doc_id) % num_shards → shard number
```

### Query Execution Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ Search Request: GET /products/_search?q=bluetooth headphones     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. Coordinating Node                                             │
│    • Receives request                                            │
│    • Determines which shards to query                            │
│    • Broadcasts query to relevant shards                         │
└──────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Shard 0      │      │ Shard 1      │      │ Shard 2      │
│ Query phase: │      │ Query phase: │      │ Query phase: │
│ • Parse      │      │ • Parse      │      │ • Parse      │
│ • Score      │      │ • Score      │      │ • Score      │
│ • Return top │      │ • Return top │      │ • Return top │
│   10 doc IDs │      │   10 doc IDs │      │   10 doc IDs │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. Coordinating Node: Merge                                      │
│    • Collect top 10 from each shard (30 total)                   │
│    • Global sort by score                                        │
│    • Select final top 10                                         │
└──────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Fetch phase: │      │ Fetch phase: │      │ Fetch phase: │
│ Shard 0:     │      │ Shard 1:     │      │ Shard 2:     │
│ Return docs  │      │ Return docs  │      │ Return docs  │
│ [42, 87]     │      │ [156]        │      │ [203, 891]   │
└──────────────┘      └──────────────┘      └──────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. Final Response: 10 documents with full content                │
└──────────────────────────────────────────────────────────────────┘
```

---

## Query Types

### Full-Text Queries

```python
# Match query - analyzed, finds relevant documents
{
    "query": {
        "match": {
            "description": "wireless bluetooth headphones"
        }
    }
}
# Finds: "bluetooth wireless earbuds", "Wireless Headphones BT"

# Match phrase - words must appear in order, adjacent
{
    "query": {
        "match_phrase": {
            "title": "quick brown fox"
        }
    }
}
# Matches: "the quick brown fox jumps"
# Doesn't match: "quick fox brown"
```

### Structured Queries

```python
# Term query - exact match, not analyzed
{
    "query": {
        "term": {
            "status": "published"
        }
    }
}

# Range query
{
    "query": {
        "range": {
            "price": {
                "gte": 100,
                "lte": 500
            }
        }
    }
}

# Bool query - combine multiple conditions
{
    "query": {
        "bool": {
            "must": [
                {"match": {"title": "elasticsearch"}}
            ],
            "filter": [
                {"term": {"status": "published"}},
                {"range": {"date": {"gte": "2024-01-01"}}}
            ],
            "should": [
                {"match": {"tags": "tutorial"}}
            ],
            "must_not": [
                {"term": {"author": "spam_bot"}}
            ]
        }
    }
}
```

**Bool query context:**
- `must`: Must match, contributes to score
- `filter`: Must match, no scoring (faster, cacheable)
- `should`: Optional, boosts score if matched
- `must_not`: Must not match, no scoring

### Fuzzy Queries

Handle typos and spelling variations:

```python
{
    "query": {
        "fuzzy": {
            "name": {
                "value": "elasticsaerch",  # Typo
                "fuzziness": "AUTO"         # Edit distance
            }
        }
    }
}
# Matches: "elasticsearch"
```

**Fuzziness levels:**
- `0`: Exact match only
- `1`: One edit (insert, delete, substitute, transpose)
- `2`: Two edits
- `AUTO`: Based on term length (0 for 1-2 chars, 1 for 3-5, 2 for 6+)

---

## Index Design Best Practices

### Mapping Definition

```python
{
    "mappings": {
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "english",
                "fields": {
                    "raw": {
                        "type": "keyword"  # For sorting/aggregations
                    }
                }
            },
            "price": {
                "type": "scaled_float",
                "scaling_factor": 100
            },
            "category": {
                "type": "keyword"
            },
            "created_at": {
                "type": "date"
            },
            "tags": {
                "type": "keyword"
            },
            "description": {
                "type": "text",
                "analyzer": "english"
            }
        }
    }
}
```

### Field Type Selection

| Data Type | Field Type | Use Case |
|-----------|------------|----------|
| Free text | `text` | Full-text search, analyzed |
| Identifiers | `keyword` | Exact match, sorting, aggregations |
| Numbers | `integer`, `long`, `float` | Range queries, sorting |
| Money | `scaled_float` | Avoid floating point issues |
| Dates | `date` | Range queries, date math |
| Boolean | `boolean` | Filters |
| Nested objects | `nested` | Query array of objects independently |

### Multi-field Mapping

```python
{
    "properties": {
        "title": {
            "type": "text",
            "analyzer": "english",
            "fields": {
                "raw": {
                    "type": "keyword"
                },
                "autocomplete": {
                    "type": "text",
                    "analyzer": "autocomplete_analyzer"
                }
            }
        }
    }
}

# Query title.raw for exact match
# Query title for full-text search
# Query title.autocomplete for suggestions
```

---

## Performance Optimization

### Index Settings

```python
{
    "settings": {
        "number_of_shards": 5,           # Plan for growth
        "number_of_replicas": 1,          # Fault tolerance + read scaling
        "refresh_interval": "30s",        # Trade freshness for indexing speed
        "index.translog.durability": "async"  # Faster indexing, slight durability risk
    }
}
```

### Query Optimization

```python
# BAD: Expensive wildcard at start
{
    "query": {
        "wildcard": {
            "email": "*@gmail.com"
        }
    }
}

# GOOD: Use keyword field + prefix
{
    "query": {
        "term": {
            "email_domain": "gmail.com"
        }
    }
}

# BAD: Deep pagination
{
    "from": 10000,
    "size": 10
}

# GOOD: search_after for deep pagination
{
    "size": 10,
    "search_after": [1609459200000, "doc_id_xyz"],
    "sort": [
        {"timestamp": "desc"},
        {"_id": "asc"}
    ]
}
```

### Caching

```
┌─────────────────────────────────────────────────────────────────┐
│ Elasticsearch Caches                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Node Query Cache (filter cache)                                  │
│ • Caches filter results (term, range filters)                    │
│ • Only filter context (not must/should)                          │
│ • Invalidated on index refresh                                   │
│                                                                  │
│ Shard Request Cache                                              │
│ • Caches entire search response                                  │
│ • Only for size=0 (aggregations only)                            │
│ • Invalidated on refresh                                         │
│                                                                  │
│ Field Data Cache                                                 │
│ • For sorting/aggregations on text fields                        │
│ • AVOID: Use keyword fields instead                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Search System Comparison

| Feature | Elasticsearch | Apache Solr | Meilisearch | Typesense | PostgreSQL FTS |
|---------|---------------|-------------|-------------|-----------|----------------|
| **Scale** | Billions of docs | Billions | Millions | Millions | Millions |
| **Setup complexity** | Medium | High | Low | Low | Already have |
| **Query latency** | 10-100ms | 10-100ms | 1-50ms | 1-20ms | 10-500ms |
| **Typo tolerance** | Config needed | Config needed | Built-in | Built-in | Limited |
| **Relevance tuning** | Extensive | Extensive | Limited | Medium | Basic |
| **Real-time** | Near (1s) | Near | Instant | Instant | Instant |
| **Operational burden** | High | High | Low | Low | Lowest |
| **Best for** | Large scale, analytics | Enterprise search | Dev-friendly | Speed-first | Simple needs |

### When to Use What

**PostgreSQL full-text search:** You already have Postgres, search is a secondary feature, < 1M documents

**Elasticsearch:** Large scale, complex queries, analytics on search data, enterprise requirements

**Meilisearch/Typesense:** Developer experience matters, need instant setup, smaller datasets, consumer-facing search

---

## Key Concepts Checklist

- [ ] Explain how inverted indexes work and why they're fast
- [ ] Describe the text analysis pipeline (tokenization, stemming, stop words)
- [ ] Calculate relevance with TF-IDF and explain BM25 improvements
- [ ] Design index mapping with appropriate field types
- [ ] Choose shard count based on data size and query patterns
- [ ] Optimize queries (avoid leading wildcards, use filter context)
- [ ] Implement deep pagination with search_after

---

## Practical Insights

**Shard sizing rules of thumb:**
- Target 10-50GB per shard
- More shards = more parallelism but higher overhead
- Too few shards = hot spots, too many = coordination overhead
- Can't change shard count after index creation (must reindex)

**Index lifecycle:**
- Use time-based indices for logs (logs-2024.01.01)
- Implement ILM policies: hot → warm → cold → delete
- Hot nodes (SSD) for recent data, warm nodes (HDD) for older data

**Search relevance is iterative:**
- Start with defaults, measure with real queries
- Use Explain API to understand scoring
- A/B test relevance changes
- Monitor "no results" and "low engagement" queries

**Operational realities:**
- Elasticsearch clusters need babysitting
- JVM heap sizing is critical (don't exceed 32GB)
- Split-brain prevention: minimum_master_nodes = (master_eligible / 2) + 1
- Back up your cluster (snapshots to S3/GCS)

**When NOT to use Elasticsearch:**
- Primary data store (it's a search engine, not a database)
- Sub-millisecond latency requirements
- Simple exact-match queries (use your database)
- Cost-sensitive with small data (operational overhead isn't worth it)
