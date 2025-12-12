# Chapter 2: Sharding & Partitioning

## The Scaling Wall

Your PostgreSQL database hits 2TB. Single node limits hit you hard:

- **Storage**: SSDs top out at ~30TB, but query performance degrades long before that
- **Memory**: Can't fit working set in RAM → disk I/O kills performance
- **CPU**: Single machine can't handle 100K+ QPS for complex queries
- **Backup/Recovery**: 2TB backup takes hours, restore takes even longer

**Vertical scaling** (bigger machine) has limits. Eventually you need **horizontal scaling** (more machines).

**Sharding**: Split data across multiple machines. Each shard holds a subset of data.

```
┌───────────────────────────────────────────────────────┐
│                   Original Database                    │
│  User 1, User 2, User 3, ... User 1M                  │
└───────────────────────────────────────────────────────┘
                          │
                          ▼ Sharding
┌─────────────────┬─────────────────┬─────────────────┐
│    Shard 1      │    Shard 2      │    Shard 3      │
│ Users 1-333K    │ Users 333K-666K │ Users 666K-1M   │
└─────────────────┴─────────────────┴─────────────────┘
```

Each shard is a complete database that can be queried independently.

---

## Choosing a Shard Key

The shard key determines which shard holds each record. This is a **critical** decision that's hard to change later.

### Good Shard Key Properties

1. **High cardinality**: Many unique values (user_id: billions, country: ~200)
2. **Even distribution**: No hot spots (created_at creates hot spots!)
3. **Query alignment**: Most queries hit single shard
4. **Immutable**: Changing shard key requires migrating data

### Shard Key Examples by Application

| Application | Good Key | Why | Bad Key | Why Bad |
|-------------|----------|-----|---------|---------|
| E-commerce | `customer_id` | Orders belong to one customer | `order_date` | Hot shard for current day |
| Social Media | `user_id` | User data colocated | `post_id` | Cross-shard queries for feeds |
| Multi-tenant SaaS | `tenant_id` | Tenant isolation | `created_at` | Time-based hot spots |
| Gaming | `game_id` | Game state colocated | `player_id` | Multiple players per game |
| IoT | `device_id` | Device data colocated | `timestamp` | All current data on one shard |

### The Shard Key Decision Framework

Ask yourself:
1. **What queries are most common?** Shard key should be in WHERE clause
2. **What data is accessed together?** Colocate related data
3. **What will grow?** High-cardinality keys scale better
4. **What won't change?** Immutable keys avoid migrations

---

## Sharding Strategies

### 1. Range-Based Sharding

Divide key space into contiguous ranges:

```
Shard 1: user_id 1-1,000,000
Shard 2: user_id 1,000,001-2,000,000
Shard 3: user_id 2,000,001-3,000,000
```

**How it works:**
```python
def get_shard(user_id):
    if user_id <= 1_000_000:
        return "shard1"
    elif user_id <= 2_000_000:
        return "shard2"
    else:
        return "shard3"
```

**Advantages:**
- Range queries efficient: `WHERE user_id BETWEEN 1000 AND 2000` hits one shard
- Easy to understand and implement
- Sequential IDs naturally distribute

**Disadvantages:**
- Hot spots for time-series: All current writes hit latest range
- Uneven distribution if keys aren't uniformly distributed
- Resharding requires moving contiguous ranges

**Example hot spot problem:**
```
Sharding by order_date:
Shard 1: 2023 orders (old, rarely accessed)
Shard 2: 2024 orders (all current traffic!)
Shard 3: 2025 orders (empty)

90% of queries hit Shard 2 → hot spot
```

### 2. Hash-Based Sharding

Hash the key, mod by number of shards:

```python
def get_shard(key, num_shards):
    return hash(key) % num_shards

# hash("user123") = 847291
# 847291 % 3 = 1
# → Goes to Shard 1
```

**Advantages:**
- Even distribution (assuming good hash function)
- No hot spots from key distribution
- Simple implementation

**Disadvantages:**
- Range queries scattered across ALL shards
- Adding shards moves most data (catastrophic!)

**The resharding problem:**
```
Before: 3 shards
hash("user123") % 3 = 1  → Shard 1

After adding 4th shard:
hash("user123") % 4 = 3  → Shard 3  (MOVED!)

When going from 3→4 shards, approximately 75% of data moves!
```

This is why simple hash sharding doesn't work for production systems.

### 3. Consistent Hashing

The solution to the resharding problem. Hash both data AND nodes onto a ring:

```
                    0°
                    │
         Node B ────●──── Node A
                   /│\
                  / │ \
                 /  │  \
                /   │   \
        Key X ●────│────● Key Y  
              │    │    │
              │    │    │
     Node C ──●────│────●── Key Z
                   │
               Node D
                   │
                  180°

Walk clockwise from key to find its node:
Key X → walks clockwise → hits Node B
Key Y → walks clockwise → hits Node D  
Key Z → walks clockwise → hits Node D
```

**Adding a new node:**
```
Add Node E between C and D:

Before:                    After:
Key Z → Node D            Key Z → Node E (only this key moved!)
Key X → Node B            Key X → Node B (unchanged)
Key Y → Node D            Key Y → Node D (unchanged)
```

Only keys between C and E move to the new node. ~1/N of data moves instead of (N-1)/N.

### Virtual Nodes (VNodes)

Problem: With 4 nodes on ring, distribution might be uneven.

Solution: Each physical node → 100-200 positions on ring

```
Physical Node A → Virtual nodes: A_0, A_1, A_2, ... A_150
Physical Node B → Virtual nodes: B_0, B_1, B_2, ... B_150

Ring now has 300 points instead of 2
Law of large numbers → even distribution
```

**Benefits:**
- Even distribution regardless of node count
- Heterogeneous hardware: More powerful node gets more vnodes
- Gradual data movement when nodes join/leave

**Used by:** Cassandra, DynamoDB, Riak, Amazon Dynamo

---

## The Hot Partition Problem (Celebrity Problem)

Even with perfect hash distribution, some keys are hotter than others.

**Instagram example:**
```
Sharding by user_id
Kylie Jenner's user_id hashes to Shard 7
She posts a photo
Her 400M followers generate feed queries
All queries for her data hit Shard 7

Regular shards: 1,000 QPS each
Kylie's shard: 400,000 QPS ← 400x load!
```

### Solutions

#### 1. Scatter-Gather for Hot Keys

```python
# For known hot keys, add random suffix to spread across shards
def write_hot_key(key, value):
    suffix = random.randint(0, 99)  # 100 sub-keys
    actual_key = f"{key}_{suffix}"
    shard = get_shard(actual_key)
    shard.write(actual_key, value)

def read_hot_key(key):
    results = []
    for suffix in range(100):
        actual_key = f"{key}_{suffix}"
        shard = get_shard(actual_key)
        results.extend(shard.read(actual_key))
    return aggregate(results)
```

Kylie's data now spread across up to 100 shards. Read is 100x more calls but parallelizable.

#### 2. Application-Level Caching

```
Cache celebrity data in front of database:

Client → Redis Cache (100K QPS capacity) → Database (rarely hit)

Cache Kylie's recent posts, follower counts, etc.
Cache hit rate for celebrities: 99%+
Database barely touched
```

#### 3. Read Replicas for Hot Shards

```
Hot Shard Primary → Replica 1
                  → Replica 2
                  → Replica 3
                  → Replica 4
                  
Load balance reads across 5 copies = 5x capacity
```

#### 4. Workload Isolation

```
Shard 1-10: Regular users (hash-based)
Shard 11: Kylie Jenner (dedicated)
Shard 12: Other top celebrities (dedicated pool)

Special routing logic for known hot users
```

---

## Cross-Shard Operations

### The Challenge

```sql
-- Single shard (good): 
SELECT * FROM orders WHERE user_id = 123;
-- user_id is shard key → hits exactly one shard

-- Cross-shard (expensive):
SELECT * FROM orders WHERE order_date > '2024-01-01';
-- order_date is NOT shard key → must query ALL shards
```

### Cross-Shard Joins

This is where sharding gets painful:

```sql
-- Assume orders sharded by user_id
-- Assume products sharded by product_id

SELECT * FROM orders o 
JOIN products p ON o.product_id = p.id
WHERE o.user_id = 123;

-- Steps:
-- 1. Find orders for user 123 (one shard - good)
-- 2. For each order, find product (scatter to product shards)
-- 3. N+1 query problem across network!
```

### Solutions for Cross-Shard Queries

#### 1. Denormalization

Store product data with order:

```json
{
  "order_id": "123",
  "user_id": "456",
  "product": {
    "id": "789",
    "name": "Widget",
    "price": 9.99
  }
}
```

**Pros:** No joins needed, single shard query
**Cons:** Data duplication, update anomalies (product price changes)

#### 2. Broadcast Tables

Replicate small reference tables to all shards:

```
Products table: 100K products, ~100MB
Replicate to all order shards
Join locally on each shard: Fast!

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Shard 1    │  │   Shard 2    │  │   Shard 3    │
│  Orders A-M  │  │  Orders N-Z  │  │  Orders ...  │
│  [Products]  │  │  [Products]  │  │  [Products]  │
└──────────────┘  └──────────────┘  └──────────────┘
     │                  │                  │
     └──────────────────┴──────────────────┘
                        │
           [Products master - updates broadcast]
```

#### 3. Colocate Related Data

Shard related tables by same key:

```
Orders sharded by user_id
Order_items sharded by user_id (via order_id → user_id)
User_profiles sharded by user_id

All of user 123's data on same shard → local joins!
```

This is the **shard colocation** pattern. Very powerful when it works.

#### 4. Scatter-Gather Query

When cross-shard is unavoidable:

```python
def cross_shard_query(sql, params):
    # Send query to all shards in parallel
    futures = []
    for shard in all_shards:
        future = executor.submit(shard.query, sql, params)
        futures.append(future)
    
    # Collect and merge results
    results = []
    for future in futures:
        results.extend(future.result())
    
    return merge_and_sort(results)
```

**Performance:** N network calls instead of 1, but parallelized.

---

## Resharding Strategies

Eventually you need more shards. How to migrate without downtime?

### 1. Double-Write Migration

```
Phase 1: Write to BOTH old and new sharding scheme
- Old write: hash(key) % 4
- New write: hash(key) % 8
- Both receive every write

Phase 2: Background copy old data to new shards
- Scan old shards
- Copy to new shard locations
- Track progress

Phase 3: Switch reads to new shards
- Reads now use new sharding
- Writes still go to both

Phase 4: Stop writes to old shards
- Verify new shards have all data
- Cut over completely

Phase 5: Decommission old shards
```

**Pros:** Zero downtime
**Cons:** 2x write load during migration, complex coordination

### 2. Virtual Sharding

Start with more shards than physical nodes:

```
Initial: 32 virtual shards on 4 physical nodes
         Node A: Shards 1-8
         Node B: Shards 9-16
         Node C: Shards 17-24
         Node D: Shards 25-32

Growth:  32 virtual shards on 8 physical nodes
         Node A: Shards 1-4
         Node B: Shards 5-8
         Node C: Shards 9-12
         ... etc

No data movement! Just reassign virtual → physical mapping
```

**Used by:** Most production systems. Plan for 10x growth.

### 3. Online Schema Change Tools

- **Vitess VReplication**: MySQL sharding solution by YouTube
- **gh-ost**: GitHub's online schema migration
- **pt-online-schema-change**: Percona toolkit

These tools handle the complexity of live migrations.

---

## Sharding Anti-Patterns

### 1. Sharding Too Early

**"Premature sharding is the root of all evil."**

A well-tuned PostgreSQL handles:
- 50K+ simple queries per second
- 10TB+ with proper indexing
- Millions of concurrent connections (with pgbouncer)

Shard when you actually need it, not before.

### 2. Wrong Shard Key

Changing shard key requires full data migration. Classic mistakes:
- Sharding by timestamp (creates hot spots)
- Sharding by non-query field (forces scatter-gather)
- Sharding by low-cardinality field (uneven distribution)

### 3. Ignoring Operational Complexity

Sharded systems require:
- Cross-shard query coordination
- Distributed transactions (or saga patterns)
- Complex backup/restore procedures
- Per-shard monitoring and alerting
- Shard-aware connection pooling

Budget 3-4x operational complexity vs single database.

### 4. Auto-Increment Primary Keys

```sql
-- Don't:
INSERT INTO orders (id, ...) VALUES (AUTO_INCREMENT, ...);
-- Creates hot spot on "current" shard for sequential writes

-- Do:
INSERT INTO orders (id, ...) VALUES (UUID(), ...);
-- Random distribution across shards

-- Or use Twitter Snowflake IDs:
-- Time-based but includes shard/worker ID
-- Sortable AND distributed
```

---

## Sharding at Scale: Real Examples

### Instagram

- Shards by user_id
- Each shard is PostgreSQL
- Uses consistent hashing with virtual nodes
- Celebrity hot spots handled by caching + read replicas

### Uber

- Shards by city/region (geographic sharding)
- Each city's data mostly independent
- Cross-city queries rare
- Easier operational isolation

### Stripe

- Shards by merchant_id
- Financial data requires strong consistency
- Uses Vitess for MySQL sharding
- Cross-shard transactions for settlements

---

## Interview Checklist

- [ ] Justify WHY sharding is needed (calculate data size, QPS)
- [ ] Choose shard key with clear reasoning
- [ ] Address hot partition mitigation
- [ ] Discuss cross-shard query strategy
- [ ] Consider operational complexity
- [ ] Plan for resharding

---

## Staff+ Insights

**When to shard:**
- Single node can't handle write throughput
- Data doesn't fit on single node (even with compression)
- Regulatory requirements (data residency)
- Operational isolation (tenant boundaries)

**When NOT to shard:**
- Read scaling only → use read replicas
- Performance issues → tune indexes, queries first
- "Future-proofing" → YAGNI applies

**Sharding complexity levels:**
1. Application-level sharding (you manage routing)
2. Proxy-based sharding (Vitess, ProxySQL)
3. Database-native sharding (CockroachDB, Spanner, Citus)

Level 3 is easiest to operate but may have limitations.
