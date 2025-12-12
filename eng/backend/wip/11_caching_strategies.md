# Chapter 11: Caching Strategies

## The Caching Imperative

**Facebook TAO Case Study:**
- Social graph: Billions of nodes, trillions of edges
- 99.8% of queries served from cache
- Cache hit: 0.5ms, Cache miss: 10ms (20x slower)
- Without cache: System would collapse under load

Caching isn't an optimization—it's a survival strategy at scale.

### The Numbers

```
Database query: 10-100ms
Redis get: 0.1-1ms
Local memory: 0.0001ms

100x-1000x speedup from caching
```

---

## Cache Patterns

### 1. Cache-Aside (Lazy Loading)

The application manages the cache explicitly.

```python
def get_user(user_id):
    # 1. Check cache first
    cache_key = f"user:{user_id}"
    cached = cache.get(cache_key)
    
    if cached:
        return cached  # Cache HIT - fast path
    
    # 2. Cache MISS - query database
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # 3. Populate cache for next time
    cache.set(cache_key, user, ttl=3600)  # 1 hour TTL
    
    return user
```

**Flow:**
```
Request → Check Cache → HIT? → Return cached data
                     → MISS? → Query DB → Store in cache → Return data
```

**Pros:**
- Simple to understand and implement
- Only caches data that's actually requested
- Cache failures don't break the system (fallback to DB)

**Cons:**
- First request always slow (cache miss)
- Cache and DB can become inconsistent
- Application must handle cache logic

**Best for:** Read-heavy workloads, data that's expensive to compute

### 2. Write-Through

Write to cache AND database together.

```python
def update_user(user_id, data):
    # Write to both cache and database
    cache_key = f"user:{user_id}"
    
    # Write to database
    db.update("UPDATE users SET ... WHERE id = ?", data, user_id)
    
    # Write to cache (same data)
    cache.set(cache_key, data, ttl=3600)
```

**Flow:**
```
Write Request → Write to DB → Write to Cache → Return success
```

**Pros:**
- Cache always consistent with DB
- No stale reads
- Subsequent reads are fast

**Cons:**
- Every write has cache overhead
- Higher write latency
- May cache data that's never read

**Best for:** Data that's read frequently after writes

### 3. Write-Behind (Write-Back)

Write to cache immediately, async write to database.

```python
def update_user(user_id, data):
    cache_key = f"user:{user_id}"
    
    # Write to cache only (immediate)
    cache.set(cache_key, data)
    
    # Queue async write to database
    write_queue.enqueue({
        "operation": "UPDATE",
        "table": "users",
        "id": user_id,
        "data": data
    })
    
    # Return immediately (don't wait for DB)
    return {"status": "success"}

# Background worker
def process_write_queue():
    while True:
        batch = write_queue.dequeue_batch(size=100)
        db.batch_write(batch)  # Efficient bulk write
```

**Flow:**
```
Write Request → Write to Cache → Queue DB Write → Return success
                                      ↓
                              Background Worker → Batch Write to DB
```

**Pros:**
- Extremely fast writes (just cache + queue)
- Can batch DB writes (much more efficient)
- Absorbs write spikes

**Cons:**
- Risk of data loss if cache fails before DB write
- Complex to implement correctly
- Eventually consistent

**Best for:** High write throughput, analytics/logging where some loss OK

### 4. Read-Through

Cache handles DB reads transparently.

```python
# Application code is simple
def get_user(user_id):
    return cache.get(f"user:{user_id}")  # Cache handles miss

# Cache implementation (e.g., custom cache layer)
class ReadThroughCache:
    def get(self, key):
        value = self.cache.get(key)
        if value is None:
            # Cache handles the DB read
            value = self.load_from_db(key)
            self.cache.set(key, value, ttl=3600)
        return value
```

**Pros:**
- Simpler application code
- Consistent caching logic
- Easy to add caching to existing systems

**Cons:**
- Requires cache infrastructure to support this
- Less control over caching behavior

### 5. Refresh-Ahead

Proactively refresh cache before expiration.

```python
class RefreshAheadCache:
    def get(self, key):
        value, ttl_remaining = self.cache.get_with_ttl(key)
        
        if ttl_remaining < self.refresh_threshold:
            # TTL running low - refresh in background
            self.async_refresh(key)
        
        return value
    
    async def async_refresh(self, key):
        # Don't block the request
        fresh_value = await self.load_from_db(key)
        self.cache.set(key, fresh_value, ttl=3600)
```

**Pros:**
- Eliminates cache miss latency for hot keys
- Predictable performance

**Cons:**
- More complex
- May refresh data that won't be requested again
- Background refresh adds load

**Best for:** Hot data with predictable access patterns

---

## Cache Stampede Prevention

### The Problem

```
Popular cache key expires at T=100
T=100.001: Request 1 - cache miss, queries DB
T=100.002: Request 2 - cache miss, queries DB
T=100.003: Request 3 - cache miss, queries DB
...
T=100.100: Request 1000 - cache miss, queries DB

1000 simultaneous DB queries for SAME data!
Database overwhelmed → cascading failure
```

This is the **cache stampede** (or thundering herd) problem.

### Solution 1: Locking (Mutex)

Only ONE request fetches, others wait.

```python
def get_with_lock(key):
    value = cache.get(key)
    if value is not None:
        return value
    
    # Try to acquire lock
    lock_key = f"lock:{key}"
    if cache.set_nx(lock_key, "1", ttl=10):  # Got lock
        try:
            value = db.query(key)
            cache.set(key, value, ttl=3600)
            return value
        finally:
            cache.delete(lock_key)
    else:
        # Someone else has lock - wait and retry
        time.sleep(0.1)
        return get_with_lock(key)  # Retry
```

**Pros:** Only one DB query
**Cons:** Adds latency for waiting requests, lock management complexity

### Solution 2: Probabilistic Early Expiration

Refresh cache BEFORE it expires, probabilistically.

```python
def get_with_early_refresh(key):
    value, ttl_remaining, total_ttl = cache.get_with_metadata(key)
    
    if value is None:
        return fetch_and_cache(key)
    
    # Probabilistic early refresh
    # As TTL decreases, probability of refresh increases
    time_to_expire = ttl_remaining
    delta = total_ttl * 0.1  # 10% of TTL
    
    if random.random() < math.exp(-time_to_expire / delta):
        # Refresh in background (don't block)
        async_refresh(key)
    
    return value
```

As expiration approaches, more requests will trigger refresh. First one to finish repopulates cache.

### Solution 3: Background Refresh

Never let cache expire—refresh proactively.

```python
# Background job runs every minute
def refresh_hot_keys():
    hot_keys = get_hot_keys()  # Track frequently accessed keys
    
    for key in hot_keys:
        ttl = cache.ttl(key)
        if ttl < 300:  # Less than 5 minutes left
            value = db.query(key)
            cache.set(key, value, ttl=3600)

# Schedule: */1 * * * * refresh_hot_keys
```

**Pros:** No stampede possible
**Cons:** Need to track hot keys, more infrastructure

### Solution 4: Stale-While-Revalidate

Serve stale data while refreshing in background.

```python
def get_with_swr(key):
    value, is_stale = cache.get_with_stale_flag(key)
    
    if value is not None:
        if is_stale:
            # Serve stale data, but refresh in background
            async_refresh(key)
        return value
    
    # No data at all - must wait
    return fetch_and_cache(key)
```

**Pros:** Always fast (serve stale while refreshing)
**Cons:** May serve stale data briefly

---

## Cache Invalidation Strategies

### Time-Based (TTL)

```python
cache.set("user:123", user_data, ttl=3600)  # Expires in 1 hour
```

Simple but data can be stale for up to TTL duration.

### Event-Based

Invalidate when data changes:

```python
def update_user(user_id, data):
    db.update(user_id, data)
    cache.delete(f"user:{user_id}")  # Invalidate immediately

# Or publish event
def update_user(user_id, data):
    db.update(user_id, data)
    event_bus.publish("user.updated", {"user_id": user_id})

# Cache service subscribes
@event_handler("user.updated")
def on_user_updated(event):
    cache.delete(f"user:{event.user_id}")
```

### Version-Based

Include version in cache key:

```python
def get_user(user_id):
    version = db.get_user_version(user_id)
    cache_key = f"user:{user_id}:v{version}"
    
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    user = db.get_user(user_id)
    cache.set(cache_key, user, ttl=86400)  # Long TTL OK
    return user

def update_user(user_id, data):
    db.update_user(user_id, data)
    db.increment_user_version(user_id)  # Old cache key now orphaned
```

Old versions naturally expire. No explicit invalidation needed.

---

## Cache Data Structures

### Simple Key-Value

```python
cache.set("user:123", {"name": "Alice", "email": "alice@example.com"})
user = cache.get("user:123")
```

### Hash (for partial updates)

```python
# Store user as hash
cache.hset("user:123", "name", "Alice")
cache.hset("user:123", "email", "alice@example.com")

# Update single field (without fetching entire object)
cache.hset("user:123", "email", "newemail@example.com")

# Get single field
email = cache.hget("user:123", "email")

# Get all fields
user = cache.hgetall("user:123")
```

### Sorted Set (for leaderboards, feeds)

```python
# Add scores
cache.zadd("leaderboard", {"player1": 1000, "player2": 1500, "player3": 800})

# Get top 10
top_10 = cache.zrevrange("leaderboard", 0, 9, withscores=True)

# Get player rank
rank = cache.zrevrank("leaderboard", "player1")
```

### List (for queues, recent items)

```python
# Add to recent views
cache.lpush("user:123:recent_views", "product:456")
cache.ltrim("user:123:recent_views", 0, 99)  # Keep last 100

# Get recent views
recent = cache.lrange("user:123:recent_views", 0, 9)  # Last 10
```

---

## Multi-Level Caching

```
┌─────────────────────────────────────────────────────┐
│                   Application                        │
│  ┌─────────────────────────────────────────────┐    │
│  │            L1: In-Memory Cache               │    │
│  │         (HashMap, Guava, Caffeine)          │    │
│  │         Latency: 0.001ms                    │    │
│  │         Size: 100MB-1GB                     │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   L2: Distributed     │
              │      (Redis)          │
              │   Latency: 0.5-1ms    │
              │   Size: 10GB-100GB    │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │      Database         │
              │   Latency: 10-100ms   │
              └───────────────────────┘
```

```python
def get_user(user_id):
    # L1: Check local cache
    user = local_cache.get(f"user:{user_id}")
    if user:
        return user
    
    # L2: Check Redis
    user = redis.get(f"user:{user_id}")
    if user:
        local_cache.set(f"user:{user_id}", user, ttl=60)  # Short L1 TTL
        return user
    
    # L3: Database
    user = db.get_user(user_id)
    redis.set(f"user:{user_id}", user, ttl=3600)
    local_cache.set(f"user:{user_id}", user, ttl=60)
    return user
```

### L1 Cache Challenges

**Cache coherence across instances:**
```
Instance A: L1 has user:123 = {name: "Alice"}
Instance B: Updates user:123 to {name: "Alicia"}
Instance B: Invalidates Redis and its own L1
Instance A: Still has stale data in L1!
```

**Solutions:**
- Short L1 TTL (60 seconds)
- Pub/sub invalidation (Redis publish, all instances subscribe)
- Accept some staleness (depends on use case)

---

## Cache Sizing and Eviction

### Eviction Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| LRU (Least Recently Used) | Evict least recently accessed | General purpose |
| LFU (Least Frequently Used) | Evict least accessed overall | Stable popularity |
| FIFO (First In First Out) | Evict oldest | Simple, predictable |
| Random | Evict random item | When access patterns unknown |
| TTL | Evict expired items | Time-sensitive data |

### Memory Estimation

```
User object: ~500 bytes
1 million users in cache: 500MB
10 million users: 5GB

With overhead (Redis): Add ~30-50%
10 million users in Redis: ~7GB
```

### Hit Rate Optimization

```
Current: 90% hit rate, 10% miss rate
DB handles: 1000 QPS * 10% = 100 QPS

Goal: 99% hit rate
DB handles: 1000 QPS * 1% = 10 QPS (10x reduction!)

How to improve:
1. Increase cache size (more items fit)
2. Better eviction policy
3. Longer TTLs
4. Cache warming
5. Prefetching
```

---

## Interview Checklist

- [ ] Choose appropriate cache pattern (cache-aside, write-through, etc.)
- [ ] Address cache invalidation strategy
- [ ] Handle cache stampede (locking, probabilistic refresh)
- [ ] Consider multi-level caching
- [ ] Plan for cache failures (fallback to DB)
- [ ] Size the cache appropriately
- [ ] Select eviction policy

---

## Staff+ Insights

**Cache is a contract:**
- What's the max staleness your product can tolerate?
- Document this for your team

**Hit rate is the metric:**
- Instrument hit/miss rates
- Alert when hit rate drops
- Investigate cache eviction patterns

**Cache warming:**
- Pre-populate cache before traffic hits
- Essential for deployments, failovers

**Don't cache everything:**
- Cache hot data (90% of requests for 10% of data)
- Long-tail data might not be worth caching
- Calculate: cost of cache miss vs cost of cache storage

**Redis vs Memcached:**
- Redis: Data structures, persistence, pub/sub
- Memcached: Simpler, multi-threaded, pure cache
- Most choose Redis for flexibility
