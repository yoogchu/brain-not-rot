# Chapter 16: Rate Limiting & Load Shedding

## Why Rate Limit?

Without rate limiting:

```
Normal traffic: 1,000 requests/second
Viral tweet links to your API: 100,000 requests/second
Result: Database overloaded → Cascading failure → Complete outage

One misbehaving client takes down service for ALL users
```

Rate limiting protects your system from:
- **Traffic spikes** (viral content, news events)
- **Misbehaving clients** (bugs, infinite loops)
- **Malicious actors** (DDoS attacks, scraping)
- **Cascading failures** (one service overload spreads)

---

## Rate Limiting Algorithms

### 1. Token Bucket

Imagine a bucket that fills with tokens at a steady rate.

```
┌─────────────────────────────┐
│    Token Bucket             │
│    Capacity: 10 tokens      │
│    Refill: 1 token/second   │
│                             │
│    ●●●●●●●○○○               │
│    (7 tokens available)     │
└─────────────────────────────┘

Request arrives:
- Have tokens? Take one, process request
- No tokens? Reject (429 Too Many Requests)
```

**Implementation:**

```python
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity        # Max tokens
        self.tokens = capacity          # Current tokens
        self.refill_rate = refill_rate  # Tokens per second
        self.last_refill = time.time()
    
    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def allow_request(self, tokens_needed=1):
        self._refill()
        if self.tokens >= tokens_needed:
            self.tokens -= tokens_needed
            return True
        return False

# Usage
bucket = TokenBucket(capacity=10, refill_rate=1)  # 10 burst, 1/sec sustained

if bucket.allow_request():
    process_request()
else:
    return "429 Too Many Requests"
```

**Characteristics:**
- Allows **bursts** up to bucket capacity
- Smooths traffic to refill rate over time
- Memory efficient (just a few numbers per client)

**Use cases:** API rate limiting, network traffic shaping

### 2. Leaky Bucket

Requests enter a queue (bucket), processed at fixed rate.

```
Requests arrive (variable rate)
         │
         ▼
┌─────────────────────────────┐
│    Queue (Bucket)           │
│    ▣ ▣ ▣ ▣ ▣                │  ← Requests waiting
│                             │
└─────────────────────────────┘
         │
         ▼ Fixed rate (leak)
    Process 1 req/sec
```

**Characteristics:**
- Output rate is **constant** (smooths bursts)
- Requests wait in queue
- Queue overflow = rejection

**Use cases:** Traffic shaping, ensuring constant processing rate

### Token Bucket vs Leaky Bucket

| Aspect | Token Bucket | Leaky Bucket |
|--------|--------------|--------------|
| Bursts | Allows (up to capacity) | Smooths out |
| Output rate | Variable (up to burst) | Constant |
| Implementation | Counter | Queue |
| Memory | Lower | Higher (stores queue) |

### 3. Fixed Window Counter

Count requests in fixed time windows.

```
Window: 1 minute

[10:00-10:01]: 45 requests  ✓ (under 100)
[10:01-10:02]: 78 requests  ✓ (under 100)
[10:02-10:03]: 102 requests ✗ (over 100, reject 2)
```

**Problem: Boundary burst**

```
Limit: 100 requests/minute

10:00:30 - 10:00:59: 100 requests (end of window)
10:01:00 - 10:01:29: 100 requests (start of window)

200 requests in 1 minute! But each window is under limit.
```

### 4. Sliding Window Log

Track exact timestamp of each request.

```python
class SlidingWindowLog:
    def __init__(self, limit, window_seconds):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests = {}  # {client_id: [timestamps]}
    
    def allow_request(self, client_id):
        now = time.time()
        window_start = now - self.window_seconds
        
        # Get client's request timestamps
        timestamps = self.requests.get(client_id, [])
        
        # Remove old timestamps
        timestamps = [t for t in timestamps if t > window_start]
        
        if len(timestamps) >= self.limit:
            return False
        
        timestamps.append(now)
        self.requests[client_id] = timestamps
        return True
```

**Pros:** Accurate, no boundary issues
**Cons:** High memory (store every timestamp)

### 5. Sliding Window Counter

Hybrid: Weighted average of current and previous window.

```
Current window: 10:01-10:02, 40 requests so far
Previous window: 10:00-10:01, 60 requests total
Current time: 10:01:30 (50% through current window)

Weighted count = (previous * 0.5) + current
               = (60 * 0.5) + 40
               = 70 requests
```

**Best balance of accuracy vs memory.**

---

## Distributed Rate Limiting

Single server rate limiting is easy. Distributed is hard.

### Problem

```
User rate limit: 100 requests/minute

┌─────────┐    ┌─────────┐    ┌─────────┐
│ Server1 │    │ Server2 │    │ Server3 │
│ 40 reqs │    │ 35 reqs │    │ 45 reqs │
└─────────┘    └─────────┘    └─────────┘

Each server thinks user is under limit
Total: 120 requests/minute → OVER LIMIT!
```

### Solution 1: Centralized Counter (Redis)

```python
def rate_limit_redis(client_id, limit, window_seconds):
    key = f"ratelimit:{client_id}:{int(time.time() / window_seconds)}"
    
    # Atomic increment and check
    pipe = redis.pipeline()
    pipe.incr(key)
    pipe.expire(key, window_seconds)
    count, _ = pipe.execute()
    
    return count <= limit
```

**Pros:** Accurate, consistent
**Cons:** Redis latency added to every request, Redis is SPOF

### Solution 2: Local Rate Limiting with Sync

```python
class LocalWithSync:
    def __init__(self, limit):
        self.local_counts = {}
        self.limit = limit
    
    def allow_request(self, client_id):
        # Fast local check
        local_count = self.local_counts.get(client_id, 0)
        if local_count > self.limit * 1.1:  # 10% buffer
            return False
        
        self.local_counts[client_id] = local_count + 1
        return True
    
    # Background sync every second
    def sync_to_redis(self):
        for client_id, count in self.local_counts.items():
            redis.incrby(f"ratelimit:{client_id}", count)
        self.local_counts.clear()
```

**Trade-off:** Less accurate but faster

### Solution 3: Sticky Sessions

Route same client to same server:

```
Client A → always Server 1 (local rate limiting works)
Client B → always Server 2 (local rate limiting works)
```

Works with IP hash or cookie-based routing.

---

## Circuit Breaker Pattern

When downstream service fails, stop hammering it.

### States

```
┌──────────────────────────────────────────────────┐
│                                                  │
│   ┌────────┐    Failures    ┌────────┐          │
│   │ CLOSED │ ────exceed────→│  OPEN  │          │
│   │        │    threshold   │        │          │
│   └────────┘                └────────┘          │
│       ▲                          │              │
│       │                     Timeout              │
│       │                          │              │
│       │    Success          ┌────────┐          │
│       └─────────────────────│ HALF-  │          │
│                             │  OPEN  │          │
│          Failure            └────────┘          │
│            │                     │              │
│            └─────────────────────┘              │
│                                                  │
└──────────────────────────────────────────────────┘
```

**CLOSED:** Normal operation, tracking failures
**OPEN:** All requests immediately rejected (fail fast)
**HALF-OPEN:** Allow one test request through

### Implementation

```python
class CircuitBreaker:
    def __init__(self, failure_threshold, recovery_timeout):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "CLOSED"
        self.last_failure_time = None
    
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF-OPEN"
            else:
                raise CircuitOpenError("Circuit is open")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
    
    def on_success(self):
        self.failures = 0
        self.state = "CLOSED"
    
    def on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"

# Usage
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

try:
    result = breaker.call(external_service.get_data, user_id)
except CircuitOpenError:
    # Fast fail, use fallback
    result = get_cached_data(user_id)
```

---

## Load Shedding

When overloaded, intentionally drop requests to protect the system.

### Priority-Based Shedding

```python
def handle_request(request):
    current_load = get_system_load()
    
    if current_load > 0.9:  # 90% capacity
        # Only serve highest priority
        if request.priority != "critical":
            return Response(status=503, body="Service overloaded")
    
    elif current_load > 0.8:  # 80% capacity
        # Shed low priority
        if request.priority == "low":
            return Response(status=503, body="Service overloaded")
    
    # Process request
    return process(request)
```

**Priority examples:**
- Critical: Payment processing
- High: User-facing reads
- Medium: Background syncs
- Low: Analytics, logging

---

## Rate Limit Response Design

### Headers

```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1640995200
Retry-After: 30

{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Please retry after 30 seconds.",
  "retry_after": 30
}
```

**Standard headers:**
| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Max requests allowed in window |
| `X-RateLimit-Remaining` | Requests remaining in window |
| `X-RateLimit-Reset` | Unix timestamp when window resets |
| `Retry-After` | Seconds until client should retry |

---

## Rate Limiting by Dimension

### By Client/User

```python
# Per API key
limit_key = f"ratelimit:apikey:{api_key}"

# Per user
limit_key = f"ratelimit:user:{user_id}"

# Per IP (anonymous)
limit_key = f"ratelimit:ip:{client_ip}"
```

### By Endpoint

```python
# Different limits for different endpoints
ENDPOINT_LIMITS = {
    "/api/search": 10,      # Expensive
    "/api/users": 100,      # Normal
    "/api/health": 1000,    # Cheap
}
```

### By Plan/Tier

```python
PLAN_LIMITS = {
    "free": 100,
    "starter": 1000,
    "pro": 10000,
    "enterprise": 100000,
}
```

---

## Key Concepts Checklist

- [ ] Choose appropriate algorithm (token bucket for API, sliding window for accuracy)
- [ ] Design for distributed environment (Redis, or accept approximation)
- [ ] Define rate limit dimensions (user, IP, endpoint, plan)
- [ ] Implement circuit breaker for downstream protection
- [ ] Plan load shedding strategy for overload
- [ ] Design informative rate limit responses

---

## Practical Insights

**Rate limiting is business logic:**
- Work with product to define limits
- Different limits for different customer tiers
- Monitor and adjust based on real usage

**Defense in depth:**
- Edge/CDN rate limiting (coarse, fast)
- API Gateway rate limiting (per-route)
- Application rate limiting (per-user, fine-grained)

**Circuit breaker tuning:**
- Too sensitive: Flapping open/closed
- Too lenient: Doesn't protect
- Start conservative, adjust based on data
