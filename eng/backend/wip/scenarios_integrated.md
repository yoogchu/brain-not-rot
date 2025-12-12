# Part 9: Integrated Problem Scenarios

## Scenario 1: Design a Local Delivery Service (Gopuff/DoorDash)

### Requirements

```
Functional:
- Users browse local store inventory
- Real-time delivery ETA
- Order placement and payment
- Driver assignment and tracking
- Delivery confirmation

Non-Functional:
- Low latency for nearby store queries
- Handle dinner rush (10x normal traffic)
- 99.9% availability
- Accurate inventory (no overselling)
```

### Scale Estimation

```
Users: 1M DAU
Orders: 100K orders/day
Peak: 10K orders/hour (dinner rush)
Stores: 1000 dark stores, 50K SKUs each
Drivers: 20K active drivers

Storage:
- Users: 10M users × 1KB = 10GB
- Orders: 100K/day × 365 days × 2KB = 73GB/year
- Products: 1000 stores × 50K × 500B = 25GB
- Driver locations: 20K × 100 updates/hour = 2M writes/hour
```

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                          Clients                                 │
│              (Mobile App, Web)                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway                                 │
│              (Auth, Rate Limiting, Routing)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Catalog    │    │    Order     │    │   Delivery   │
│   Service    │    │   Service    │    │   Service    │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Inventory   │    │   Payment    │    │   Driver     │
│   Service    │    │   Service    │    │  Location    │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Key Design Decisions

**1. Store/Product Discovery:**
```
User location → Find nearby stores → Get products

Geospatial indexing:
- Store locations in PostGIS or Redis Geo
- Query: "Stores within 5 miles of (lat, lng)"
- Cache popular areas heavily

Response < 100ms requirement:
- CDN for static catalog
- Redis cache for inventory
- Precompute store assignments by zip code
```

**2. Inventory Management:**
```
Challenge: Don't oversell during rush

Option A: Pessimistic locking
BEGIN;
SELECT quantity FROM inventory WHERE sku = X FOR UPDATE;
-- If quantity >= ordered_quantity
UPDATE inventory SET quantity = quantity - N WHERE sku = X;
COMMIT;

Pros: Strong consistency
Cons: Lock contention during rush

Option B: Optimistic with reservation
1. Reserve inventory (soft lock, TTL)
2. Process payment
3. Confirm reservation or release

Better for high concurrency
```

**3. Driver Assignment:**
```
When order placed:
1. Find available drivers near store
2. Score by: distance, current route, acceptance rate
3. Send offer to best driver
4. Timeout → offer to next driver

Real-time location:
- Drivers send location every 5 seconds
- Store in Redis with TTL
- Geospatial query for nearby drivers
```

**4. ETA Calculation:**
```
ETA = Store prep time + Driver pickup + Transit time

Factors:
- Historical prep time by store/order size
- Current driver location
- Traffic (Google Maps API)
- Weather adjustments

Update ETA every minute, push to client via WebSocket
```

---

## Scenario 2: Design a Real-Time Bidding System

### Requirements

```
Functional:
- Receive ad requests from publishers (100ms SLA)
- Query multiple demand partners
- Select winning bid
- Track impressions and clicks

Non-Functional:
- 1M requests/second peak
- < 100ms end-to-end latency
- No duplicate billings
- Real-time budget tracking
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Publisher Ad Request                          │
│              (User visits page, ad slot available)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Ad Exchange Service                           │
│              (Receives request, 10ms budget)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ DSP 1 (30ms)   │  │ DSP 2 (30ms)   │  │ DSP 3 (30ms)   │
│ Bidder         │  │ Bidder         │  │ Bidder         │
└────────────────┘  └────────────────┘  └────────────────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Auction Engine (20ms)                         │
│              (Second-price auction, fraud check)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Ad Server (10ms)                              │
│              (Return winning creative)                           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**1. Low Latency Bidding:**
```
Total budget: 100ms
- Request parsing: 5ms
- Fan out to DSPs: 30ms (parallel, with timeout)
- Auction: 5ms
- Response: 5ms
- Buffer: 55ms

DSP timeout handling:
- Set hard 30ms timeout
- Return best bids received
- Log late responses for analysis
```

**2. Budget Tracking:**
```
Challenge: 1M rps, real-time budget enforcement

Approach: Approximate budgets locally, reconcile periodically

Each bidder node:
- Local budget counter (in-memory)
- Decrement on bid
- Sync with central store every 100ms
- Stop bidding if local budget exhausted

Central budget store:
- Redis with atomic operations
- Campaign ID → remaining budget
- Periodic reconciliation with billing
```

**3. Deduplication:**
```
Problem: Network retries cause duplicate impressions

Solution: Idempotent impression logging

impression_id = hash(request_id + creative_id + timestamp_bucket)

Redis:
SETNX impression:{id} 1 EX 3600
- If set succeeds: new impression, log it
- If set fails: duplicate, ignore
```

---

## Scenario 3: Design a Global Chat Application

### Requirements

```
Functional:
- 1:1 and group messaging (up to 1000 members)
- Message delivery and read receipts
- Online/offline status
- Push notifications
- Message history

Non-Functional:
- Real-time delivery (< 500ms)
- Global users (minimize latency)
- Message ordering within conversation
- Offline message queuing
```

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Global                                    │
│                                                                   │
│   US-West          US-East          EU-West          Asia        │
│  ┌───────┐        ┌───────┐        ┌───────┐       ┌───────┐    │
│  │ Chat  │◄──────►│ Chat  │◄──────►│ Chat  │◄─────►│ Chat  │    │
│  │Cluster│        │Cluster│        │Cluster│       │Cluster│    │
│  └───────┘        └───────┘        └───────┘       └───────┘    │
│      │                │                │               │         │
│      └────────────────┴────────────────┴───────────────┘         │
│                              │                                    │
│                    Cross-Region Kafka                            │
│                    (async replication)                           │
└──────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**1. Message Delivery:**
```
Sender → Chat Service → Kafka → Recipient's Chat Service → WebSocket → Recipient

Message flow:
1. Client sends message via WebSocket
2. Chat service validates, assigns message ID
3. Publish to Kafka topic (partitioned by conversation_id)
4. Consumer routes to recipient's connection
5. If offline: Store, send push notification

Ordering: Kafka partition per conversation ensures order
```

**2. Presence (Online/Offline):**
```
Challenge: Track millions of online users

Approach: Regional presence with gossip

Each region:
- User connects → Add to local presence set
- User disconnects → Remove after timeout (handle reconnects)
- Heartbeat every 30 seconds

Cross-region:
- Gossip protocol shares presence summaries
- "User X is online in EU-West"
- Query local first, then remote regions
```

**3. Group Messages:**
```
Small groups (< 100):
- Fan-out on write
- Send message to all members immediately
- Simple, low latency

Large groups (100-1000):
- Fan-out on read
- Write to group inbox
- Members fetch from inbox
- More scalable, slightly higher latency
```

**4. Message Storage:**
```
Hot data (recent messages):
- Cassandra with time-based partitioning
- conversation_id as partition key
- message_timestamp as clustering key

Cold data (old messages):
- Archive to S3 after 30 days
- Load on demand if user scrolls back
```

---

## Scenario 4: Design a Video Processing Pipeline

### Requirements

```
Functional:
- Accept video uploads (up to 10GB)
- Transcode to multiple resolutions/formats
- Generate thumbnails
- Extract metadata
- Deliver via CDN

Non-Functional:
- Process 10K videos/day
- Resilient to failures (no lost videos)
- Cost efficient (spot instances)
- Progress tracking
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Video Upload                                │
│              (Multipart upload to S3)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                         S3 Event
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Processing Queue (SQS)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Workflow Orchestrator                         │
│                    (AWS Step Functions)                          │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Validate │─►│Transcode │─►│Thumbnail │─►│ Publish  │        │
│  │          │  │ (parallel)│  │          │  │          │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                     │                                            │
│            ┌────────┼────────┐                                   │
│            ▼        ▼        ▼                                   │
│         1080p    720p      480p                                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CDN Distribution                              │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**1. Reliable Upload:**
```
Large files need resumable uploads:

Multipart upload:
1. Client requests upload URL from API
2. API creates multipart upload, returns upload_id
3. Client uploads parts (5MB each)
4. If part fails: retry just that part
5. Client completes upload
6. S3 assembles parts

Track progress:
- Store part status in DynamoDB
- Resume from last successful part
```

**2. Processing Workflow:**
```
Step Functions state machine:

StartProcessing
    ↓
Validate (file type, size, duration)
    ↓
Parallel:
    ├── Transcode 1080p
    ├── Transcode 720p
    ├── Transcode 480p
    └── Generate thumbnails
    ↓
Wait for all
    ↓
Update database
    ↓
Notify user
    ↓
End

Retries: Built-in per step
Failures: Dead letter queue for manual review
```

**3. Cost Optimization:**
```
Transcoding is CPU-intensive:

Spot instances:
- 70-90% cost savings
- Handle interruptions gracefully
- Checkpoint progress to S3

Right-sizing:
- Match instance to video resolution
- 4K → Large instance
- 480p → Small instance

Batch processing:
- Queue videos, process in batches
- Better resource utilization
```

---

## Scenario 5: Design a Distributed Rate Limiter

### Requirements

```
Functional:
- Rate limit API requests per user/API key
- Support multiple limit tiers
- Return remaining quota in response headers
- Allow burst within limits

Non-Functional:
- Sub-millisecond latency overhead
- Work across multiple API servers
- Handle 100K+ rate limit checks/second
- Graceful degradation if rate limiter fails
```

### Design

```
┌─────────────────────────────────────────────────────────────────┐
│                      API Request                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API Gateway / LB                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Rate Limit Middleware                         │
│                                                                   │
│  1. Extract client ID (API key, user ID, IP)                    │
│  2. Check rate limit                                             │
│  3. If allowed: Increment counter, proceed                       │
│  4. If exceeded: Return 429                                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Redis Cluster                                 │
│              (Sliding window counters)                           │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# Sliding window counter in Redis
import redis
import time

def check_rate_limit(redis_client, key: str, limit: int, window: int) -> tuple[bool, int]:
    """
    Returns (allowed, remaining_quota)
    """
    now = time.time()
    window_start = now - window
    
    # Use sorted set with timestamp as score
    pipe = redis_client.pipeline()
    
    # Remove old entries
    pipe.zremrangebyscore(key, 0, window_start)
    
    # Count current entries
    pipe.zcard(key)
    
    # Add current request
    pipe.zadd(key, {f"{now}:{uuid4()}": now})
    
    # Set expiry
    pipe.expire(key, window)
    
    _, current_count, _, _ = pipe.execute()
    
    remaining = max(0, limit - current_count)
    allowed = current_count < limit
    
    return allowed, remaining
```

### Handling Failures

```
Redis down? Options:

1. Fail open (allow all requests):
   - Risk: No rate limiting
   - Use: When availability > protection

2. Fail closed (reject all requests):
   - Risk: Complete outage
   - Use: When protection critical

3. Local rate limiting:
   - Each server tracks locally
   - Less accurate but functional
   - Use: Best of both worlds

4. Circuit breaker on Redis:
   - After N failures, fall back to local
   - Periodically retry Redis
```

---

## Interview Checklist for Design Scenarios

- [ ] Clarify requirements (functional and non-functional)
- [ ] Estimate scale (users, requests, storage)
- [ ] Draw high-level architecture
- [ ] Identify critical paths and bottlenecks
- [ ] Deep dive into 2-3 key components
- [ ] Discuss trade-offs of design decisions
- [ ] Address failure modes and recovery
- [ ] Consider monitoring and observability
