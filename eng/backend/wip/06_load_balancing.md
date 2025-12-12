# Chapter 6: Load Balancing

## Why Load Balancing?

Single server has hard limits:

- **Availability**: Server dies = complete outage
- **Scalability**: One server has finite CPU, memory, network
- **Maintenance**: Can't upgrade without downtime

Load balancer distributes requests across multiple servers:

```
                    ┌─────────────┐
                    │   Client    │
                    └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │    Load     │
                   │  Balancer   │
                   └─────────────┘
                    /     │     \
                   ▼      ▼      ▼
            ┌──────┐ ┌──────┐ ┌──────┐
            │ Srv1 │ │ Srv2 │ │ Srv3 │
            └──────┘ └──────┘ └──────┘
```

Now you can:
- Handle more traffic (horizontal scaling)
- Survive server failures (remove unhealthy servers)
- Deploy without downtime (rolling updates)

---

## Layer 4 vs Layer 7 Load Balancing

The "layer" refers to the OSI network model.

### Layer 4 (Transport Layer)

Operates on **TCP/UDP** level. Sees only network information.

```
What L4 LB sees:
- Source IP: 203.0.113.50
- Destination IP: 10.0.0.1
- Source Port: 52431
- Destination Port: 443
- Protocol: TCP

What L4 LB does NOT see:
- HTTP headers
- URLs  
- Cookies
- Request body
```

**How it works:**
```
1. Client connects to Load Balancer IP:443
2. LB picks a backend server (e.g., server-2)
3. LB rewrites packet destination to server-2's IP (NAT)
4. Packets flow to server-2
5. Response packets NAT'd back through LB to client
6. Client thinks it's talking to LB the whole time
```

**Characteristics:**
| Aspect | L4 LB |
|--------|-------|
| Speed | Very fast (just packet forwarding) |
| Protocol awareness | None (doesn't understand HTTP) |
| SSL termination | No (passes encrypted traffic through) |
| Routing decisions | IP/port based only |
| Connection state | Per TCP connection |

**Use cases:**
- TCP services (databases, Redis, message queues)
- Very high throughput requirements (millions of connections)
- When you don't need content-based routing
- Gaming servers, MQTT brokers

### Layer 7 (Application Layer)

Operates on **HTTP/HTTPS** level. Understands application protocols.

```
What L7 LB sees:
- Full HTTP request
- URL: /api/users/123
- Headers: Authorization, Accept-Language, Cookie
- HTTP method: GET, POST, etc.
- Request body (if needed)
```

**How it works:**
```
1. Client sends HTTP request to Load Balancer
2. LB TERMINATES the TCP connection (full HTTP parsing)
3. LB inspects HTTP request (URL, headers, etc.)
4. LB establishes NEW TCP connection to backend
5. LB forwards request (potentially modified)
6. LB receives response from backend
7. LB forwards response to client
```

**Capabilities:**
```
URL-based routing:
  /api/*     → API servers
  /static/*  → CDN origin
  /admin/*   → Admin servers

Header-based routing:
  Accept-Language: fr → French servers
  User-Agent: Mobile* → Mobile-optimized servers

Cookie-based affinity:
  session_id=abc123 → Always same server

Request modification:
  Add X-Request-ID header
  Rewrite URLs
  Inject security headers

Caching:
  Cache GET responses at LB
  
Compression:
  Compress responses before sending to client
```

**Use cases:**
- HTTP/HTTPS web services
- Microservices routing
- A/B testing, canary deployments
- API gateway functionality

### Comparison

| Feature | Layer 4 | Layer 7 |
|---------|---------|---------|
| Speed | Faster | Slower (more processing) |
| SSL termination | No | Yes |
| Content routing | No | Yes |
| WebSocket support | Pass-through | Full support |
| Connection overhead | Lower | Higher (two connections) |
| Cost | Cheaper | More expensive |

---

## Load Balancing Algorithms

### 1. Round Robin

Simplest algorithm. Rotate through servers in order.

```
Request 1 → Server A
Request 2 → Server B
Request 3 → Server C
Request 4 → Server A
Request 5 → Server B
...
```

**Pros:** Simple, even distribution of request COUNT
**Cons:** Assumes all servers equal, all requests equal cost

**When to use:** Stateless services with homogeneous servers

### 2. Weighted Round Robin

Servers have different capacities? Give them weights.

```
Weights: Server A=5, Server B=3, Server C=2
Total weight: 10

Distribution of 10 requests:
A, A, A, A, A, B, B, B, C, C

Server A handles 50% (5/10)
Server B handles 30% (3/10)
Server C handles 20% (2/10)
```

**Use case:** Mixed hardware (new server 2x capacity of old)

### 3. Least Connections

Route to server with fewest active connections.

```
Current active connections:
Server A: 50 connections
Server B: 30 connections  ← Fewest
Server C: 45 connections

New request → Server B
```

**Why it's better than Round Robin:**
```
Round Robin problem:
- Request 1 (10ms) → A
- Request 2 (10000ms slow query) → B
- Request 3 (10ms) → C
- Request 4 (10ms) → A
- Request 5 (10ms) → B  ← B still processing slow query!

Server B gets overloaded despite "fair" distribution.

Least Connections:
After Request 2, B has 1 connection, others have 0
Request 3, 4, 5, 6... all go to A and C
Server B handles slow query without piling up
```

**Use case:** Requests with varying processing times, long-lived connections

### 4. Weighted Least Connections

Combines weights with connection count.

```
Score = active_connections / weight

Server A: 50 connections, weight 5 → score = 10
Server B: 30 connections, weight 3 → score = 10  
Server C: 18 connections, weight 2 → score = 9  ← Lowest

New request → Server C
```

### 5. Least Response Time

Route to server responding fastest.

```
Recent average response times:
Server A: 50ms
Server B: 30ms  ← Fastest
Server C: 100ms

New request → Server B
```

Often combined with connection count:
`score = active_connections * avg_response_time`

**Use case:** Heterogeneous backends, detecting degraded servers

### 6. IP Hash

Hash client IP to determine server. Same client always hits same server.

```python
def get_server(client_ip, servers):
    return servers[hash(client_ip) % len(servers)]

# Client 1.2.3.4 → always Server B
# Client 5.6.7.8 → always Server A
```

**Pros:** Session stickiness without cookies
**Cons:** Uneven distribution if clients have different request rates

**Use case:** Simple session affinity for legacy apps

### 7. Consistent Hashing

Like IP hash, but adding/removing servers doesn't reshuffle everything.

```
Hash ring with servers and client IPs:

        0°
        │
     A ─●─ B
       /│\
      / │ \
     /  │  \
Client1 │ Client2
        │
     C ─●─ D
        │
       180°

Client1 → walks clockwise → hits Server A
Client2 → walks clockwise → hits Server D

Add Server E between C and D:
- Only clients between C and E move to E
- Everyone else unchanged
```

**Use case:** Caching layers, stateful services where reshuffling is expensive

---

## Session Affinity (Sticky Sessions)

### The Problem

Stateful applications store session in server memory:

```
Request 1 → Server A
  Server A: Creates session, stores user_id=123

Request 2 → Server B (round robin)
  Server B: "Who are you? No session!" 
  User: Gets logged out 😤
```

### Solution 1: Load Balancer Sticky Sessions

LB tracks which server each client should use:

```
First request from Client X:
1. LB routes to Server A (normal algorithm)
2. LB remembers: Client X → Server A
3. LB inserts cookie: SERVERID=server-a

Subsequent requests from Client X:
1. LB sees cookie: SERVERID=server-a
2. LB routes to Server A
```

**Problem:** Server A dies = all sessions lost

### Solution 2: Shared Session Store

All servers use external session storage:

```
┌─────────────────────────────────────────────────┐
│                    Redis                         │
│  session:abc123 = {user_id: 123, cart: [...]}   │
└─────────────────────────────────────────────────┘
       ▲              ▲              ▲
       │              │              │
   Server A       Server B       Server C
```

Any server can handle any request:
```python
def handle_request(request):
    session_id = request.cookies.get('session_id')
    session = redis.get(f"session:{session_id}")
    # Same session data regardless of which server
```

**Problem:** Redis becomes critical dependency

### Solution 3: Stateless Design (Best)

Store session in signed token (JWT) on client:

```
JWT Token contains:
{
  "user_id": 123,
  "email": "user@example.com",
  "roles": ["admin"],
  "exp": 1704067200
}
+ cryptographic signature

Each request includes token in header
Any server can validate signature and extract data
No session storage needed!
```

**Problem:** Can't invalidate tokens easily (logout)
**Solution:** Short expiry + refresh tokens, or token blacklist

---

## Health Checks

How does the LB know if a server is healthy?

### Active Health Checks

LB periodically pings backends:

```
Every 10 seconds:
LB → Server A: GET /health
Server A → LB: 200 OK {"status": "healthy"}

LB → Server B: GET /health
Server B → LB: 503 Service Unavailable

LB → Server C: GET /health
[timeout - no response]

Result:
Server A: healthy ✓
Server B: unhealthy (bad response) ✗
Server C: unhealthy (timeout) ✗
```

**Configuration example:**
```yaml
health_check:
  path: /health
  interval: 10s           # Check every 10 seconds
  timeout: 5s             # Fail if no response in 5s
  healthy_threshold: 2    # 2 consecutive successes = healthy
  unhealthy_threshold: 3  # 3 consecutive failures = unhealthy
```

**What should /health check?**
```python
@app.get("/health")
def health_check():
    # Check critical dependencies
    try:
        db.execute("SELECT 1")  # Database alive?
        redis.ping()            # Cache alive?
        return {"status": "healthy"}, 200
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 503
```

### Passive Health Checks

LB monitors actual request traffic:

```
LB observes:
- Server A: 0.1% error rate → healthy
- Server B: 15% error rate → degraded, reduce traffic
- Server C: 100% connection refused → unhealthy, remove

No separate health endpoint needed
Real-world traffic = real health signal
```

**Best practice:** Use BOTH active and passive health checks

---

## High Availability for Load Balancers

The Load Balancer is itself a Single Point of Failure!

### Solution 1: DNS Round Robin

```
api.example.com → [10.0.0.1, 10.0.0.2, 10.0.0.3]
                   (LB1)     (LB2)     (LB3)

Client DNS resolution returns random LB
```

**Problem:** DNS caching causes uneven distribution, slow failover

### Solution 2: Active-Passive with VRRP

```
┌─────────────────────────┐
│    Virtual IP (VIP)     │
│      192.168.1.100      │
└─────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌──────┐  ┌──────┐
│ LB1  │  │ LB2  │
│ACTIVE│  │STANDBY│
└──────┘  └──────┘

Normal operation:
- LB1 owns VIP, serves all traffic
- LB2 monitors LB1 with heartbeat

Failover:
- LB1 fails, LB2 detects missing heartbeat
- LB2 claims VIP (gratuitous ARP)
- Failover in seconds
```

**VRRP (Virtual Router Redundancy Protocol):** Standard protocol for this

### Solution 3: Active-Active with Anycast

```
Multiple LBs advertise SAME IP via BGP:

┌─────────────────────────────────────┐
│         Same IP: 1.2.3.4            │
│   Advertised by LBs worldwide       │
└─────────────────────────────────────┘
     │              │              │
     ▼              ▼              ▼
┌────────┐    ┌────────┐    ┌────────┐
│ LB-US  │    │ LB-EU  │    │ LB-AP  │
└────────┘    └────────┘    └────────┘

Routing:
- US users → LB-US (shortest BGP path)
- EU users → LB-EU
- AP users → LB-AP

One LB fails:
- BGP withdraws route
- Traffic automatically reroutes to others
```

**Used by:** Cloudflare, AWS Global Accelerator, all major CDNs

---

## Cloud Load Balancers

### AWS

| Service | Layer | Features |
|---------|-------|----------|
| ALB (Application) | 7 | HTTP/HTTPS, WebSocket, path routing |
| NLB (Network) | 4 | TCP/UDP, static IP, extreme performance |
| CLB (Classic) | 4/7 | Legacy, both layers |
| GWLB (Gateway) | 3 | For security appliances |

### GCP

| Service | Layer | Features |
|---------|-------|----------|
| HTTP(S) LB | 7 | Global anycast, managed SSL |
| TCP Proxy LB | 4 | TLS termination |
| Network LB | 4 | Regional, pass-through |

### Azure

| Service | Layer | Features |
|---------|-------|----------|
| Application Gateway | 7 | WAF included |
| Azure Load Balancer | 4 | Regional |
| Traffic Manager | DNS | Global DNS-based |
| Front Door | 7 | Global, CDN + LB |

---

## Interview Checklist

- [ ] Choose L4 vs L7 based on requirements
- [ ] Select appropriate algorithm (least connections for variable-duration requests)
- [ ] Design health checks (active + passive)
- [ ] Address load balancer HA (active-passive or anycast)
- [ ] Consider cloud provider options
- [ ] Handle session affinity if needed (prefer stateless design)

---

## Staff+ Insights

**L4 vs L7 decision:**
- Default to L7 for HTTP services (more visibility, features)
- Use L4 when L7 overhead is unacceptable (gaming, high-frequency trading)
- L4 for TCP services (databases, message queues)

**Health check tuning:**
- Too aggressive: Flapping (healthy → unhealthy → healthy)
- Too relaxed: Slow failover, requests hit dead servers
- Start with: 10s interval, 5s timeout, 3 failures to unhealthy

**Connection draining:**
- When removing server, let existing connections finish
- AWS calls this "deregistration delay"
- Typical: 30-300 seconds depending on request duration

**Metrics to watch:**
- Request rate per backend
- Error rate per backend
- Active connections per backend
- Health check status
- LB CPU/memory (can become bottleneck)
