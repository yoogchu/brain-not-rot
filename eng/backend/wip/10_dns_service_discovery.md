# Chapter 10: DNS & Service Discovery

## DNS: The Internet's Phone Book

### The Problem DNS Solves

```
Without DNS:
"Connect to 142.250.185.206" (What's that?)

With DNS:
"Connect to google.com" (Oh, Google!)
```

DNS translates human-readable domain names to IP addresses.

---

## DNS Resolution Process

```
User types: api.example.com

Step 1: Browser cache
        → Found? Use it. Done.
        → Not found? Continue.

Step 2: Operating System cache
        → Found? Use it. Done.
        → Not found? Continue.

Step 3: Local DNS Resolver (ISP or corporate)
        → Found in cache? Return it.
        → Not found? Query hierarchy.

Step 4: Root DNS Servers (13 root server clusters)
        "I don't know api.example.com,
         but .com TLD servers are at 192.5.6.30"

Step 5: TLD Servers (.com)
        "I don't know api.example.com,
         but example.com nameservers are at ns1.example.com"

Step 6: Authoritative Nameserver (example.com)
        "api.example.com is 93.184.216.34"

Step 7: Cache at each level based on TTL
        Return to user.
```

**Visual representation:**
```
                    Browser Cache
                         │
                         ▼
                    OS DNS Cache
                         │
                         ▼
              ┌──────────────────────┐
              │   Local Resolver     │
              │   (ISP/Corporate)    │
              └──────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ Root    │    │  .com   │    │Authorit-│
    │ Servers │───►│   TLD   │───►│  ative  │
    └─────────┘    └─────────┘    └─────────┘
                                       │
                                       ▼
                              IP: 93.184.216.34
```

---

## DNS Record Types

| Type | Purpose | Example |
|------|---------|---------|
| **A** | IPv4 address | `api.example.com → 93.184.216.34` |
| **AAAA** | IPv6 address | `api.example.com → 2606:2800:220:1:...` |
| **CNAME** | Alias to another name | `www.example.com → example.com` |
| **MX** | Mail server | `example.com → mail.example.com (priority 10)` |
| **TXT** | Text records | SPF, DKIM, domain verification |
| **NS** | Nameserver | `example.com NS ns1.example.com` |
| **SRV** | Service location | `_http._tcp.example.com → server:port:priority:weight` |
| **PTR** | Reverse DNS | `34.216.184.93 → api.example.com` |

### A Record Example

```
$ dig api.example.com A

;; ANSWER SECTION:
api.example.com.    300    IN    A    93.184.216.34
                    └─TTL (seconds)
```

### CNAME Example

```
$ dig www.example.com

;; ANSWER SECTION:
www.example.com.    3600   IN    CNAME    example.com.
example.com.        300    IN    A        93.184.216.34
```

**CNAME chain:** `www.example.com → example.com → 93.184.216.34`

### SRV Record (Service Discovery)

```
_http._tcp.api.example.com. 300 IN SRV 10 5 8080 server1.example.com.
                                     │  │  │    │
                                     │  │  │    └─ Target host
                                     │  │  └─ Port
                                     │  └─ Weight (load distribution)
                                     └─ Priority (lower = preferred)
```

---

## TTL and Caching

**TTL (Time To Live):** How long resolvers cache the record.

```
Low TTL (60s):
├── Faster DNS changes
├── More DNS queries
└── Good for: Failover, load balancing

High TTL (86400s = 1 day):
├── Fewer DNS queries
├── Slower DNS changes
└── Good for: Stable records, lower load
```

**Propagation delay:**
```
You change: api.example.com → new IP

Timeline:
├── T+0: Authoritative updated
├── T+0 to T+TTL: Some resolvers have old IP
├── T+TTL: All caches expired, new IP everywhere

With TTL=3600, full propagation takes up to 1 hour
```

**Strategy for zero-downtime migration:**
```
1. Lower TTL to 60s (wait for old TTL to expire)
2. Make DNS change
3. Wait for propagation (~60s)
4. Verify new IP is serving
5. Raise TTL back to 3600s
```

---

## DNS for Load Balancing

### Round-Robin DNS

```
$ dig api.example.com

;; ANSWER SECTION:
api.example.com.    60    IN    A    192.168.1.1
api.example.com.    60    IN    A    192.168.1.2
api.example.com.    60    IN    A    192.168.1.3
```

Clients rotate through IPs. Simple but limited:
- No health checks
- No weighted distribution
- Client may cache one IP

### Geographic DNS (GeoDNS)

```
User in NYC:
$ dig api.example.com
→ 10.0.1.1 (US-East datacenter)

User in Tokyo:
$ dig api.example.com
→ 10.0.2.1 (Asia datacenter)
```

**Providers:** Route53, Cloudflare, NS1

### Weighted DNS

```
api.example.com:
├── 70% → 10.0.1.1 (primary)
└── 30% → 10.0.2.1 (secondary/canary)
```

Useful for gradual traffic shifting, canary deployments.

---

## Service Discovery Patterns

### The Microservices Problem

```
Order Service needs to call Payment Service.

Static configuration:
PAYMENT_SERVICE_URL = "http://10.0.1.50:8080"

Problems:
1. Payment Service IP changes → update config, redeploy
2. Payment Service scales → which instance?
3. Payment Service fails → still calling dead instance
```

### Pattern 1: Client-Side Discovery

```
┌─────────────┐
│   Client    │
│  (Order Svc)│
└─────────────┘
      │
      │ 1. Query "payment-service"
      ▼
┌─────────────────────┐
│  Service Registry   │
│                     │
│ payment-service:    │
│   - 10.0.1.50:8080  │
│   - 10.0.1.51:8080  │
│   - 10.0.1.52:8080  │
└─────────────────────┘
      │
      │ 2. Returns list of instances
      ▼
┌─────────────┐
│   Client    │
│  (Order Svc)│ 3. Client picks one (load balancing logic)
└─────────────┘
      │
      │ 4. Direct call
      ▼
┌─────────────────┐
│ Payment Service │
│  (10.0.1.51)    │
└─────────────────┘
```

**Pros:**
- No proxy in path (lower latency)
- Client controls load balancing

**Cons:**
- Every client needs discovery logic
- Every client needs LB logic
- Tightly coupled to registry

**Tools:** Netflix Eureka, Consul (client mode), etcd

### Pattern 2: Server-Side Discovery

```
┌─────────────┐
│   Client    │
│  (Order Svc)│
└─────────────┘
      │
      │ 1. Call payment-service
      ▼
┌─────────────────────┐     ┌─────────────────────┐
│   Load Balancer     │◄────│  Service Registry   │
│                     │     │                     │
│ Routes to healthy   │     │ payment-service:    │
│ instance            │     │   - 10.0.1.50:8080  │
└─────────────────────┘     └─────────────────────┘
      │
      │ 2. Forward to instance
      ▼
┌─────────────────┐
│ Payment Service │
│  (10.0.1.51)    │
└─────────────────┘
```

**Pros:**
- Client is simple (just call LB)
- Centralized LB logic
- Easy to add health checks

**Cons:**
- Extra network hop
- LB can be bottleneck
- LB needs HA

**Tools:** AWS ALB + ECS, Kubernetes Services, Consul + NGINX

### Pattern 3: DNS-Based Discovery

```
┌─────────────┐
│   Client    │
│  (Order Svc)│
└─────────────┘
      │
      │ 1. DNS query: payment-service.internal
      ▼
┌─────────────────────┐
│    Internal DNS     │
│                     │
│ payment-service:    │
│   A 10.0.1.50       │
│   A 10.0.1.51       │
│   A 10.0.1.52       │
└─────────────────────┘
      │
      │ 2. Returns IP(s)
      ▼
┌─────────────┐
│   Client    │ 3. Direct call
│  (Order Svc)│
└─────────────┘
      │
      │
      ▼
┌─────────────────┐
│ Payment Service │
└─────────────────┘
```

**Kubernetes DNS example:**
```
Service name: payment-service
Namespace: production

DNS name: payment-service.production.svc.cluster.local

Pod calls: http://payment-service:8080
Kubernetes DNS resolves to ClusterIP
kube-proxy routes to healthy pod
```

---

## Service Registration

### Self-Registration

```
┌─────────────────┐
│ Payment Service │
│   (instance)    │
└─────────────────┘
        │
        │ On startup: Register myself
        │ Periodically: Send heartbeat
        │ On shutdown: Deregister
        ▼
┌─────────────────────┐
│  Service Registry   │
│                     │
│ payment-service:    │
│   - 10.0.1.50 (me!) │
└─────────────────────┘
```

**Implementation:**
```python
class ServiceInstance:
    def __init__(self, registry, service_name, host, port):
        self.registry = registry
        self.service_name = service_name
        self.host = host
        self.port = port
        self.instance_id = f"{host}:{port}"
    
    def start(self):
        # Register on startup
        self.registry.register(
            self.service_name,
            self.instance_id,
            self.host,
            self.port
        )
        
        # Start heartbeat
        self.heartbeat_thread = threading.Thread(target=self._heartbeat)
        self.heartbeat_thread.start()
    
    def _heartbeat(self):
        while self.running:
            self.registry.heartbeat(self.service_name, self.instance_id)
            time.sleep(10)
    
    def stop(self):
        self.running = False
        self.registry.deregister(self.service_name, self.instance_id)
```

### Third-Party Registration

```
┌─────────────────┐    ┌─────────────────┐
│ Payment Service │    │   Registrar     │
│   (instance)    │    │   (sidecar/     │
│                 │◄───│    platform)    │
│  Just runs,     │    │                 │
│  no registry    │    │ Watches health, │
│  awareness      │    │ manages registry│
└─────────────────┘    └─────────────────┘
                              │
                              │ Register/deregister
                              ▼
                    ┌─────────────────────┐
                    │  Service Registry   │
                    └─────────────────────┘
```

**Kubernetes model:**
```yaml
# Deployment creates pods
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: payment
        image: payment:v1
        ports:
        - containerPort: 8080

# Service auto-discovers pods via labels
apiVersion: v1
kind: Service
spec:
  selector:
    app: payment  # Matches pods with this label
  ports:
  - port: 8080
```

Kubernetes automatically:
- Watches pod lifecycle
- Updates endpoints
- Updates DNS records

---

## Health Checks

### Active Health Checks

```
Registry/LB periodically probes each instance:

┌─────────────┐     GET /health      ┌─────────────┐
│   Registry  │────────────────────► │   Service   │
│             │◄──────────────────── │  Instance   │
│             │     200 OK           │             │
└─────────────┘                      └─────────────┘
        │
        │ If 3 consecutive failures:
        │ Mark instance unhealthy
        ▼
┌─────────────────────┐
│ Healthy instances:  │
│   - 10.0.1.50 ✓     │
│   - 10.0.1.51 ✓     │
│   - 10.0.1.52 ✗     │ ← Removed from rotation
└─────────────────────┘
```

### Passive Health Checks

```
Monitor actual traffic for failures:

Request → Instance → 500 Error
Request → Instance → 500 Error
Request → Instance → 500 Error

3 errors in 30s? Mark unhealthy.
```

### Health Check Endpoint

```python
@app.route('/health')
def health():
    checks = {
        'database': check_db_connection(),
        'redis': check_redis_connection(),
        'disk_space': check_disk_space(),
    }
    
    all_healthy = all(checks.values())
    
    return jsonify({
        'status': 'healthy' if all_healthy else 'unhealthy',
        'checks': checks
    }), 200 if all_healthy else 503
```

### Liveness vs Readiness

**Liveness:** Is the process alive?
- Fail → restart container
- Check: Is the app responding at all?

**Readiness:** Can it serve traffic?
- Fail → remove from load balancer
- Check: Are dependencies connected?

```yaml
# Kubernetes example
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 3
```

---

## Service Discovery Tools

| Tool | Type | Consistency | Best For |
|------|------|-------------|----------|
| **Consul** | CP (Raft) | Strong | Multi-DC, health checks |
| **etcd** | CP (Raft) | Strong | Kubernetes, config |
| **ZooKeeper** | CP (ZAB) | Strong | Hadoop ecosystem |
| **Eureka** | AP | Eventual | Netflix OSS stack |
| **Kubernetes** | Built-in | Strong | K8s native apps |

---

## Interview Checklist

- [ ] Explain DNS resolution hierarchy
- [ ] Describe common record types (A, CNAME, SRV)
- [ ] Explain TTL trade-offs
- [ ] Compare client-side vs server-side discovery
- [ ] Describe service registration patterns
- [ ] Design health check strategy
- [ ] Know when to use which discovery tool

---

## Staff+ Insights

**DNS gotchas:**
- JVM caches DNS forever by default (set `networkaddress.cache.ttl`)
- Some clients don't respect TTL
- DNS failover is slow (TTL + client cache)

**Service discovery selection:**
- Kubernetes? Use built-in DNS + Services
- Multi-cloud/hybrid? Consider Consul
- Simple setup? DNS-based may be enough

**Health check anti-patterns:**
- Health check checks nothing meaningful
- Health check does expensive work
- Health check doesn't check dependencies
- Single failure = immediate removal (need threshold)

**Registration patterns:**
- Self-registration: More control, more code
- Platform registration: Less code, platform lock-in
- Most modern: Let Kubernetes/ECS handle it

**DNS-based discovery limits:**
- No real-time updates (TTL delay)
- No health-based routing
- No traffic splitting
- Good for: Initial discovery, simple cases
- Use service mesh for: Advanced routing, observability
