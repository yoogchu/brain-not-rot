# Chapter 8: API Gateway & Service Mesh

## The Microservices Problem

Without a gateway, clients face complexity:

```
Mobile App needs to call:
├── User Service (port 8001) → authentication
├── Product Service (port 8002) → authentication
├── Order Service (port 8003) → authentication
├── Payment Service (port 8004) → authentication
└── Notification Service (port 8005) → authentication

Problems:
1. Client knows all service addresses
2. Each service implements authentication
3. Each service implements rate limiting
4. Cross-cutting concerns duplicated everywhere
5. Protocol translation (mobile wants REST, services use gRPC)
```

---

## API Gateway: Single Entry Point

```
                    ┌─────────────────────────────────┐
                    │          API Gateway            │
Mobile App ────────→│  - Authentication               │
Web App ───────────→│  - Rate Limiting                │
Partner API ───────→│  - Request Routing              │
                    │  - Protocol Translation         │
                    │  - Response Aggregation         │
                    │  - Logging & Monitoring         │
                    └─────────────────────────────────┘
                        │         │           │
                        ▼         ▼           ▼
                   User Svc  Product Svc  Order Svc
                   (internal) (internal)  (internal)
```

**Key insight:** Gateway handles all cross-cutting concerns. Services focus on business logic only.

---

## API Gateway Functions

### 1. Request Routing

```yaml
# Route configuration
routes:
  - path: /api/v1/users/*
    service: user-service
    strip_prefix: /api/v1
    
  - path: /api/v1/products/*
    service: product-service
    methods: [GET]
    
  - path: /api/v1/orders/*
    service: order-service
    timeout: 30s
    
  - path: /api/v2/*
    service: api-v2-service
    # Version routing
```

**Path-based routing:**
```
/users/123 → user-service
/products/456 → product-service
/orders/789 → order-service
```

**Header-based routing:**
```
Accept-Language: fr → french-content-service
X-API-Version: 2 → api-v2-service
```

### 2. Authentication & Authorization

```
┌─────────┐      ┌─────────────┐      ┌─────────────┐
│ Client  │──────│ API Gateway │──────│   Backend   │
└─────────┘      └─────────────┘      └─────────────┘
     │                  │                    │
     │  1. Request +    │                    │
     │     JWT Token    │                    │
     │─────────────────→│                    │
     │                  │                    │
     │          2. Validate JWT              │
     │             signature                 │
     │          3. Check expiry              │
     │          4. Extract claims            │
     │          5. Check permissions         │
     │                  │                    │
     │                  │  6. Forward with   │
     │                  │     user context   │
     │                  │───────────────────→│
     │                  │                    │
     │                  │  7. Response       │
     │                  │←───────────────────│
     │                  │                    │
     │  8. Response     │                    │
     │←─────────────────│                    │
```

**Implementation:**
```python
# Gateway authentication middleware
def authenticate(request):
    token = request.headers.get("Authorization")
    if not token:
        return Response(status=401, body="Missing token")
    
    try:
        # Validate JWT
        claims = jwt.decode(token, PUBLIC_KEY, algorithms=["RS256"])
        
        # Check expiry
        if claims["exp"] < time.time():
            return Response(status=401, body="Token expired")
        
        # Add user context for backends
        request.headers["X-User-ID"] = claims["sub"]
        request.headers["X-User-Roles"] = ",".join(claims["roles"])
        
        return forward_to_backend(request)
        
    except jwt.InvalidTokenError:
        return Response(status=401, body="Invalid token")
```

### 3. Rate Limiting

```yaml
rate_limits:
  # Global rate limit
  - scope: global
    rate: 10000/second
    
  # Per-client rate limit
  - scope: client_ip
    rate: 100/minute
    
  # Per-API-key rate limit
  - scope: api_key
    rate: 1000/minute
    
  # Per-endpoint rate limit
  - path: /api/search
    scope: api_key
    rate: 10/minute  # Expensive operation
    
  # Tiered limits
  - path: /api/*
    scope: api_key
    tiers:
      free: 100/hour
      pro: 1000/hour
      enterprise: 10000/hour
```

### 4. Request/Response Transformation

**Protocol Translation:**
```
Client: REST/JSON
     │
     ▼
┌─────────────┐
│   Gateway   │  Translates JSON ↔ Protobuf
└─────────────┘
     │
     ▼
Backend: gRPC/Protobuf
```

**Response Aggregation:**
```python
# Client wants user dashboard in one call
# Gateway aggregates from multiple services

async def get_user_dashboard(user_id):
    # Parallel calls to services
    user, orders, notifications = await asyncio.gather(
        user_service.get_user(user_id),
        order_service.get_recent_orders(user_id),
        notification_service.get_unread(user_id)
    )
    
    # Aggregate response
    return {
        "user": user,
        "recent_orders": orders,
        "notifications": notifications
    }
```

**Request Enrichment:**
```python
# Add headers before forwarding
def enrich_request(request):
    request.headers["X-Request-ID"] = generate_uuid()
    request.headers["X-Forwarded-For"] = request.client_ip
    request.headers["X-Request-Start"] = time.time()
    return request
```

### 5. Caching

```yaml
cache:
  # Cache GET requests for products
  - path: /api/products/*
    methods: [GET]
    ttl: 300s  # 5 minutes
    vary_by: [Accept-Language, Accept-Encoding]
    
  # Cache search results
  - path: /api/search
    methods: [GET]
    ttl: 60s
    vary_by: [query_params]
    
  # Never cache user-specific data
  - path: /api/users/*
    cache: false
```

---

## Popular API Gateways

| Gateway | Type | Strengths | Best For |
|---------|------|-----------|----------|
| **Kong** | Open source | Plugin ecosystem, Lua extensibility | General purpose |
| **AWS API Gateway** | Managed | AWS integration, Lambda triggers | AWS-native apps |
| **Apigee** | Enterprise | Analytics, developer portal | API monetization |
| **NGINX** | Open source | High performance, battle-tested | High throughput |
| **Traefik** | Cloud native | Auto-discovery, Let's Encrypt | Kubernetes |
| **Ambassador** | Kubernetes | Envoy-based, GitOps friendly | Kubernetes |

---

## Service Mesh: Service-to-Service Communication

### The Internal Communication Problem

```
Services need to communicate with each other:

┌─────────┐     ┌─────────┐     ┌─────────┐
│ Order   │────→│ Payment │────→│ Notify  │
│ Service │     │ Service │     │ Service │
└─────────┘     └─────────┘     └─────────┘

Each service must implement:
- Service discovery (where is Payment Service?)
- Load balancing (which instance?)
- Retries and timeouts
- Circuit breaking
- Encryption (mTLS)
- Observability (traces, metrics)

Duplicated in every service = inconsistent, error-prone
```

### Service Mesh Solution

**Sidecar proxy pattern:**

```
┌────────────────────────────────────────────────────────┐
│                     Service Mesh                        │
│                                                         │
│  ┌────────────────────┐      ┌────────────────────┐    │
│  │   Order Service    │      │  Payment Service   │    │
│  │ ┌──────┬────────┐  │      │ ┌────────┬──────┐  │    │
│  │ │ App  │ Envoy  │◄─┼──────┼─►│ Envoy  │ App  │  │    │
│  │ │      │ Proxy  │  │ mTLS │ │ Proxy  │      │  │    │
│  │ └──────┴────────┘  │      │ └────────┴──────┘  │    │
│  └────────────────────┘      └────────────────────┘    │
│                                                         │
│  App talks to localhost:port                           │
│  Proxy handles ALL networking                          │
│                                                         │
│  ┌──────────────────────────────────────────────┐      │
│  │              Control Plane                    │      │
│  │  - Push configuration to all proxies         │      │
│  │  - Certificate management                    │      │
│  │  - Policy enforcement                        │      │
│  │  - Service discovery                         │      │
│  └──────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────┘
```

**Key insight:** Application code is unaware of mesh. It just calls `localhost:port`. Sidecar handles everything.

---

## Service Mesh Features

### 1. Mutual TLS (mTLS)

```
Without mesh:
Service A ──── plain HTTP ────► Service B
             (anyone can intercept)

With mesh:
Service A ─► Envoy ═══ mTLS encrypted ═══► Envoy ─► Service B
              │                              │
              └── Both authenticate each other ──┘
```

**How mTLS works:**
```
1. Control plane issues certificates to each service
2. Envoy A presents cert to Envoy B
3. Envoy B verifies A's cert against CA
4. Envoy B presents cert to Envoy A
5. Envoy A verifies B's cert
6. Encrypted channel established
7. Automatic rotation (no manual cert management)
```

**Zero-trust networking:** Every call is authenticated and encrypted, even inside the datacenter.

### 2. Traffic Management

**Canary Deployments:**
```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: payment-service
spec:
  http:
  - route:
    - destination:
        host: payment-service
        subset: v1
      weight: 95
    - destination:
        host: payment-service
        subset: v2
      weight: 5  # 5% to new version
```

**Header-based routing (testing in production):**
```yaml
http:
- match:
  - headers:
      x-test-user:
        exact: "true"
  route:
  - destination:
      host: payment-service
      subset: v2  # Test users get v2
- route:
  - destination:
      host: payment-service
      subset: v1  # Everyone else gets v1
```

**Fault injection (chaos engineering):**
```yaml
http:
- fault:
    delay:
      percentage:
        value: 10
      fixedDelay: 5s  # 10% of requests delayed 5s
    abort:
      percentage:
        value: 5
      httpStatus: 500  # 5% of requests fail
  route:
  - destination:
      host: payment-service
```

### 3. Resilience

**Retries:**
```yaml
http:
- retries:
    attempts: 3
    perTryTimeout: 2s
    retryOn: "5xx,reset,connect-failure"
  route:
  - destination:
      host: payment-service
```

**Circuit Breaking:**
```yaml
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
spec:
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http2MaxRequests: 1000
        maxRequestsPerConnection: 10
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 60s
      maxEjectionPercent: 50
```

### 4. Observability

**Automatic metrics:**
```
- Request rate (requests/second)
- Error rate (4xx, 5xx)
- Latency (p50, p95, p99)
- Saturation (connections, memory)
```

**Distributed tracing:**
```
Order Service → Payment Service → Fraud Service → Bank API

Trace ID: abc123
├── Span 1: Order Service (50ms)
│   └── Span 2: Payment Service (30ms)
│       └── Span 3: Fraud Service (15ms)
│           └── Span 4: Bank API (100ms)
Total: 195ms (bottleneck: Bank API)
```

---

## Service Mesh Options

| Mesh | Data Plane | Control Plane | Complexity | Best For |
|------|------------|---------------|------------|----------|
| **Istio** | Envoy | Istiod | High | Full features |
| **Linkerd** | linkerd2-proxy (Rust) | Linkerd | Medium | Simplicity |
| **Consul Connect** | Envoy or built-in | Consul | Medium | HashiCorp stack |
| **AWS App Mesh** | Envoy | AWS managed | Low | AWS native |
| **Cilium** | eBPF (no sidecar) | Cilium | Medium | Performance |

---

## API Gateway vs Service Mesh

```
                 External Traffic (North-South)
                         │
                         ▼
                ┌─────────────────┐
                │   API Gateway   │
                │ • Auth          │
                │ • Rate limiting │
                │ • Public APIs   │
                └─────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   ┌─────────┐      ┌─────────┐      ┌─────────┐
   │  Svc A  │◄────►│  Svc B  │◄────►│  Svc C  │
   └─────────┘      └─────────┘      └─────────┘
        │                │                │
        └────────────────┴────────────────┘
              Internal Traffic (East-West)
                   Service Mesh
                 • mTLS
                 • Traffic management
                 • Internal observability
```

**Use API Gateway for:**
- External clients → your services
- Public API management
- Authentication of external users
- External rate limiting
- API versioning for clients

**Use Service Mesh for:**
- Service → service communication
- mTLS between internal services
- Canary/blue-green between service versions
- Internal resilience (retries, circuit breaking)
- Distributed tracing

**They complement each other, not replace:**
```
Client → API Gateway → Service A → (via mesh) → Service B
```

---

## Key Concepts Checklist

- [ ] Explain API Gateway's role (single entry point, cross-cutting concerns)
- [ ] Describe key gateway functions (routing, auth, rate limiting)
- [ ] Explain service mesh sidecar pattern
- [ ] Describe mTLS and zero-trust networking
- [ ] Know traffic management features (canary, circuit breaking)
- [ ] Distinguish north-south vs east-west traffic
- [ ] Choose appropriate gateway vs mesh based on requirements

---

## Practical Insights

**Gateway anti-patterns:**
- Gateway becomes a monolith (too much logic)
- Tight coupling (gateway knows too much about services)
- Single point of failure (no gateway HA)

**Mesh considerations:**
- Sidecar overhead (CPU, memory, latency)
- Complexity of debugging through proxies
- Operational burden of control plane
- Consider if you actually need it (10 services? Maybe not)

**When to adopt service mesh:**
- More than ~20 microservices
- Strong security requirements (mTLS everywhere)
- Complex traffic management needs
- Need unified observability

**Start simple:**
- API Gateway first
- Add service mesh when pain is clear
- Don't add mesh "just in case"
