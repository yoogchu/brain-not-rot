# Chapter 20: Microservices Patterns

## The Microservices Migration That Wasn't

Your monolithic e-commerce platform serves 10,000 orders/day with a 6-person engineering team.

```
Before: Monolith
Deployment time: 15 minutes
Dependencies: All code in one repo
Bugs: Easy to debug with stack traces
Team velocity: 5 features/week

After 6 months of "microservices transformation":
- 47 services (nobody knows what 12 of them do)
- 3-hop distributed transactions to create an order
- Deployment time: 2 hours (coordinating service versions)
- Debugging: "Which service logged this error?"
- Team velocity: 1 feature/week
- New hires: 3 weeks to understand system architecture

CEO asks: "Why did velocity drop 80%?"
```

Microservices solve real problems at scale. They also create new problems. Understanding WHEN and HOW to use them is critical.

---

## Conway's Law and Service Boundaries

**Conway's Law:** Your system architecture mirrors your organization's communication structure.

```
Monolith Team:            Microservices Team:
┌───────────────┐        ┌─────┐  ┌─────┐  ┌─────┐
│  6 engineers  │        │Order│  │User │  │Pay- │
│  One codebase │        │Team │  │Team │  │ment │
│  One database │        │  2  │  │  2  │  │Team │
└───────────────┘        │devs │  │devs │  │  2  │
                         └─────┘  └─────┘  │devs │
Daily standup                               └─────┘
 ↓                       Weekly sync meeting
"Add feature X"          ↓
Changes 3 files          "Need API change"
Ships in 2 days          Cross-team tickets
                         Ships in 2 weeks
```

**The insight:** Don't design microservices until you have teams to own them.

---

## Service Decomposition Strategies

### Domain-Driven Design (DDD)

**The Problem:** How do you split a monolith without creating a distributed mess?

**Bounded Contexts:**

```
E-commerce Domain Map:

┌──────────────────────────────────────────────────┐
│ Order Management Context                         │
│ - Order (aggregate root)                         │
│ - OrderLine, ShippingAddress                     │
│ - Commands: CreateOrder, CancelOrder             │
└──────────────────────────────────────────────────┘
                  │
                  │ Order.customerId
                  ▼
┌──────────────────────────────────────────────────┐
│ Customer Context                                 │
│ - Customer (aggregate root)                      │
│ - CustomerProfile, PaymentMethod                 │
│ - Commands: CreateCustomer, UpdateProfile        │
└──────────────────────────────────────────────────┘
                  │
                  │ Order.paymentId
                  ▼
┌──────────────────────────────────────────────────┐
│ Payment Context                                  │
│ - Payment (aggregate root)                       │
│ - Transaction, Refund                            │
│ - Commands: AuthorizePayment, CapturePayment     │
└──────────────────────────────────────────────────┘
```

**Data Ownership Rules:**
- Each context owns its data completely
- No direct database access across contexts
- Only communicate via APIs or events

**Implementation:**

```python
# Order Service - owns orders
class OrderService:
    def __init__(self, db, customer_client, payment_client):
        self.db = db
        self.customer_client = customer_client  # HTTP client
        self.payment_client = payment_client

    async def create_order(self, customer_id, items):
        # Validate customer exists (cross-service call)
        customer = await self.customer_client.get_customer(customer_id)
        if not customer:
            raise CustomerNotFoundError()

        # Calculate total
        total = sum(item.price * item.quantity for item in items)

        # Create order in own database
        order = Order(
            customer_id=customer_id,
            items=items,
            total=total,
            status="PENDING"
        )
        await self.db.save(order)

        # Publish event for other services
        await self.event_bus.publish(OrderCreatedEvent(order))

        return order
```

### By Team (Conway's Law)

```
Team Structure → Service Boundaries

Frontend Team:
- BFF (Backend for Frontend) service
- Aggregates data from backend services

Checkout Team:
- Order Service
- Cart Service
- Owns checkout flow end-to-end

Payments Team:
- Payment Service
- Fraud Detection Service
- Owns money movement

Each team:
- Deploys independently
- Has on-call rotation for their services
- Decides tech stack within guidelines
```

**Trade-offs:**

| Aspect | Monolith | Microservices | Modular Monolith |
|--------|----------|---------------|------------------|
| Deployment | One unit | Independent services | One unit, modular code |
| Scaling | Scale entire app | Scale services independently | Scale entire app |
| Team autonomy | Low (code conflicts) | High (service ownership) | Medium (module ownership) |
| Complexity | Low | High (distributed) | Medium |
| Data consistency | Easy (ACID) | Hard (eventual) | Easy (ACID) |
| Debugging | Easy (one stack trace) | Hard (distributed traces) | Easy (one stack trace) |
| Initial velocity | High | Low (infrastructure) | High |
| Scale velocity | Low (merge conflicts) | High (parallel work) | Medium |

**When to use:**
- Microservices: 5+ independent teams, different scaling needs, need deployment independence
- Modular Monolith: 2-5 teams, want team boundaries without distributed complexity
- Monolith: 1 team, unclear domain boundaries, <50K lines of code

**When NOT to use microservices:**
- Fewer than 3 teams
- Domain boundaries unclear
- No DevOps maturity (CI/CD, observability)
- Startup finding product-market fit

---

## Saga Pattern: Distributed Transactions

**The Problem:** You need to update data across multiple services atomically, but distributed transactions (2PC) don't scale.

### Choreography-Based Saga

Services react to events, no central coordinator.

```
Order Creation Flow (Choreography):

┌─────────────┐
│Order Service│
└──────┬──────┘
       │ 1. Create order (status=PENDING)
       │ 2. Publish OrderCreated event
       ▼
┌──────────────────────────────────────────┐
│          Event Bus (Kafka)               │
└──────────────────────────────────────────┘
       │                      │
       ▼                      ▼
┌─────────────┐        ┌─────────────┐
│Payment Svc  │        │Inventory Svc│
└──────┬──────┘        └──────┬──────┘
       │                      │
       │ 3. Process payment   │ 5. Reserve items
       │ 4. Publish           │ 6. Publish
       │    PaymentSucceeded  │    ItemsReserved
       ▼                      ▼
┌──────────────────────────────────────────┐
│          Event Bus (Kafka)               │
└──────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│Order Service│  7. Update order (status=CONFIRMED)
└─────────────┘
```

**Failure Handling (Compensation):**

```
If payment fails:
PaymentFailed event → Order Service sets status=CANCELLED

If inventory reservation fails:
ItemsNotAvailable event → Payment Service refunds → Order Service cancels
```

**Implementation:**

```python
# Order Service
class OrderService:
    async def on_order_created(self, event):
        order = await self.db.get_order(event.order_id)
        # Saga state machine
        order.saga_status = "PAYMENT_PENDING"
        await self.db.save(order)
        await self.event_bus.publish(OrderCreatedEvent(order))

    async def on_payment_succeeded(self, event):
        order = await self.db.get_order(event.order_id)
        order.saga_status = "INVENTORY_PENDING"
        await self.db.save(order)

    async def on_items_reserved(self, event):
        order = await self.db.get_order(event.order_id)
        order.saga_status = "COMPLETED"
        order.status = "CONFIRMED"
        await self.db.save(order)

    async def on_payment_failed(self, event):
        order = await self.db.get_order(event.order_id)
        order.saga_status = "CANCELLED"
        order.status = "CANCELLED"
        await self.db.save(order)
```

### Orchestration-Based Saga

Central coordinator manages the saga flow.

```
Order Creation Flow (Orchestration):

┌──────────────────────────────────────────┐
│      Order Saga Orchestrator             │
│  (state machine tracking saga progress)  │
└──────────────────────────────────────────┘
       │              │              │
       │ 1. Reserve   │ 2. Process   │ 3. Update
       │    inventory │    payment   │    order
       ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│Inventory Svc│ │Payment Svc  │ │Order Service│
└─────────────┘ └─────────────┘ └─────────────┘
       │              │              │
       └──── Reply ───┴──── Reply ───┘
                     │
                     ▼
              Orchestrator decides:
              Success → Next step
              Failure → Compensate
```

**Implementation:**

```python
class OrderSagaOrchestrator:
    def __init__(self, inventory_client, payment_client, order_client):
        self.inventory_client = inventory_client
        self.payment_client = payment_client
        self.order_client = order_client

    async def execute_order_saga(self, order_id):
        saga_state = SagaState(order_id)

        try:
            # Step 1: Reserve inventory
            saga_state.step = "RESERVE_INVENTORY"
            reservation = await self.inventory_client.reserve_items(
                order_id, items
            )
            saga_state.reservation_id = reservation.id

            # Step 2: Process payment
            saga_state.step = "PROCESS_PAYMENT"
            payment = await self.payment_client.charge(
                order_id, amount
            )
            saga_state.payment_id = payment.id

            # Step 3: Confirm order
            saga_state.step = "CONFIRM_ORDER"
            await self.order_client.confirm_order(order_id)

            saga_state.status = "COMPLETED"
            return saga_state

        except InventoryNotAvailableError:
            saga_state.status = "CANCELLED"
            await self.order_client.cancel_order(order_id)
            raise

        except PaymentFailedError:
            saga_state.status = "COMPENSATING"
            # Compensate: Release inventory reservation
            await self.inventory_client.release_reservation(
                saga_state.reservation_id
            )
            await self.order_client.cancel_order(order_id)
            saga_state.status = "CANCELLED"
            raise

        except Exception as e:
            # Compensate all previous steps
            saga_state.status = "COMPENSATING"
            await self.compensate(saga_state)
            saga_state.status = "FAILED"
            raise

    async def compensate(self, saga_state):
        # Undo in reverse order
        if saga_state.payment_id:
            await self.payment_client.refund(saga_state.payment_id)
        if saga_state.reservation_id:
            await self.inventory_client.release_reservation(
                saga_state.reservation_id
            )
        await self.order_client.cancel_order(saga_state.order_id)
```

**Choreography vs Orchestration:**

| Aspect | Choreography | Orchestration |
|--------|--------------|---------------|
| Coordination | Decentralized (events) | Centralized (orchestrator) |
| Complexity | Spread across services | Concentrated in orchestrator |
| Coupling | Loose | Tighter (services know orchestrator) |
| Debugging | Hard (follow event chain) | Easier (one place to check) |
| Single point of failure | No | Yes (orchestrator) |

**When to use:**
- Choreography: Simple workflows (2-3 steps), loosely coupled services
- Orchestration: Complex workflows (5+ steps), need visibility/control

**When NOT to use:**
- Don't use sagas for read-heavy operations (use eventual consistency/caching)
- Don't use for real-time consistency needs (use monolith or synchronous 2PC if <10 TPS)

---

## Circuit Breaker Pattern

Expanded from Chapter 16 with microservices focus.

**The Problem:** Service A depends on Service B. Service B starts failing. Service A keeps retrying, exhausting its thread pool, and becomes unavailable too. Cascading failure.

```
Without Circuit Breaker:

Service A (Order Service)
  │
  ├─ Request 1 → Service B (Payment) → Timeout (30s)
  ├─ Request 2 → Service B (Payment) → Timeout (30s)
  ├─ Request 3 → Service B (Payment) → Timeout (30s)
  └─ ...

Service A thread pool exhausted → Service A DOWN
All users see errors, even for operations not needing payments
```

**With Circuit Breaker:**

```python
class MicroserviceCircuitBreaker:
    def __init__(
        self,
        failure_threshold=5,
        recovery_timeout=30,
        success_threshold=2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.failures = 0
        self.successes = 0
        self.state = "CLOSED"
        self.last_failure_time = None

    async def call(self, service_func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF-OPEN"
                self.successes = 0
            else:
                raise CircuitOpenError(
                    f"Circuit open, retry after {self.recovery_timeout}s"
                )

        try:
            result = await asyncio.wait_for(
                service_func(*args, **kwargs),
                timeout=5.0  # Fail fast
            )
            self.on_success()
            return result
        except (asyncio.TimeoutError, Exception) as e:
            self.on_failure()
            raise

    def on_success(self):
        if self.state == "HALF-OPEN":
            self.successes += 1
            if self.successes >= self.success_threshold:
                self.state = "CLOSED"
                self.failures = 0
        else:
            self.failures = 0

    def on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()

        if self.state == "HALF-OPEN":
            self.state = "OPEN"
        elif self.failures >= self.failure_threshold:
            self.state = "OPEN"

# Usage with fallback
payment_breaker = MicroserviceCircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30
)

async def process_order_with_fallback(order_id):
    try:
        payment = await payment_breaker.call(
            payment_service.charge,
            order_id
        )
        return {"status": "success", "payment": payment}
    except CircuitOpenError:
        # Fallback: Accept order, charge later
        await queue.enqueue_delayed_payment(order_id)
        return {"status": "pending_payment", "order_id": order_id}
```

**Key metrics to monitor:**

```python
# Circuit breaker metrics
circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)"
)

circuit_breaker_failures = Counter(
    "circuit_breaker_failures_total",
    "Total failures that contributed to circuit opening"
)

circuit_breaker_fallbacks = Counter(
    "circuit_breaker_fallbacks_total",
    "Total fallback responses served"
)
```

---

## Sidecar Pattern and Service Mesh

**The Problem:** Every service needs: logging, metrics, tracing, circuit breaking, retries, TLS. Writing this in each service = duplicated code and inconsistency.

**Sidecar Pattern:**

```
┌─────────────────────────────────────┐
│           Pod (Kubernetes)          │
│                                     │
│  ┌──────────────┐  ┌─────────────┐ │
│  │  Order Svc   │  │   Envoy     │ │
│  │  (business   │◄─┤  (sidecar)  │◄┼─── Incoming
│  │   logic)     │  │             │ │    traffic
│  └──────┬───────┘  └─────────────┘ │
│         │                           │
│         └─────────────────────────► │
│              Outgoing traffic       │
│         (goes through Envoy)        │
└─────────────────────────────────────┘

Envoy handles:
- Load balancing
- Circuit breaking
- Retries
- Timeouts
- Metrics (request rate, latency)
- Distributed tracing
- TLS termination
```

**Service Mesh (Istio/Linkerd):**

```
Control Plane:
┌─────────────────────────────────────┐
│  Istio Control Plane                │
│  - Service discovery                │
│  - Traffic rules configuration      │
│  - Certificate management           │
└─────────────────────────────────────┘
       │ Configures
       ▼
Data Plane (Envoy sidecars):
┌─────────┐    ┌─────────┐    ┌─────────┐
│Order Svc│    │Payment  │    │User Svc │
│ + Envoy │───►│+ Envoy  │───►│ + Envoy │
└─────────┘    └─────────┘    └─────────┘

All service-to-service traffic goes through sidecars
```

**Configuration Example (Istio):**

```yaml
# Circuit breaker for payment service
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: payment-circuit-breaker
spec:
  host: payment-service
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        maxRequestsPerConnection: 2
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
```

**Trade-offs:**

| Aspect | In-app libraries | Sidecar/Service Mesh |
|--------|------------------|----------------------|
| Latency | None | +1-2ms per hop |
| Memory | Negligible | ~50-100MB per sidecar |
| Consistency | Each team implements | Centrally enforced |
| Language lock-in | Yes (Java, Python, etc.) | No (any language) |
| Learning curve | Low | High (new infra) |
| Debugging | Straightforward | Complex (proxy layer) |

**When to use:** 10+ microservices, polyglot environment, need centralized observability
**When NOT to use:** <5 services, latency-critical (<5ms), small team (can't operate mesh)

---

## API Composition and BFF

**The Problem:** Mobile app needs data from 5 services. Making 5 HTTP calls from mobile = slow, drains battery.

### API Gateway (Simple Composition)

```
Mobile App
    │
    │ 1 request
    ▼
┌────────────────────┐
│   API Gateway      │
│                    │
│  async def get_home_screen(user_id):
│    user, orders, recs, cart, promo = await asyncio.gather(
│      user_svc.get(user_id),
│      order_svc.recent(user_id),
│      rec_svc.recommendations(user_id),
│      cart_svc.get(user_id),
│      promo_svc.active(user_id)
│    )
│    return {
│      "user": user,
│      "recent_orders": orders,
│      "recommendations": recs,
│      "cart": cart,
│      "promotions": promo
│    }
└────────────────────┘
    │  │  │  │  │
    └──┴──┴──┴──┴──► 5 parallel requests to backend services
```

### Backend for Frontend (BFF)

Different clients need different data shapes.

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  iOS BFF     │  │  Android BFF │  │  Web BFF     │
│              │  │              │  │              │
│ Minimal data │  │ Minimal data │  │ Full data    │
│ for mobile   │  │ for mobile   │  │ for desktop  │
└──────────────┘  └──────────────┘  └──────────────┘
       │                 │                  │
       └─────────────────┴──────────────────┘
                         │
           ┌─────────────┴─────────────┐
           │   Backend Microservices   │
           └───────────────────────────┘
```

**Implementation:**

```python
# Mobile BFF (minimal data)
class MobileBFF:
    async def get_product_details(self, product_id, user_id):
        product, reviews_summary = await asyncio.gather(
            product_service.get(product_id),
            review_service.get_summary(product_id)  # Summary only
        )

        return {
            "id": product.id,
            "name": product.name,
            "price": product.price,
            "image_url": product.mobile_image_url,  # Smaller image
            "rating": reviews_summary.avg_rating,
            "review_count": reviews_summary.total_reviews
            # No full review text to save bandwidth
        }

# Web BFF (rich data)
class WebBFF:
    async def get_product_details(self, product_id, user_id):
        product, reviews, recommendations, inventory = await asyncio.gather(
            product_service.get(product_id),
            review_service.get_recent(product_id, limit=10),
            rec_service.similar_products(product_id),
            inventory_service.check(product_id)
        )

        return {
            "product": product.to_dict(),  # Full product details
            "reviews": [r.to_dict() for r in reviews],  # Full reviews
            "similar_products": recommendations,
            "in_stock": inventory.quantity > 0,
            "estimated_delivery": inventory.estimated_delivery
        }
```

**When to use BFF:**
- Multiple client types (mobile, web, IoT) with different needs
- Client-specific logic (A/B tests, feature flags per platform)
- Need to version API per client

**When NOT to use:**
- Single client type (just use API gateway)
- Clients can handle multiple requests efficiently (GraphQL might be better)

---

## Service Discovery

**The Problem:** Payment service runs on 5 instances at dynamic IPs. Order service needs to find them.

### Client-Side Discovery

```
┌─────────────────────────────────────────────────┐
│  Order Service                                  │
│                                                 │
│  1. Query registry for "payment-service"        │
│  2. Get list: [10.0.1.5:8080, 10.0.1.6:8080]    │
│  3. Load balance (round-robin)                  │
│  4. Make HTTP request                           │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Service Registry (Consul/etcd/Eureka)          │
│                                                 │
│  payment-service:                               │
│    - instance-1: 10.0.1.5:8080  (healthy)       │
│    - instance-2: 10.0.1.6:8080  (healthy)       │
│    - instance-3: 10.0.1.7:8080  (unhealthy)     │
└─────────────────────────────────────────────────┘
```

**Implementation:**

```python
import consul

class ServiceDiscoveryClient:
    def __init__(self, consul_host="localhost"):
        self.consul = consul.Consul(host=consul_host)
        self.cache = {}
        self.cache_ttl = 30  # seconds

    def get_service_instances(self, service_name):
        # Check cache
        if service_name in self.cache:
            cached_time, instances = self.cache[service_name]
            if time.time() - cached_time < self.cache_ttl:
                return instances

        # Query Consul
        index, services = self.consul.health.service(
            service_name,
            passing=True  # Only healthy instances
        )

        instances = [
            {
                "host": s["Service"]["Address"],
                "port": s["Service"]["Port"]
            }
            for s in services
        ]

        # Cache result
        self.cache[service_name] = (time.time(), instances)
        return instances

    def call_service(self, service_name, path):
        instances = self.get_service_instances(service_name)
        if not instances:
            raise ServiceUnavailableError(f"{service_name} not available")

        # Simple round-robin
        instance = random.choice(instances)
        url = f"http://{instance['host']}:{instance['port']}{path}"

        return requests.get(url)

# Usage
discovery = ServiceDiscoveryClient()
response = discovery.call_service("payment-service", "/charge")
```

### Server-Side Discovery (Load Balancer)

```
Order Service
    │
    │ http://payment-service/charge
    ▼
┌─────────────────────────────────────────────────┐
│  Load Balancer (AWS ALB / Nginx / Envoy)        │
│  - Queries service registry                     │
│  - Routes to healthy instance                   │
└─────────────────────────────────────────────────┘
    │
    └─► Routes to one of:
        - payment-service-1: 10.0.1.5
        - payment-service-2: 10.0.1.6
```

**Kubernetes Service Discovery:**

```yaml
# Payment service instances (Deployment)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payment-service
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: payment-service
    spec:
      containers:
      - name: payment
        image: payment-service:v1

---
# Service (DNS entry)
apiVersion: v1
kind: Service
metadata:
  name: payment-service
spec:
  selector:
    app: payment-service
  ports:
  - port: 80
    targetPort: 8080

# Order service calls: http://payment-service/charge
# Kubernetes DNS resolves to one of the 3 pods
```

---

## Distributed Tracing

**The Problem:** Request touches 6 services. Which one is slow?

```
User: "Checkout took 5 seconds, fix it!"

Without tracing:
- Check Order Service logs? Normal.
- Check Payment Service logs? Normal.
- Check Inventory Service logs? Normal.
Hours of guessing...

With tracing:
- Look at trace ID: abc123
- See full request flow with timing
- Find: Inventory Service → Database query took 4.5s
Fixed in 10 minutes.
```

**Trace Structure:**

```
Trace (Request ID: abc123)
│
├─ Span: API Gateway (100ms)
│  └─ Span: Order Service (3000ms)
│     ├─ Span: User Service.getUser (50ms)
│     ├─ Span: Inventory Service.checkStock (2800ms)
│     │  └─ Span: Database Query (2750ms)  ← SLOW!
│     └─ Span: Payment Service.charge (100ms)
│        └─ Span: Stripe API call (80ms)
```

**Implementation with OpenTelemetry:**

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

tracer = trace.get_tracer(__name__)

# Order Service
class OrderService:
    async def create_order(self, customer_id, items):
        with tracer.start_as_current_span("create_order") as span:
            span.set_attribute("customer_id", customer_id)
            span.set_attribute("item_count", len(items))

            # Call User Service
            with tracer.start_as_current_span("get_customer"):
                customer = await self.user_client.get(customer_id)

            # Call Inventory Service
            with tracer.start_as_current_span("check_inventory"):
                available = await self.inventory_client.check(items)

            # Call Payment Service
            with tracer.start_as_current_span("process_payment"):
                payment = await self.payment_client.charge(
                    customer_id,
                    total
                )

            span.set_attribute("order_status", "success")
            return order

# Propagate trace context via HTTP headers
import aiohttp

async def call_service(url, trace_context):
    headers = {
        "traceparent": trace_context  # W3C Trace Context standard
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            return await response.json()
```

**Key trace attributes to include:**

```python
# Business context
span.set_attribute("customer_id", customer_id)
span.set_attribute("order_id", order_id)
span.set_attribute("order_total", total)

# Technical context
span.set_attribute("http.method", "POST")
span.set_attribute("http.status_code", 200)
span.set_attribute("db.statement", "SELECT * FROM orders WHERE id = ?")

# Outcomes
span.set_attribute("error", False)
span.set_attribute("payment_gateway", "stripe")
```

---

## Monolith vs Microservices vs Modular Monolith

| Criteria | Monolith | Modular Monolith | Microservices |
|----------|----------|------------------|---------------|
| **Codebase** | Single repo | Single repo, modular code | Multiple repos |
| **Deployment** | Single binary | Single binary | Independent services |
| **Database** | Shared | Shared (logical separation) | Separate per service |
| **Scaling** | Scale entire app | Scale entire app | Scale services independently |
| **Team size** | 1-5 engineers | 5-20 engineers | 20+ engineers across teams |
| **Learning curve** | Low | Medium | High |
| **Operational complexity** | Low | Low | High (monitoring, tracing, service mesh) |
| **Consistency** | ACID transactions | ACID transactions | Eventual consistency |
| **Deployment risk** | High (all or nothing) | High (all or nothing) | Low (isolated deployments) |
| **Cross-cutting changes** | Easy (refactor) | Medium (module boundaries) | Hard (coordinate services) |
| **Testing** | Simple (integration tests) | Simple (integration tests) | Complex (contract tests, E2E) |
| **Latency** | No network calls | No network calls | Network hops add latency |
| **Failure isolation** | None (one crash = all down) | None (one crash = all down) | Strong (circuit breakers) |
| **Technology diversity** | Single stack | Single stack | Polyglot (choose per service) |
| **Best for** | Startups, simple domains | Growing teams, clear modules | Large orgs, complex domains |

---

## Key Concepts Checklist

- [ ] Define service boundaries using domain-driven design or team ownership
- [ ] Choose saga pattern (choreography vs orchestration) for distributed transactions
- [ ] Implement circuit breakers for inter-service calls with fallback strategies
- [ ] Decide on sidecar/service mesh vs in-app libraries for cross-cutting concerns
- [ ] Design BFF pattern if multiple client types with different data needs
- [ ] Set up service discovery (client-side vs server-side)
- [ ] Implement distributed tracing with trace context propagation
- [ ] Understand when NOT to use microservices (small team, unclear boundaries)

---

## Practical Insights

**Start with a modular monolith:**
- Split code into modules with clear boundaries
- Separate databases logically (schemas/tables per module)
- Deploy as monolith, but enforce module boundaries in code reviews
- Extract to microservices only when you have 2+ teams or need independent scaling
- Companies that succeed: Shopify, GitHub (monoliths with 1000+ engineers)

**Data ownership is non-negotiable:**
```
Bad: Order Service queries User database directly
Good: Order Service calls User Service API

Why? When User Service team changes their schema, Order Service breaks.
Solution: API contract = stability boundary
```

**Distributed transactions are expensive:**
- Saga for order creation: 3-10x slower than monolith transaction
- Use sagas only when you NEED service independence
- For read-heavy: Replicate data locally (event-driven), avoid cross-service calls

**Circuit breaker thresholds:**
```
failure_threshold = 5-10 requests
recovery_timeout = 10-30 seconds (match service recovery time)
success_threshold = 2-3 requests (don't close too early)

Monitor: Circuit open time > 1 minute = investigate dependency
```

**Service mesh adds latency:**
- Envoy sidecar: +0.5-2ms per hop
- For 5-hop request: +2.5-10ms total
- Is centralized observability worth the latency? Depends on your p99 targets.
- Rule of thumb: If p99 < 50ms, measure sidecar impact carefully

**Trace sampling in production:**
```
100% sampling = too much data (expensive, slows system)
1% sampling = miss rare bugs

Strategy:
- Always trace errors (100% of failures)
- Sample slow requests (p95+) at 100%
- Sample normal requests at 1-10%
- Adaptive: Increase sampling when detecting issues
```

**Conway's Law is inevitable:**
- If you split services but not teams, you get distributed monolith (worst of both worlds)
- If you split teams but not services, you get merge conflicts and slow deploys
- Align services with team ownership FIRST, then extract code
