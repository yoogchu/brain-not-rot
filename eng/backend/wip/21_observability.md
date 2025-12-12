# Chapter 21: Observability - Metrics, Logs, and Traces

## Why Observability?

```
3 AM Alert: "Payment service latency > 5 seconds"

Without observability:
- Which service is slow? 🤷
- What changed? 🤷
- How many users affected? 🤷
- Is it getting worse? 🤷

With observability:
- Metrics: 95th percentile latency spiked at 2:47 AM
- Traces: Slow requests spending 4.8s in database call
- Logs: "Connection timeout to payments-db-replica-3"

Root cause found in 5 minutes: Database replica failed over
```

---

## The Three Pillars

```
┌─────────────────────────────────────────────────────────┐
│                    Observability                         │
├─────────────────┬─────────────────┬────────────────────┤
│     METRICS     │      LOGS       │      TRACES        │
│   (Aggregated   │   (Discrete     │  (Request flow     │
│    numbers)     │    events)      │   across services) │
├─────────────────┼─────────────────┼────────────────────┤
│ • Request rate  │ • Error details │ • Service A → B    │
│ • Error rate    │ • Debug info    │ • Latency per hop  │
│ • Latency p99   │ • User actions  │ • Bottleneck ID    │
│ • CPU/Memory    │ • Audit trail   │ • Error location   │
├─────────────────┼─────────────────┼────────────────────┤
│ What's broken?  │ Why is it       │ Where in the       │
│ How bad?        │ broken?         │ request flow?      │
└─────────────────┴─────────────────┴────────────────────┘
```

---

## Metrics

### The RED Method (Request-focused)

```
Rate:     Requests per second
Errors:   Errors per second  
Duration: Latency distribution (p50, p95, p99)

Perfect for: Services, APIs, microservices
```

### The USE Method (Resource-focused)

```
Utilization: % time resource is busy
Saturation:  Queue depth / backlog
Errors:      Error count

Perfect for: Infrastructure, databases, queues
```

### Metric Types

**Counter:**
```
Monotonically increasing value.

http_requests_total{method="GET", status="200"} 15234
http_requests_total{method="GET", status="500"} 42

Use for: Request counts, error counts, bytes sent
Calculate: Rate of change (requests/second)
```

**Gauge:**
```
Value that goes up and down.

memory_usage_bytes 734003200
active_connections 47
queue_depth 156

Use for: Current values, queue sizes, temperatures
```

**Histogram:**
```
Distribution of values in buckets.

http_request_duration_seconds_bucket{le="0.1"} 24054
http_request_duration_seconds_bucket{le="0.5"} 33444
http_request_duration_seconds_bucket{le="1.0"} 34001
http_request_duration_seconds_bucket{le="+Inf"} 34325
http_request_duration_seconds_sum 8953.42
http_request_duration_seconds_count 34325

Use for: Latency distributions, request sizes
Calculate: Percentiles (p50, p95, p99)
```

### Prometheus Example

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

ACTIVE_REQUESTS = Gauge(
    'http_active_requests',
    'Currently processing requests'
)

# Use in code
@app.route('/api/users')
def get_users():
    ACTIVE_REQUESTS.inc()
    start = time.time()
    
    try:
        result = fetch_users()
        REQUEST_COUNT.labels('GET', '/api/users', '200').inc()
        return result
    except Exception as e:
        REQUEST_COUNT.labels('GET', '/api/users', '500').inc()
        raise
    finally:
        REQUEST_LATENCY.labels('GET', '/api/users').observe(time.time() - start)
        ACTIVE_REQUESTS.dec()
```

### Key Metrics to Track

**Service Level:**
```
Request rate (QPS)
Error rate (% or count)
Latency percentiles (p50, p95, p99, p999)
Availability (uptime %)
```

**Infrastructure Level:**
```
CPU utilization
Memory usage
Disk I/O
Network I/O
Container restarts
```

**Business Level:**
```
Orders per minute
Revenue per hour
Active users
Conversion rate
```

---

## Logs

### Structured Logging

```python
# Bad: Unstructured
logger.info(f"User {user_id} placed order {order_id} for ${amount}")

# Good: Structured JSON
logger.info("order_placed", extra={
    "user_id": user_id,
    "order_id": order_id,
    "amount": amount,
    "currency": "USD",
    "items_count": len(items)
})

# Output:
{
  "timestamp": "2024-01-15T10:23:45.123Z",
  "level": "INFO",
  "message": "order_placed",
  "user_id": "u-123",
  "order_id": "ord-456",
  "amount": 99.99,
  "currency": "USD",
  "items_count": 3,
  "service": "order-service",
  "trace_id": "abc123"
}
```

**Why structured?**
- Searchable: `user_id:u-123 AND level:ERROR`
- Aggregatable: Count orders by user
- Parseable: Automated alerting

### Log Levels

```
TRACE   Detailed debugging (rarely in production)
DEBUG   Development debugging
INFO    Normal operations (request handled, job completed)
WARN    Unexpected but handled (retry succeeded, deprecated API)
ERROR   Failures requiring attention (request failed, exception)
FATAL   System cannot continue (startup failure, OOM)
```

**Production recommendation:**
```
Development: DEBUG and above
Production:  INFO and above
Debugging:   Temporarily enable DEBUG for specific component
```

### Correlation IDs

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Gateway   │────►│   Order     │────►│   Payment   │
│             │     │   Service   │     │   Service   │
└─────────────┘     └─────────────┘     └─────────────┘

Request ID: req-abc123 (propagated through all services)

Gateway log:
{"request_id": "req-abc123", "message": "Received order request"}

Order Service log:
{"request_id": "req-abc123", "message": "Creating order"}

Payment Service log:
{"request_id": "req-abc123", "message": "Charging payment"}

Search: request_id:req-abc123 → See entire request flow
```

### Log Aggregation Architecture

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Service A  │  │  Service B  │  │  Service C  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       ▼                ▼                ▼
┌─────────────────────────────────────────────────┐
│            Log Shipper (Fluentd/Filebeat)       │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│            Message Queue (Kafka)                │
│            (buffer for spikes)                  │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│         Log Storage (Elasticsearch/Loki)        │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              UI (Kibana/Grafana)                │
└─────────────────────────────────────────────────┘
```

---

## Distributed Tracing

### The Problem

```
User complaint: "Checkout is slow"

Checkout involves:
Gateway → Cart Service → Inventory Service → Payment Service → Order Service

Which one is slow? Logs show each service took "some time" but no correlation.
```

### Trace Structure

```
Trace ID: trace-abc123 (entire request)
├── Span 1: Gateway (parent=none)
│   ├── service: gateway
│   ├── operation: handle_checkout
│   ├── duration: 850ms
│   └── status: ok
│
├── Span 2: Cart Service (parent=Span 1)
│   ├── service: cart-service
│   ├── operation: get_cart
│   ├── duration: 45ms
│   └── status: ok
│
├── Span 3: Inventory Service (parent=Span 1)
│   ├── service: inventory-service
│   ├── operation: check_availability
│   ├── duration: 120ms
│   └── status: ok
│
├── Span 4: Payment Service (parent=Span 1)
│   ├── service: payment-service
│   ├── operation: charge_card
│   ├── duration: 650ms  ← BOTTLENECK!
│   └── status: ok
│
└── Span 5: Order Service (parent=Span 1)
    ├── service: order-service
    ├── operation: create_order
    ├── duration: 30ms
    └── status: ok
```

### OpenTelemetry Example

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# Setup
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(agent_host_name="jaeger", agent_port=6831)
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(jaeger_exporter))

tracer = trace.get_tracer(__name__)

# Create spans
@app.route('/checkout')
def checkout():
    with tracer.start_as_current_span("checkout") as span:
        span.set_attribute("user_id", current_user.id)
        
        # Child span
        with tracer.start_as_current_span("get_cart"):
            cart = cart_service.get_cart(current_user.id)
        
        # Another child span
        with tracer.start_as_current_span("process_payment"):
            payment = payment_service.charge(cart.total)
        
        return {"order_id": order.id}
```

### Context Propagation

```
Service A calls Service B:

1. Service A creates span, injects context into headers:
   headers["traceparent"] = "00-trace_id-span_id-01"

2. Service B extracts context from headers:
   parent_context = extract(headers)
   
3. Service B creates child span with parent context:
   with tracer.start_as_current_span("operation", context=parent_context):
       ...
```

### Sampling

```
At high traffic, tracing everything is expensive.

Sampling strategies:

1. Head-based (decide at start):
   - Sample 1% of all traces
   - Simple but may miss interesting traces

2. Tail-based (decide at end):
   - Keep traces with errors
   - Keep traces slower than threshold
   - More complex, requires buffering

3. Adaptive:
   - Sample more when traffic is low
   - Sample less when traffic is high
```

---

## Connecting the Three Pillars

```
Alert fires: Error rate > 5%

1. METRICS: Dashboard shows errors spiking at 14:32
   → Which service? payment-service

2. LOGS: Filter by service=payment-service, level=ERROR, time=14:32
   → "Connection refused: payments-db-primary:5432"

3. TRACES: Find slow/errored traces at 14:32
   → Database connection span timing out

Root cause: Database failover, connection pool not updated
```

### Exemplars (Metrics → Traces)

```
Prometheus metric with trace link:

http_request_duration_seconds{method="GET"} 2.5 # trace_id=abc123

High latency metric → Click → Jump to exact trace
```

---

## Alerting

### Alert on Symptoms, Not Causes

```
Bad alerts (causes):
- CPU > 80%
- Memory > 90%
- Disk > 85%

Good alerts (symptoms):
- Error rate > 1%
- Latency p99 > 500ms
- Success rate < 99%

Why? Users don't care about CPU. They care if the service works.
```

### Alert Fatigue Prevention

```
Problem: 100 alerts/day → Team ignores all alerts

Solutions:

1. Actionable: Every alert should have a clear response
2. Deduplicated: Group related alerts
3. Prioritized: P1 (page), P2 (ticket), P3 (review weekly)
4. Owned: Each alert has a clear owner
5. Reviewed: Delete alerts that aren't acted on
```

### Alert Example

```yaml
# Prometheus alerting rule
groups:
- name: payment-service
  rules:
  - alert: HighErrorRate
    expr: |
      sum(rate(http_requests_total{service="payment", status=~"5.."}[5m]))
      /
      sum(rate(http_requests_total{service="payment"}[5m]))
      > 0.01
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Payment service error rate > 1%"
      description: "Error rate is {{ $value | humanizePercentage }}"
      runbook: "https://runbooks.example.com/payment-errors"
```

---

## Key Concepts Checklist

- [ ] Explain the three pillars and when to use each
- [ ] Describe RED and USE methods
- [ ] Know metric types (counter, gauge, histogram)
- [ ] Explain structured logging benefits
- [ ] Describe distributed tracing and context propagation
- [ ] Design alerting strategy (symptoms vs causes)

---

## Practical Insights

**Observability costs:**
- Metrics: Low (aggregated)
- Logs: High (store everything)
- Traces: Medium (sample intelligently)

**Cardinality explosion:**
```
Bad: metric{user_id="..."} → millions of time series
Good: metric{tier="free|pro|enterprise"} → 3 time series

High cardinality kills Prometheus/metrics systems
```

**Log retention strategy:**
```
Hot:  7 days  (fast storage, full access)
Warm: 30 days (slower, queryable)
Cold: 1 year  (archive, restore to query)
```

**Trace sampling in production:**
- Always trace errors
- Always trace slow requests (> p99)
- Sample 1-10% of normal traffic
- 100% for specific user IDs (debugging)

**SLO-based alerting:**
```
Instead of: "Error rate > 1%"
Alert on: "Burning error budget faster than expected"

If monthly budget is 0.1% errors:
Alert when: Projected to exhaust budget in < 3 days
```
