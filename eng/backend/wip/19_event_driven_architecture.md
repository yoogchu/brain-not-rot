# Chapter 19: Event-Driven Architecture

## Why Event-Driven Architecture?

Your e-commerce platform processes orders:

```
Order placed:
1. Save order to database
2. Charge credit card
3. Send confirmation email
4. Update inventory
5. Trigger shipment
6. Update analytics

Synchronous approach:
- Request takes 8 seconds (serial operations)
- Payment gateway timeout = entire order fails
- Email service down = order can't be placed
- Tight coupling between all services
```

With event-driven architecture:

```
Order placed → Publish "OrderCreated" event → Return 200 OK (200ms)

Event consumers (async):
- Payment service charges card
- Email service sends confirmation
- Inventory service updates stock
- Shipping service creates label
- Analytics service records metrics

Each service processes independently
Failures don't cascade
System stays responsive
```

---

## Event Sourcing

**The Problem:**
Traditional CRUD stores current state. You lose history: Why is this account balance $1,000? What sequence of transactions led here? How do we rebuild state if the database corrupts?

**How It Works:**

Instead of storing current state, store the sequence of events that led to it.

```
Traditional CRUD:
┌─────────────────────────┐
│ accounts table          │
│ id | balance            │
│ 42 | $1,000             │  ← Only current state
└─────────────────────────┘

Event Sourcing:
┌─────────────────────────────────────────────┐
│ events table (append-only)                  │
│ id | type              | data      | ts     │
│ 1  | AccountCreated    | {id: 42}  | 10:00  │
│ 2  | MoneyDeposited    | {$500}    | 10:05  │
│ 3  | MoneyWithdrawn    | {$200}    | 10:10  │
│ 4  | MoneyDeposited    | {$700}    | 10:15  │
└─────────────────────────────────────────────┘
         │
         ▼ Replay events to compute state
   Balance = 0 + 500 - 200 + 700 = $1,000
```

**Implementation:**

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any

@dataclass
class Event:
    aggregate_id: str
    event_type: str
    data: Dict[str, Any]
    version: int  # For optimistic locking

class BankAccount:
    def __init__(self, account_id: str):
        self.account_id = account_id
        self.balance = 0
        self.version = 0
        self.pending_events = []

    def deposit(self, amount: float):
        event = Event(
            aggregate_id=self.account_id,
            event_type="MoneyDeposited",
            data={"amount": amount},
            version=self.version + 1
        )
        self._apply_event(event)
        self.pending_events.append(event)

    def _apply_event(self, event: Event):
        if event.event_type == "MoneyDeposited":
            self.balance += event.data["amount"]
        elif event.event_type == "MoneyWithdrawn":
            self.balance -= event.data["amount"]
        self.version += 1

    @classmethod
    def from_events(cls, account_id: str, events: List[Event]):
        """Rebuild state by replaying events"""
        account = cls(account_id)
        for event in events:
            account._apply_event(event)
        return account
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Auditability | Complete history, perfect audit trail | More storage needed |
| Debugging | Replay events to reproduce bugs | Need to handle schema evolution |
| State Recovery | Rebuild from events after corruption | Replay can be slow for long histories |
| Temporal Queries | "What was balance at 10AM yesterday?" | Query complexity increases |

**When to use:** Financial systems, auditing requirements, complex domains where history matters

**When NOT to use:** Simple CRUD apps, performance-critical reads (use CQRS), frequently changing business rules

---

## CQRS (Command Query Responsibility Segregation)

**The Problem:**
Event sourcing gives great writes but slow reads (replay all events). Also, read and write requirements often differ: writes need strong consistency, reads need speed and denormalization.

**How It Works:**

Separate read and write models completely.

```
┌──────────────────────────────────────────────────────┐
│                    Client                            │
└──────────────────────────────────────────────────────┘
         │                               │
         │ Commands                      │ Queries
         │ (mutations)                   │ (reads)
         ▼                               ▼
┌─────────────────┐              ┌──────────────────┐
│  Write Model    │              │   Read Model     │
│  (Event Store)  │   Events     │   (Projections)  │
│  ┌───────────┐  │──────────────►  ┌────────────┐  │
│  │  Events   │  │              │  │ account_view│ │
│  │ (append)  │  │              │  │   balance   │  │
│  └───────────┘  │              │  └────────────┘  │
│                 │              │                  │
│  Optimized for: │              │  Optimized for: │
│  - Consistency  │              │  - Fast reads   │
│  - Validation   │              │  - Denormalized │
└─────────────────┘              └──────────────────┘
```

**Implementation:**

```python
# Write side (commands)
class AccountCommandService:
    def __init__(self, event_store, event_bus):
        self.event_store = event_store
        self.event_bus = event_bus

    async def deposit_money(self, account_id: str, amount: float):
        # Load current state from events
        events = self.event_store.get_events(account_id)
        account = BankAccount.from_events(account_id, events)

        # Execute command
        account.deposit(amount)

        # Persist and publish events
        for event in account.pending_events:
            self.event_store.append_event(event)
            await self.event_bus.publish(event)

# Read side (queries)
class AccountQueryService:
    def __init__(self):
        self.account_views = {}  # Denormalized read models

    def get_account_balance(self, account_id: str) -> Dict:
        """Fast read from denormalized view"""
        return self.account_views.get(account_id, {"balance": 0})

    def get_accounts_with_balance_over(self, amount: float):
        """Complex query on read model"""
        return [a for a in self.account_views.values() if a["balance"] > amount]

# Projection: Update read models from events
class AccountProjection:
    def __init__(self, query_service):
        self.query_service = query_service

    async def handle_event(self, event: Event):
        """Update read model when events occur"""
        account_id = event.aggregate_id

        if account_id not in self.query_service.account_views:
            self.query_service.account_views[account_id] = {"balance": 0}

        view = self.query_service.account_views[account_id]

        if event.event_type == "MoneyDeposited":
            view["balance"] += event.data["amount"]
        elif event.event_type == "MoneyWithdrawn":
            view["balance"] -= event.data["amount"]
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Read Performance | Fast, denormalized | Eventual consistency |
| Scalability | Independent scaling of reads/writes | More complexity |
| Flexibility | Multiple read models from same events | Data duplication |

**When to use:** High read/write ratio, complex queries, need to scale reads independently

**When NOT to use:** Simple CRUD, strong consistency required, small team

---

## Outbox Pattern

**The Problem:**
You need to atomically update database AND publish event. If DB commit succeeds but event publish fails, data is inconsistent.

```
Bad approach:
1. Save order to database ✓
2. Publish "OrderCreated" event ✗ (Kafka down)
Result: Order in DB, but no one knows about it
```

**How It Works:**

Store events in database table, publish them in separate transaction.

```
┌─────────────────────────────────────────────┐
│              Application                     │
└─────────────────────────────────────────────┘
         │
         ▼ Single DB transaction
┌─────────────────────────────────────────────┐
│              Database                        │
│  ┌──────────────┐      ┌─────────────────┐  │
│  │  orders      │      │  outbox_events  │  │
│  │  id: 123     │      │  event_id: 1    │  │
│  │  status: new │      │  type: Order... │  │
│  └──────────────┘      │  published: 0   │  │
│                        └─────────────────┘  │
└─────────────────────────────────────────────┘
         │
         ▼ Separate process polls outbox
┌─────────────────────────────────────────────┐
│          Message Relay                       │
│  1. Read unpublished events                  │
│  2. Publish to Kafka                         │
│  3. Mark as published                        │
└─────────────────────────────────────────────┘
```

**Implementation:**

```python
from sqlalchemy import Column, Integer, String, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Order(Base):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    customer_id = Column(String)
    total = Column(Integer)

class OutboxEvent(Base):
    __tablename__ = 'outbox_events'
    id = Column(Integer, primary_key=True)
    event_type = Column(String)
    aggregate_id = Column(String)
    payload = Column(Text)
    published = Column(Boolean, default=False)

# Save entity + event in single transaction
def create_order(session, customer_id: str, total: int):
    order = Order(customer_id=customer_id, total=total)
    session.add(order)
    session.flush()  # Get order.id

    # Create outbox event in SAME transaction
    event = OutboxEvent(
        event_type="OrderCreated",
        aggregate_id=str(order.id),
        payload=json.dumps({"order_id": order.id, "total": total})
    )
    session.add(event)
    session.commit()  # Both committed atomically

# Message relay: Poll and publish unpublished events
async def poll_and_publish(session, kafka_producer):
    while True:
        events = session.query(OutboxEvent).filter_by(published=False).limit(100).all()
        for event in events:
            await kafka_producer.send(topic=event.event_type, value=event.payload)
            event.published = True
            session.commit()
        await asyncio.sleep(1)
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Reliability | Guaranteed event publishing | Small delay (eventual consistency) |
| Atomicity | DB + event in single transaction | Extra table, polling overhead |
| Ordering | Events published in order | Need to handle duplicates |

**When to use:** Must guarantee events are published, can tolerate small delay

**When NOT to use:** Need immediate consistency, extremely high event volume

---

## Event Schema Evolution

**The Problem:**
Events are append-only. You can't change historical events. But business requirements change.

**Schema Evolution Strategies:**

```python
# Version 1: Original schema
{"type": "OrderCreated", "version": 1, "data": {"order_id": "123", "total": 5000}}

# Version 2: Add optional field (backward compatible)
{"type": "OrderCreated", "version": 2, "data": {"order_id": "123", "total": 5000, "currency": "USD"}}

# Upcasting: Transform old events to new format
class EventUpcaster:
    def upcast(self, event: dict) -> dict:
        version = event.get("version", 1)
        if version == 1:
            event["data"]["currency"] = "USD"  # Add default
            event["version"] = 2
        return event
```

**Using Avro for Schema Evolution:**

```python
# Schema V2 adds optional "currency" field with default "USD"
# Old V1 events read with V2 schema auto-fill default values
schema_v2 = {
    "type": "record",
    "name": "OrderCreated",
    "fields": [
        {"name": "order_id", "type": "string"},
        {"name": "total", "type": "int"},
        {"name": "currency", "type": "string", "default": "USD"}  # New field
    ]
}
```

**Schema Registry Pattern:**

```
┌──────────────┐
│  Producer    │─────1. Register schema────►┌──────────────────┐
└──────────────┘                            │ Schema Registry  │
       │◄────────2. Get schema ID───────────└──────────────────┘
       ▼
┌──────────────┐
│   Kafka      │  [schema_id + payload]
└──────────────┘
       │
       ▼
┌──────────────┐──────3. Fetch schema──────►┌──────────────────┐
│  Consumer    │                             │ Schema Registry  │
└──────────────┘                             └──────────────────┘
```

**Trade-offs:**

| Approach | Pros | Cons |
|----------|------|------|
| Avro | Compact, schema evolution built-in | Binary format (not human-readable) |
| Protobuf | Efficient, code generation | More complex than JSON |
| JSON + versioning | Simple, human-readable | No schema enforcement, verbose |
| Upcasting | Handles any change | Must maintain upcasters forever |

---

## Idempotency & Exactly-Once Processing

**The Problem:**
Networks are unreliable. Events may be delivered multiple times. Processing them twice causes bugs.

```
Event: "MoneyDeposited $100"

Processed once:  Balance = $1000 + $100 = $1100 ✓
Processed twice: Balance = $1100 + $100 = $1200 ✗ (wrong!)
```

**Solution: Idempotency Keys**

```python
class IdempotentEventProcessor:
    def process_event(self, event: Event):
        event_id = f"{event.aggregate_id}:{event.version}"

        if self.session.query(ProcessedEvent).filter_by(event_id=event_id).first():
            return  # Already processed, skip

        self._handle_event(event)
        self.session.add(ProcessedEvent(event_id=event_id))
        self.session.commit()

# Using Redis for idempotency
async def handle_payment(payment_id: str, amount: float, idempotency_key: str):
    existing = await redis.get(f"payment:{idempotency_key}")
    if existing:
        return json.loads(existing)  # Return cached result

    result = await process_payment(payment_id, amount)
    await redis.setex(f"payment:{idempotency_key}", 86400, json.dumps(result))
    return result
```

**Kafka Exactly-Once:**

```python
# Producer: Transactional writes
producer = KafkaProducer(
    transactional_id="my-transactional-id",
    enable_idempotence=True
)

producer.begin_transaction()
producer.send("orders", key="order-123", value=event_data)
producer.commit_transaction()

# Consumer: Read committed only
consumer = KafkaConsumer(
    "orders",
    isolation_level="read_committed"  # Only see committed events
)
```

---

## Saga Pattern

**The Problem:**
Distributed transaction across multiple services. If one fails, need to rollback others. But 2PC is slow and locks resources.

**How It Works:**

Chain of local transactions with compensating actions.

```
Order Saga (success):
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Order      │───►│   Payment    │───►│  Inventory   │
│   Service    │    │   Service    │    │   Service    │
└──────────────┘    └──────────────┘    └──────────────┘
CreateOrder ✓       ChargeCard ✓        ReserveItems ✓

Order Saga (payment fails):
┌──────────────┐    ┌──────────────┐
│   Order      │◄───│   Payment    │
│   Service    │    │   Service    │
└──────────────┘    └──────────────┘
CancelOrder ✓       ChargeCard ✗ (compensate)
```

**Orchestration: Central coordinator**

```
┌────────────────────────────────────┐
│      Saga Orchestrator             │
│                                    │
│  1. CreateOrder      → Order       │
│  2. ChargeCard       → Payment     │
│  3. ReserveItems     → Inventory   │
│                                    │
│  Failure? Run compensations:       │
│  - RefundCard                      │
│  - CancelOrder                     │
└────────────────────────────────────┘
```

**Implementation:**

```python
from enum import Enum
from typing import List, Callable

class SagaStep:
    def __init__(self, name: str, action: Callable, compensation: Callable):
        self.name = name
        self.action = action
        self.compensation = compensation

class SagaOrchestrator:
    def __init__(self, steps: List[SagaStep]):
        self.steps = steps
        self.completed_steps = []

    async def execute(self):
        """Execute saga steps in order"""
        try:
            for step in self.steps:
                await step.action()
                self.completed_steps.append(step)
            return {"status": "success"}

        except Exception as e:
            await self._compensate()
            return {"status": "failed", "error": str(e)}

    async def _compensate(self):
        """Run compensating transactions in reverse order"""
        for step in reversed(self.completed_steps):
            await step.compensation()

# Usage
saga = SagaOrchestrator([
    SagaStep("CreateOrder",
             lambda: create_order(order_data),
             lambda: cancel_order(order_id)),
    SagaStep("ChargeCard",
             lambda: charge_card(payment_data),
             lambda: refund_card(payment_id)),
    SagaStep("ReserveInventory",
             lambda: reserve_inventory(items),
             lambda: release_inventory(reservation_id))
])

result = await saga.execute()
```

**Trade-offs:**

| Aspect | Choreography | Orchestration |
|--------|--------------|---------------|
| Coupling | Low (services independent) | Higher (central coordinator) |
| Complexity | Distributed (hard to debug) | Centralized (easier to reason) |
| Observability | Hard (events scattered) | Easy (orchestrator knows all) |

**When to use:** Multi-service transactions, can tolerate eventual consistency

**When NOT to use:** Need ACID guarantees, compensations are impossible (can't unring a bell)

---

## Event Sourcing vs Traditional CRUD

| Aspect | Event Sourcing | Traditional CRUD |
|--------|----------------|------------------|
| History | Complete audit trail | Only current state |
| Debugging | Replay events to reproduce | Logs + gut feeling |
| Storage | More (all events) | Less (current state) |
| Reads | Slow (replay) or CQRS | Fast (direct lookup) |
| Writes | Fast (append-only) | Moderate (update in place) |
| Complexity | High (event handlers, projections) | Low (simple queries) |
| Schema Changes | Hard (events immutable) | Easy (ALTER TABLE) |
| Best For | Financial, audit-heavy domains | General CRUD apps |

---

## Key Concepts Checklist

- [ ] Explain event sourcing: storing state as event log
- [ ] Rebuild aggregate state by replaying events
- [ ] Describe CQRS: separate read/write models
- [ ] Implement outbox pattern for reliable event publishing
- [ ] Handle event schema evolution (Avro, versioning, upcasting)
- [ ] Design idempotent event handlers
- [ ] Implement saga pattern for distributed transactions
- [ ] Know when NOT to use event sourcing (simple CRUD, performance-critical)

---

## Practical Insights

**Event store optimization:**
- Snapshot aggregates every N events to avoid replaying thousands
- Example: Snapshot bank account every 100 transactions
- Replay from latest snapshot: `snapshot.balance + events_since_snapshot`
- Rule of thumb: Snapshot when `events_since_snapshot > 1000`

**Debugging event-sourced systems:**
```
Problem: Account balance is wrong

Debug steps:
1. Get all events for account
2. Replay events locally
3. Find which event caused incorrect state
4. Check event handler logic for that event type

This is IMPOSSIBLE with CRUD (no history)
```

**Projection lag monitoring:**
```python
# Write side: Record event timestamp
event.created_at = now()

# Read side: Track projection lag
projection_lag = now() - last_processed_event.created_at

Alert if lag > 5 seconds (read model is stale)
```

**Event versioning in practice:**
- Store version in event: `{"version": 2, "type": "OrderCreated", ...}`
- Consumers handle multiple versions or upcast to latest
- Never delete old event handlers (old events still exist in store)
- Plan for version migration: "90% of events are V2, drop V1 handler next quarter"

**When NOT to use event sourcing:**
- **Simple CRUD:** User profile with name, email, avatar - just use UPDATE
- **Frequently changing business rules:** Event replay becomes nightmare
- **Performance-critical reads:** Either use CQRS or avoid event sourcing
- **Small team:** Learning curve and maintenance overhead too high
- **Privacy requirements:** "Delete my data" is hard (events are immutable)

**Saga timeout handling:**
```python
async def execute_step(step):
    try:
        # Don't wait forever for downstream service
        await asyncio.wait_for(step.action(), timeout=30)
    except asyncio.TimeoutError:
        # Compensate and fail saga
        raise SagaTimeoutError(f"{step.name} timed out")
```
