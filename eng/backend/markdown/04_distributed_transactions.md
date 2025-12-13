# Chapter 4: Distributed Transactions

## The Double-Charge Problem

```
User clicks "Buy $100 item"
1. Server receives request
2. Server charges credit card: SUCCESS
3. Network connection drops
4. Server tries to respond: FAILS
5. User doesn't see confirmation
6. User clicks again
7. Server charges AGAIN: $200 charged!
```

This happens thousands of times daily at Stripe's scale. Distributed transactions are HARD.

---

## Idempotency: The Foundation

### Definition

An **idempotent** operation produces the same result regardless of how many times it's executed.

```python
# Naturally idempotent:
SET user.email = "new@email.com"  # Same result if run 10 times
DELETE FROM cart WHERE user_id = 123  # Still deleted after 10 runs

# NOT idempotent:
counter.increment()  # 1, 2, 3, 4... different each time
INSERT INTO logs (message) VALUES ('clicked')  # Duplicates!
```

### Why Idempotency Matters

Networks are unreliable. Clients retry. Servers crash mid-operation. Without idempotency:

```
Client → Server: "Charge $100"
Server: Charges card, crashes before responding
Client: No response, retry
Client → Server: "Charge $100"
Server: Charges card again!
User: $200 charged 😱
```

### Implementing Idempotency with Keys

```python
def make_payment(amount, idempotency_key):
    # Check if we've seen this request before
    existing = db.get(f"idempotency:{idempotency_key}")
    if existing:
        return existing  # Return cached result (same as before)
    
    # First time seeing this request - process it
    result = payment_provider.charge(amount)
    
    # Store result for future duplicates
    db.set(f"idempotency:{idempotency_key}", result, ttl=72*3600)  # 72 hour TTL
    
    return result
```

**Client generates the key:**
```javascript
// Client-side
const idempotencyKey = generateUUID();

// First attempt
fetch('/api/charge', {
  headers: { 'Idempotency-Key': idempotencyKey },
  body: JSON.stringify({ amount: 100 })
});

// Retry (network failed) - SAME key
fetch('/api/charge', {
  headers: { 'Idempotency-Key': idempotencyKey },  // Same!
  body: JSON.stringify({ amount: 100 })
});
```

### Idempotency Key Guidelines

| Aspect | Recommendation |
|--------|----------------|
| Generator | Client-generated UUID |
| Scope | Per-user, per-operation type |
| TTL | 24-72 hours (outlast retry storms) |
| Storage | Full request + response for verification |
| Collision | Return 409 Conflict if key reused with different params |

### The Request Verification Problem

```python
# What if client reuses key with different amount?
# Request 1: {key: "abc", amount: 100}
# Request 2: {key: "abc", amount: 200}  # Different!

def make_payment(amount, idempotency_key):
    existing = db.get(f"idempotency:{idempotency_key}")
    if existing:
        # Verify request matches
        if existing.request_hash != hash(amount):
            raise ConflictError("Idempotency key reused with different params")
        return existing.result
    
    # ... rest of logic
```

> **Common Mistake:** "Database transactions solve this." 
> 
> NO. The failure can happen AFTER the DB transaction commits but BEFORE the client receives the response. The commit succeeded, data is persisted, but client doesn't know.

---

## Two-Phase Commit (2PC)

### The Problem

Multiple databases/services must commit or abort **together**.

```
Order Service: Create order in orders_db
Payment Service: Charge customer in payments_db
Inventory Service: Reserve items in inventory_db

All must succeed, or all must rollback.
Partial success = inconsistent state.
```

### The Protocol

```
              Coordinator
                  │
    ┌─────────────┼─────────────┐
    │             │             │
Participant A  Participant B  Participant C

PHASE 1: PREPARE
─────────────────
Coordinator → All: "Prepare to commit transaction T1"
Each participant:
  - Acquires locks
  - Writes to WAL (durable)
  - Votes YES (can commit) or NO (must abort)
All → Coordinator: "YES" or "NO"

PHASE 2A: COMMIT (if all voted YES)
────────────────────────────────────
Coordinator: Writes "COMMIT T1" to own WAL
Coordinator → All: "Commit"
Each participant:
  - Applies changes
  - Releases locks
All → Coordinator: "Committed"

PHASE 2B: ABORT (if any voted NO)
─────────────────────────────────
Coordinator → All: "Abort"
Each participant:
  - Rollback changes
  - Releases locks
All → Coordinator: "Aborted"
```

### The Visual Timeline

```
Phase 1 (Prepare):
Coordinator: ──────[Prepare]──────────────────────>
     │
     ├──> Participant A: [Prepare] → [Vote YES] ──>
     ├──> Participant B: [Prepare] → [Vote YES] ──>
     └──> Participant C: [Prepare] → [Vote YES] ──>

Phase 2 (Commit):
Coordinator: [All YES] ──────[Commit]──────────────>
     │
     ├──> Participant A: [Commit] → [Done] ──>
     ├──> Participant B: [Commit] → [Done] ──>
     └──> Participant C: [Commit] → [Done] ──>
```

### 2PC Problems

#### Problem 1: Blocking

```
Timeline:
T0: Coordinator sends PREPARE
T1: Participants vote YES, acquire locks
T2: Coordinator writes COMMIT to WAL
T3: Coordinator crashes before sending COMMIT
T4: Participants stuck!
    - Can't commit (no instruction)
    - Can't abort (might have committed)
    - Holding locks indefinitely 🔒
```

The blocking window can last until coordinator recovers—minutes or hours.

#### Problem 2: Latency

```
Minimum round trips:
1. Coordinator → Participants: PREPARE
2. Participants → Coordinator: VOTE
3. Coordinator → Participants: COMMIT
4. Participants → Coordinator: DONE

4 sequential network hops minimum
Cross-datacenter: 400ms+ per transaction
```

#### Problem 3: Availability

Any participant down = transaction fails.

```
3 participants, each 99.9% available
P(all up) = 0.999³ = 99.7%

10 participants, each 99.9% available  
P(all up) = 0.999¹⁰ = 99.0%

More participants = lower availability
```

### When to Use 2PC

✅ **Good fit:**
- Same database vendor (distributed PostgreSQL, MySQL clusters)
- Critical financial transactions that MUST be atomic
- Low-frequency, high-value operations
- Single datacenter

❌ **Bad fit:**
- Microservices with different databases
- High-throughput systems (latency kills)
- Cross-datacenter operations
- When any participant might be slow

---

## The Saga Pattern

### Key Insight

Don't try to make distributed operations atomic. Instead:
- Break into **local transactions**
- Define **compensating transactions** for rollback

```
Traditional ACID: One big atomic transaction
┌─────────────────────────────────────────┐
│ Create Order + Charge + Reserve Inventory │
│              ALL OR NOTHING               │
└─────────────────────────────────────────┘

Saga: Sequence of local transactions with compensation
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Create Order │ → │ Charge Card  │ → │ Reserve Item │
└──────────────┘   └──────────────┘   └──────────────┘
       ↓                  ↓                  ↓
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Cancel Order │   │ Refund Card  │   │ Release Item │
│(compensation)│   │(compensation)│   │(compensation)│
└──────────────┘   └──────────────┘   └──────────────┘
```

### Saga Execution: Happy Path

```
1. Create Order (status: PENDING)      ✓
2. Reserve Payment                     ✓
3. Reserve Inventory                   ✓
4. Confirm Order (status: CONFIRMED)   ✓

All steps succeeded → Done!
```

### Saga Execution: Failure & Compensation

```
1. Create Order (status: PENDING)      ✓
2. Reserve Payment                     ✓
3. Reserve Inventory                   ✗ (out of stock!)

Compensation kicks in:
4. Refund Payment (undo step 2)        ✓
5. Cancel Order (undo step 1)          ✓

Order status: CANCELLED
User sees: "Item out of stock, payment refunded"
```

---

## Saga Coordination Patterns

### Pattern 1: Choreography (Event-Driven)

Each service listens to events and decides what to do:

```
┌────────────┐    order.created    ┌──────────────┐
│   Order    │ ─────────────────►  │   Payment    │
│  Service   │                     │   Service    │
└────────────┘                     └──────────────┘
      ▲                                   │
      │                       payment.reserved
      │                                   │
      │                                   ▼
      │                            ┌──────────────┐
      │                            │  Inventory   │
      │                            │   Service    │
      │                            └──────────────┘
      │                                   │
      │         inventory.reserved         │
      └────────────────────────────────────┘

Each service:
- Listens for relevant events
- Performs local transaction
- Emits outcome event
- Handles failure events (compensate)
```

**Implementation:**
```python
# Order Service
@event_handler("payment.reserved")
def on_payment_reserved(event):
    order = db.get_order(event.order_id)
    order.status = "PAYMENT_RESERVED"
    db.save(order)

@event_handler("payment.failed")
def on_payment_failed(event):
    order = db.get_order(event.order_id)
    order.status = "CANCELLED"
    db.save(order)
    emit("order.cancelled", order_id=event.order_id)

# Payment Service
@event_handler("order.created")
def on_order_created(event):
    try:
        reservation = reserve_payment(event.customer_id, event.amount)
        emit("payment.reserved", order_id=event.order_id)
    except InsufficientFunds:
        emit("payment.failed", order_id=event.order_id)

@event_handler("inventory.failed")
def on_inventory_failed(event):
    # Compensate: refund the payment
    refund_payment(event.order_id)
    emit("payment.refunded", order_id=event.order_id)
```

**Choreography Pros:**
- Simple, decoupled services
- No single point of failure
- Easy to add new services

**Choreography Cons:**
- Hard to understand full flow (scattered logic)
- No central view of saga state
- Complex failure handling (who compensates whom?)
- Risk of cyclic dependencies

### Pattern 2: Orchestration (Central Coordinator)

One service manages the entire saga:

```
┌────────────────────────────────────────────────┐
│            Saga Orchestrator                    │
│  (Knows full flow, manages state machine)       │
│                                                 │
│  Step 1: CreateOrder() ──────────────────────┐  │
│  Step 2: ReservePayment() ───────────────┐   │  │
│  Step 3: ReserveInventory() ──────────┐  │   │  │
│                                       │  │   │  │
└───────────────────────────────────────┼──┼───┼──┘
                                        │  │   │
                                        ▼  ▼   ▼
                                  ┌─────┬─────┬─────┐
                                  │Order│Pay- │Inv- │
                                  │Svc  │ment │ntory│
                                  └─────┴─────┴─────┘
```

**Implementation:**
```python
class OrderSagaOrchestrator:
    def __init__(self):
        self.steps = [
            SagaStep(
                action=self.create_order,
                compensation=self.cancel_order
            ),
            SagaStep(
                action=self.reserve_payment,
                compensation=self.refund_payment
            ),
            SagaStep(
                action=self.reserve_inventory,
                compensation=self.release_inventory
            ),
            SagaStep(
                action=self.confirm_order,
                compensation=None  # Final step, no compensation
            )
        ]
    
    async def execute(self, order_request):
        saga_state = SagaState(order_request)
        completed_steps = []
        
        for step in self.steps:
            try:
                await step.action(saga_state)
                completed_steps.append(step)
            except Exception as e:
                # Failure! Run compensations in reverse order
                for completed_step in reversed(completed_steps):
                    if completed_step.compensation:
                        await completed_step.compensation(saga_state)
                raise SagaFailedError(e)
        
        return saga_state.result
```

**Orchestration Pros:**
- Clear visibility into saga state
- Easier to add/modify steps
- Centralized error handling
- Better for complex business logic

**Orchestration Cons:**
- Orchestrator is potential single point of failure
- Can become bottleneck
- Tighter coupling to orchestrator
- More infrastructure to manage

---

## Designing Compensating Transactions

### Requirements

**1. Idempotent:** Safe to retry
```python
def refund_payment(order_id):
    payment = db.get_payment(order_id)
    if payment.status == "REFUNDED":
        return  # Already done, idempotent!
    
    stripe.refund(payment.charge_id)
    payment.status = "REFUNDED"
    db.save(payment)
```

**2. Commutative:** Order shouldn't matter
```python
# If refund and release happen concurrently,
# final state should be the same regardless of order
```

**3. Eventual:** May take time to complete
```python
# Compensation might involve:
# - External APIs (might be slow)
# - Manual intervention (might take days)
# - Batch processes
```

### Semantic vs Physical Compensation

**Physical Compensation:** Undo the exact operation
```
Charge $100 → Refund $100
Reserve 5 items → Unreserve 5 items
```

**Semantic Compensation:** Business-level undo
```
Create shipment → Cancel shipment (different API, might incur fees)
Send email → Send "oops, ignore that" email (can't unsend)
Generate report → Mark report as invalid
Publish article → Publish retraction
```

Physical is simpler but not always possible. Design for semantic compensation from the start.

---

## Saga State Management

### State Machine

```
                    ┌──────────────────┐
                    │     STARTED      │
                    └──────────────────┘
                            │
                            ▼
                    ┌──────────────────┐
                    │ ORDER_CREATED    │
                    └──────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ┌──────────────────┐        ┌──────────────────┐
    │ PAYMENT_RESERVED │        │ PAYMENT_FAILED   │
    └──────────────────┘        └──────────────────┘
              │                           │
              ▼                           ▼
    ┌──────────────────┐        ┌──────────────────┐
    │INVENTORY_RESERVED│        │ COMPENSATING     │
    └──────────────────┘        └──────────────────┘
              │                           │
              ▼                           ▼
    ┌──────────────────┐        ┌──────────────────┐
    │    COMPLETED     │        │     FAILED       │
    └──────────────────┘        └──────────────────┘
```

### Persistence

Store saga state in database:

```sql
CREATE TABLE saga_state (
    saga_id UUID PRIMARY KEY,
    saga_type VARCHAR(50),
    current_step INTEGER,
    state VARCHAR(20),
    payload JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE saga_step_log (
    saga_id UUID REFERENCES saga_state(saga_id),
    step_number INTEGER,
    step_name VARCHAR(50),
    action_status VARCHAR(20),
    compensation_status VARCHAR(20),
    error_message TEXT,
    executed_at TIMESTAMP
);
```

**Why persist?**
- Recovery after crash
- Debugging and auditing
- Retry stuck sagas
- Monitor saga health

---

## Handling the "In-Between" State

Sagas have windows where the system is inconsistent:

```
T0: Order created (ORDER service) ──────────────────────┐
T1: Payment processing... ◀─── User might see order     │ INCONSISTENT
T2: Payment confirmed            without payment!       │ WINDOW
T3: Inventory reserving... ◀─── Payment done,           │
T4: Saga complete               but no inventory yet   ─┘
```

### Solutions

**1. Show "Processing" Status**
```json
{
  "order_id": "123",
  "status": "PROCESSING",
  "message": "Your order is being confirmed..."
}
```

User sees clear state. Better than confusing partial state.

**2. Eventual Confirmation**
```
Immediately: "Order received! Confirmation coming soon."
Later (async): Email/notification when saga completes

User expectations set correctly.
```

**3. Timeout and Retry**
```python
def check_stuck_sagas():
    stuck = db.query("""
        SELECT * FROM saga_state 
        WHERE state = 'PROCESSING' 
        AND updated_at < NOW() - INTERVAL '5 minutes'
    """)
    
    for saga in stuck:
        if saga.retry_count < MAX_RETRIES:
            retry_saga(saga)
        else:
            escalate_to_human(saga)
```

---

## Try-Confirm/Cancel (TCC)

A variation of Saga for resource reservation:

```
TCC Three Phases:
┌─────────────────────────────────────────────────┐
│  TRY: Reserve resources tentatively              │
│       - Lock inventory (soft lock)               │
│       - Reserve payment (auth, not capture)      │
│       - Create pending order                     │
├─────────────────────────────────────────────────┤
│  CONFIRM: Finalize if all Try succeeded          │
│       - Consume reserved inventory               │
│       - Capture payment                          │
│       - Confirm order                            │
├─────────────────────────────────────────────────┤
│  CANCEL: Release if any Try failed               │
│       - Release inventory lock                   │
│       - Void payment auth                        │
│       - Cancel order                             │
└─────────────────────────────────────────────────┘
```

**Advantage over Saga:** Resources reserved upfront, less compensation complexity
**Disadvantage:** Requires all services to support Try/Confirm/Cancel pattern

---

## Distributed Transaction Patterns Comparison

| Pattern | Consistency | Availability | Complexity | Use Case |
|---------|-------------|--------------|------------|----------|
| 2PC | Strong (ACID) | Low | Medium | Same DB vendor, critical transactions |
| Saga Choreography | Eventual | High | Medium | Simple flows, microservices |
| Saga Orchestration | Eventual | High | Higher | Complex flows, visibility needed |
| TCC | Strong-ish | Medium | High | Resource reservation scenarios |

---

## Key Concepts Checklist

- [ ] Identify if distributed transaction is actually needed
- [ ] Design idempotent operations with idempotency keys
- [ ] Choose between 2PC and Saga based on requirements
- [ ] Design compensating transactions for each step
- [ ] Handle the "in-between" inconsistent state
- [ ] Consider timeout and retry strategies
- [ ] Plan for partial failures and manual intervention

---

## Practical Insights

**Avoid distributed transactions when possible:**
- Can you redesign to avoid cross-service transactions?
- Can you use eventual consistency with reconciliation?
- Can you batch operations?

**Saga debugging is hard:**
- Log every step with correlation ID
- Store full request/response payloads
- Build admin UI to view saga state
- Alert on stuck sagas

**Compensation is business logic:**
- Not just technical rollback
- May involve customer communication
- May have financial implications (refund fees)
- Design compensation with product team
