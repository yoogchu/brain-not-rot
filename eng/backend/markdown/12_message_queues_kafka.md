# Chapter 12: Message Queues & Kafka

## Why Message Queues?

### The Synchronous Problem

```
Order Service (synchronous):

def create_order(order):
    save_to_db(order)              # 50ms
    charge_payment(order)          # 200ms - SLOW!
    update_inventory(order)        # 100ms
    send_email(order)              # 300ms - SLOW!
    send_sms(order)                # 250ms - SLOW!
    update_analytics(order)        # 100ms
    return "Order created"         # Total: 1000ms!

Problems:
1. User waits 1 second
2. Email service down = order fails
3. SMS slow = everything slow
4. Can't scale independently
```

### The Asynchronous Solution

```
Order Service (with queue):

def create_order(order):
    save_to_db(order)              # 50ms
    queue.publish("order.created", order)  # 5ms
    return "Order created"         # Total: 55ms!

# Separate consumers (run independently):
PaymentWorker:    subscribe("order.created") → charge_payment()
InventoryWorker:  subscribe("order.created") → update_inventory()
EmailWorker:      subscribe("order.created") → send_email()
SMSWorker:        subscribe("order.created") → send_sms()
AnalyticsWorker:  subscribe("order.created") → update_analytics()
```

**Benefits:**
- User response: 55ms (not 1000ms)
- Email service down? Messages queue up, process later
- Scale workers independently
- Add new consumers without changing producer

---

## Message Queue Patterns

### Point-to-Point (Queue)

```
                    ┌─────────────────┐
Producer ──────────►│     Queue       │──────────► Consumer
                    │  [msg1][msg2]   │
                    └─────────────────┘

Each message delivered to ONE consumer.
Used for: Work distribution, task queues
```

### Publish-Subscribe (Topic)

```
                    ┌─────────────────┐
                    │     Topic       │
Publisher ─────────►│   "order.new"   │
                    └─────────────────┘
                     │       │       │
                     ▼       ▼       ▼
                Consumer Consumer Consumer
                   A        B        C

Each message delivered to ALL subscribers.
Used for: Event broadcasting, notifications
```

### Consumer Groups (Kafka pattern)

```
                    ┌─────────────────────────────────┐
                    │        Topic: orders            │
                    │  Partition 0: [m1][m4][m7]      │
                    │  Partition 1: [m2][m5][m8]      │
                    │  Partition 2: [m3][m6][m9]      │
                    └─────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │Consumer A│   │Consumer B│   │Consumer C│
        │(P0: m1,  │   │(P1: m2,  │   │(P2: m3,  │
        │ m4, m7)  │   │ m5, m8)  │   │ m6, m9)  │
        └──────────┘   └──────────┘   └──────────┘
              └───────────────┴───────────────┘
                      Consumer Group: "payments"

Within a consumer group:
- Each partition → exactly one consumer
- Scale consumers up to partition count
- Messages load-balanced across group
```

---

## Apache Kafka Deep Dive

### Why Kafka is Fast

**1. Sequential I/O (not random):**
```
Traditional DB:
Random writes: ~100 writes/second per disk

Kafka:
Sequential writes: ~100,000 writes/second per disk

Kafka appends to end of log file (sequential).
No seeks, no random access.
```

**2. Zero-Copy Transfer:**
```
Traditional:
Disk → Kernel Buffer → User Buffer → Socket Buffer → Network

Kafka (sendfile syscall):
Disk → Kernel Buffer ─────────────────────────────► Network

No data copying through application!
```

**3. Batching:**
```
Without batching:
Message 1: [headers][data] → send
Message 2: [headers][data] → send
Message 3: [headers][data] → send

With batching:
[headers][msg1][msg2][msg3] → send (one network call)

Also enables compression across batch.
```

**4. Page Cache:**
```
Producer writes:
Disk ← OS Page Cache ← Kafka

Consumer reads (recent data):
Disk   OS Page Cache → Kafka
       └─ Already in memory!

Hot data served from RAM, not disk.
```

### Kafka Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Kafka Cluster                           │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    Topic: orders                         │  │
│  │                                                          │  │
│  │  Broker 1          Broker 2          Broker 3           │  │
│  │  ┌──────────┐     ┌──────────┐     ┌──────────┐        │  │
│  │  │ P0       │     │ P0       │     │ P1       │        │  │
│  │  │ (leader) │     │ (replica)│     │ (leader) │        │  │
│  │  └──────────┘     └──────────┘     └──────────┘        │  │
│  │  ┌──────────┐     ┌──────────┐     ┌──────────┐        │  │
│  │  │ P1       │     │ P2       │     │ P2       │        │  │
│  │  │ (replica)│     │ (leader) │     │ (replica)│        │  │
│  │  └──────────┘     └──────────┘     └──────────┘        │  │
│  │                                                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  ZooKeeper (or KRaft): Cluster coordination, leader election  │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

**Partition:** Unit of parallelism
**Replication Factor:** Number of copies (typically 3)
**Leader:** Handles all reads/writes for partition
**Follower:** Replicates from leader, ready for failover

### Producers

```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=['kafka1:9092', 'kafka2:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    acks='all',  # Wait for all replicas
    retries=3,
    batch_size=16384,  # Batch size in bytes
    linger_ms=5,  # Wait up to 5ms to batch
)

# Send message
producer.send(
    'orders',
    key=b'user-123',  # Key determines partition
    value={'order_id': 456, 'amount': 99.99}
)

# Ensure delivery
producer.flush()
```

**Acknowledgment modes:**
| acks | Durability | Latency |
|------|------------|---------|
| `0` | None (fire and forget) | Lowest |
| `1` | Leader only | Medium |
| `all` | All in-sync replicas | Highest |

### Consumers

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'orders',
    bootstrap_servers=['kafka1:9092', 'kafka2:9092'],
    group_id='payment-processor',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',  # Start from beginning if no offset
    enable_auto_commit=True,
    auto_commit_interval_ms=5000,
)

for message in consumer:
    order = message.value
    print(f"Processing order: {order['order_id']}")
    process_payment(order)
    # Offset auto-committed every 5 seconds
```

### Consumer Offset Management

```
Topic: orders, Partition 0

Messages:  [0][1][2][3][4][5][6][7][8][9]
                         ▲           ▲
                         │           │
                    Committed     Current
                     Offset       Position

Consumer Group: "payments"
Last committed offset: 5
Currently processing: 9

If consumer crashes at 9:
- Restart from offset 5
- Messages 5-8 processed again
- Need idempotent processing!
```

**Manual offset commit:**
```python
consumer = KafkaConsumer(
    'orders',
    group_id='payment-processor',
    enable_auto_commit=False,  # Manual commit
)

for message in consumer:
    try:
        process_payment(message.value)
        # Commit only after successful processing
        consumer.commit()
    except Exception as e:
        # Don't commit, will retry on restart
        log.error(f"Failed: {e}")
```

---

## Delivery Semantics

### At-Most-Once

```
1. Receive message
2. Commit offset
3. Process message ← If this fails, message lost!

Code:
for message in consumer:
    consumer.commit()  # Commit first
    process(message)   # Then process (might fail)
```

### At-Least-Once

```
1. Receive message
2. Process message
3. Commit offset ← If crash here, message reprocessed

Code:
for message in consumer:
    process(message)   # Process first
    consumer.commit()  # Then commit (might replay)
```

**Most common choice.** Handle duplicates with idempotent processing.

### Exactly-Once

```
Kafka Transactions (producer):
producer.init_transactions()
producer.begin_transaction()
producer.send('output-topic', result)
producer.send_offsets_to_transaction(...)
producer.commit_transaction()  # Atomic: both or neither

Requirements:
- Kafka 0.11+
- Consumer read_committed isolation
- Higher latency
```

**Alternative: Idempotent consumers**
```python
def process_order(order):
    # Check if already processed
    if db.exists(f"processed:{order['order_id']}"):
        return  # Idempotent: skip duplicate
    
    # Process
    charge_payment(order)
    
    # Mark as processed
    db.set(f"processed:{order['order_id']}", True)
```

---

## Kafka vs Traditional Queues

| Feature | Kafka | RabbitMQ/SQS |
|---------|-------|--------------|
| Model | Log (append-only) | Queue (consume and delete) |
| Retention | Time-based (days/weeks) | Until consumed |
| Replay | Yes (seek to offset) | No (once consumed, gone) |
| Ordering | Per partition | Queue-level |
| Throughput | Very high (100K+ msg/s) | High (10K+ msg/s) |
| Consumer groups | Native | Requires setup |
| Use case | Event streaming, logs | Task queues, RPC |

---

## Partitioning Strategies

### Key-Based (Default)

```python
# Same key → same partition → ordering guaranteed
producer.send('orders', key=b'user-123', value=order1)
producer.send('orders', key=b'user-123', value=order2)
# order1 always before order2 for user-123

# Different keys may go to different partitions
producer.send('orders', key=b'user-456', value=order3)
# order3 could be processed before order1 (different partition)
```

### Round-Robin (No Key)

```python
# No key → round-robin across partitions
producer.send('orders', value=order1)  # Partition 0
producer.send('orders', value=order2)  # Partition 1
producer.send('orders', value=order3)  # Partition 2
# No ordering guarantee
```

### Custom Partitioner

```python
class GeographicPartitioner:
    def __call__(self, key, all_partitions, available):
        if key.startswith(b'US-'):
            return 0
        elif key.startswith(b'EU-'):
            return 1
        elif key.startswith(b'APAC-'):
            return 2
        return hash(key) % len(all_partitions)

producer = KafkaProducer(partitioner=GeographicPartitioner())
```

---

## Common Patterns

### Event Sourcing

```
Instead of storing current state, store all events:

Events:
[OrderCreated(id=1, items=[...])]
[ItemAdded(id=1, item=...)]
[ItemRemoved(id=1, item=...)]
[OrderPaid(id=1, amount=...)]
[OrderShipped(id=1, tracking=...)]

Current state = replay all events

Benefits:
- Complete audit trail
- Time travel (state at any point)
- Event replay for new projections
```

### CQRS (Command Query Responsibility Segregation)

```
┌─────────────┐       ┌─────────────┐
│  Commands   │       │   Queries   │
│  (writes)   │       │   (reads)   │
└─────────────┘       └─────────────┘
       │                     ▲
       ▼                     │
┌─────────────┐       ┌─────────────┐
│   Kafka     │──────►│  Read DB    │
│  (events)   │       │ (optimized) │
└─────────────┘       └─────────────┘
       │
       ▼
┌─────────────┐
│  Write DB   │
│ (normalized)│
└─────────────┘

Writes: Append events to Kafka
Reads: Query optimized read model
Consumer: Projects events to read model
```

### Saga Orchestration

```
┌────────────────────────────────────────────────────────┐
│                    Saga Orchestrator                    │
│   (consumes from all topics, coordinates flow)          │
└────────────────────────────────────────────────────────┘
        │           │           │           │
        ▼           ▼           ▼           ▼
   order.cmd   payment.cmd  inventory.cmd  ship.cmd
        │           │           │           │
        ▼           ▼           ▼           ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Order   │  │ Payment  │  │Inventory │  │Shipping  │
│ Service  │  │ Service  │  │ Service  │  │ Service  │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
        │           │           │           │
        ▼           ▼           ▼           ▼
   order.evt   payment.evt  inventory.evt  ship.evt
        │           │           │           │
        └───────────┴───────────┴───────────┘
                        │
                        ▼
                 Saga Orchestrator
              (handles completions/failures)
```

---

## Key Concepts Checklist

- [ ] Explain Kafka's performance characteristics (sequential I/O, zero-copy)
- [ ] Describe partitions and consumer groups
- [ ] Explain offset management and delivery semantics
- [ ] Compare at-most-once, at-least-once, exactly-once
- [ ] Design partition key strategy for ordering requirements
- [ ] Know when to use Kafka vs traditional queues

---

## Practical Insights

**Partition count:**
- More partitions = more parallelism
- But: More file handles, longer recovery
- Rule of thumb: Start with `max(expected_throughput / 10MB_per_partition_per_sec, num_consumers)`

**Consumer lag monitoring:**
```
Consumer lag = Latest offset - Consumer offset

High lag = Consumer can't keep up
Alert on: Lag growing over time
```

**Retention and compaction:**
```
# Time-based retention
log.retention.hours=168  # 7 days

# Log compaction (keep latest per key)
cleanup.policy=compact

Use compaction for: Changelogs, caches
Use retention for: Event streams, logs
```

**Replication and durability:**
```
replication.factor=3
min.insync.replicas=2
acks=all

This means:
- 3 copies of data
- At least 2 must ACK write
- Can lose 1 broker without data loss
```

**Kafka Connect for integrations:**
```
Source connectors: DB → Kafka (CDC)
Sink connectors: Kafka → DB/Search/Analytics

Don't reinvent: Use existing connectors
```
