# Chapter 18: Batch vs Stream Processing

## The Processing Time Paradox

Your data team reports success:

```
Monday 9 AM: "Sales report completed! Revenue from yesterday: $2.3M"

You check the live dashboard.
Current revenue (Monday 9 AM): Already $500K

The business asks: "What was revenue at 8 AM? At 8:30 AM?"
Batch job answer: "I don't know. I only run once daily."

Meanwhile, fraud detection needs to block suspicious transactions BEFORE they complete.
Batch processing: "I'll tell you tomorrow if yesterday's transactions were fraudulent."
Result: Millions lost. Customers angry. Regulators investigating.
```

Batch processing trades freshness for simplicity. Stream processing trades simplicity for freshness. The right choice depends on whether you can afford to wait.

---

## Batch Processing Fundamentals

### MapReduce Model

**The Problem:**
Process terabytes of data on a single machine: Takes weeks. Process in parallel across 1000 machines: Need to coordinate, handle failures, aggregate results.

**How It Works:**

```
Input Data (10 TB)
       │
       ├────────────────┬─────────────┬───────────────┐
       ▼                ▼             ▼               ▼
  ┌─────────┐     ┌─────────┐   ┌─────────┐    ┌─────────┐
  │  Map 1  │     │  Map 2  │   │  Map 3  │... │  Map N  │
  │ (chunk) │     │ (chunk) │   │ (chunk) │    │ (chunk) │
  └─────────┘     └─────────┘   └─────────┘    └─────────┘
       │                │             │               │
       └────────────────┴──────┬──────┴───────────────┘
                               │ Shuffle & Sort
       ┌────────────────┬──────┴──────┬───────────────┐
       ▼                ▼             ▼               ▼
  ┌─────────┐     ┌─────────┐   ┌─────────┐    ┌─────────┐
  │Reduce 1 │     │Reduce 2 │   │Reduce 3 │... │Reduce M │
  │ (key A) │     │ (key B) │   │ (key C) │    │ (key Z) │
  └─────────┘     └─────────┘   └─────────┘    └─────────┘
       │                │             │               │
       └────────────────┴─────────────┴───────────────┘
                               │
                               ▼
                          Output Data
```

**Implementation (Word Count):**

```python
from collections import defaultdict
from typing import Iterator, Tuple

def map_function(document: str) -> Iterator[Tuple[str, int]]:
    """Map: Split document into words, emit (word, 1) for each."""
    for word in document.split():
        word = word.lower().strip('.,!?;:')
        yield (word, 1)

def reduce_function(word: str, counts: Iterator[int]) -> Tuple[str, int]:
    """Reduce: Sum all counts for a given word."""
    return (word, sum(counts))

# MapReduce framework handles:
# - Distributing input chunks to mappers
# - Shuffling map output by key to reducers
# - Handling failures, retries, stragglers

# Simplified single-machine version to show the pattern:
def mapreduce(documents, map_fn, reduce_fn):
    # Map phase
    intermediate = defaultdict(list)
    for doc in documents:
        for key, value in map_fn(doc):
            intermediate[key].append(value)

    # Reduce phase
    results = []
    for key, values in intermediate.items():
        results.append(reduce_fn(key, values))

    return results

# Usage
documents = [
    "hello world hello",
    "world of batch processing",
    "hello batch world"
]

word_counts = mapreduce(documents, map_function, reduce_function)
# Result: [('hello', 3), ('world', 3), ('batch', 2), ('of', 1), ('processing', 1)]
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Latency | Amortize cost over large batches | Hours to days delay |
| Throughput | Very high (process TBs/hour) | Not applicable for real-time |
| Fault tolerance | Recompute lost partitions | Entire job fails if output write fails |
| Resource usage | Can use spot instances, batch jobs | Wastes resources if data trickles in |

**When to use:** Daily/hourly reports, ETL jobs, model training, log analysis where latency doesn't matter.

**When NOT to use:** Real-time dashboards, fraud detection, alerting, user-facing features.

---

### Apache Spark Architecture

**The Problem:**
MapReduce writes intermediate results to disk between stages. For iterative algorithms (machine learning, graph processing), this is extremely slow.

**How It Works:**

Spark keeps intermediate data in memory using Resilient Distributed Datasets (RDDs).

```
Driver Program
┌────────────────────────────────────────────┐
│  SparkContext                              │
│  - Creates execution plan                 │
│  - Schedules tasks                         │
└────────────────────────────────────────────┘
         │
         ├──────────────┬──────────────┐
         ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐
    │Worker 1 │   │Worker 2 │   │Worker 3 │
    │         │   │         │   │         │
    │Executor │   │Executor │   │Executor │
    │ Cache   │   │ Cache   │   │ Cache   │
    │ Task    │   │ Task    │   │ Task    │
    └─────────┘   └─────────┘   └─────────┘
         │              │              │
         └──────────────┴──────────────┘
                        │
                        ▼
              Distributed Storage
              (HDFS, S3, etc.)
```

**Implementation (PySpark):**

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window, count, avg

# Initialize Spark
spark = SparkSession.builder \
    .appName("SalesAnalysis") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

# Read data (lazy - doesn't execute yet)
sales_df = spark.read.parquet("s3://bucket/sales/2024-01-*")

# Transformations (lazy - builds execution plan)
daily_revenue = sales_df \
    .groupBy("date", "product_id") \
    .agg(
        count("*").alias("num_orders"),
        sum("amount").alias("revenue")
    ) \
    .filter(col("revenue") > 1000) \
    .orderBy(col("revenue").desc())

# Action (triggers execution)
daily_revenue.write.parquet("s3://bucket/reports/daily_revenue")

# Caching for iterative algorithms
user_features = spark.read.parquet("s3://bucket/users")
user_features.cache()  # Keep in memory across iterations

for iteration in range(10):
    # Reuses cached data - 10x faster than re-reading from disk
    model = train_model(user_features)
    user_features = update_features(user_features, model)
```

**Key concepts:**

- **Lazy evaluation:** Build execution plan, optimize, then execute
- **Lineage:** Track transformations to recompute lost partitions
- **Caching:** Persist intermediate results in memory
- **Wide vs narrow transformations:**
  - Narrow: `map`, `filter` (no shuffle needed)
  - Wide: `groupBy`, `join` (require shuffle across network)

---

## Stream Processing Fundamentals

### Event Time vs Processing Time

**The Problem:**
Events generate at time T1, arrive at processor at time T2. Which timestamp matters?

```
Event: User clicks "buy" at 11:59:50 PM Dec 31
Network delay, server queue → Arrives 11:00:10 AM Jan 1

Question: Was this a December sale or January sale?

Processing time: January (wrong)
Event time: December (correct)
```

**Event time** = When event actually occurred (critical for correctness)
**Processing time** = When event was processed (easier to implement)

```python
class Event:
    def __init__(self, user_id, action, event_time):
        self.user_id = user_id
        self.action = action
        self.event_time = event_time  # When it happened
        self.processing_time = None    # When we processed it

    def process(self):
        self.processing_time = time.time()

# Event time matters for:
# - Financial transactions (regulation requires accurate timestamps)
# - Analytics (daily active users should count by user's timezone)
# - SLA tracking (measure from when request started, not when we saw it)
```

---

### Windowing Strategies

**The Problem:**
Streams are infinite. To compute aggregates (count, sum, average), need to group events into finite chunks.

**1. Tumbling Windows**

Fixed-size, non-overlapping windows.

```
Time:  0s  1s  2s  3s  4s  5s  6s  7s  8s  9s  10s
       ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
       │ Window 1  │ Window 2  │ Window 3  │
       │  (0-3s)   │  (3-6s)   │  (6-9s)   │
       └───────────┴───────────┴───────────┘

Events: A B   C D   E F   G H   I J
Window 1: A, B, C
Window 2: D, E, F
Window 3: G, H, I
```

**Use case:** Throughput metrics (requests per minute), hourly aggregates.

**2. Sliding Windows**

Fixed-size, overlapping windows.

```
Time:  0s  1s  2s  3s  4s  5s  6s  7s  8s
       ├───┼───┼───┼───┼───┼───┼───┼───┤
Window 1:  ├───────────┤ (0-3s)
Window 2:      ├───────────┤ (1-4s)
Window 3:          ├───────────┤ (2-5s)
Window 4:              ├───────────┤ (3-6s)

Slide: 1s, Size: 3s

Events at 1s, 2s, 3s appear in multiple windows
```

**Use case:** Moving averages, rolling metrics (CPU utilization over last 5 minutes).

**3. Session Windows**

Dynamic size based on inactivity gap.

```
User activity:
A────B───C─────────D──E────────F───G──H
│Session 1│ gap>30s │Session 2│gap>30s│Session 3│

Session 1: A, B, C (user active, then 30s gap)
Session 2: D, E (30s gap)
Session 3: F, G, H
```

**Use case:** User sessions, shopping cart analysis, clickstream tracking.

**Implementation (Tumbling Window):**

```python
from collections import defaultdict
from datetime import datetime, timedelta

class TumblingWindow:
    def __init__(self, window_size_seconds):
        self.window_size = window_size_seconds
        self.windows = defaultdict(list)  # {window_start: [events]}

    def get_window_start(self, event_time):
        """Align event to window start."""
        timestamp = event_time.timestamp()
        window_num = int(timestamp // self.window_size)
        return window_num * self.window_size

    def add_event(self, event):
        window_start = self.get_window_start(event.event_time)
        self.windows[window_start].append(event)

    def get_complete_windows(self, current_time):
        """Return windows that have closed."""
        current_window = self.get_window_start(current_time)
        complete = {}

        for window_start, events in list(self.windows.items()):
            if window_start < current_window:
                complete[window_start] = events
                del self.windows[window_start]

        return complete

# Usage
window = TumblingWindow(window_size_seconds=60)  # 1-minute windows

# Process stream
for event in event_stream:
    window.add_event(event)

    # Periodically emit completed windows
    for window_start, events in window.get_complete_windows(datetime.now()):
        count = len(events)
        print(f"Window {window_start}: {count} events")
```

---

### Watermarks and Late Data

**The Problem:**
In distributed systems, events arrive out of order. How long do you wait for late data before computing a window result?

```
Window: 10:00-10:01 (event time)

Events arrive (processing time):
10:01:05 - Event from 10:00:30 ✓ (on time)
10:01:10 - Event from 10:00:45 ✓ (on time)
10:01:15 - Event from 10:00:20 ✓ (late, but within grace period)
10:05:00 - Event from 10:00:55 ? (very late - include or drop?)
```

**Watermark:** A threshold saying "I've seen all events up to time T."

```
Watermark Strategy: event_time - 1 minute

┌─────────────────────────────────────────────┐
│         Event Time Timeline                 │
│  10:00      10:01      10:02      10:03     │
└─────────────────────────────────────────────┘

Current processing time: 10:03
Watermark: 10:02 (= 10:03 - 1 min)

Meaning: "All events before 10:02 have arrived"
         "Safe to emit results for 10:00-10:01 window"

Event from 10:01:30 arrives at 10:04
→ After watermark → Late data
```

**Handling late data:**

```python
class WatermarkedWindow:
    def __init__(self, window_size, allowed_lateness):
        self.window_size = window_size
        self.allowed_lateness = allowed_lateness
        self.windows = defaultdict(list)
        self.emitted_windows = set()

    def process_event(self, event, current_time):
        window_start = self.get_window_start(event.event_time)
        watermark = current_time - self.allowed_lateness

        # Event is too late?
        if event.event_time < watermark - self.window_size:
            return ("dropped", None)

        # Add to window
        self.windows[window_start].append(event)

        # Emit windows that passed watermark
        results = []
        for ws in list(self.windows.keys()):
            window_end = ws + self.window_size

            if window_end < watermark and ws not in self.emitted_windows:
                count = len(self.windows[ws])
                results.append((ws, count))
                self.emitted_windows.add(ws)

        return ("processed", results)

# Trade-off:
# - Small allowed_lateness: Results fast, but drop late data
# - Large allowed_lateness: Include late data, but results delayed
```

**When to use event time:**
- Financial transactions (accuracy critical)
- Analytics across timezones
- Processing historical data (backfilling)

**When to use processing time:**
- Monitoring (current system state)
- Simple aggregations where slight inaccuracy is acceptable

---

## Exactly-Once Semantics

**The Problem:**
Stream processor crashes after processing event but before marking it as consumed. After restart: Process again? Skip?

```
Event: "Transfer $100 from Account A to Account B"

At-most-once: Might lose event → Money disappears
At-least-once: Might duplicate event → $100 transferred twice
Exactly-once: Event processed exactly once → $100 transferred once
```

### Approaches

**1. Idempotent Processing**

Make duplicate processing safe.

```python
# Not idempotent
def process_transfer(event):
    account_a.balance -= event.amount
    account_b.balance += event.amount

# If processed twice: Double transfer!

# Idempotent
def process_transfer_idempotent(event):
    if event.id in processed_events:
        return  # Already done

    with database.transaction():
        account_a.balance -= event.amount
        account_b.balance += event.amount
        processed_events.add(event.id)

# If processed twice: Second time is no-op
```

**2. Transactional Processing (Kafka + Database)**

Atomically commit both consumption offset and processing result.

```python
from kafka import KafkaConsumer
from sqlalchemy import create_engine

consumer = KafkaConsumer('transfers',
                         enable_auto_commit=False,
                         isolation_level='read_committed')

engine = create_engine('postgresql://...')

for message in consumer:
    event = parse(message.value)

    with engine.begin() as conn:
        # Process event
        conn.execute(
            "UPDATE accounts SET balance = balance - %s WHERE id = %s",
            (event.amount, event.from_account)
        )
        conn.execute(
            "UPDATE accounts SET balance = balance + %s WHERE id = %s",
            (event.amount, event.to_account)
        )

        # Store offset in same transaction
        conn.execute(
            "INSERT INTO kafka_offsets (topic, partition, offset) VALUES (%s, %s, %s) "
            "ON CONFLICT (topic, partition) DO UPDATE SET offset = %s",
            ('transfers', message.partition, message.offset, message.offset)
        )

        # If commit succeeds: Both processing and offset stored
        # If commit fails: Neither stored, message reprocessed
```

**3. Two-Phase Commit (Flink Checkpointing)**

Periodically snapshot entire processing state.

```
Time:  0s      5s      10s     15s (crash)   20s (restart)
       │       │       │       │             │
       ├───────┴───────┴───────┤             │
    Checkpoint 1   Checkpoint 2              │
                       │                     │
                       └─────────────────────┘
                       Restore from Checkpoint 2
                       Replay events since 10s
```

---

## Architecture Patterns

### Lambda Architecture

**The Problem:**
Batch processing is accurate but slow. Stream processing is fast but complex. Can we have both?

```
                        ┌─────────────────┐
                        │   Data Source   │
                        │  (Kafka, etc.)  │
                        └─────────────────┘
                         ╱               ╲
                        ╱                 ╲
                       ▼                   ▼
           ┌────────────────────┐  ┌──────────────────┐
           │   Batch Layer      │  │   Speed Layer    │
           │   (Spark)          │  │   (Flink)        │
           │                    │  │                  │
           │ - Complete data    │  │ - Recent data    │
           │ - High latency     │  │ - Low latency    │
           │ - Accurate         │  │ - Approximate    │
           └────────────────────┘  └──────────────────┘
                       │                   │
                       ▼                   ▼
           ┌────────────────────┐  ┌──────────────────┐
           │  Batch Views       │  │  Real-time Views │
           │  (Hive, Parquet)   │  │  (Redis, Druid)  │
           └────────────────────┘  └──────────────────┘
                       ╲                   ╱
                        ╲                 ╱
                         ▼               ▼
                        ┌─────────────────┐
                        │  Serving Layer  │
                        │  (Merge views)  │
                        └─────────────────┘
```

**How it works:**

```python
def query_revenue(date, product_id):
    # Get batch view (complete, historical)
    batch_revenue = query_batch_view(
        "SELECT SUM(revenue) FROM daily_revenue "
        "WHERE date < today() AND product_id = %s",
        product_id
    )

    # Get real-time view (incomplete, recent)
    realtime_revenue = query_realtime_view(
        "GET revenue:today:{product_id}"
    )

    # Merge
    total = batch_revenue + realtime_revenue
    return total

# Batch layer recomputes every 24 hours
# Speed layer maintains last 24 hours
# Query time: Merge both
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Accuracy | Batch layer is source of truth | Must maintain two codebases |
| Latency | Speed layer provides real-time results | Complex merge logic at query time |
| Operational | Batch jobs are easier to debug | Doubled infrastructure cost |

**When to use:** Data analytics platforms where you need both historical accuracy and real-time updates.

**When NOT to use:** When you can achieve latency goals with pure stream or pure batch.

---

### Kappa Architecture

**The Problem:**
Lambda architecture requires maintaining two processing pipelines. Can we use only stream processing?

```
                        ┌─────────────────┐
                        │   Data Source   │
                        │  (Kafka)        │
                        └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │ Stream Layer    │
                        │ (Flink)         │
                        │                 │
                        │ - All data      │
                        │ - Low latency   │
                        └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │  Serving Layer  │
                        │  (DB, Cache)    │
                        └─────────────────┘

Historical reprocessing:
- Keep events in Kafka (long retention)
- Deploy new version of stream processor
- Replay from beginning
```

**When to use:**
- Stream processing latency is acceptable for all use cases
- Can afford Kafka storage for full history
- Want single codebase

**When NOT to use:**
- Need complex batch algorithms (ML training, graph analysis)
- Can't afford stream processing infrastructure for all data

---

## Stream Processing Frameworks Comparison

### Apache Kafka Streams

**Architecture:** Library (not framework), runs in your application.

```python
# Note: Python pseudocode (actual Kafka Streams is Java)
from kafka import KafkaConsumer, KafkaProducer

consumer = KafkaConsumer('page-views')
producer = KafkaProducer()

# State store (local RocksDB)
user_counts = {}

for message in consumer:
    user_id = message.value['user_id']

    # Update state
    user_counts[user_id] = user_counts.get(user_id, 0) + 1

    # Emit result
    producer.send('page-view-counts', {
        'user_id': user_id,
        'count': user_counts[user_id]
    })
```

**Characteristics:**
- No separate cluster (deploys with your app)
- Exactly-once via Kafka transactions
- State stored in local RocksDB + changelog topic
- Scales by adding more app instances

---

### Apache Flink

**Architecture:** Separate cluster with JobManager and TaskManagers.

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import MapFunction, KeyedProcessFunction

env = StreamExecutionEnvironment.get_execution_environment()
env.enable_checkpointing(60000)  # Checkpoint every 60s

# Define stream
events = env.add_source(KafkaSource(...))

# Transformations
results = events \
    .map(lambda e: (e['user_id'], 1)) \
    .key_by(lambda x: x[0]) \
    .sum(1) \
    .add_sink(KafkaSink(...))

env.execute("PageViewCount")
```

**Characteristics:**
- True streaming (not micro-batches)
- Very low latency (milliseconds)
- Advanced features: event time, watermarks, state
- Exactly-once via distributed snapshots
- Complex to operate

---

### Apache Spark Streaming (Structured Streaming)

**Architecture:** Micro-batch processing (treats stream as incremental batch).

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import window, col

spark = SparkSession.builder.appName("PageViews").getOrCreate()

# Read stream
events = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "page-views") \
    .load()

# Windowed aggregation
windowed_counts = events \
    .groupBy(
        window(col("timestamp"), "1 minute"),
        col("user_id")
    ) \
    .count()

# Write stream
query = windowed_counts.writeStream \
    .outputMode("update") \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "page-view-counts") \
    .option("checkpointLocation", "/tmp/checkpoint") \
    .start()

query.awaitTermination()
```

**Characteristics:**
- Micro-batch (100ms+ latency)
- Same API as batch Spark (easy to learn)
- Exactly-once via write-ahead log + checkpoints
- Good for ~1 second latency requirements

---

## Technology Comparison

| Feature | Spark | Flink | Kafka Streams | Beam |
|---------|-------|-------|---------------|------|
| **Processing Model** | Micro-batch | True streaming | True streaming | Abstraction layer |
| **Latency** | 100ms - 1s | Sub-second | Sub-second | Depends on runner |
| **Throughput** | Very high | High | Medium | Depends on runner |
| **Exactly-once** | Yes | Yes | Yes | Depends on runner |
| **State Management** | In-memory + checkpoints | RocksDB + snapshots | RocksDB + changelog | Depends on runner |
| **Deployment** | Cluster (YARN, K8s) | Cluster (K8s, standalone) | Library (no cluster) | Depends on runner |
| **Operational Complexity** | Medium | High | Low | Medium |
| **Event Time** | Yes | Yes (best-in-class) | Yes | Yes |
| **SQL Support** | Excellent | Good | KSQL (separate) | Basic |
| **Best For** | Batch + Stream hybrid | Low-latency streaming | Microservices + streaming | Multi-cloud portability |

**Choosing a framework:**

```
Need < 100ms latency? → Flink
Already using Spark for batch? → Spark Structured Streaming
Simple stateful processing in microservices? → Kafka Streams
Want cloud portability? → Beam (runs on Dataflow, Flink, Spark)
```

---

## Checkpointing and Fault Tolerance

**The Problem:**
Stream processor crashes. How do you resume without losing data or duplicating processing?

### Flink's Checkpoint Mechanism

```
Processing pipeline:
Source → Transform → Sink

Checkpoint process:
1. JobManager triggers checkpoint N
2. Source marks current position (Kafka offset)
3. Each operator saves state to durable storage
4. Sink saves output position
5. Checkpoint N complete

Crash recovery:
1. Restore state from latest checkpoint N
2. Rewind source to checkpoint N offset
3. Replay events from checkpoint to now
```

**Implementation:**

```python
from pyflink.datastream import StreamExecutionEnvironment

env = StreamExecutionEnvironment.get_execution_environment()

# Checkpoint configuration
env.enable_checkpointing(60000)  # Every 60 seconds
env.get_checkpoint_config().set_checkpoint_storage_dir("s3://bucket/checkpoints")
env.get_checkpoint_config().set_min_pause_between_checkpoints(30000)
env.get_checkpoint_config().set_checkpoint_timeout(600000)

# Failure handling
env.set_restart_strategy(
    RestartStrategies.fixed_delay_restart(
        3,      # Try 3 times
        10000   # Wait 10s between attempts
    )
)
```

**Checkpoint tuning:**

```
Checkpoint interval:
- Too frequent: High overhead, slow processing
- Too infrequent: Long recovery time after failure
- Rule of thumb: 1-5 minutes for most workloads

Checkpoint size:
- Large state → Long checkpoint time
- Solution: Incremental checkpoints (only changed state)

State backend:
- MemoryStateBackend: Fast, limited by heap
- FsStateBackend: Slower, no size limit
- RocksDBStateBackend: Largest datasets, disk-based
```

---

## State Management

**The Problem:**
Stateless processing is easy (map, filter). Stateful processing (count, join, sessionize) needs to store data across events.

### State Types

**1. Keyed State**

State partitioned by key (user_id, session_id).

```python
from pyflink.datastream.functions import KeyedProcessFunction, RuntimeContext

class UserActivityCounter(KeyedProcessFunction):
    def open(self, runtime_context: RuntimeContext):
        # Create state descriptor
        self.count_state = runtime_context.get_state(
            ValueStateDescriptor("count", Types.INT())
        )

    def process_element(self, event, ctx):
        # Read current count for this key
        current_count = self.count_state.value()
        if current_count is None:
            current_count = 0

        # Update
        current_count += 1
        self.count_state.update(current_count)

        # Emit
        yield (event.user_id, current_count)
```

**2. Operator State**

State shared across all keys (useful for source/sink offsets).

```python
class KafkaSourceWithCheckpoint:
    def __init__(self):
        self.offset_state = None

    def init_state(self, context):
        # Restore offset from checkpoint
        self.offset_state = context.get_operator_state("kafka_offset")
        self.current_offset = self.offset_state.value() or 0

    def next_element(self):
        message = kafka.poll(self.current_offset)
        self.current_offset += 1
        return message

    def snapshot_state(self, checkpoint_id):
        # Save offset to checkpoint
        self.offset_state.update(self.current_offset)
```

**State size management:**

```python
# Problem: State grows unbounded
user_events = {}  # Will eventually OOM

# Solution 1: TTL (Time To Live)
from pyflink.datastream.state import StateTtlConfig

ttl_config = StateTtlConfig.new_builder(Time.days(7)) \
    .set_update_type(StateTtlConfig.UpdateType.OnCreateAndWrite) \
    .set_state_visibility(StateTtlConfig.StateVisibility.NeverReturnExpired) \
    .build()

state_descriptor = ValueStateDescriptor("events", Types.LIST(Types.STRING()))
state_descriptor.enable_time_to_live(ttl_config)

# Solution 2: Manual cleanup
class StatefulProcessor(KeyedProcessFunction):
    def process_element(self, event, ctx):
        # Register timer to cleanup old data
        ctx.timer_service().register_event_time_timer(
            event.timestamp + timedelta(days=7)
        )

    def on_timer(self, timestamp, ctx):
        # Cleanup state older than 7 days
        self.state.clear()
```

---

## Key Concepts Checklist

- [ ] Understand when to use batch vs stream processing
- [ ] Explain MapReduce model and Spark's improvements (in-memory, lazy evaluation)
- [ ] Describe windowing strategies (tumbling, sliding, session)
- [ ] Explain event time vs processing time and watermarks
- [ ] Compare exactly-once approaches (idempotency, transactions, checkpointing)
- [ ] Contrast Lambda vs Kappa architecture
- [ ] Compare Spark vs Flink vs Kafka Streams for different use cases
- [ ] Design checkpointing strategy for fault tolerance
- [ ] Manage state size and lifecycle in stream processing

---

## Practical Insights

**Batch vs stream is a spectrum, not binary:**
- Micro-batching (Spark): Batch every 100ms
- True streaming (Flink): Process each event individually
- Hybrid (Lambda): Both batch and stream
- Choose based on latency requirements, not ideology

**The event time/watermark trap:**
```
You implement event time processing.
Events arrive out of order by hours (mobile devices offline).
Watermark strategy: Allow 1-hour lateness.
Result: All windows delayed by 1 hour → Defeats purpose of streaming.

Solution:
- Use processing time for real-time dashboards
- Use event time for accurate financial/analytics
- Separate hot path (recent, fast) from cold path (historical, accurate)
```

**State is the hardest part:**
- State grows unbounded → OOM
- State takes forever to checkpoint → Processing stalls
- State gets corrupted → Silent data errors
- Monitor state size per key. Alert on outliers.
- Use TTL aggressively. Most stream analytics don't need infinite history.
- Test recovery: Kill random workers, verify results are correct.

**Backpressure and resource tuning:**
```
Source produces 100k events/sec
Processor handles 50k events/sec
Result: Unbounded growth in queues → OOM

Solutions:
1. Scale out: More parallel instances
2. Scale up: More CPU/memory per instance
3. Backpressure: Slow down source (Kafka consumer lag)
4. Load shedding: Drop low-priority events

Monitor:
- Consumer lag (source can't keep up)
- Task backpressure time (operator can't keep up)
- Checkpoint duration (state too large)
- GC time (heap pressure)
```

**Choosing checkpoint intervals:**
```
Checkpoint overhead:
- Pause processing: 100ms - 5s (depends on state size)
- Write to S3/HDFS: Network bandwidth cost

Recovery time:
- Restore from checkpoint: 1-10 minutes
- Replay events since checkpoint: Proportional to interval

If checkpoint every 1 minute:
- Overhead: 100ms / 60s = 0.16% of CPU
- Recovery: Replay 1 minute of data

If checkpoint every 30 minutes:
- Overhead: 100ms / 1800s = 0.005% of CPU (better)
- Recovery: Replay 30 minutes of data (worse)

Rule of thumb:
- High-throughput: 5-10 minute checkpoints
- Low-throughput: 1-2 minute checkpoints
- Critical systems: 30-60 second checkpoints + incremental mode
```

**When NOT to use stream processing:**
- ML model training (need full dataset, iterative, batch is better)
- Complex joins across many datasets (batch SQL is simpler)
- Ad-hoc analysis (analysts querying data warehouse)
- When you can tolerate daily/hourly latency (batch is cheaper and simpler)

**Kafka retention for Kappa architecture:**
```
1 TB/day ingestion
7-day retention = 7 TB storage
30-day retention = 30 TB storage

Kafka storage is cheap (replication factor 3 = 90 TB total).
But: Reprocessing 30 days takes 30x longer than 1 day.

Strategy:
- Hot tier (Kafka): 7-14 days, fast reprocessing
- Cold tier (S3/HDFS): Infinite, batch reprocessing
- Use Kafka for bug fixes, S3 for major rewrites
```
