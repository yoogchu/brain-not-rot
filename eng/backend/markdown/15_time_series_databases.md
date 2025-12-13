# Chapter 15: Time-Series Databases

## The Problem

Your IoT platform monitors 50,000 smart thermostats. Each device sends:

```
Temperature, humidity, battery level every 60 seconds
= 3 metrics × 50,000 devices × 60 readings/hour
= 9 million data points per hour
= 216 million per day
```

After 3 months using PostgreSQL:

```
SELECT avg(temperature)
FROM readings
WHERE device_id = 'device_123'
  AND timestamp > NOW() - INTERVAL '7 days';

Query time: 45 seconds (table scan of 19 billion rows)
Disk usage: 4.2 TB (mostly indexes)
Write throughput: Degrading (500 inserts/sec → 150 inserts/sec)
```

This is time-series data. Regular databases were not built for this.

---

## What Makes Time-Series Data Special?

Time-series data has unique characteristics:

**1. Append-only writes**
- Data arrives in time order
- Never update past values (immutable)
- DELETE is rare (retention policies, not random deletes)

**2. Time-ordered queries**
- Almost all queries have time range filters
- Recent data accessed far more than old data
- Aggregations over time windows (hourly, daily averages)

**3. High cardinality**
- Many unique series (devices × metrics)
- Each series = unique combination of tags/labels
- Example: `temperature{device="dev_123", room="bedroom", floor="2"}`

**4. High write volume, lower read volume**
- Millions of writes per second
- Reads are analytical (dashboards, alerts)

Regular databases (PostgreSQL, MySQL) optimize for:
- Random reads/writes
- Updates in place
- Complex joins
- ACID transactions

Time-series databases optimize for:
- Sequential writes
- Immutable data
- Efficient time-range scans
- Compression

---

## Storage Optimization

### Columnar Storage

**The Problem:**
Row-oriented storage wastes I/O when querying single columns.

```
Row-oriented (traditional):
┌───────────────────────────────────────────────────┐
│ Row 1: [timestamp=T1, device=D1, temp=20, hum=45] │
│ Row 2: [timestamp=T2, device=D1, temp=21, hum=46] │
│ Row 3: [timestamp=T3, device=D2, temp=19, hum=50] │
└───────────────────────────────────────────────────┘

Query: SELECT avg(temp) FROM readings
Must read ALL columns, even though only need temp
```

**Columnar storage:**

```
┌──────────────────┐ ┌──────────────┐ ┌──────────┐
│ timestamp        │ │ device       │ │ temp     │
├──────────────────┤ ├──────────────┤ ├──────────┤
│ T1               │ │ D1           │ │ 20       │
│ T2               │ │ D1           │ │ 21       │
│ T3               │ │ D2           │ │ 19       │
└──────────────────┘ └──────────────┘ └──────────┘

Query: SELECT avg(temp)
Only read temp column → Much less I/O
```

### Compression Techniques

**Delta-of-delta encoding:**

Temperature readings are usually close to previous values.

```python
# Raw values (4 bytes per value)
timestamps = [1609459200, 1609459260, 1609459320, 1609459380]

# Delta (difference from previous)
deltas = [1609459200, 60, 60, 60]

# Delta-of-delta (difference of deltas)
delta_of_deltas = [1609459200, 60, 0, 0, 0]
# Store base, first delta, then mostly zeros
# Zeros compress extremely well
```

**Result:** 96% compression ratio

**Gorilla compression (Facebook's algorithm):**

Used by Prometheus, InfluxDB for floats.

```python
# Temperature readings
values = [20.5, 20.6, 20.5, 20.7, 20.6]

# Store first value as-is: 20.5
# For subsequent values, XOR with previous:
#   20.6 XOR 20.5 → Small number (similar bits)
#   Store only the differing bits

# Typical compression: 1.37 bytes per data point
# (vs 8 bytes for float64)
```

**Run-length encoding:**

For repeated values.

```python
status = [1, 1, 1, 1, 1, 0, 0, 0, 1, 1]

# Store as: [(1, 5), (0, 3), (1, 2)]
# (value, count) pairs
```

---

## Downsampling and Retention

**The Problem:**
Storing raw data forever is expensive and unnecessary.

```
Raw data (1-second resolution):
  Day 1-7: Keep for detailed debugging
  = 7 days × 86,400 seconds = 604,800 points

After 7 days:
  Hourly averages sufficient
  = 24 hours × 365 days = 8,760 points

After 1 year:
  Daily aggregates sufficient
  = 365 points
```

### Continuous Aggregation

```python
# InfluxDB retention policies and continuous queries
CREATE RETENTION POLICY "raw" ON "sensors"
  DURATION 7d
  REPLICATION 1
  DEFAULT

CREATE RETENTION POLICY "hourly" ON "sensors"
  DURATION 90d
  REPLICATION 1

CREATE RETENTION POLICY "daily" ON "sensors"
  DURATION 5y
  REPLICATION 1

# Auto-downsample raw → hourly
CREATE CONTINUOUS QUERY "downsample_hourly" ON "sensors"
BEGIN
  SELECT mean(temperature) AS temperature_mean,
         max(temperature) AS temperature_max,
         min(temperature) AS temperature_min
  INTO "hourly"."temperature_hourly"
  FROM "raw"."temperature"
  GROUP BY time(1h), device_id
END
```

### Python Implementation

```python
import time
from datetime import datetime, timedelta

class RetentionManager:
    def __init__(self, db):
        self.db = db
        self.policies = [
            {"name": "raw", "duration_days": 7, "resolution": "1m"},
            {"name": "hourly", "duration_days": 90, "resolution": "1h"},
            {"name": "daily", "duration_days": 1825, "resolution": "1d"},
        ]

    def downsample_hourly(self):
        """Aggregate raw data into hourly buckets."""
        cutoff = datetime.now() - timedelta(hours=1)

        # Aggregate last hour of raw data
        query = """
        INSERT INTO metrics_hourly (timestamp, device_id, avg_temp, max_temp, min_temp)
        SELECT
            date_trunc('hour', timestamp) as hour,
            device_id,
            avg(temperature) as avg_temp,
            max(temperature) as max_temp,
            min(temperature) as min_temp
        FROM metrics_raw
        WHERE timestamp >= %s AND timestamp < %s
        GROUP BY hour, device_id
        """
        self.db.execute(query, (cutoff - timedelta(hours=1), cutoff))

    def expire_old_data(self):
        """Delete data outside retention windows."""
        for policy in self.policies:
            cutoff = datetime.now() - timedelta(days=policy["duration_days"])
            table = f"metrics_{policy['name']}"

            deleted = self.db.execute(
                f"DELETE FROM {table} WHERE timestamp < %s",
                (cutoff,)
            )
            print(f"Expired {deleted} rows from {table}")

# Run as cron job
manager = RetentionManager(db)
manager.downsample_hourly()  # Every hour
manager.expire_old_data()    # Daily
```

---

## InfluxDB Architecture

### Data Model: Tags vs Fields

```
measurement: The "table" (e.g., temperature)
tags: Indexed metadata (device_id, location, room)
fields: Actual values (temperature, humidity)
timestamp: Nanosecond precision
```

**Example:**

```
temperature,device=dev_123,room=bedroom,floor=2 value=20.5 1609459200000000000
│           │                                  │          │
│           └─ tags (indexed)                 │          └─ timestamp (ns)
│                                              └─ fields (not indexed)
└─ measurement
```

### Write Path

```
┌──────────────────────────────────────────────────┐
│                Write Request                      │
└──────────────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Write Ahead Log     │  ← Durability
         │   (WAL)               │
         └───────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   In-Memory Cache     │  ← Fast writes
         │   (time-ordered)      │
         └───────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Flush to TSM Files  │  ← Compressed columnar
         │   (Time-Structured    │     storage on disk
         │    Merge tree)        │
         └───────────────────────┘
```

### Line Protocol

```python
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

client = InfluxDBClient(url="http://localhost:8086", token="my-token", org="my-org")
write_api = client.write_api(write_options=SYNCHRONOUS)

# Write single point
point = Point("temperature") \
    .tag("device", "dev_123") \
    .tag("room", "bedroom") \
    .field("value", 20.5) \
    .time(datetime.utcnow())

write_api.write(bucket="sensors", record=point)

# Batch write (more efficient)
points = []
for i in range(1000):
    points.append(
        Point("temperature")
        .tag("device", f"dev_{i}")
        .tag("room", "bedroom")
        .field("value", 20.0 + i * 0.1)
        .time(datetime.utcnow())
    )

write_api.write(bucket="sensors", record=points)

# Query with Flux
query = '''
from(bucket: "sensors")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "temperature")
  |> filter(fn: (r) => r.device == "dev_123")
  |> mean()
'''

tables = client.query_api().query(query)
for table in tables:
    for record in table.records:
        print(f"Average temp: {record.get_value()}")
```

---

## TimescaleDB: SQL for Time-Series

TimescaleDB is a PostgreSQL extension. You get SQL + time-series optimizations.

### Hypertables

**Problem:** Single PostgreSQL table becomes slow with billions of rows.

**Solution:** Automatic partitioning by time (chunks).

```
┌─────────────────────────────────────────────────┐
│           Hypertable: sensor_data               │
│         (appears as single table to user)       │
└─────────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Chunk 1      │ │ Chunk 2      │ │ Chunk 3      │
│ Jan 1-7      │ │ Jan 8-14     │ │ Jan 15-21    │
│              │ │              │ │              │
│ Regular PG   │ │ Regular PG   │ │ Regular PG   │
│ table        │ │ table        │ │ table        │
└──────────────┘ └──────────────┘ └──────────────┘
```

**Query optimization:**

```sql
SELECT avg(temperature)
FROM sensor_data
WHERE timestamp > NOW() - INTERVAL '7 days';

-- TimescaleDB only scans relevant chunks (1 week)
-- Not entire table
```

### Setup and Usage

```python
import psycopg2
from datetime import datetime, timedelta

conn = psycopg2.connect("dbname=metrics user=postgres")
cur = conn.cursor()

# Create hypertable
cur.execute("""
    CREATE TABLE sensor_data (
        timestamp TIMESTAMPTZ NOT NULL,
        device_id TEXT NOT NULL,
        temperature DOUBLE PRECISION,
        humidity DOUBLE PRECISION
    );
""")

cur.execute("SELECT create_hypertable('sensor_data', 'timestamp');")

# Insert data (same as regular PostgreSQL)
cur.execute("""
    INSERT INTO sensor_data (timestamp, device_id, temperature, humidity)
    VALUES (%s, %s, %s, %s)
""", (datetime.now(), 'dev_123', 20.5, 45.2))

# Continuous aggregates (materialized views)
cur.execute("""
    CREATE MATERIALIZED VIEW sensor_data_hourly
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 hour', timestamp) AS hour,
        device_id,
        avg(temperature) AS avg_temp,
        max(temperature) AS max_temp,
        min(temperature) AS min_temp
    FROM sensor_data
    GROUP BY hour, device_id;
""")

# Refresh policy (auto-update)
cur.execute("""
    SELECT add_continuous_aggregate_policy('sensor_data_hourly',
        start_offset => INTERVAL '1 day',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
""")

# Retention policy (auto-delete old data)
cur.execute("""
    SELECT add_retention_policy('sensor_data', INTERVAL '90 days');
""")

# Query (standard SQL)
cur.execute("""
    SELECT hour, avg_temp
    FROM sensor_data_hourly
    WHERE device_id = 'dev_123'
      AND hour > NOW() - INTERVAL '7 days'
    ORDER BY hour;
""")

rows = cur.fetchall()
for row in rows:
    print(f"{row[0]}: {row[1]}°C")

conn.commit()
cur.close()
conn.close()
```

**When to use TimescaleDB:**
- You already use PostgreSQL
- Need SQL and existing PostgreSQL ecosystem (pgAdmin, etc.)
- Want joins with relational data
- Need ACID transactions

**When NOT to use:**
- Pure time-series workload (InfluxDB may be faster)
- Don't need SQL (InfluxDB's Flux is simpler for time-series)

---

## Prometheus: Metrics Monitoring

Prometheus is designed for monitoring infrastructure and applications.

### Data Model

```
metric_name{label1="value1", label2="value2"} value timestamp

http_requests_total{method="GET", endpoint="/api/users", status="200"} 1234 1609459200
│                   │                                                 │    │
│                   └─ labels (high cardinality)                     │    └─ timestamp
│                                                                     └─ counter value
└─ metric name
```

### Metric Types

**Counter:** Only increases (resets on restart)

```python
from prometheus_client import Counter

http_requests = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])

# Increment
http_requests.labels(method='GET', endpoint='/api/users').inc()
http_requests.labels(method='POST', endpoint='/api/users').inc()
```

**Gauge:** Can go up or down

```python
from prometheus_client import Gauge

temperature = Gauge('room_temperature_celsius', 'Room temperature', ['room'])

temperature.labels(room='bedroom').set(20.5)
temperature.labels(room='kitchen').set(22.3)
```

**Histogram:** Buckets for distributions

```python
from prometheus_client import Histogram

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Observe values
with request_duration.labels(method='GET', endpoint='/api/users').time():
    # Process request
    process_request()

# Generates:
# http_request_duration_seconds_bucket{le="0.1"} 100
# http_request_duration_seconds_bucket{le="0.5"} 250
# http_request_duration_seconds_bucket{le="1.0"} 400
# ...
```

### PromQL (Query Language)

```promql
# Rate of HTTP requests per second (last 5 minutes)
rate(http_requests_total[5m])

# 95th percentile request duration
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# CPU usage by instance
sum by (instance) (rate(cpu_usage_seconds_total[5m]))

# Alert: High error rate
rate(http_requests_total{status=~"5.."}[5m]) > 0.05
```

### Pull vs Push

Prometheus **pulls** metrics from targets:

```
┌───────────────────┐
│   Prometheus      │
│   Server          │
└───────────────────┘
         │
         │ HTTP GET /metrics (every 15s)
         │
         ▼
┌───────────────────┐
│   Application     │
│   :9090/metrics   │ ← Exposes metrics endpoint
└───────────────────┘
```

**Why pull?**
- Prometheus controls scrape frequency
- Service discovery (auto-find targets)
- Detect when target is down

**For short-lived jobs** (batch, cron):
Use Pushgateway to push metrics.

---

## Cardinality Explosion

**The Silent Killer of TSDBs**

Cardinality = Number of unique time series.

```
Metric: http_requests_total
Labels: method, endpoint, status, user_id

If you have:
- 5 methods (GET, POST, PUT, DELETE, PATCH)
- 100 endpoints
- 5 status codes (200, 400, 404, 500, 503)
- 10,000 users

Cardinality = 5 × 100 × 5 × 10,000 = 25,000,000 unique series!
```

**Problem:** Each series uses memory for indexing.

```
InfluxDB: ~1-3 KB per series in memory
25M series × 2KB = 50 GB RAM just for index!
```

### Anti-Pattern: User ID in Labels

```python
# WRONG: Don't do this
http_requests.labels(
    method='GET',
    endpoint='/api/users',
    user_id='user_12345'  # ← DANGER: Unbounded cardinality
).inc()

# With 1M users = 1M unique series
```

### Correct Pattern: Use Fields, Not Tags

```python
# InfluxDB: Use fields for high-cardinality data
point = Point("http_request") \
    .tag("method", "GET") \
    .tag("endpoint", "/api/users") \
    .field("user_id", "user_12345") \  # ← Field, not tag
    .field("duration_ms", 45)

# Prometheus: Aggregate at query time
# Don't use user_id as label
# Instead: Track total requests, filter logs for specific users
```

### Monitoring Cardinality

```python
# InfluxDB: Check series cardinality
SHOW SERIES CARDINALITY

# Prometheus: Check metrics
curl http://localhost:9090/api/v1/status/tsdb

# Alert on high cardinality
{
  "metric": "prometheus_tsdb_symbol_table_size_bytes",
  "alert_threshold": 1000000  # 1M series
}
```

**Rules of thumb:**
- Cardinality per metric < 100,000
- Total cardinality < 10 million
- If you need user-level data, use logs or separate analytics DB

---

## Comparison Table

| Feature | InfluxDB | TimescaleDB | Prometheus | ClickHouse | QuestDB |
|---------|----------|-------------|------------|------------|---------|
| **Query Language** | Flux, InfluxQL | SQL | PromQL | SQL | SQL |
| **Storage** | TSM (custom) | PostgreSQL tables | Custom TSDB | Columnar | Columnar |
| **Compression** | Gorilla, delta-of-delta | PostgreSQL + custom | Gorilla | LZ4, ZSTD | LZ4 |
| **Max Cardinality** | ~10M series | ~100M series | ~10M series | Billions | ~10M series |
| **Write Throughput** | 500K points/sec | 150K rows/sec | 1M samples/sec | 1M+ rows/sec | 1.4M rows/sec |
| **SQL Support** | No (Flux) | Full PostgreSQL | No (PromQL) | Full SQL | Full SQL |
| **Best For** | General time-series | SQL + time-series hybrid | Metrics monitoring | Analytics, high cardinality | Financial, IoT |
| **Clustering** | Enterprise only | Timescale Cloud | Federation | Built-in | Community (limited) |
| **Retention Policies** | Built-in | Built-in | Built-in | TTL | Manual |
| **Learning Curve** | Medium | Low (if know SQL) | Medium | Medium | Low |

**InfluxDB:**
- Use for: General-purpose time-series, IoT, sensor data
- Avoid for: Requiring SQL, ultra-high cardinality

**TimescaleDB:**
- Use for: Need SQL, join with relational data, existing PostgreSQL stack
- Avoid for: Pure time-series (InfluxDB may be faster)

**Prometheus:**
- Use for: Monitoring infrastructure, Kubernetes, microservices
- Avoid for: Long-term storage (use Thanos/Cortex), general time-series

**ClickHouse:**
- Use for: Analytics, high cardinality, complex aggregations
- Avoid for: Simple time-series (overkill)

**QuestDB:**
- Use for: Financial data (tick data), ultra-fast ingestion
- Avoid for: Complex queries, need mature ecosystem

---

## Key Concepts Checklist

- [ ] Explain time-series data characteristics (append-only, time-ordered, high cardinality)
- [ ] Describe columnar storage and compression techniques (delta-of-delta, Gorilla)
- [ ] Design retention policies and downsampling strategies
- [ ] Compare InfluxDB tags vs fields data model
- [ ] Explain TimescaleDB hypertables and continuous aggregates
- [ ] Describe Prometheus metric types (counter, gauge, histogram)
- [ ] Identify cardinality explosion risks and mitigation
- [ ] Choose appropriate TSDB for use case (InfluxDB vs TimescaleDB vs Prometheus)

---

## Practical Insights

**Cardinality is your enemy:**

Monitor series count religiously. A single bad metric with unbounded labels (user IDs, request IDs, IP addresses) can bring down your TSDB. Use fields/values, not tags/labels, for high-cardinality data. In Prometheus, if you need per-user metrics, you're using the wrong tool—use application logs or a dedicated analytics database instead.

**Retention strategy from day one:**

```
Raw data: 7-30 days (debugging, incident response)
Hourly rollups: 90 days (trend analysis)
Daily rollups: 2-5 years (long-term trends, capacity planning)
```

Don't wait until you have 10 TB of data to implement retention. Disk is cheap, but TSDB performance degrades with dataset size. Delete old data proactively.

**Batch writes, not individual:**

```python
# Slow: 1,000 individual writes = 1,000 network round-trips
for reading in readings:
    influx.write(reading)

# Fast: 1 batch write = 1 network round-trip
influx.write_batch(readings)  # 10-100x faster
```

Batch size sweet spot: 5,000-10,000 points per batch. Larger = risk of timeout/memory issues.

**Pre-aggregate when possible:**

Don't store every single data point if you don't need it. If you only ever query hourly averages, store hourly averages. Raw data is useful for debugging, but 99% of queries are aggregates. Use continuous aggregates (TimescaleDB) or recording rules (Prometheus).

**Know when NOT to use a TSDB:**

TSDBs are optimized for time-series queries. If you need:
- Complex joins across multiple entities → Use PostgreSQL/MySQL
- Full-text search → Use Elasticsearch
- User behavior analytics → Use ClickHouse or data warehouse
- Real-time alerting on complex patterns → Use stream processor (Flink)

TSDBs are great at what they do, but they're not a universal solution. Many production systems use TSDB for metrics + PostgreSQL for metadata + S3 for raw event logs.

**Monitoring the monitoring system:**

Your TSDB is critical infrastructure. Monitor:
```
- Write throughput (points/sec)
- Query latency (p95, p99)
- Disk usage growth rate
- Series cardinality
- Memory usage (index size)
```

Alert on degradation before it becomes an outage. A slow TSDB means blind operators during an incident—the worst possible time to lose observability.
