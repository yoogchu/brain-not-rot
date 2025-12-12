# Chapter 14: Database Internals

## Why Understanding Internals Matters

```
Scenario: Query runs in 50ms in dev, 30 seconds in production

Without internals knowledge:
"It's slow. Add more servers?"

With internals knowledge:
"Query isn't using the index. B-tree lookup would be O(log n),
but it's doing full table scan O(n). The index exists but
statistics are stale. Run ANALYZE."

Problem solved in 5 minutes, not 5 days.
```

---

## Storage Engines: B-Trees vs LSM-Trees

### B-Trees (PostgreSQL, MySQL InnoDB, SQLite)

**Structure:**
```
                    ┌─────────────────────────┐
                    │   Root Node             │
                    │  [10 | 20 | 30 | 40]    │
                    └─────────────────────────┘
                     /    |      |      \
                    /     |      |       \
         ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
         │ <10     │ │ 10-20   │ │ 20-30   │ │ >40     │
         │ [3,7,9] │ │[12,15,18]│ │[22,25,28]│ │[45,50,55]│
         └─────────┘ └─────────┘ └─────────┘ └─────────┘
              |           |           |            |
              ▼           ▼           ▼            ▼
          ┌──────┐   ┌──────┐   ┌──────┐    ┌──────┐
          │ Leaf │   │ Leaf │   │ Leaf │    │ Leaf │
          │ Data │   │ Data │   │ Data │    │ Data │
          └──────┘   └──────┘   └──────┘    └──────┘

Lookup: id = 25
1. Root: 25 > 20, < 30 → middle-right branch
2. Branch: Find 25 in [22,25,28]
3. Leaf: Read row data

O(log n) lookups - typically 3-4 disk reads
```

**B-Tree Write Process:**
```
INSERT id=23

1. Find correct leaf node
2. If leaf has space: Insert in sorted order
3. If leaf full: Split into two nodes
   - Update parent pointers
   - May cascade splits up the tree
4. Write to disk (random I/O)
```

**B-Tree Characteristics:**
- Read-optimized (one location per key)
- Random I/O on writes
- In-place updates
- Good for read-heavy OLTP

### LSM-Trees (Cassandra, LevelDB, RocksDB, ScyllaDB)

**Structure:**
```
Write Path:
                                    
New Write ──► Memtable (in memory, sorted)
                   │
                   │ (when memtable full)
                   ▼
              ┌─────────┐
              │ SSTable │ (Sorted String Table, immutable)
              │  (L0)   │
              └─────────┘
                   │
                   │ (compaction)
                   ▼
              ┌─────────┐
              │ SSTable │ (larger, sorted)
              │  (L1)   │
              └─────────┘
                   │
                   │ (more compaction)
                   ▼
              ┌─────────────────────┐
              │    SSTable (L2)     │ (even larger)
              └─────────────────────┘
```

**LSM Write Process:**
```
INSERT id=23

1. Write to Write-Ahead Log (durability)
2. Insert into Memtable (in-memory sorted structure)
3. Return success (very fast!)

Background:
4. When memtable full → flush to SSTable on disk
5. Periodically compact SSTables (merge + deduplicate)
```

**LSM Read Process:**
```
READ id=23

1. Check Memtable (newest data)
2. Check Bloom filters for each SSTable
   - Bloom filter: "Definitely not here" or "Maybe here"
3. Check SSTables from newest to oldest
4. First match wins (newest version)

Worst case: Check all levels
```

**LSM-Tree Characteristics:**
- Write-optimized (sequential I/O)
- Read amplification (check multiple files)
- Space amplification (during compaction)
- Good for write-heavy workloads

### B-Tree vs LSM-Tree Comparison

| Aspect | B-Tree | LSM-Tree |
|--------|--------|----------|
| Write performance | Slower (random I/O) | Faster (sequential) |
| Read performance | Faster (one location) | Slower (multiple files) |
| Space efficiency | Higher | Lower (compaction) |
| Write amplification | Lower | Higher (rewrite on compact) |
| Use case | Read-heavy OLTP | Write-heavy, logs, time-series |

---

## Indexes

### Why Indexes Matter

```
Without index:
SELECT * FROM users WHERE email = 'alice@example.com';
→ Full table scan: O(n) - Check every row

With index on email:
→ B-tree lookup: O(log n) - 3-4 disk reads

1 million rows:
- Full scan: ~1 million row reads
- Index lookup: ~20 row reads (log₂ 1M ≈ 20)
```

### Index Types

**B-Tree Index (default):**
```sql
CREATE INDEX idx_email ON users(email);

Good for:
- Equality: WHERE email = 'x'
- Range: WHERE created_at > '2024-01-01'
- Prefix: WHERE name LIKE 'Al%'
- Sorting: ORDER BY created_at

Bad for:
- Suffix/contains: WHERE name LIKE '%ice'
- Low cardinality: WHERE is_active = true (50% of rows)
```

**Hash Index:**
```sql
CREATE INDEX idx_email ON users USING HASH (email);

Good for:
- Exact equality: WHERE email = 'x'

Bad for:
- Range queries
- Sorting

Faster than B-tree for exact lookups, but limited.
```

**GIN (Generalized Inverted Index):**
```sql
CREATE INDEX idx_tags ON posts USING GIN (tags);

Good for:
- Array contains: WHERE tags @> ARRAY['python']
- Full-text search: WHERE to_tsvector(body) @@ 'search'
- JSONB containment

Used for: Arrays, JSONB, full-text search
```

**GiST (Generalized Search Tree):**
```sql
CREATE INDEX idx_location ON stores USING GIST (location);

Good for:
- Geometric queries: WHERE location <-> point '(x,y)' < 10
- Range types: WHERE period && '[2024-01-01, 2024-12-31]'
- PostGIS spatial queries
```

### Composite Indexes

```sql
CREATE INDEX idx_user_date ON orders(user_id, created_at);

Order matters!

Uses index:
WHERE user_id = 123                              ✓
WHERE user_id = 123 AND created_at > '2024-01-01' ✓
WHERE user_id = 123 ORDER BY created_at          ✓

Does NOT use index:
WHERE created_at > '2024-01-01'                  ✗ (first column missing)
ORDER BY created_at                              ✗
```

**Index ordering (leftmost prefix rule):**
```
Index on (A, B, C) supports:
- WHERE A = ?
- WHERE A = ? AND B = ?
- WHERE A = ? AND B = ? AND C = ?
- WHERE A = ? ORDER BY B
- WHERE A = ? AND B = ? ORDER BY C

Does NOT support:
- WHERE B = ?        (A missing)
- WHERE C = ?        (A, B missing)
- ORDER BY B         (A missing)
```

### Covering Indexes

```sql
-- Query
SELECT user_id, email FROM users WHERE email = 'x';

-- Regular index
CREATE INDEX idx_email ON users(email);
→ Index lookup + table lookup (to get user_id)

-- Covering index
CREATE INDEX idx_email_userid ON users(email, user_id);
→ Index-only scan (all data in index, no table lookup)
```

PostgreSQL: `INCLUDE` clause for non-key columns:
```sql
CREATE INDEX idx_email ON users(email) INCLUDE (user_id, name);
```

---

## Query Execution

### EXPLAIN ANALYZE

```sql
EXPLAIN ANALYZE
SELECT * FROM orders
WHERE user_id = 123 AND status = 'pending';

                                    QUERY PLAN
-------------------------------------------------------------------------------
Index Scan using idx_orders_user_id on orders  (cost=0.43..8.45 rows=1 width=100)
  Index Cond: (user_id = 123)
  Filter: (status = 'pending')
  Rows Removed by Filter: 5
  Planning Time: 0.1 ms
  Execution Time: 0.05 ms
```

**Key things to look for:**
- **Seq Scan:** Full table scan (usually bad for large tables)
- **Index Scan:** Using index (good)
- **Index Only Scan:** All data from index (best)
- **Rows Removed by Filter:** Index not selective enough
- **Nested Loop:** Can be slow for large joins

### Join Algorithms

**Nested Loop:**
```
For each row in Table A:
    For each row in Table B:
        If join condition matches:
            Output row

O(n × m) - Good for: Small tables, indexed lookups
```

**Hash Join:**
```
Build hash table from smaller table (Table A)
For each row in Table B:
    Lookup in hash table
    If match: Output row

O(n + m) - Good for: Equality joins, no index
Requires: Memory for hash table
```

**Merge Join:**
```
Sort both tables by join key
Walk through both simultaneously
Output matching rows

O(n log n + m log m) - Good for: Already sorted data
```

---

## Write-Ahead Logging (WAL)

### The Durability Problem

```
Write without WAL:
1. Modify data in memory
2. Acknowledge to client
3. Later: Flush to disk

Crash between 2 and 3 → DATA LOST!
```

### WAL Solution

```
Write with WAL:
1. Write change to WAL (sequential, fast)
2. Acknowledge to client
3. Modify data in memory
4. Later: Flush to disk (background)

Crash at any point:
- Replay WAL on recovery
- No data loss!
```

**WAL structure:**
```
WAL Segment File:
┌─────────────────────────────────────────────────┐
│ Record 1: INSERT INTO users (id=1, name='Alice')│
│ Record 2: UPDATE users SET name='Bob' WHERE id=1│
│ Record 3: DELETE FROM users WHERE id=1         │
│ Record 4: COMMIT                               │
└─────────────────────────────────────────────────┘

On crash recovery:
1. Read WAL from last checkpoint
2. Replay all committed transactions
3. Discard uncommitted transactions
```

---

## MVCC (Multi-Version Concurrency Control)

### The Concurrency Problem

```
Without MVCC:
Transaction A: SELECT * FROM accounts WHERE id = 1;  (balance = 100)
Transaction B: UPDATE accounts SET balance = 50 WHERE id = 1;
Transaction A: SELECT * FROM accounts WHERE id = 1;  (balance = 50!)

Transaction A sees inconsistent data (non-repeatable read)
```

### MVCC Solution

```
With MVCC:
Every row has hidden columns: (xmin, xmax)
xmin = Transaction that created this version
xmax = Transaction that deleted/updated this version

Row versions:
┌──────────────────────────────────────────────┐
│ id=1, balance=100, xmin=100, xmax=200       │ (old version)
│ id=1, balance=50,  xmin=200, xmax=null      │ (current version)
└──────────────────────────────────────────────┘

Transaction A (started at xid=150):
- Sees version where xmin <= 150 < xmax
- Sees balance=100 (consistent snapshot)

Transaction B (xid=200):
- Creates new version with xmin=200
- Sets xmax=200 on old version
```

**Benefits:**
- Readers don't block writers
- Writers don't block readers
- Consistent snapshots without locks

**Cost:**
- Dead tuples (old versions) need cleanup (VACUUM)
- More storage for multiple versions

---

## Vacuum and Maintenance

### Why Vacuum?

```
After UPDATE/DELETE with MVCC:
- Old row versions still exist (dead tuples)
- Space not reclaimed
- Index bloat

Without vacuum:
- Table grows forever
- Queries slow down
- Disk fills up
```

### Autovacuum

```sql
-- Check autovacuum stats
SELECT relname, n_dead_tup, n_live_tup, last_autovacuum
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000;

-- Tune autovacuum
ALTER TABLE orders SET (
    autovacuum_vacuum_threshold = 50,
    autovacuum_vacuum_scale_factor = 0.1
);
-- Vacuum when dead_tuples > 50 + 0.1 * n_live_tup
```

### Statistics and ANALYZE

```sql
-- Query planner uses statistics for cost estimation
-- Stale statistics → bad query plans

-- Manual analyze
ANALYZE users;

-- Check statistics
SELECT * FROM pg_stats WHERE tablename = 'users';
```

---

## Connection Pooling

### The Problem

```
Creating a PostgreSQL connection:
1. TCP handshake: ~1ms
2. SSL handshake: ~5ms
3. Authentication: ~2ms
4. Fork backend process: ~10ms
Total: ~20ms per connection

With 100 requests/second, each creating new connection:
100 × 20ms = 2 seconds of overhead per second
Server overwhelmed!
```

### Connection Pool Solution

```
┌─────────────────────────────────────────────────────────┐
│                    Connection Pool                       │
│                      (PgBouncer)                        │
│                                                          │
│  Pool: [conn1][conn2][conn3][conn4][conn5]              │
│         (pre-established, reused)                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
         ▲       ▲       ▲       ▲       ▲
         │       │       │       │       │
    ┌────┴──┐ ┌──┴───┐ ┌─┴──┐ ┌──┴──┐ ┌──┴──┐
    │Client1│ │Client2│ │...│ │...  │ │...  │
    └───────┘ └──────┘ └────┘ └─────┘ └─────┘
         (hundreds of clients share few connections)
```

**Pool modes:**
- **Session:** Client keeps connection for entire session
- **Transaction:** Client gets connection per transaction
- **Statement:** Client gets connection per statement

```
# pgbouncer.ini
[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 20
```

---

## Interview Checklist

- [ ] Explain B-tree vs LSM-tree trade-offs
- [ ] Know when to create indexes (and when not to)
- [ ] Read and interpret EXPLAIN ANALYZE output
- [ ] Explain MVCC and its benefits/costs
- [ ] Describe WAL and crash recovery
- [ ] Understand connection pooling necessity

---

## Staff+ Insights

**Index strategy:**
- Don't over-index (writes slow, storage cost)
- Monitor unused indexes: `pg_stat_user_indexes`
- Partial indexes for hot data: `WHERE is_active = true`
- Consider covering indexes for critical queries

**Query optimization:**
- Start with EXPLAIN ANALYZE
- Look for Seq Scans on large tables
- Check for missing indexes
- Watch for implicit casts that prevent index use

**Maintenance:**
- Monitor dead tuple ratio
- Tune autovacuum for write-heavy tables
- Regular ANALYZE after bulk loads
- pg_repack for table/index bloat

**Connection management:**
- Always use connection pooling in production
- Size pool to match actual concurrency needs
- Transaction pooling for stateless apps
- Monitor connection wait times
