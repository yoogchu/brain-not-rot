# Chapter 3: Consistency Models

## The Bank Account Problem

```
Account Balance: $1,000

T0: NYC ATM reads balance: $1,000
T0: London ATM reads balance: $1,000
T1: NYC ATM withdraws $800
T2: London ATM withdraws $500
T3: Final balance: ???

Without consistency guarantees: -$300 overdraft!
```

This isn't hypothetical—it's the exact problem that drives distributed systems design.

Consistency models define what guarantees the system provides about the order and visibility of operations.

---

## The Consistency Spectrum

From strongest to weakest:

```
STRONG ◄─────────────────────────────────────────► WEAK

Linearizability > Sequential > Causal > Eventual
     │              │            │         │
     │              │            │         └─ Maximum availability
     │              │            └─ Respects cause/effect
     │              └─ Global order (but not real-time)
     └─ Appears as single copy in real-time
```

Each level trades some consistency for better performance and availability.

---

## Linearizability (Strongest)

### Definition

System behaves as if there's a **single copy of data** with operations taking effect **atomically** at some point between invocation and response.

```
T0────────T1────────T2────────T3────────T4
    Write X=1
    ├─────────┤
          │
          └─ Somewhere in this interval, write "takes effect"
          
              Read X
              ├──────┤
              Must return 1 (write completed before read started)
```

### The Key Guarantee: Real-Time Ordering

If operation A completes before operation B starts, then A's effect is visible to B.

```
Process 1: Write X=1 [─────────] ← completes at T1
Process 2:                       Read X [────] ← starts at T2

T1 < T2 → Read MUST see X=1
```

### What Linearizability Looks Like in Practice

```python
# Client 1 writes
db.write("balance", 1000)  # Returns success at T1

# Client 2 reads (starts after T1)
value = db.read("balance")  # MUST return 1000

# If Client 2 saw 0 or old value, that's NOT linearizable
```

### Implementation Requirements

Achieving linearizability requires coordination:

**Option 1: Single leader with synchronous replication**
```
All reads/writes go through one node
That node is the "single copy"
Synchronous replication ensures durability
```

**Option 2: Consensus protocol (Raft, Paxos)**
```
Replicas agree on operation order
Majority must acknowledge before success
etcd, ZooKeeper, Consul use this
```

**Option 3: Single node**
```
One database, no replication
Trivially linearizable
But: no fault tolerance, no scaling
```

### The Cost of Linearizability

**Latency:**
- Every operation requires coordination
- Cross-datacenter: 100-200ms minimum per operation
- Network round-trips for consensus

**Availability:**
- During network partitions, must choose consistency over availability
- Minority partition can't make progress
- CAP theorem in action

**Throughput:**
- Single point of serialization limits throughput
- Can't horizontally scale writes

### Use Cases for Linearizability

- **Leader election**: Exactly one leader at a time
- **Distributed locks**: Exclusive access to resources
- **Unique ID generation**: No duplicate IDs
- **Financial transactions**: Account balance never negative
- **Inventory**: Don't oversell items

---

## Sequential Consistency

### Definition

All operations appear in **some sequential order** consistent with each process's **program order**.

The key difference from linearizability: **No real-time guarantee.**

### Example

```
Process A: Write X=1, Write Y=2
Process B: Read Y, Read X

Valid sequential orderings:
1. W(X=1), W(Y=2), R(Y)→2, R(X)→1  ✓
   A's writes happen, then B reads
   
2. W(X=1), R(Y)→0, W(Y=2), R(X)→1  ✓
   B starts reading before A finishes writing Y
   
3. R(Y)→2, R(X)→0  ✗
   INVALID: If B sees Y=2, A's writes must have happened
   But X should be 1 if Y is 2 (A wrote X before Y)
```

### The Subtle Difference from Linearizability

```
Real time:
T0        T1        T2        T3
Client A: Write X=1 [────]
Client B:           [────] Write X=2
Client C:                            Read X

Linearizable: C must read 2 (B completed before C started)
Sequentially consistent: C could read 1 OR 2
  (as long as all clients see same order)
```

Sequential consistency allows reordering as long as each process's operations stay in order.

### Implementation

- Total order broadcast
- Single sequencer assigns sequence numbers
- All replicas apply operations in sequence order

### Use Cases

- Less common in databases
- Some memory models (older CPUs)
- Systems where real-time ordering doesn't matter

---

## Causal Consistency

### Definition

Causally related operations are seen in the same order by all processes. Concurrent (unrelated) operations can be seen in different orders.

### What Makes Operations "Causally Related"?

1. **Same process**: A then B in same process → A happens-before B
2. **Message passing**: A sends message received by B → A happens-before B
3. **Transitivity**: A → B and B → C implies A → C

```
Process A: Write X=1
Process B: Read X=1, Write Y=2  (Y=2 causally depends on X=1)
Process C: Read Y=2, Read X=???

Causal consistency guarantees: C must read X=1
(Can't see effect Y=2 without seeing its cause X=1)

BUT if two writes are independent:
Process D: Write Z=100
Process E: Write W=200
Process F: Might see Z=100 then W=200
Process G: Might see W=200 then Z=100
(No causal relationship → order can differ)
```

### Implementation: Vector Clocks

```python
# Each process maintains vector of logical clocks
# Vector has one entry per process

Process A: [A:1, B:0, C:0] → Write X=1
           # A increments its own clock

Process B: Receives X=1 with [A:1, B:0, C:0]
           Updates: [A:1, B:1, C:0] → Write Y=2
           # B takes max of received + increments own

Process C: Receives Y=2 with [A:1, B:1, C:0]
           Knows X=1 must exist (A:1 in vector)
           Must wait for X=1 before applying Y=2
```

**Comparing vectors:**
```
V1 = [A:2, B:1, C:3]
V2 = [A:1, B:2, C:2]

V1 < V2?  A:2 > A:1  NO
V2 < V1?  B:2 > B:1  NO
Neither dominates → CONCURRENT (can be applied in either order)

V3 = [A:3, B:2, C:4]
V3 > V1?  All components ≥ V1 and at least one >  YES
V3 happens after V1
```

### Version Vectors for Replicated Data

```python
# Each replica maintains version vector
# On write, increment own component

Replica 1: {R1: 5, R2: 3, R3: 4}  "I've seen 5 of my writes, 
                                   3 from R2, 4 from R3"

Replica 2: {R1: 4, R2: 6, R3: 4}  "I'm behind on R1's updates"

When R2 receives data with {R1: 5, ...}:
"Oh, I need R1's 5th update before this is consistent"
```

### Use Cases

- Social media feeds (see post before replies)
- Collaborative editing (see edit before acknowledgment)
- Chat applications (messages in order per conversation)

**Advantage over sequential:** Higher availability, lower latency
**Advantage over eventual:** Respects cause/effect relationships

---

## Eventual Consistency

### Definition

If no new writes occur, **all replicas will eventually converge** to the same value.

```
T0: Write X=1 to replica A
T1: Replica B still has X=0 (stale)
T2: Replica C still has X=0 (stale)
... replication propagates ...
T100: All replicas have X=1 (eventually!)
```

### What's NOT Guaranteed

- **How long** until convergence (could be seconds or hours)
- **Order** of operations
- **What value** you'll read at any point
- **Whether reads are monotonic** (might go backwards!)

### The Problems with Pure Eventual Consistency

```
Scenario: Shopping cart

T0: User adds item on replica A
T1: User views cart on replica B (empty! stale replica)
T2: User thinks item wasn't added
T3: User adds item again
T4: Eventually both replicas sync → TWO items!
```

This is why pure eventual consistency is often not enough.

### Strengthening Eventual Consistency

| Variant | Guarantee | Implementation |
|---------|-----------|----------------|
| Read-your-writes | You see your own writes | Sticky sessions or version tracking |
| Monotonic reads | Once you see X=1, never see X=0 | Stick to same replica or version floor |
| Monotonic writes | Your writes apply in order | Queue writes, apply sequentially |
| Consistent prefix | See operations in order they occurred | Ordered replication log |

**Read-your-writes:**
```python
def read_with_your_writes(user_id, key):
    last_write_version = get_user_last_write_version(user_id, key)
    # Wait for replica to have at least this version
    replica = get_replica_with_version(last_write_version)
    return replica.read(key)
```

**Monotonic reads:**
```python
def monotonic_read(user_id, key):
    last_read_version = get_user_last_read_version(user_id, key)
    replica = get_replica_with_version(last_read_version)
    value, version = replica.read(key)
    set_user_last_read_version(user_id, key, version)
    return value
```

### Where Eventual Consistency Works

- **DNS**: Can take hours to propagate, that's OK
- **CDN caches**: Stale content is acceptable
- **Social media feeds**: Seeing a post 10 seconds late is fine
- **Product catalog**: Price might be stale briefly
- **Session data**: Lost session is annoying but recoverable

### Where Eventual Consistency Fails

- **Banking**: Can't have stale balance
- **Inventory**: Can't oversell
- **Leader election**: Must have exactly one leader
- **Unique constraints**: Can't have duplicate usernames

---

## CAP and PACELC

### CAP Theorem

During a network **P**artition, you must choose:
- **C**onsistency: Every read receives the most recent write
- **A**vailability: Every request receives a response

You can't have both during a partition.

```
         ┌─────────────────────┐
         │                     │
         │    Consistency      │
         │         │           │
         │    ┌────┴────┐      │
         │    │ During  │      │
         │    │ partition│      │
         │    │ choose  │      │
         │    │  ONE    │      │
         │    └────┬────┘      │
         │    Availability     │
         │                     │
         │      Partition      │
         │      Tolerance      │
         │    (must handle     │
         │  network failures)  │
         └─────────────────────┘
```

**Note:** "Partition Tolerance" isn't optional for distributed systems. Network failures WILL happen.

### CP vs AP Systems

**CP (Consistent, Partition-tolerant):**
```
During partition, minority side stops accepting writes
Guarantees consistency at cost of availability
Examples: ZooKeeper, etcd, HBase, MongoDB (default)
```

**AP (Available, Partition-tolerant):**
```
During partition, all sides continue operating
May return stale data
Examples: Cassandra, DynamoDB, CouchDB
```

### The Reality: Not Pure CP or AP

Most systems let you choose per-operation:

**ZooKeeper:**
```
Writes: Always CP (must reach majority)
Reads: Can be stale (AP) or linearizable (CP)
```

**DynamoDB:**
```
Default: Eventually consistent reads (AP)
Optional: Strongly consistent reads (CP)
```

**Cassandra:**
```
Consistency level per query:
- ONE: AP (any replica)
- QUORUM: CP-ish (majority)
- ALL: Strong CP (all replicas)
```

### PACELC: The Extended Model

CAP only talks about partitions. What about normal operation?

**PACELC:** 
```
if (Partition) {
    choose: Availability or Consistency
} else {
    choose: Latency or Consistency
}
```

Most of the time there's no partition. You're trading **latency vs consistency**.

| System | Partition Behavior | Normal Behavior |
|--------|-------------------|-----------------|
| Dynamo/Cassandra | A (available) | L (low latency) |
| MongoDB | C (consistent) | L (low latency) |
| VoltDB | C (consistent) | C (consistent) |
| PNUTS | C (consistent) | L (low latency) |

---

## Implementing Consistency in Practice

### MongoDB

**Read Concerns:**
```javascript
// Local: Read from node, might be stale
db.collection.find().readConcern("local")

// Majority: Read data acknowledged by majority
db.collection.find().readConcern("majority")

// Linearizable: Strongest, highest latency
db.collection.find().readConcern("linearizable")
```

**Write Concerns:**
```javascript
// Acknowledged by 1 node (fast, risky)
db.collection.insert({...}, {w: 1})

// Acknowledged by majority (safe)
db.collection.insert({...}, {w: "majority"})

// Acknowledged by all (very safe, slow)
db.collection.insert({...}, {w: <number_of_replicas>})
```

### Cassandra

```sql
-- Per-query consistency
SELECT * FROM users WHERE id = 123 
USING CONSISTENCY QUORUM;

-- Options:
-- ONE: Fast but might be stale
-- QUORUM: Majority nodes agree
-- ALL: All replicas respond
-- LOCAL_QUORUM: Majority in local DC (for multi-DC)
```

### PostgreSQL with Replicas

```sql
-- Synchronous replication config
synchronous_standby_names = 'replica1'  -- Wait for this replica
synchronous_commit = on  -- Don't return until synced

-- Or per-transaction:
SET synchronous_commit = on;  -- This transaction waits
SET synchronous_commit = off;  -- This one doesn't
```

---

## Choosing the Right Consistency Level

### Decision Framework

```
Is data loss ever acceptable?
├─ No → Strong consistency
│       └─ How bad is high latency?
│           ├─ Unacceptable → Maybe reconsider requirements
│           └─ Acceptable → Linearizable/Sequential
│
└─ Temporary inconsistency OK?
    ├─ Yes, but cause/effect matters → Causal
    │
    └─ Yes, eventually correct is fine → Eventual
        └─ Need read-your-writes? → Eventual + session stickiness
```

### Consistency by Use Case

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Bank balance | Linearizable | Never show wrong balance |
| Leader election | Linearizable | Exactly one leader |
| Social feed | Causal | See replies after posts |
| Product catalog | Eventual | Stale price briefly OK |
| User session | Read-your-writes | See your own actions |
| Analytics | Eventual | Approximate is fine |
| Shopping cart | Causal or stronger | Don't lose items |

---

## Consistency Tradeoffs Summary

| Model | Availability | Latency | Complexity | Use Case |
|-------|--------------|---------|------------|----------|
| Linearizable | Low | High | High | Locks, leader election, banking |
| Sequential | Medium | Medium | Medium | Total ordering needed |
| Causal | High | Low | Medium | Social apps, collaboration |
| Eventual | Highest | Lowest | Low | Caching, CDNs, analytics |

---

## Key Concepts Checklist

- [ ] Clarify consistency requirements for each operation type
- [ ] Distinguish read vs write consistency needs
- [ ] Consider geographic distribution impact
- [ ] Discuss CAP/PACELC tradeoffs explicitly
- [ ] Mention specific consistency implementations (MongoDB read concerns, Cassandra consistency levels)
- [ ] Know when to use each consistency level

---

## Practical Insights

**Linearizability is expensive:**
Cross-datacenter linearizability can add 100-300ms to every operation. Often better to use causal consistency + conflict resolution.

**Consistency is per-operation, not per-system:**
Even in "eventually consistent" systems, you can do strongly consistent reads when needed. Design for flexibility.

**Most bugs are consistency bugs:**
"Works on my machine" often means "worked with one replica." Test with realistic replication lag.

**Spanner's approach:**
Google Spanner uses atomic clocks + GPS for global strong consistency with ~7ms commit wait. Expensive infrastructure, but removes consistency headaches.
