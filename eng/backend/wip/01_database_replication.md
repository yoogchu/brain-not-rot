# Chapter 1: Database Replication

## The Core Problem

Your production database serves 50,000 QPS. At 3 AM, the primary server's SSD fails. Without replication:
- You restore from last night's backup: 8 hours of transactions lost
- Recovery takes 2+ hours while customers can't transact
- Revenue loss could be catastrophic

With properly configured replication:
- A replica promotes to primary in seconds
- Zero data loss with synchronous replication
- Customers experience brief blip, not extended outage

But replication isn't just disaster recovery. It enables:
- **Read scaling**: Route reads to replicas, writes to primary
- **Geographic distribution**: Serve users from nearby replicas
- **Analytics isolation**: Run heavy queries on dedicated replicas
- **Rolling upgrades**: Upgrade replicas one-by-one

---

## Single-Leader Replication

### The Basic Idea

One node (leader/primary/master) accepts all writes. Changes propagate to followers via Write-Ahead Log (WAL) shipping or logical replication.

```
┌─────────────┐     WAL Stream      ┌─────────────┐
│   Leader    │ ──────────────────> │  Follower 1 │
│  (Writes)   │                     │   (Reads)   │
└─────────────┘ ──────────────────> ┌─────────────┐
                                    │  Follower 2 │
                                    │   (Reads)   │
                                    └─────────────┘
```

Think of it like a newspaper: there's one editor-in-chief (the leader) who approves all changes, and then those changes get distributed to all the newsstands (followers).

### How WAL Shipping Works

The Write-Ahead Log is the database's journal. Before any change happens to actual data pages, it's first written to the WAL. This ensures durability—if the system crashes, it can replay the WAL to recover.

```
1. Client: UPDATE users SET balance = 100 WHERE id = 123
2. Leader writes to WAL: "Set balance=100 for user 123"
3. Leader applies change to data pages
4. Leader streams WAL entry to followers
5. Followers replay WAL entry against their data pages
```

**Physical vs Logical Replication:**

| Type | What's Replicated | Pros | Cons |
|------|-------------------|------|------|
| Physical (WAL) | Raw byte changes | Exact copy, simpler | Same version required |
| Logical | SQL-level changes | Cross-version, selective | More complex, slower |

---

## Synchronous vs Asynchronous Replication

This is one of the most important tradeoffs in distributed systems.

### Synchronous Replication

```
Timeline:
T0: Client sends write to leader
T1: Leader writes to local disk
T2: Leader sends to follower
T3: Follower writes to disk
T4: Follower sends ACK to leader
T5: Leader sends ACK to client
```

The client doesn't get a success response until the follower has confirmed it received the data.

**What this guarantees:**
- If leader crashes after ACK, data is safe on follower
- Follower can be promoted with zero data loss

**What this costs:**
- Latency: Every write waits for network round-trip to follower
- Availability: If follower is slow or down, writes block

### Asynchronous Replication

```
Timeline:
T0: Client sends write to leader
T1: Leader writes to local disk
T2: Leader sends ACK to client (DONE!)
T3: Leader eventually sends to follower
T4: Follower writes to disk
```

The client gets success immediately after leader persists.

**What this guarantees:**
- Low latency writes
- Leader can operate even if followers are down

**What this risks:**
- If leader crashes between T2 and T4, data is LOST
- Followers may be minutes behind during high load

### Semi-Synchronous: The Middle Ground

Wait for ONE follower to ACK, async to the rest.

```
Leader ──sync──> Follower 1 (must ACK before client success)
       ──async─> Follower 2 (eventual)
       ──async─> Follower 3 (eventual)
```

MySQL's semi-sync replication does this. PostgreSQL's `synchronous_standby_names` can be configured similarly.

**Trade-off summary:**

| Aspect | Synchronous | Asynchronous | Semi-Sync |
|--------|-------------|--------------|-----------|
| Durability | Strong | Weak | Medium |
| Write Latency | High (+ RTT) | Low | Medium |
| Availability | Lower | Higher | Medium |
| Use Case | Financial | High-throughput | General |

---

## Replication Lag: The Sneaky Problem

Asynchronous replication creates a window where follower data is stale:

```
Timeline:
T0: User updates profile picture on leader
T1: Leader returns success to user
T2: User refreshes page, request goes to follower
T3: Follower hasn't received update yet
→ User sees OLD picture (WTF moment)
```

This isn't a bug—it's the inherent nature of async replication. But it creates terrible UX.

### Read-After-Write Consistency Patterns

**Pattern 1: Read from leader after writes**
```python
def get_user_profile(user_id, just_updated=False):
    if just_updated:
        # Route to leader for fresh data
        return leader_db.get_user(user_id)
    else:
        # Route to replica for scalability
        return replica_db.get_user(user_id)
```

Simple but requires tracking "just updated" state. Often done with:
- Session flag that expires after N seconds
- Check if user's write timestamp is within lag window

**Pattern 2: Monotonic reads**
```python
# Pin user to same replica for entire session
def get_replica_for_user(user_id):
    return replicas[hash(user_id) % len(replicas)]
```

User always reads from same replica. They might see stale data, but never go "backwards" in time.

**Pattern 3: Versioned reads**
```python
# Include write timestamp in response
response = {
    "data": user_profile,
    "version": "2024-01-15T10:30:00Z"
}

# Client sends version in next request
# Replica waits until it has that version before responding
```

More complex but provides strong guarantees.

---

## Failover: The Hard Problem

When the leader dies, a follower must become the new leader. This sounds simple but is surprisingly tricky.

### Detection: Is the Leader Actually Dead?

**The problem:** Network partition vs actual crash look identical from outside.

```
Scenario 1: Leader crashed
Leader ──✗──> [no heartbeat] ──> Follower: "Leader is dead"
Correct action: Elect new leader

Scenario 2: Network partition
Leader ──✗ network ✗──> Follower: "Leader is dead"
Reality: Leader is fine, serving writes
Dangerous action: Elect new leader → SPLIT BRAIN
```

**Typical approach:** Heartbeat timeout
- Leader sends heartbeat every 5 seconds
- If no heartbeat for 30 seconds, assume dead

**Trade-offs:**
- Too short timeout: False positives during GC pauses, network blips
- Too long timeout: Extended downtime during real failures

### Election: Which Follower Becomes Leader?

**Option 1: Most up-to-date replica**
```
Follower A: Has data up to WAL position 1000
Follower B: Has data up to WAL position 1050  ← Winner
Follower C: Has data up to WAL position 1020
```

Minimizes data loss. Most common approach.

**Option 2: Predetermined priority**
```
Config: priority_list = [follower_a, follower_b, follower_c]
# Always try A first, then B, then C
```

Predictable, but might elect a less up-to-date replica.

**Option 3: Consensus among replicas**
Use Raft/Paxos to vote. More complex but handles edge cases better.

### Client Reconfiguration

All clients must discover the new leader. Options:

**DNS update:**
```
Before: primary.db.example.com → 10.0.0.1 (old leader)
After:  primary.db.example.com → 10.0.0.2 (new leader)
```
Problem: DNS caching causes clients to use stale address for minutes.

**Service discovery (Consul, ZooKeeper):**
```
Clients watch: /services/database/primary
Value changes from "10.0.0.1" to "10.0.0.2"
Clients immediately reconnect
```

**Proxy layer:**
```
Client → HAProxy → Leader
            ↓
         HAProxy health-checks all nodes
         Routes to current leader automatically
```

### Split-Brain: The Nightmare Scenario

```
Original state:
Leader A ←──── Follower B
         ←──── Follower C

Network partition:
┌─────────────┐         ┌─────────────┐
│   Leader A  │    ✗    │  Follower B │
│  (isolated) │─────────│  Follower C │
└─────────────┘         └─────────────┘

What happens:
- A thinks it's still leader, continues accepting writes
- B gets elected as new leader, also accepts writes
- Both accept writes → DATA DIVERGES
```

**Solutions:**

**1. Fencing tokens:**
```
Old leader has token: 42
New leader gets token: 43

Storage system rejects writes with token < 43
Old leader's writes fail, even if network heals
```

**2. STONITH (Shoot The Other Node In The Head):**
```
Before promoting new leader:
1. Send power-off command to old leader's management interface
2. Wait for confirmation that old leader is OFF
3. Only then promote new leader
```

Sounds violent, but it's the only way to be CERTAIN old leader can't write.

**3. Quorum-based leadership:**
```
To be leader, must maintain connections to majority of nodes
5-node cluster: Need 3 nodes agreeing you're leader

If partition splits 2|3:
- Side with 2 nodes: Can't get majority, steps down
- Side with 3 nodes: Has majority, continues as leader
```

---

## Multi-Leader Replication

### When Single-Leader Isn't Enough

**Multi-datacenter deployment:**
```
Users in NYC → DC in NYC → Leader → Cross-Atlantic → DC in London
                                    (150ms latency!)
```

Every write from London users has 300ms+ round-trip. Unacceptable for some applications.

**Offline-capable clients:**
Mobile apps need to work offline. Each device is essentially a "leader" for its local data.

**High write throughput:**
Single leader becomes CPU bottleneck at extreme scale.

### Architecture

```
┌─────────────────┐         ┌─────────────────┐
│   DC: US-East   │ <─────> │   DC: EU-West   │
│    Leader A     │  Async  │    Leader B     │
│   (US writes)   │ Repl.   │  (EU writes)    │
└─────────────────┘         └─────────────────┘
        ↑                           ↑
   US Users                    EU Users
   (low latency)              (low latency)
```

### The Fundamental Challenge: Conflicts

When two leaders modify the same data simultaneously:

```
Timeline (wall clock):
T0: User in NYC sets title = "Hello" in US-East
T0: User in London sets title = "Bonjour" in EU-West
T1: Changes replicate asynchronously
T2: Both DCs have CONFLICTING versions

Which value wins?
```

### Conflict Resolution Strategies

#### 1. Last-Write-Wins (LWW)

```
Use timestamps to pick winner:
- US write: {title: "Hello", ts: 1000000001}
- EU write: {title: "Bonjour", ts: 1000000002}
- EU wins because timestamp is higher
```

**The problem:** Clocks aren't perfectly synchronized.

```
Actual event order: US wrote first
But EU's clock is 5ms ahead
EU timestamp is higher → EU "wins"
US user's write silently DISAPPEARS
```

NTP provides ~100ms accuracy over internet, ~1-10ms on LAN. That's a large conflict window.

**When LWW is acceptable:**
- Idempotent operations (SET x = 5)
- Immutable event logs (append-only)
- Cache updates (staleness is OK)

**Used by:** Cassandra, DynamoDB (default)

#### 2. CRDTs (Conflict-free Replicated Data Types)

Data structures mathematically guaranteed to converge regardless of operation order.

**G-Counter (Grow-only Counter):**
```python
# Each node maintains its own count
class GCounter:
    def __init__(self, node_id):
        self.counts = {}  # {node_id: count}
        self.node_id = node_id
    
    def increment(self):
        self.counts[self.node_id] = self.counts.get(self.node_id, 0) + 1
    
    def value(self):
        return sum(self.counts.values())
    
    def merge(self, other):
        # Take max of each node's count
        for node, count in other.counts.items():
            self.counts[node] = max(self.counts.get(node, 0), count)

# Example:
# Node A: {A: 5, B: 0}  total = 5
# Node B: {A: 3, B: 7}  total = 10
# After merge: {A: 5, B: 7}  total = 12  ← CORRECT!
```

**Why this works:** Each node only increments its own counter. Merge takes max. No conflicts possible.

**Common CRDTs:**
| Type | Description | Use Case |
|------|-------------|----------|
| G-Counter | Grow-only counter | Like counts |
| PN-Counter | Supports increment AND decrement | Account balances |
| G-Set | Add-only set | Tag collections |
| OR-Set | Add and remove | Shopping carts |
| LWW-Register | Single value with timestamp | Profile fields |

**Used by:** Riak, Redis (CRDT mode), Automerge, Yjs

#### 3. Custom Application Resolution

Store ALL conflicting versions, let application decide.

```json
{
  "_id": "doc123",
  "_conflicts": ["2-abc", "2-def"],
  "versions": {
    "2-abc": {"title": "Hello", "author": "UserA"},
    "2-def": {"title": "Bonjour", "author": "UserB"}
  }
}
```

Application can:
- Merge intelligently (combine fields)
- Present conflict to user ("Which version do you want?")
- Apply domain-specific rules

**Used by:** CouchDB, PouchDB

---

## Leaderless Replication

### Dynamo-Style Architecture

No leader. Any node can accept reads or writes.

```
Client Write Request
        │
        ├─────────> Node A (write)
        ├─────────> Node B (write)  
        └─────────> Node C (write)
        
ACK when W nodes respond
```

**Used by:** Cassandra, DynamoDB, Riak, Voldemort

### Quorum Mathematics

Given:
- **N** = total replicas
- **W** = write quorum (nodes that must ACK write)
- **R** = read quorum (nodes to read from)

**Guarantee:** If W + R > N, at least one read node has latest write.

```
Example: N=3, W=2, R=2

Write to A, B (W=2 satisfied)
Read from B, C (R=2 satisfied)
B has latest → read succeeds

      A ──── B ──── C
   [write] [write]  [stale]
              ↑
         overlap guarantees
         fresh read
```

### Tuning Quorums

| Configuration | Property | Use Case |
|--------------|----------|----------|
| W=N, R=1 | Write all, read any | Read-heavy, high availability reads |
| W=1, R=N | Write any, read all | Write-heavy, high availability writes |
| W=N/2+1, R=N/2+1 | Balanced | General purpose |
| W=1, R=1 | No guarantee | Best effort, maximum availability |

### Sloppy Quorums and Hinted Handoff

**What if a required node is unreachable during write?**

**Strict Quorum:** Write fails if W designated nodes unavailable

**Sloppy Quorum:** Write to W nodes, even if not the designated replicas

```
Designated replicas: A, B, C
A is down during write

Strict quorum: FAIL (can't reach designated nodes)
Sloppy quorum: Write to B, C, D (D is "hint" node)
D stores: {data, hint: "belongs to A"}

When A recovers:
D checks: "Is A back?"
D transfers data to A → "hinted handoff"
D deletes its copy
```

**Trade-off:**

| Aspect | Strict Quorum | Sloppy Quorum |
|--------|---------------|---------------|
| Availability | Lower | Higher |
| Consistency | Stronger (W + R > N holds) | Weaker |
| Use Case | Banking, inventory | Shopping carts |

**Key insight:** Sloppy quorum breaks W + R > N guarantee because you might write to {B, C, D} and read from {A, B, C}—only B overlaps.

### Anti-Entropy and Read Repair

How do stale replicas catch up?

**Read Repair:**
```python
def read_with_repair(key):
    responses = [node.read(key) for node in replicas[:R]]
    latest = max(responses, key=lambda r: r.version)
    
    # Update any stale replicas
    for response in responses:
        if response.version < latest.version:
            response.node.write(key, latest.value, latest.version)
    
    return latest.value
```

**Anti-Entropy Process:** Background Merkle tree comparison

```
Node A's Merkle Tree:        Node B's Merkle Tree:
       [H_root]                     [H_root]
       /      \                     /      \
    [H_01]  [H_23]              [H_01]  [H_23'] ← different!
    /   \    /   \              /   \    /   \
  [H0] [H1] [H2] [H3]         [H0] [H1] [H2'] [H3]

Compare roots → different
Drill down → H_23 differs
Drill down → H2 differs
Sync only keys in bucket 2 (efficient!)
```

---

## Replication Topology Summary

| Topology | Consistency | Availability | Write Latency | Complexity |
|----------|-------------|--------------|---------------|------------|
| Single-Leader Sync | Strong | Lower | High | Low |
| Single-Leader Async | Eventual | Higher | Low | Low |
| Multi-Leader | Eventual + Conflicts | High | Low (local) | High |
| Leaderless | Tunable | Highest | Tunable | Medium |

---

## Interview Checklist

When discussing replication in interviews:

- [ ] Clarify durability requirements (can we lose data?)
- [ ] Discuss failover strategy and timing
- [ ] Address replication lag consequences
- [ ] Consider geographic distribution
- [ ] Mention specific technology (Postgres, Cassandra, etc.)
- [ ] Quantify latency impact (sync replication adds RTT)

---

## Staff+ Insights

**AWS RDS Multi-AZ failover** takes 60-120 seconds. For faster failover:
- Aurora: ~30 seconds
- Aurora Global Database: ~1 minute RPO, ~1 minute RTO

**PostgreSQL streaming replication** lag is typically <1 second under normal load, but can spike to minutes during heavy writes or network issues.

**Cassandra's default consistency** is ONE for both reads and writes—eventual consistency by default. Use QUORUM for stronger guarantees.
