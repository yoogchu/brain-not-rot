# Chapter 5: Consensus & Coordination

## The Split-Brain Problem

```
Database Cluster with 3 nodes: A, B, C
A is leader, B and C are followers

Network partition occurs:
┌─────────────┐         ┌─────────────┐
│   Node A    │    ✗    │   Node B    │
│  (leader)   │─────────│  (follower) │
│             │         │   Node C    │
└─────────────┘         │  (follower) │
                        └─────────────┘

Without consensus:
- A continues as leader (isolated, alone)
- B and C can't reach A, elect B as new leader
- Now TWO leaders accepting writes
- Data diverges catastrophically
- When partition heals: which data wins?
```

This is **split-brain**—the nightmare scenario that consensus algorithms prevent.

---

## Why Consensus Matters

Consensus enables distributed systems to:

1. **Leader Election**: Ensure exactly one leader at any time
2. **Atomic Broadcast**: Deliver messages in same order to all nodes
3. **Distributed Locks**: Exclusive access to resources
4. **Configuration Management**: Agree on cluster membership
5. **Replicated State Machines**: Keep replicas perfectly in sync

Without consensus, you can't build reliable distributed systems.

---

## The Raft Consensus Algorithm

Raft was designed for **understandability**. It's used by etcd, Consul, CockroachDB, TiKV.

### Three Roles

```
┌─────────────┐
│   Leader    │ ← Handles all client requests
│             │   Replicates to followers
└─────────────┘
      │
      ├──────────────────────────────────┐
      ▼                                  ▼
┌─────────────┐                  ┌─────────────┐
│  Follower   │                  │  Follower   │
│             │ ← Passive        │             │
│             │   Responds to    │             │
└─────────────┘   leader         └─────────────┘

During election:
┌─────────────┐
│  Candidate  │ ← Requesting votes
│             │   to become leader
└─────────────┘
```

### State Machine

```
            ┌───────────────────┐
            │     Follower      │
            └───────────────────┘
                    │
          Election timeout
          (no heartbeat from leader)
                    │
                    ▼
            ┌───────────────────┐
      ┌─────│     Candidate     │─────┐
      │     └───────────────────┘     │
      │             │                  │
  Discovers    Wins election      Split vote
  current      (majority)         (tie, retry)
  leader           │                   │
      │            │                   │
      ▼            ▼                   │
┌───────────┐  ┌───────────┐          │
│ Follower  │  │  Leader   │──────────┘
└───────────┘  └───────────┘
                    │
            Discovers higher term
            (newer leader exists)
                    │
                    ▼
            ┌───────────────────┐
            │     Follower      │
            └───────────────────┘
```

### Terms: Logical Time

Terms are Raft's logical clock. They monotonically increase.

```
Term 1: Node A is leader
        A sends heartbeats with term=1
        Everyone agrees A is leader
        
        [Network partition happens]
        
Term 2: A is unreachable
        B starts election, increments term to 2
        C votes for B (term=2 > term=1 is valid)
        B becomes leader with term=2
        
Term 3: A rejoins network
        A receives message with term=2
        A sees: "term 2 > my term 1"
        A: "I'm stale!" → steps down to follower
        A updates its term to 2
```

**Key rule:** Node with higher term wins. Lower term operations are rejected.

---

## Leader Election in Detail

### The Process

```
Initial state: All nodes start as followers
               A, B, C have term=0

Step 1: Election timeout
─────────────────────────
A's election timeout fires (randomized 150-300ms)
A hasn't heard from leader
A: "I'll try to become leader"

Step 2: Become candidate
────────────────────────
A increments term: term=1
A votes for itself
A sends RequestVote(term=1, candidateId=A) to B, C

Step 3: Voting
──────────────
B receives RequestVote(term=1, candidateId=A)
B checks: "term=1 > my term=0? Yes"
B checks: "Have I voted in term 1? No"
B: Grants vote to A, updates term to 1

C receives RequestVote(term=1, candidateId=A)
C: Same logic, grants vote to A

Step 4: Majority wins
─────────────────────
A receives 2 votes (B and C) + self-vote = 3 votes
3 of 3 nodes = majority
A becomes leader!
A immediately sends heartbeats to B, C
B, C: "Received heartbeat from leader A, stay follower"
```

### Why Randomized Timeouts?

```
Problem without randomization:
A timeout: 200ms
B timeout: 200ms
C timeout: 200ms

All timeout simultaneously → all become candidates
→ all vote for themselves → 3-way split → no winner
→ all timeout again → repeat forever

Solution with randomization:
A timeout: 180ms  ← fires first
B timeout: 250ms
C timeout: 230ms

A becomes candidate first → requests votes
B and C haven't timed out → still followers → grant votes
A wins before B or C try
```

Typical range: 150-300ms. Should be >> heartbeat interval (50ms).

---

## Log Replication

Once elected, the leader handles all writes:

```
Client sends: SET X=5

Leader's Log:
┌─────┬─────┬─────┬─────┐
│ 1:A │ 2:B │ 3:C │ 4:X │  (index:command)
└─────┴─────┴─────┴─────┘
                     ↑
                  Uncommitted (new entry)

Leader sends AppendEntries to followers:
AppendEntries(
  term=1,
  prevLogIndex=3,
  prevLogTerm=1,
  entries=[{index=4, term=1, cmd="SET X=5"}]
)

Follower B's Log before:
┌─────┬─────┬─────┐
│ 1:A │ 2:B │ 3:C │
└─────┴─────┴─────┘

Follower B:
- Checks: Do I have entry 3 with term 1? Yes
- Appends entry 4
- Sends ACK to leader

Follower B's Log after:
┌─────┬─────┬─────┬─────┐
│ 1:A │ 2:B │ 3:C │ 4:X │
└─────┴─────┴─────┴─────┘

Leader receives ACK from B and C (majority):
- Marks entry 4 as COMMITTED
- Applies to state machine
- Returns success to client
- Next heartbeat tells followers: "commit up to 4"
```

### The Commitment Rule

An entry is **committed** when stored on a **majority** of nodes.

```
5-node cluster: Need 3 nodes
3-node cluster: Need 2 nodes

Why majority?
- Any two majorities overlap by at least 1 node
- Future leader election requires majority
- Therefore, future leader WILL have committed entries
```

---

## Raft Safety Guarantees

### Election Safety

**At most one leader per term.**

```
Proof:
- Winning requires majority vote
- Each node votes at most once per term
- Majority = more than half
- Can't have two majorities (they'd overlap)
- Therefore: at most one winner per term
```

### Leader Completeness

**If entry committed in term T, present in all leaders for term > T.**

```
Entry committed → stored on majority
New leader elected → needed votes from majority
Majorities overlap → at least one voter has entry
Raft only elects candidates with most up-to-date log
→ New leader has committed entry
```

### State Machine Safety

**All nodes apply same commands in same order.**

```
Committed entries never change (leader completeness)
All nodes eventually receive all committed entries
Same sequence of commands → same final state
```

---

## Cluster Membership

### Why Odd Numbers?

```
5-node cluster:
Majority = 3
Can tolerate: 5 - 3 = 2 failures

4-node cluster:
Majority = 3 (need more than half of 4)
Can tolerate: 4 - 3 = 1 failure

Wait... 4 nodes tolerates FEWER failures than 5 nodes?

Yes! Adding even node doesn't help:
- 3 nodes: majority = 2, tolerate 1 failure
- 4 nodes: majority = 3, tolerate 1 failure  ← SAME!
- 5 nodes: majority = 3, tolerate 2 failures

Optimal cluster sizes:
- 3 nodes: tolerate 1 failure
- 5 nodes: tolerate 2 failures  
- 7 nodes: tolerate 3 failures
```

### Cluster Reconfiguration: The Danger

Changing membership is dangerous:

```
Bad approach (direct switch):
Old config: A, B, C (majority = 2)
New config: A, B, C, D, E (majority = 3)

During transition, both configs active:
- A, B could elect leader with old config (2 of 3)
- C, D, E could elect leader with new config (3 of 5)
- TWO LEADERS!
```

### Joint Consensus Solution

```
Step 1: Leader proposes C_old,new config
        (operations need majority of BOTH configs)

Step 2: Replicate C_old,new to majority
        Old config majority: A, B (2 of 3) ✓
        New config majority: A, B, C (3 of 5) ✓
        Both agree → committed

Step 3: Leader proposes C_new config
        Now only new config matters

Step 4: Replicate C_new to new majority
        Committed when 3 of 5 have it

Step 5: Remove nodes not in C_new
```

During joint consensus, can't have two leaders because any leader needs both majorities.

---

## ZooKeeper

Coordination service built on **ZAB** (ZooKeeper Atomic Broadcast) consensus.

### Data Model

Hierarchical namespace (like a filesystem):

```
/
├── services
│   ├── api-server
│   │   ├── instance-1: "host1:8080"
│   │   └── instance-2: "host2:8080"
│   └── database
│       └── leader: "host3:5432"
├── locks
│   └── job-scheduler: "worker-7"
└── config
    └── feature-flags: "{...}"
```

### Node Types

**Persistent Nodes:** Survive client disconnect, require explicit deletion
```python
zk.create("/config/db-url", b"postgres://localhost:5432")
# Survives forever until explicitly deleted
```

**Ephemeral Nodes:** Auto-deleted when client session ends
```python
zk.create("/services/api/instance-1", b"host1:8080", ephemeral=True)
# If this client dies or disconnects, node disappears
# Perfect for service discovery!
```

**Sequential Nodes:** Auto-appended incrementing counter
```python
zk.create("/locks/job-", sequential=True)
# Creates: /locks/job-0000000001
# Next:    /locks/job-0000000002
# Perfect for distributed locks and queues!
```

### Watches

One-time notifications when nodes change:

```python
def watch_callback(event):
    print(f"Node changed: {event.type} on {event.path}")
    # Re-register watch for continuous monitoring
    zk.get("/config/feature-flags", watch=watch_callback)

# Register watch
data, stat = zk.get("/config/feature-flags", watch=watch_callback)
# Callback fires ONCE when node changes
```

---

## Common ZooKeeper Patterns

### Service Discovery

```python
# Service registers itself (ephemeral = auto-cleanup on death)
def register_service(service_name, host, port):
    instance_id = f"{host}:{port}"
    zk.create(
        f"/services/{service_name}/{instance_id}",
        b"",
        ephemeral=True
    )

# Client discovers services
def discover_services(service_name):
    instances = zk.get_children(f"/services/{service_name}")
    return instances  # ["host1:8080", "host2:8080"]

# If host1 crashes, its ephemeral node disappears
# Next discover_services() call won't include it
```

### Leader Election

```python
def participate_in_election(election_path):
    # Create sequential ephemeral node
    my_node = zk.create(
        f"{election_path}/candidate-",
        ephemeral=True,
        sequence=True
    )
    # Creates: /election/candidate-0000000042
    
    while True:
        # Get all candidates, sorted
        candidates = sorted(zk.get_children(election_path))
        my_name = my_node.split("/")[-1]
        
        if my_name == candidates[0]:
            # I'm the smallest = I'm the leader!
            return "LEADER"
        else:
            # Watch the candidate just before me
            my_index = candidates.index(my_name)
            predecessor = candidates[my_index - 1]
            
            # Wait for predecessor to die
            if zk.exists(f"{election_path}/{predecessor}", watch=election_watch):
                wait_for_watch()
            # Loop and check again
```

**Why watch predecessor, not leader?**
- If 100 nodes watch leader, leader death triggers 100 watches → thundering herd
- If each watches predecessor, leader death triggers 1 watch → graceful

### Distributed Lock

```python
def acquire_lock(lock_path):
    # Same as leader election!
    lock_node = zk.create(
        f"{lock_path}/lock-",
        ephemeral=True,
        sequence=True
    )
    
    while True:
        children = sorted(zk.get_children(lock_path))
        my_name = lock_node.split("/")[-1]
        
        if my_name == children[0]:
            # I have the lock
            return lock_node
        else:
            # Wait for predecessor
            my_index = children.index(my_name)
            predecessor = children[my_index - 1]
            wait_for_node_deletion(f"{lock_path}/{predecessor}")

def release_lock(lock_node):
    zk.delete(lock_node)
    # Or just disconnect - ephemeral node auto-deletes
```

---

## Comparison: Raft vs Paxos vs ZAB

| Aspect | Raft | Paxos | ZAB |
|--------|------|-------|-----|
| Understandability | High (designed for it) | Low (notoriously complex) | Medium |
| Leader-based | Yes, single leader | Multi-Paxos has leader | Yes, single leader |
| Reconfiguration | Joint consensus | Complex | Dynamic membership |
| Used by | etcd, Consul, TiKV | Spanner, Chubby | ZooKeeper |
| Learning curve | Days | Weeks/Months | Days |

### When to Use Each

**Raft/etcd/Consul:**
- Configuration storage
- Service discovery
- Leader election
- Small, critical data (< 1GB typically)

**ZooKeeper:**
- Coordination between services
- Distributed locks
- Barrier synchronization
- Mature ecosystem, battle-tested

**Paxos/Spanner:**
- Global-scale databases
- When you need Google-level engineering
- Rarely implemented directly

---

## Consensus Limitations

### Performance Costs

**Latency:**
```
Minimum: 1 RTT for leader to get majority ACK
Cross-datacenter (NYC ↔ London): 70ms RTT
Every write: 70ms+ latency

Compare to single-node: <1ms
```

**Throughput:**
```
All writes go through one leader
Leader becomes bottleneck
Typical: 10K-50K writes/second
Compare to sharded system: millions/second
```

### Availability Constraints

```
Need majority online to make progress

3-node cluster: Need 2 online
5-node cluster: Need 3 online

During leader election: No writes processed
Election time: 150-500ms typically
```

### When NOT to Use Consensus

| Scenario | Better Alternative |
|----------|-------------------|
| Read-heavy workloads | Eventual consistency with read replicas |
| Cross-region writes | CRDTs or last-write-wins |
| High-throughput counters | CRDTs (G-Counter) |
| Cache coordination | TTL-based expiration |
| Fire-and-forget events | Message queue |

---

## Key Concepts Checklist

- [ ] Explain why consensus is needed (avoid split-brain)
- [ ] Describe leader election mechanism
- [ ] Discuss log replication and commitment
- [ ] Explain majority quorum math (why odd numbers)
- [ ] Know when NOT to use consensus
- [ ] Mention specific implementations (Raft in etcd, ZAB in ZooKeeper)

---

## Practical Insights

**Consensus is a building block, not a solution:**
- Don't use ZooKeeper for everything
- It's for coordination, not data storage
- Keep consensus cluster small (3-5 nodes)

**Network partitions are rare but devastating:**
- Test partition scenarios explicitly
- Know your system's behavior during partition
- Have runbooks for split-brain recovery

**etcd vs ZooKeeper:**
- etcd: Simpler API, gRPC, Kubernetes native
- ZooKeeper: More features (watches, sequences), older ecosystem
- For new projects: etcd is usually the choice

**Raft implementations vary:**
- etcd's Raft is battle-tested
- DIY Raft is surprisingly hard to get right
- Use existing implementations when possible
