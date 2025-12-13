# Chapter 36: Concurrency Primitives

## The Core Problem

Your API server handles 10,000 concurrent requests. Each request needs to:
- Read from a shared cache
- Update a counter
- Write to a connection pool

Without proper synchronization:

```
Thread A: reads counter = 100
Thread B: reads counter = 100
Thread A: writes counter = 101
Thread B: writes counter = 101  ← LOST UPDATE! Should be 102
```

At 10,000 RPS, you're losing hundreds of updates per second. Your metrics are wrong, your rate limiter is broken, and your connection pool is corrupted.

Concurrency primitives are the building blocks that prevent this chaos. Master them, and you can build thread-safe systems. Misuse them, and you'll create deadlocks, race conditions, and subtle bugs that only appear in production at 3 AM.

---

## Race Conditions: The Fundamental Problem

A race condition occurs when the behavior of code depends on the relative timing of events—typically when multiple threads access shared data without proper synchronization.

### The Classic Read-Modify-Write Race

```python
# BROKEN: Race condition
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        # Three operations, not atomic!
        temp = self.value    # 1. READ
        temp = temp + 1      # 2. MODIFY
        self.value = temp    # 3. WRITE
```

```
Thread interleaving that causes lost update:

Thread A                    Thread B
────────                    ────────
read value (0)
                            read value (0)
add 1 (temp=1)
                            add 1 (temp=1)
write value (1)
                            write value (1)  ← Overwrites!

Expected: 2, Actual: 1
```

### Data Races vs Race Conditions

| Type | Definition | Example |
|------|------------|---------|
| Data Race | Two threads access same memory, at least one writes, no synchronization | Concurrent read/write to `self.value` |
| Race Condition | Correctness depends on timing | Check-then-act on file existence |

A program can have race conditions without data races (using locks incorrectly) and data races without race conditions (benign concurrent writes of same value).

---

## Mutex (Mutual Exclusion Lock)

The most fundamental synchronization primitive. Only one thread can hold the mutex at a time.

### How It Works

```
┌─────────────────────────────────────────────────────┐
│                     MUTEX STATE                      │
├─────────────────────────────────────────────────────┤
│  locked = False, owner = None, waiters = []         │
└─────────────────────────────────────────────────────┘

Thread A calls lock():
┌─────────────────────────────────────────────────────┐
│  locked = True, owner = A, waiters = []             │
└─────────────────────────────────────────────────────┘

Thread B calls lock():
┌─────────────────────────────────────────────────────┐
│  locked = True, owner = A, waiters = [B]            │
│  Thread B is BLOCKED (sleeping)                     │
└─────────────────────────────────────────────────────┘

Thread A calls unlock():
┌─────────────────────────────────────────────────────┐
│  locked = True, owner = B, waiters = []             │
│  Thread B is WOKEN and now owns the mutex           │
└─────────────────────────────────────────────────────┘
```

### Python Implementation

```python
import threading

class ThreadSafeCounter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:  # Acquires on enter, releases on exit
            self.value += 1

    def get(self):
        with self.lock:
            return self.value

# Alternative: explicit acquire/release
def increment_explicit(self):
    self.lock.acquire()
    try:
        self.value += 1
    finally:
        self.lock.release()  # ALWAYS release in finally!
```

### Mutex Properties

| Property | Description |
|----------|-------------|
| Mutual Exclusion | Only one thread holds lock at a time |
| Progress | If no thread holds lock, one waiting thread will acquire it |
| Bounded Waiting | A thread won't wait forever (usually FIFO, but not guaranteed) |

### Common Mutex Pitfalls

**1. Forgetting to unlock (deadlock with self):**
```python
# BROKEN: Exception prevents unlock
def broken_increment(self):
    self.lock.acquire()
    self.value += 1  # If this raises, lock is never released!
    self.lock.release()
```

**2. Double locking (deadlock):**
```python
# BROKEN: Non-reentrant mutex
def outer(self):
    with self.lock:
        self.inner()  # Deadlock! Already holding lock

def inner(self):
    with self.lock:  # Blocks forever waiting for ourselves
        pass
```

**3. Lock ordering violation (deadlock between threads):**
```python
# Thread A: locks X, then Y
# Thread B: locks Y, then X
# → Circular wait → Deadlock
```

---

## Reentrant Lock (Recursive Mutex)

A mutex that can be acquired multiple times by the same thread.

```python
import threading

class ReentrantExample:
    def __init__(self):
        self.lock = threading.RLock()  # Reentrant lock
        self.data = []

    def add(self, item):
        with self.lock:
            self.data.append(item)
            self.log(f"Added {item}")  # Calls another locked method

    def log(self, message):
        with self.lock:  # Same thread can acquire again!
            print(f"[LOG] {message}, size={len(self.data)}")
```

### Reentrant Lock Internals

```
┌────────────────────────────────────────────────────┐
│  RLock State                                        │
├────────────────────────────────────────────────────┤
│  owner = Thread-A                                   │
│  count = 2        ← Acquired twice by Thread-A      │
│  waiters = [Thread-B]                               │
└────────────────────────────────────────────────────┘

Thread-A must call unlock() twice to fully release
```

| Mutex Type | Same Thread Re-acquire | Use Case |
|------------|----------------------|----------|
| Non-reentrant (Lock) | Deadlock | Simple critical sections |
| Reentrant (RLock) | Allowed | Recursive algorithms, nested calls |

**Trade-off:** RLock has slightly more overhead (must track owner and count).

---

## Semaphore

A generalized lock that allows N concurrent accessors.

```
┌─────────────────────────────────────────────────────┐
│                    SEMAPHORE                         │
├─────────────────────────────────────────────────────┤
│  count = 3   (3 permits available)                  │
│  waiters = []                                        │
└─────────────────────────────────────────────────────┘

acquire() → count-- (if count > 0)
release() → count++ (wakes one waiter if any)
```

### Use Case: Connection Pool

```python
import threading
import queue

class ConnectionPool:
    def __init__(self, max_connections=10):
        self.semaphore = threading.Semaphore(max_connections)
        self.connections = queue.Queue()

        # Pre-create connections
        for _ in range(max_connections):
            self.connections.put(self._create_connection())

    def get_connection(self):
        self.semaphore.acquire()  # Block if all connections in use
        return self.connections.get()

    def return_connection(self, conn):
        self.connections.put(conn)
        self.semaphore.release()  # Allow another thread to acquire

    def _create_connection(self):
        return {"id": id(self), "status": "open"}

# Usage
pool = ConnectionPool(max_connections=5)

def worker():
    conn = pool.get_connection()
    try:
        # Use connection...
        pass
    finally:
        pool.return_connection(conn)
```

### Binary Semaphore vs Mutex

A semaphore with count=1 looks like a mutex, but there's a key difference:

| Aspect | Mutex | Binary Semaphore |
|--------|-------|------------------|
| Owner | Has owner thread | No owner concept |
| Unlock | Only owner can unlock | Any thread can release |
| Use Case | Protect critical section | Signaling between threads |

```python
# Semaphore for signaling (producer-consumer)
ready = threading.Semaphore(0)  # Starts at 0!

def producer():
    data = produce_data()
    buffer.put(data)
    ready.release()  # Signal: data is ready

def consumer():
    ready.acquire()  # Wait for signal
    data = buffer.get()
    process(data)
```

---

## Condition Variable

Allows threads to wait for a specific condition to become true.

```
┌─────────────────────────────────────────────────────┐
│              CONDITION VARIABLE                      │
├─────────────────────────────────────────────────────┤
│  Associated lock: mutex                              │
│  Wait queue: [Thread-B, Thread-C]                    │
└─────────────────────────────────────────────────────┘

wait():  Release lock, sleep, re-acquire lock when woken
notify(): Wake one waiting thread
notify_all(): Wake all waiting threads
```

### Producer-Consumer with Condition Variable

```python
import threading
from collections import deque

class BoundedQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = deque()
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

    def put(self, item):
        with self.not_full:
            while len(self.queue) >= self.capacity:
                self.not_full.wait()  # Release lock, wait, re-acquire

            self.queue.append(item)
            self.not_empty.notify()  # Wake one consumer

    def get(self):
        with self.not_empty:
            while len(self.queue) == 0:
                self.not_empty.wait()

            item = self.queue.popleft()
            self.not_full.notify()  # Wake one producer
            return item
```

### Why "while" Instead of "if"?

```python
# BROKEN: Using if
if len(self.queue) == 0:
    self.not_empty.wait()
# After waking, queue might be empty again! (spurious wakeup)

# CORRECT: Using while
while len(self.queue) == 0:
    self.not_empty.wait()
# Re-check condition after every wakeup
```

**Spurious wakeups** can occur due to:
- OS implementation details
- `notify_all()` waking multiple threads when only one can proceed
- Signal interrupts

---

## Read-Write Lock (RWLock)

Optimizes for read-heavy workloads by allowing multiple concurrent readers.

```
┌─────────────────────────────────────────────────────┐
│                   RWLOCK STATE                       │
├─────────────────────────────────────────────────────┤
│  readers = 3                                         │
│  writer = None                                       │
│  writer_waiting = False                              │
└─────────────────────────────────────────────────────┘

Read lock: Allowed if no writer (readers++)
Write lock: Allowed only if readers=0 and no writer
```

### Implementation

```python
import threading

class ReadWriteLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.readers = 0
        self.writers_waiting = 0
        self.writer_active = False
        self.read_ok = threading.Condition(self.lock)
        self.write_ok = threading.Condition(self.lock)

    def acquire_read(self):
        with self.lock:
            while self.writer_active or self.writers_waiting > 0:
                self.read_ok.wait()
            self.readers += 1

    def release_read(self):
        with self.lock:
            self.readers -= 1
            if self.readers == 0:
                self.write_ok.notify()

    def acquire_write(self):
        with self.lock:
            self.writers_waiting += 1
            while self.readers > 0 or self.writer_active:
                self.write_ok.wait()
            self.writers_waiting -= 1
            self.writer_active = True

    def release_write(self):
        with self.lock:
            self.writer_active = False
            self.write_ok.notify()
            self.read_ok.notify_all()

# Usage
class Cache:
    def __init__(self):
        self.data = {}
        self.rwlock = ReadWriteLock()

    def get(self, key):
        self.rwlock.acquire_read()
        try:
            return self.data.get(key)
        finally:
            self.rwlock.release_read()

    def set(self, key, value):
        self.rwlock.acquire_write()
        try:
            self.data[key] = value
        finally:
            self.rwlock.release_write()
```

### Reader vs Writer Priority

| Policy | Behavior | Trade-off |
|--------|----------|-----------|
| Reader-preference | New readers allowed even if writer waiting | Writers can starve |
| Writer-preference | No new readers if writer waiting | Readers can starve |
| Fair | FIFO ordering for all | More complex, lower throughput |

---

## Spinlock

A lock that busy-waits instead of sleeping. Used when lock hold time is very short.

```python
import threading
import time

class SpinLock:
    def __init__(self):
        self.locked = False

    def acquire(self):
        while True:
            # Atomic test-and-set
            if not self.locked:
                self.locked = True
                return
            # Spin (busy-wait)

    def release(self):
        self.locked = False
```

**Note:** This Python example is illustrative. Real spinlocks require atomic CPU instructions.

### When to Use Spinlock vs Mutex

```
Lock Hold Time vs Context Switch Cost:

Short hold time (< 1μs):
┌─────────────────────────────────────────────────────┐
│ Spinlock wins: spinning cheaper than context switch │
└─────────────────────────────────────────────────────┘

Long hold time (> 10μs):
┌─────────────────────────────────────────────────────┐
│ Mutex wins: sleeping frees CPU for other work       │
└─────────────────────────────────────────────────────┘
```

| Aspect | Spinlock | Mutex |
|--------|----------|-------|
| Wait method | Busy-wait (burns CPU) | Sleep (yields CPU) |
| Overhead | No syscall | Context switch (~1-10μs) |
| Best for | Very short critical sections | Longer critical sections |
| Preemption | Must disable interrupts | Works with preemption |
| Use case | OS kernels, lock-free structures | Application code |

---

## Barriers

Synchronization point where threads wait until all have arrived.

```
Thread 1: ──────────────┐
                        │
Thread 2: ─────────┐    │
                   │    │
Thread 3: ────┐    │    ├──── BARRIER ────> All continue
              │    │    │
Thread 4: ───────────┐  │
                     │  │
                     ▼  ▼
              All 4 threads arrive, then proceed
```

### Use Case: Parallel Computation Phases

```python
import threading

def parallel_computation(data, barrier, thread_id, results):
    # Phase 1: Each thread processes its chunk
    results[thread_id] = process_chunk(data[thread_id])

    barrier.wait()  # Wait for all threads to finish Phase 1

    # Phase 2: Combine results (all threads see all Phase 1 results)
    if thread_id == 0:
        final_result = combine(results)

# Create barrier for 4 threads
barrier = threading.Barrier(4)
threads = []
results = [None] * 4

for i in range(4):
    t = threading.Thread(
        target=parallel_computation,
        args=(data_chunks, barrier, i, results)
    )
    threads.append(t)
    t.start()
```

---

## Deadlock

A situation where threads are blocked forever, each waiting for the other.

### Four Conditions for Deadlock

All four must be present:

1. **Mutual Exclusion**: Resources can't be shared
2. **Hold and Wait**: Thread holds resources while waiting for more
3. **No Preemption**: Resources can't be forcibly taken
4. **Circular Wait**: Thread A waits for B, B waits for A

```
┌──────────┐         ┌──────────┐
│ Thread A │         │ Thread B │
│ holds X  │◄───────►│ holds Y  │
│ wants Y  │         │ wants X  │
└──────────┘         └──────────┘
     ▲                     │
     └─────────────────────┘
           Circular Wait
```

### Prevention Strategies

**1. Lock Ordering (Break Circular Wait):**
```python
# BROKEN: Inconsistent order
def transfer_broken(from_acc, to_acc, amount):
    with from_acc.lock:        # A locks Account1
        with to_acc.lock:       # A waits for Account2
            # ...               # B locks Account2, waits for Account1

# FIXED: Consistent order by ID
def transfer_fixed(from_acc, to_acc, amount):
    first, second = sorted([from_acc, to_acc], key=lambda a: a.id)
    with first.lock:
        with second.lock:
            from_acc.balance -= amount
            to_acc.balance += amount
```

**2. Lock Timeout (Break Hold and Wait):**
```python
def try_transfer(from_acc, to_acc, amount, timeout=1.0):
    if from_acc.lock.acquire(timeout=timeout):
        try:
            if to_acc.lock.acquire(timeout=timeout):
                try:
                    from_acc.balance -= amount
                    to_acc.balance += amount
                    return True
                finally:
                    to_acc.lock.release()
        finally:
            from_acc.lock.release()
    return False  # Failed to acquire locks, retry later
```

**3. Deadlock Detection:**
```python
# Build wait-for graph, detect cycles
# If cycle found: abort one transaction, release its locks
```

---

## Lock-Free and Wait-Free Algorithms

Avoid locks entirely using atomic operations.

### Atomic Operations

```python
# Python's GIL makes simple operations atomic
# For explicit atomics, use ctypes or atomic libraries

from threading import Lock

# Compare-and-swap (CAS) pseudocode
def compare_and_swap(memory, expected, new_value):
    """Atomically: if memory == expected, set memory = new_value"""
    if memory.value == expected:
        memory.value = new_value
        return True
    return False
```

### Lock-Free Counter

```python
import threading
from ctypes import c_long

class LockFreeCounter:
    """Lock-free counter using CAS (conceptual)"""

    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()  # Simulating CAS

    def increment(self):
        while True:
            current = self.value
            # CAS: if value hasn't changed, update it
            with self._lock:  # Simulate atomic CAS
                if self.value == current:
                    self.value = current + 1
                    return
            # CAS failed, another thread modified; retry
```

### Lock-Free vs Wait-Free

| Type | Guarantee | Trade-off |
|------|-----------|-----------|
| Blocking | Thread may wait forever | Simple, uses mutexes |
| Lock-Free | System makes progress | Some threads may starve |
| Wait-Free | Every thread makes progress | Most complex, highest overhead |

---

## Python's Global Interpreter Lock (GIL)

Python (CPython) has a GIL that prevents true parallel execution of Python bytecode.

```
┌─────────────────────────────────────────────────────┐
│                   PYTHON GIL                         │
├─────────────────────────────────────────────────────┤
│  Only ONE thread executes Python bytecode at a time │
│  Released during I/O operations and C extensions    │
└─────────────────────────────────────────────────────┘

CPU-bound: Threads don't help (use multiprocessing)
I/O-bound: Threads help (GIL released during I/O)
```

### When Threading Helps in Python

```python
import threading
import time
import requests

# I/O-BOUND: Threading helps
def fetch_urls_threaded(urls):
    def fetch(url):
        return requests.get(url).status_code

    threads = [threading.Thread(target=fetch, args=(url,)) for url in urls]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

# CPU-BOUND: Use multiprocessing instead
from multiprocessing import Pool

def compute_heavy(n):
    return sum(i * i for i in range(n))

def compute_parallel(numbers):
    with Pool(4) as pool:
        return pool.map(compute_heavy, numbers)
```

### Implications for Concurrency Primitives

| Primitive | Still Needed? | Why |
|-----------|--------------|-----|
| Mutex | Yes | Compound operations (read-modify-write) |
| Condition | Yes | Thread coordination |
| Semaphore | Yes | Resource limiting |
| RWLock | Marginal | GIL already serializes, less benefit |

---

## Comparison Table

| Primitive | Concurrent Access | Use Case | Overhead |
|-----------|------------------|----------|----------|
| Mutex | 1 | General critical sections | Low |
| RLock | 1 (same thread multiple) | Recursive/nested locking | Low-Medium |
| Semaphore | N | Connection pools, rate limiting | Low |
| RWLock | N readers OR 1 writer | Read-heavy caches | Medium |
| Condition | N/A (coordination) | Producer-consumer, events | Low |
| Barrier | N/A (sync point) | Parallel phase coordination | Low |
| Spinlock | 1 | Very short critical sections | High CPU |

---

## Key Concepts Checklist

- [ ] Identify race conditions in concurrent code
- [ ] Choose between mutex, semaphore, and condition variable
- [ ] Implement producer-consumer with proper synchronization
- [ ] Prevent deadlocks with lock ordering
- [ ] Understand when to use RWLock vs Mutex
- [ ] Know Python GIL implications for threading
- [ ] Explain difference between lock-free and wait-free

---

## Practical Insights

**Lock granularity matters.** Coarse-grained locks (one lock for entire data structure) are simple but limit concurrency. Fine-grained locks (per-element locks) increase concurrency but add complexity and overhead. Start coarse, measure, then refine.

**Lock contention is often the bottleneck.** Profile your application before adding more locks. High contention on a single lock means threads spend most time waiting, not working. Consider lock striping (multiple locks for different portions of data).

**Deadlock debugging is painful.** Always acquire locks in a consistent global order. Document the order. Use lock timeouts in production to prevent indefinite hangs—better to fail fast than hang forever.

**Condition variables need the while-loop pattern.** Always re-check the condition after waking. Spurious wakeups are real, and notify_all() can wake multiple threads when only one should proceed.

**Python threading is for I/O, multiprocessing is for CPU.** The GIL means CPU-bound Python threads don't parallelize. For CPU-intensive work, use `multiprocessing` or move to native extensions (NumPy, Cython).

**Spinlocks are rarely appropriate in application code.** The break-even point is around 1μs—shorter than most database queries, API calls, or even memory allocations. Use regular mutexes unless you're writing OS kernels or lock-free data structures.
