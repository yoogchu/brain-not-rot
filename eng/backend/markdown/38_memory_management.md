# Chapter 38: Memory Management

## The Core Problem

Your production service handles 10,000 QPS with 200ms p99 latency. Over the weekend, traffic spikes to 15,000 QPS. Suddenly:
- Latency jumps to 3 seconds
- CPU spikes to 100% but most time is in garbage collection
- Server starts swapping to disk (death spiral)
- OOM killer terminates your process
- Complete outage

Without understanding memory management:
- You blindly increase heap size → longer GC pauses
- You restart services → temporary fix, problem returns
- You scale horizontally → expensive, doesn't solve root cause

With proper memory management:
- Identify memory leaks before they cause outages
- Tune GC for your workload (throughput vs latency)
- Use memory pools to reduce allocation overhead
- Understand cache effects that make code 10x faster
- Know when to use off-heap storage

Memory is the foundation. Everything else builds on it.

---

## Stack vs Heap Memory

### The Fundamental Split

Every program divides memory into two regions with opposite characteristics:

```
┌──────────────────────────────────────────────────┐
│               Process Memory                      │
├──────────────────────────────────────────────────┤
│                                                   │
│  ┌────────────────────────────────┐              │
│  │          Stack                 │              │
│  │  (grows downward ▼)            │              │
│  │  ┌──────────────────────┐      │              │
│  │  │ local_var = 42       │      │ ← Fast       │
│  │  │ function_call()      │      │ ← Automatic  │
│  │  └──────────────────────┘      │ ← Limited    │
│  │                                 │              │
│  └────────────────────────────────┘              │
│               ↕                                   │
│         Free space                                │
│               ↕                                   │
│  ┌────────────────────────────────┐              │
│  │          Heap                  │              │
│  │  (grows upward ▲)              │              │
│  │  ┌──────────────────────┐      │              │
│  │  │ malloc(1024)         │      │ ← Flexible   │
│  │  │ objects, arrays      │      │ ← Manual     │
│  │  └──────────────────────┘      │ ← Unlimited* │
│  │                                 │              │
│  └────────────────────────────────┘              │
│                                                   │
└──────────────────────────────────────────────────┘
```

### Stack: The Fast Path

**How it works:**
- Stack pointer register (ESP/RSP) tracks top of stack
- Function call: Push return address, allocate frame
- Function return: Pop frame, restore stack pointer
- All in CPU registers (nanosecond operations)

**Memory layout:**
```
Function call: foo(x, y) → bar(z)

High addresses
┌────────────────────┐
│  foo's local vars  │ ← foo's stack frame
├────────────────────┤
│  return address    │
├────────────────────┤
│  saved registers   │
├────────────────────┤
│  bar's local vars  │ ← bar's stack frame (current)
├────────────────────┤ ← Stack pointer (ESP)
│                    │
└────────────────────┘
Low addresses
```

**Example:**
```python
def calculate(x, y):
    # All these variables live on the stack
    result = x + y      # 8 bytes (int64)
    doubled = result * 2  # 8 bytes
    return doubled      # Frame destroyed on return

# Stack frame created and destroyed in nanoseconds
calculate(10, 20)
```

**Characteristics:**
| Aspect | Detail |
|--------|--------|
| Speed | Extremely fast (just adjust stack pointer) |
| Size | Limited (typically 1-8 MB per thread) |
| Lifetime | Automatic (destroyed on function return) |
| Fragmentation | None (always contiguous) |
| Thread safety | Each thread has own stack |

**When to use:**
- Local variables
- Function parameters
- Small arrays (< few KB)
- Short-lived data

**When NOT to use:**
- Large objects (will overflow stack)
- Data that outlives function call
- Dynamically sized data
- Shared between threads

### Heap: The Flexible Alternative

**How it works:**
- Managed by allocator (malloc/free, or GC)
- Allocator tracks free regions via free lists or bitmaps
- Allocation finds suitable free block
- Deallocation marks block as free for reuse

**Example allocator state:**
```
Heap memory:
┌──────────┬─────────┬──────────┬─────────┬──────────┐
│ Used     │ Free    │ Used     │ Free    │ Used     │
│ 24 bytes │ 64 bytes│ 48 bytes │ 128 B   │ 16 bytes │
└──────────┴─────────┴──────────┴─────────┴──────────┘
           ▲                    ▲
           │                    │
        Free list: 64 bytes → 128 bytes → NULL
```

**Python example:**
```python
def process_data():
    # This list lives on the heap
    large_data = [0] * 1000000  # 8 MB array

    # Can return heap-allocated data
    result = {"processed": large_data}
    return result  # Stack frame destroyed, heap data survives

# Heap allocation persists
data = process_data()
```

**Characteristics:**
| Aspect | Detail |
|--------|--------|
| Speed | Slower (search free lists, metadata overhead) |
| Size | Limited only by RAM/virtual memory |
| Lifetime | Manual (C) or GC (Python/Java) |
| Fragmentation | Common problem |
| Thread safety | Requires synchronization |

**Trade-off summary:**

| Use Case | Stack | Heap |
|----------|-------|------|
| Local counter | ✓ | |
| 10 MB buffer | | ✓ |
| Return from function | | ✓ |
| Per-thread cache | ✓ | |
| Recursive call data | ✓ | ✗ (overflow risk) |

---

## Memory Allocation Strategies

### The malloc Problem

**What malloc must do:**
1. Find a free block of sufficient size
2. Split block if too large
3. Return pointer to usable memory
4. Track allocation metadata

**Cost:** 100-1000 CPU cycles (vs 1-2 cycles for stack)

### Strategy 1: Free List

```
┌──────────────────────────────────────────────────┐
│                 Heap                              │
│                                                   │
│  Free list: Head → 32 bytes → 64 bytes → NULL    │
│                      │           │                │
│  ┌─────┬────────┬───▼───┬──┬────▼─────┬─────┐   │
│  │Used │  Used  │ Free  │U │  Free    │Used │   │
│  │ 16B │  24B   │ 32B   │8B│  64B     │ 40B │   │
│  └─────┴────────┴───────┴──┴──────────┴─────┘   │
│                                                   │
└──────────────────────────────────────────────────┘
```

**Allocation strategies:**

**First-fit:** Find first block large enough
```python
def malloc_first_fit(size):
    for block in free_list:
        if block.size >= size:
            allocate(block, size)
            return block.address
    return None  # Out of memory
```
- Fast (stops at first match)
- Poor memory utilization (leaves small gaps)

**Best-fit:** Find smallest block that fits
```python
def malloc_best_fit(size):
    best = None
    for block in free_list:
        if block.size >= size:
            if best is None or block.size < best.size:
                best = block
    if best:
        allocate(best, size)
        return best.address
    return None
```
- Better utilization
- Slower (must scan entire list)
- Creates tiny unusable fragments

**Worst-fit:** Find largest block
- Idea: Leave large fragments for future large allocations
- Rarely used (poor performance in practice)

### Strategy 2: Segregated Free Lists

Separate free lists per size class:

```
Size classes:
┌────────────────────────────────────────────────┐
│  16 bytes:  ● → ● → ● → NULL                  │
│  32 bytes:  ● → ● → NULL                      │
│  64 bytes:  ● → NULL                          │
│  128 bytes: ● → ● → ● → ● → NULL              │
│  256 bytes: ● → NULL                          │
│  512 bytes: ● → ● → NULL                      │
│  Large:     ● → NULL                          │
└────────────────────────────────────────────────┘
```

**Allocation:**
```python
SIZE_CLASSES = [16, 32, 64, 128, 256, 512]

def malloc_segregated(size):
    # Round up to nearest size class
    size_class = next(s for s in SIZE_CLASSES if s >= size)

    free_list = free_lists[size_class]
    if free_list:
        return free_list.pop()

    # Allocate new chunk from OS
    return allocate_from_os(size_class)
```

**Advantages:**
- O(1) allocation (no searching)
- Reduced fragmentation (same-size allocations)
- Cache-friendly (similar objects together)

**Used by:** tcmalloc, jemalloc, glibc malloc

### Strategy 3: Memory Pools (Arena Allocation)

Pre-allocate large chunk, carve into fixed-size blocks:

```python
class MemoryPool:
    def __init__(self, block_size, num_blocks):
        self.block_size = block_size
        self.pool = bytearray(block_size * num_blocks)

        # Initialize free list
        self.free_list = []
        for i in range(num_blocks):
            offset = i * block_size
            self.free_list.append(offset)

    def allocate(self):
        if not self.free_list:
            raise MemoryError("Pool exhausted")

        offset = self.free_list.pop()
        return memoryview(self.pool)[offset:offset + self.block_size]

    def free(self, offset):
        self.free_list.append(offset)

# Usage
pool = MemoryPool(block_size=64, num_blocks=1000)

# Fast allocation (just pop from list)
obj1 = pool.allocate()
obj2 = pool.allocate()

# Fast deallocation (just push to list)
pool.free(obj1)
```

**When to use:**
- Many same-size allocations (e.g., fixed-size packets)
- High allocation/deallocation frequency
- Need deterministic allocation time

**Real-world example:**
```python
# Web server connection pool
class Connection:
    __slots__ = ['socket', 'buffer', 'state']  # Fixed size

connection_pool = MemoryPool(sizeof(Connection), 10000)

def handle_new_connection(socket):
    # No malloc overhead!
    conn = connection_pool.allocate()
    conn.socket = socket
    conn.buffer = bytearray(4096)
    conn.state = 'ACTIVE'
    return conn
```

---

## Garbage Collection Algorithms

### Why GC Exists

Manual memory management (C/C++):
```c
char* data = malloc(1024);  // Allocate
// ... use data ...
free(data);  // Must remember to free!
```

**Problems:**
- Forget to free → memory leak
- Use after free → crash or security bug
- Double free → corruption

**GC trade-off:** Automatic safety vs performance overhead

### Reference Counting

Track number of references to each object:

```
Object A: ref_count = 2
         ▲      ▲
         │      │
    var_x    var_y

When var_x deleted:
Object A: ref_count = 1 (still alive)

When var_y deleted:
Object A: ref_count = 0 → DEALLOCATE
```

**Implementation:**
```python
class RefCountedObject:
    def __init__(self, value):
        self.value = value
        self.ref_count = 1  # Created with 1 reference

    def incref(self):
        self.ref_count += 1

    def decref(self):
        self.ref_count -= 1
        if self.ref_count == 0:
            self.cleanup()  # Deallocate

    def cleanup(self):
        print(f"Deallocating {self.value}")
        # Free memory

# Usage
obj = RefCountedObject("data")  # ref_count = 1
x = obj
obj.incref()  # ref_count = 2

del obj
x.decref()  # ref_count = 1

del x
x.decref()  # ref_count = 0 → cleanup called
```

**Pros:**
- Immediate reclamation (no pauses)
- Deterministic (know exactly when freed)
- Simple to understand

**Cons:**
- Reference counting overhead on every assignment
- Cannot handle cycles:
```
Object A: ref_count = 1
    │         ▲
    ▼         │
Object B: ref_count = 1
```
Both have ref_count = 1 but are unreachable from program!

**Used by:** Python (primary GC), Swift, Objective-C

### Mark-and-Sweep

**Two phases:**
1. **Mark:** Traverse all reachable objects from roots
2. **Sweep:** Free all unmarked objects

```
Initial state:
Roots → [A] → [B] → [C]
        [D] → [E]
        [F] (unreachable)

Mark phase (depth-first search):
A.marked = True
B.marked = True (reached from A)
C.marked = True (reached from B)
D.marked = True (root)
E.marked = True (reached from D)
F.marked = False (never reached)

Sweep phase:
Free all objects where marked = False
→ Free F
```

**Implementation:**
```python
class MarkSweepGC:
    def __init__(self):
        self.all_objects = []
        self.roots = []

    def mark(self):
        # Clear all marks
        for obj in self.all_objects:
            obj.marked = False

        # Mark from roots (DFS)
        def mark_recursive(obj):
            if obj.marked:
                return
            obj.marked = True
            for child in obj.children:
                mark_recursive(child)

        for root in self.roots:
            mark_recursive(root)

    def sweep(self):
        alive = []
        for obj in self.all_objects:
            if obj.marked:
                alive.append(obj)
            else:
                obj.cleanup()  # Free memory

        self.all_objects = alive

    def collect(self):
        self.mark()
        self.sweep()
```

**Pros:**
- Handles cycles (if unreachable from roots, gets swept)
- No per-assignment overhead

**Cons:**
- "Stop-the-world" pause (entire program frozen during collection)
- Heap fragmentation (creates gaps)

**Pause time example:**
```
10 million objects in heap
Mark phase: 50ms (traverse object graph)
Sweep phase: 20ms (scan all objects)
Total pause: 70ms (user sees latency spike!)
```

### Generational GC

**Key insight:** Most objects die young.

```
Empirical observation:
90% of objects become garbage within seconds
5% live a bit longer
5% live for the entire program lifetime
```

**Divide heap into generations:**

```
┌──────────────────────────────────────────────────┐
│                                                   │
│  ┌──────────────────┐                            │
│  │   Young Gen      │ ← Most allocations         │
│  │   (Eden + S0/S1) │ ← Frequent, fast GC        │
│  │   Size: 10 MB    │ ← 100ms pauses             │
│  └──────────────────┘                            │
│           │                                       │
│           │ Survivors promoted ▼                 │
│           │                                       │
│  ┌──────────────────┐                            │
│  │   Old Gen        │ ← Long-lived objects       │
│  │   Size: 1 GB     │ ← Infrequent GC            │
│  │                  │ ← 1000ms+ pauses           │
│  └──────────────────┘                            │
│                                                   │
└──────────────────────────────────────────────────┘
```

**Algorithm:**
```python
class GenerationalGC:
    def __init__(self):
        self.young = []
        self.old = []
        self.minor_gc_threshold = 10_000  # objects
        self.major_gc_threshold = 100_000

    def allocate(self, obj):
        self.young.append(obj)

        if len(self.young) > self.minor_gc_threshold:
            self.minor_gc()

    def minor_gc(self):
        """Collect young generation only"""
        print("Minor GC (young gen)")

        # Mark from roots, but STOP at old gen
        survivors = mark_from_roots(
            self.young,
            stop_at=self.old
        )

        # Promote survivors to old gen
        for obj in survivors:
            if obj.age > 3:  # Survived 3 collections
                self.old.append(obj)
            else:
                obj.age += 1
                survivors_kept.append(obj)

        self.young = survivors_kept

        # Check if old gen needs collection
        if len(self.old) > self.major_gc_threshold:
            self.major_gc()

    def major_gc(self):
        """Collect all generations (expensive!)"""
        print("Major GC (full heap)")

        # Mark and sweep entire heap
        all_objects = self.young + self.old
        survivors = mark_and_sweep(all_objects)

        # Rebuild generations
        self.young = [o for o in survivors if o.age <= 3]
        self.old = [o for o in survivors if o.age > 3]
```

**GC frequency example:**
```
Minor GC: Every 100ms (10ms pause)
Major GC: Every 60 seconds (500ms pause)

99.9% of GC pauses are short (10ms)
0.1% of GC pauses are long (500ms)
```

**Used by:** JVM (G1GC, ZGC), .NET, Python (for cycles)

### Copying GC

Divide heap into two semi-spaces. Only one is active at a time.

```
Initial state (using From-space):
┌─────────────────┐  ┌─────────────────┐
│  From-space     │  │   To-space      │
│  [A][B][_][C]   │  │   [_][_][_][_]  │
│  [_][_][D][_]   │  │   [_][_][_][_]  │
└─────────────────┘  └─────────────────┘
   ▲ In use              Empty

GC triggered:
1. Copy live objects (A, B, C, D) to To-space
2. Compact them (no gaps)
3. Swap roles

After GC (now using To-space):
┌─────────────────┐  ┌─────────────────┐
│  From-space     │  │   To-space      │
│  [_][_][_][_]   │  │  [A][B][C][D]   │
│  [_][_][_][_]   │  │  [_][_][_][_]   │
└─────────────────┘  └─────────────────┘
   Empty                ▲ In use (compacted!)
```

**Implementation:**
```python
class CopyingGC:
    def __init__(self, heap_size):
        self.from_space = bytearray(heap_size // 2)
        self.to_space = bytearray(heap_size // 2)
        self.alloc_ptr = 0

    def allocate(self, size):
        if self.alloc_ptr + size > len(self.from_space):
            self.collect()

        addr = self.alloc_ptr
        self.alloc_ptr += size
        return addr

    def collect(self):
        # Copy live objects from from_space to to_space
        new_alloc_ptr = 0

        for obj in self.get_live_objects():
            # Copy to to_space
            size = obj.size
            memcpy(
                self.to_space[new_alloc_ptr:],
                self.from_space[obj.addr:obj.addr + size]
            )

            # Update forwarding pointer
            obj.new_addr = new_alloc_ptr
            new_alloc_ptr += size

        # Swap spaces
        self.from_space, self.to_space = self.to_space, self.from_space
        self.alloc_ptr = new_alloc_ptr

        # to_space is now empty, ready for next allocation
```

**Pros:**
- Automatic compaction (no fragmentation!)
- Fast allocation (just bump pointer)
- Only touches live objects (ignores garbage)

**Cons:**
- Only use 50% of heap at any time
- Pause time proportional to live set size

**When to use:**
- Young generation GC (mostly garbage, little live data)
- Low fragmentation requirement

---

## Memory Leaks

### What Is a Memory Leak?

Memory that's allocated but never freed, despite being unreachable.

**Example (Python):**
```python
# Global cache that never shrinks
cache = {}

def process_request(user_id):
    # Cache grows unbounded
    if user_id not in cache:
        cache[user_id] = expensive_computation(user_id)

    return cache[user_id]

# After 1 million users:
# cache has 1 million entries, even if most users never return
# Memory leak!
```

### Detecting Leaks

**Symptom checklist:**
- [ ] Memory usage grows over time
- [ ] Memory never decreases, even under low load
- [ ] Eventually crashes with OOM
- [ ] Restarting temporarily fixes the issue

**Detection tools:**

**1. Memory profiling:**
```python
import tracemalloc

tracemalloc.start()

# Run code
process_requests()

# Take snapshot
snapshot = tracemalloc.take_snapshot()

# Top 10 memory allocators
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

Output:
```
/app/cache.py:42: 245 MB
/app/database.py:103: 89 MB
/app/handlers.py:205: 34 MB
```

**2. Heap dump analysis:**
```bash
# Take heap dump (Python)
python -m guppy

# JVM heap dump
jmap -dump:format=b,file=heap.bin <pid>

# Analyze with Eclipse Memory Analyzer
# Look for: Objects with high retained size
```

**3. Growth rate monitoring:**
```python
import psutil
import time

def monitor_memory():
    process = psutil.Process()
    baseline = process.memory_info().rss

    while True:
        time.sleep(60)
        current = process.memory_info().rss
        growth = (current - baseline) / baseline * 100

        print(f"Memory growth: {growth:.1f}%")

        if growth > 50:  # 50% growth per hour
            alert("Possible memory leak!")
```

### Common Leak Patterns

**1. Event listeners not removed:**
```python
class Dashboard:
    def __init__(self, event_bus):
        # Register listener
        event_bus.subscribe('update', self.on_update)

    def close(self):
        # BUG: Forgot to unsubscribe!
        # event_bus holds reference to self
        # Dashboard never gets garbage collected
        pass

# Fix:
def close(self):
    event_bus.unsubscribe('update', self.on_update)
```

**2. Circular references (Python with __del__):**
```python
class Parent:
    def __init__(self):
        self.child = Child(self)

    def __del__(self):
        print("Parent deleted")

class Child:
    def __init__(self, parent):
        self.parent = parent

    def __del__(self):
        print("Child deleted")

# Creates cycle: Parent → Child → Parent
# If __del__ is defined, GC won't break cycle!
p = Parent()
del p  # Not deleted! Leak!
```

**3. Unbounded cache:**
```python
# BAD: Unbounded cache
cache = {}

def get_user(user_id):
    if user_id not in cache:
        cache[user_id] = db.fetch_user(user_id)
    return cache[user_id]

# GOOD: LRU cache with max size
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_user(user_id):
    return db.fetch_user(user_id)
```

---

## Memory Fragmentation

### The Problem

```
After many allocations/deallocations:

┌──────────────────────────────────────────────────┐
│  [Used][Free][Used][Free][Used][Free][Used]      │
│   32B   8B   16B   12B   24B   4B    40B         │
└──────────────────────────────────────────────────┘
        ▲           ▲           ▲
     8B free     12B free    4B free

Total free: 24 bytes
But cannot satisfy: malloc(16) → FAIL!
Free space is fragmented into small chunks
```

### External vs Internal Fragmentation

**External fragmentation:**
- Free space divided into small, unusable gaps
- Caused by varying allocation sizes

**Internal fragmentation:**
- Waste within allocated blocks
- Allocator rounds up (e.g., 25 bytes → allocate 32 bytes)

**Example:**
```python
# Request 25 bytes
# Allocator uses 32-byte size class
ptr = malloc(25)

# Actual allocation:
# [25 bytes used | 7 bytes wasted]
# 7 bytes of internal fragmentation
```

### Solutions

**1. Compacting GC:**
Move objects to eliminate gaps (copying GC does this automatically)

**2. Memory pools:**
Fixed-size allocations avoid fragmentation

**3. Buddy allocator:**
```
Split blocks in powers of 2:

128 bytes available
Request 20 bytes (rounds to 32)

Split: 128 → 64 + 64
Split: 64 → 32 + 32
Allocate: 32 bytes

Remaining free:
32, 64 bytes (clean power-of-2 chunks)
```

---

## Memory-Mapped Files

### The Concept

Map file directly into process virtual memory:

```
┌──────────────────────────────────────────────────┐
│           Process Virtual Memory                  │
│                                                   │
│  ┌────────────────────────────────┐              │
│  │    Code segment                │              │
│  ├────────────────────────────────┤              │
│  │    Heap                        │              │
│  ├────────────────────────────────┤              │
│  │    Memory-mapped file          │ ◄────┐       │
│  │    (appears as byte array)     │      │       │
│  │    [0][1][2][3]...[1000000]    │      │       │
│  └────────────────────────────────┘      │       │
│                                           │       │
└───────────────────────────────────────────┼───────┘
                                            │
                                            │
                                  ┌─────────▼──────┐
                                  │  Disk file     │
                                  │  /data/log.txt │
                                  └────────────────┘
```

**Usage:**
```python
import mmap
import os

# Open file
fd = os.open('large_file.dat', os.O_RDWR)

# Memory-map it
mm = mmap.mmap(fd, 0)  # Map entire file

# Access like array (no read() calls!)
byte = mm[1000]  # Read byte 1000
mm[1000] = 42    # Write byte 1000

# OS handles caching, paging automatically
mm.close()
```

**When to use:**
- Large files (> 100 MB)
- Random access patterns
- Shared memory between processes
- Database implementations

**Benefits:**
- OS handles caching (no manual buffering)
- Multiple processes share same physical pages
- Lazy loading (only load pages when accessed)

**Example (shared counter):**
```python
# Process 1
mm = mmap.mmap(-1, 8)  # Anonymous shared memory
mm[0:8] = (0).to_bytes(8, 'little')

# Process 2 (shares same memory)
mm = mmap.mmap(-1, 8)
counter = int.from_bytes(mm[0:8], 'little')
counter += 1
mm[0:8] = counter.to_bytes(8, 'little')

# Both processes see the same counter!
```

---

## Cache Locality and Memory Access Patterns

### Why Locality Matters

**Memory hierarchy:**
```
┌────────────────────────────────────────┐
│  CPU Register: 1 cycle (0.3 ns)       │ ← Fastest
├────────────────────────────────────────┤
│  L1 Cache: 4 cycles (1 ns)            │ ← 32 KB
├────────────────────────────────────────┤
│  L2 Cache: 12 cycles (3 ns)           │ ← 256 KB
├────────────────────────────────────────┤
│  L3 Cache: 40 cycles (10 ns)          │ ← 8 MB
├────────────────────────────────────────┤
│  RAM: 200 cycles (60 ns)              │ ← Slow!
├────────────────────────────────────────┤
│  SSD: 50,000 cycles (15 µs)           │ ← 250x slower
├────────────────────────────────────────┤
│  HDD: 10,000,000 cycles (3 ms)        │ ← 50,000x slower
└────────────────────────────────────────┘
```

**Cache line:** 64 bytes loaded at once

### Sequential vs Random Access

**Sequential access (cache-friendly):**
```python
import time

# Array of 10 million integers
arr = list(range(10_000_000))

start = time.time()
total = 0
for i in range(len(arr)):
    total += arr[i]  # Sequential access

elapsed = time.time() - start
print(f"Sequential: {elapsed:.3f}s")  # 0.150s
```

**Random access (cache-hostile):**
```python
import random

indices = list(range(10_000_000))
random.shuffle(indices)

start = time.time()
total = 0
for i in indices:
    total += arr[i]  # Random access

elapsed = time.time() - start
print(f"Random: {elapsed:.3f}s")  # 1.500s (10x slower!)
```

**Why?** Sequential access: Next element likely in cache
Random access: Every access might be a cache miss

### Struct of Arrays vs Array of Structs

**Array of Structs (AoS):**
```python
# Each particle is a struct
particles = [
    {'x': 1.0, 'y': 2.0, 'vx': 0.1, 'vy': 0.2},
    {'x': 1.5, 'y': 2.5, 'vx': 0.3, 'vy': 0.4},
    # ... 1 million particles
]

# Update positions
for p in particles:
    p['x'] += p['vx']  # Access x, vx (scattered in memory)
    p['y'] += p['vy']  # Access y, vy (scattered in memory)
```

Memory layout:
```
[x y vx vy | x y vx vy | x y vx vy | ...]
 ▲  access x → cache loads all 4 fields (waste!)
```

**Struct of Arrays (SoA):**
```python
# Separate arrays for each field
particles = {
    'x': [1.0, 1.5, ...],   # 1 million x values
    'y': [2.0, 2.5, ...],   # 1 million y values
    'vx': [0.1, 0.3, ...],  # 1 million vx values
    'vy': [0.2, 0.4, ...],  # 1 million vy values
}

# Update positions
n = len(particles['x'])
for i in range(n):
    particles['x'][i] += particles['vx'][i]  # Sequential!
    particles['y'][i] += particles['vy'][i]  # Sequential!
```

Memory layout:
```
x: [1.0 | 1.5 | 2.0 | 2.5 | ...]  ← Sequential access
vx: [0.1 | 0.3 | 0.5 | ...]       ← Sequential access
```

**Performance comparison:**
```
AoS: 150ms (cache misses on scattered fields)
SoA: 50ms (cache hits on sequential arrays)
3x speedup from layout change alone!
```

---

## False Sharing

### The Problem

```
Two threads, two separate variables:

Thread 1 writes: counter_a
Thread 2 writes: counter_b

CPU Core 1               CPU Core 2
┌──────────┐            ┌──────────┐
│ L1 Cache │            │ L1 Cache │
│ [a|b]    │ ◄────┐     │ [a|b]    │
└──────────┘      │     └──────────┘
                  │
            Cache line (64 bytes)
            contains BOTH a and b!
```

**What happens:**
1. Core 1 modifies `counter_a`
2. Cache line invalidated on Core 2
3. Core 2 must reload cache line (even though it only needs `counter_b`)
4. Core 2 modifies `counter_b`
5. Cache line invalidated on Core 1
6. Ping-pong effect: Constant cache invalidation

**Performance impact:**
```python
import threading

# BAD: False sharing
class Counters:
    def __init__(self):
        self.a = 0  # Adjacent in memory
        self.b = 0  # Same cache line!

counters = Counters()

def increment_a():
    for _ in range(10_000_000):
        counters.a += 1  # Invalidates b's cache!

def increment_b():
    for _ in range(10_000_000):
        counters.b += 1  # Invalidates a's cache!

t1 = threading.Thread(target=increment_a)
t2 = threading.Thread(target=increment_b)

# Time: 2.5 seconds (cache thrashing!)
```

### Solution: Padding

```python
# GOOD: Separate cache lines
class Counters:
    def __init__(self):
        self.a = 0
        self._padding = [0] * 8  # 64 bytes padding
        self.b = 0

# Now a and b are on different cache lines
# Time: 0.3 seconds (8x faster!)
```

**Cache line alignment:**
```
Memory layout (64-byte cache lines):

Without padding:
┌────────────────────────────────────────┐
│ [a] [b]                                 │ ← Same cache line
└────────────────────────────────────────┘

With padding:
┌────────────────────────────────────────┐
│ [a] [padding........................]  │ ← Cache line 1
└────────────────────────────────────────┘
┌────────────────────────────────────────┐
│ [b]                                     │ ← Cache line 2
└────────────────────────────────────────┘
```

---

## Python Memory Management

### Reference Counting

Every Python object has a reference count:

```python
import sys

x = []
print(sys.getrefcount(x))  # 2 (x + argument to getrefcount)

y = x
print(sys.getrefcount(x))  # 3 (x, y, argument)

del y
print(sys.getrefcount(x))  # 2
```

**CPython implementation:**
```c
typedef struct {
    Py_ssize_t ob_refcnt;  // Reference count
    PyTypeObject *ob_type;  // Type
    // ... object data
} PyObject;

// Every assignment increments refcount
void Py_INCREF(PyObject *obj) {
    obj->ob_refcnt++;
}

// Every delete decrements refcount
void Py_DECREF(PyObject *obj) {
    if (--obj->ob_refcnt == 0) {
        free(obj);  // Deallocate immediately
    }
}
```

### Cycle Detection

Reference counting can't handle cycles:

```python
a = []
b = []
a.append(b)  # a → b
b.append(a)  # b → a (cycle!)

del a
del b

# Cycle still exists in memory!
# ref(a) = 1 (from b)
# ref(b) = 1 (from a)
```

**Cyclic GC:**
```python
import gc

# Force garbage collection
collected = gc.collect()
print(f"Collected {collected} objects")

# Disable for performance (if no cycles)
gc.disable()
```

### Memory Pools (pymalloc)

Python uses custom allocator for small objects (< 512 bytes):

```
┌──────────────────────────────────────────────────┐
│                Python Memory                      │
│                                                   │
│  Small objects (< 512B): pymalloc                │
│    ┌──────────────────────────────┐              │
│    │  Size class 16:  ●●●●●●       │             │
│    │  Size class 32:  ●●●●         │             │
│    │  Size class 64:  ●●●          │             │
│    └──────────────────────────────┘              │
│                                                   │
│  Large objects (≥ 512B): malloc/free             │
│    ┌──────────────────────────────┐              │
│    │  [1024B] [2048B] [4096B]     │             │
│    └──────────────────────────────┘              │
│                                                   │
└──────────────────────────────────────────────────┘
```

**Optimization:**
```python
# Reuse objects
class ObjectPool:
    def __init__(self):
        self.pool = []

    def acquire(self):
        if self.pool:
            return self.pool.pop()
        return {}  # New allocation

    def release(self, obj):
        obj.clear()  # Reset state
        self.pool.append(obj)  # Reuse later

pool = ObjectPool()

# Fast: Reuse existing objects
for _ in range(1_000_000):
    obj = pool.acquire()
    # ... use obj ...
    pool.release(obj)
```

---

## JVM Memory Model and GC Tuning

### JVM Memory Regions

```
┌──────────────────────────────────────────────────┐
│                   JVM Heap                        │
│                                                   │
│  ┌─────────────────────────────────────────┐     │
│  │          Young Generation               │     │
│  │  ┌──────┬──────────┬──────────┐         │     │
│  │  │ Eden │ From (S0)│ To (S1)  │         │     │
│  │  │ 8/10 │   1/10   │   1/10   │         │     │
│  │  └──────┴──────────┴──────────┘         │     │
│  │  Most objects die here                  │     │
│  └─────────────────────────────────────────┘     │
│              ▼ Promotion                         │
│  ┌─────────────────────────────────────────┐     │
│  │          Old Generation                 │     │
│  │  Long-lived objects                     │     │
│  │  Tenured after N young GC cycles        │     │
│  └─────────────────────────────────────────┘     │
│                                                   │
│  ┌─────────────────────────────────────────┐     │
│  │          Metaspace                      │     │
│  │  Class metadata (was PermGen)           │     │
│  │  Not in heap (native memory)            │     │
│  └─────────────────────────────────────────┘     │
│                                                   │
└──────────────────────────────────────────────────┘
```

### GC Algorithms

**1. Serial GC** (`-XX:+UseSerialGC`)
- Single-threaded
- Stop-the-world
- Good for: Single-core, < 100 MB heap

**2. Parallel GC** (`-XX:+UseParallelGC`)
- Multi-threaded GC
- Optimizes throughput
- Good for: Batch processing

**3. G1 GC** (`-XX:+UseG1GC`)
- Divides heap into regions
- Predictable pause times
- Good for: Large heaps (> 4 GB), low latency

**4. ZGC** (`-XX:+UseZGC`)
- Sub-10ms pauses
- Concurrent (no stop-the-world)
- Good for: Ultra-low latency (< 10 ms p99)

### GC Tuning

**Common flags:**
```bash
java -Xms2G \          # Initial heap
     -Xmx8G \          # Max heap
     -XX:+UseG1GC \    # Use G1 collector
     -XX:MaxGCPauseMillis=200 \  # Target 200ms pauses
     -XX:+PrintGCDetails \       # Log GC activity
     -XX:+PrintGCDateStamps \
     -Xloggc:gc.log \
     MyApp
```

**Analyzing GC logs:**
```
[GC pause (G1 Evacuation Pause) (young), 0.0234567 secs]
   [Eden: 512M(512M)->0B(480M)     # Young gen collected
    Survivors: 32M->64M
    Heap: 2048M(4096M)->1600M(4096M)]
```

**Tuning strategy:**
1. Monitor GC frequency and pause times
2. If frequent minor GC: Increase young gen size
3. If long major GC pauses: Increase heap size or use different GC
4. If low throughput: Reduce GC overhead (larger heap)

**Example scenario:**
```
Problem: 500ms GC pauses every 2 seconds

Analysis:
- Old gen filling up quickly
- Objects promoted too early

Solution:
# Increase young gen, reduce promotions
-XX:NewRatio=1  # Young:Old = 1:1 (was 1:2)
-XX:MaxTenuringThreshold=15  # More cycles before promotion

Result: 50ms pauses every 10 seconds
```

---

## Key Concepts Checklist

- [ ] Explain stack vs heap trade-offs (speed vs flexibility)
- [ ] Describe at least two memory allocation strategies
- [ ] Compare reference counting vs mark-sweep GC
- [ ] Discuss generational GC and why it works
- [ ] Identify common memory leak patterns
- [ ] Explain memory fragmentation and solutions
- [ ] Describe cache locality impact on performance
- [ ] Know how to tune GC for your language/runtime

---

## Practical Insights

**Memory is the bottleneck, not CPU.** Modern systems spend 50%+ time waiting for memory. A cache miss costs 200 CPU cycles—optimize for locality first, algorithms second.

**Monitor allocator performance.** Tools like `jemalloc` can be 2-3x faster than default malloc for certain workloads. Profile before choosing. Consider `tcmalloc` for multi-threaded servers, `mimalloc` for general use.

**GC tuning is workload-specific.** Don't copy settings from blog posts. Run YOUR workload, measure YOUR pauses, tune based on YOUR p99 latency requirements. Start with defaults, change one flag at a time, measure impact.

**Object pools pay off at high scale.** Below 10,000 allocations/second, pools add complexity without benefit. Above 100,000/sec, pools can reduce GC pressure by 10x and eliminate tail latency from allocation storms.

**False sharing kills multi-core performance.** If your multi-threaded code doesn't scale linearly with cores, check for false sharing. Use cache line padding for frequently updated counters. Tools: `perf c2c` (Linux), Intel VTune.

**Python's GIL makes memory management easier but limits parallelism.** Reference counting prevents data races, but means CPU-bound Python is single-threaded. Use multiprocessing for parallelism, threading for I/O. Or migrate hot paths to C extensions.
