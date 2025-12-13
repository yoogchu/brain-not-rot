# Chapter 43: CPU Architecture & Performance

## The Core Problem

Your API server handles 50,000 requests per second. You've optimized the database queries. You've tuned the network stack. You've added caching everywhere. But response times are still inconsistent—P99 latency spikes from 10ms to 200ms under load.

You profile the application and discover the bottleneck: **CPU cache misses**.

```
Scenario: Processing user session data

Array layout 1 (cache-friendly):
struct UserSession {
    user_id, session_token, last_active  // All accessed together
}
Processing time: 2ms for 10,000 sessions

Array layout 2 (cache-hostile):
Separate arrays: user_ids[], tokens[], timestamps[]
Processing time: 15ms for 10,000 sessions (7.5x slower!)

Same algorithm, same data, different memory layout.
The difference: CPU cache hit rate drops from 95% to 60%
```

A CPU cache miss costs 100-300 CPU cycles. At 50,000 RPS with poor cache locality, you're wasting **billions of cycles per second**. Understanding CPU architecture isn't optional at scale—it's the difference between handling 10k RPS and 100k RPS on the same hardware.

---

## Memory Hierarchy and Latency

Modern CPUs access memory through a multi-level hierarchy. Each level trades off size for speed.

### The Hierarchy

```
┌────────────────────────────────────────────────────────┐
│  CPU Core                                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Registers: 64 bytes, <1 cycle                   │  │
│  └──────────────────────────────────────────────────┘  │
│                          │                              │
│                          ▼                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │  L1 Cache: 32-64KB, ~4 cycles (1ns)             │  │
│  │  Split: L1d (data), L1i (instruction)           │  │
│  └──────────────────────────────────────────────────┘  │
│                          │                              │
│                          ▼                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │  L2 Cache: 256KB-1MB, ~12 cycles (3ns)          │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│  L3 Cache: 8-64MB, ~40 cycles (10ns)                   │
│  Shared across all cores                               │
└────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│  Main Memory (RAM): 16-512GB, ~200 cycles (100ns)      │
└────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│  SSD: TB scale, ~100,000 cycles (100μs)                │
└────────────────────────────────────────────────────────┘
```

### Latency Numbers Every Engineer Should Know (2025)

```python
# Relative latencies in CPU cycles (3 GHz CPU)

LATENCIES = {
    "L1 cache hit": 4,              # 1.3ns
    "L2 cache hit": 12,             # 4ns
    "L3 cache hit": 40,             # 13ns
    "Main memory": 200,             # 67ns
    "Mutex lock/unlock": 100,       # 33ns
    "Context switch": 10_000,       # 3μs
    "SSD random read": 450_000,     # 150μs
    "Network within datacenter": 1_500_000,  # 0.5ms
}

# Cache miss penalty
l1_miss_penalty = LATENCIES["L3 cache hit"] - LATENCIES["L1 cache hit"]
# 36 cycles wasted per L1 miss

# At 1M cache misses per second:
wasted_cycles_per_sec = 1_000_000 * l1_miss_penalty
# 36 million cycles wasted = 12ms of CPU time per second
```

**Key insight:** Going to main memory is 50x slower than L1. Optimizing for cache locality can have a bigger impact than algorithmic optimizations.

---

## Cache Lines: The Fundamental Unit

CPUs don't fetch individual bytes—they fetch **cache lines** (typically 64 bytes).

### How Cache Lines Work

```
Memory address: 0x1000 to 0x103F (64 bytes)
    ┌─────────────────────────────────────────────────┐
    │  Byte 0-7  │  8-15  │  16-23  │ ... │  56-63   │
    └─────────────────────────────────────────────────┘
         ▲
         │
    Access any byte → Entire 64-byte line loaded into cache
```

**Example: Array traversal**

```python
import numpy as np
import time

def measure_access_pattern(size, stride):
    """Measure time to access array with given stride"""
    arr = np.arange(size, dtype=np.int64)  # 8 bytes per element

    start = time.perf_counter()
    total = 0
    # Access every 'stride'th element
    for i in range(0, size, stride):
        total += arr[i]
    elapsed = time.perf_counter() - start

    return elapsed, total

# Test different strides
size = 10_000_000

# Stride 1: Sequential access (cache-friendly)
# Accesses: [0, 1, 2, 3, 4, 5, 6, 7, 8, ...]
# Cache line (64 bytes = 8 int64s) loaded once, used 8 times
time_stride1, _ = measure_access_pattern(size, 1)

# Stride 8: Every 8th element (still cache-friendly)
# Accesses: [0, 8, 16, 24, ...]
# Each access in different cache line, but sequential
time_stride8, _ = measure_access_pattern(size, 8)

# Stride 1024: Large jumps (cache-hostile)
# Accesses: [0, 1024, 2048, ...]
# Each access likely evicts previous from cache
time_stride1024, _ = measure_access_pattern(size, 1024)

print(f"Stride 1: {time_stride1:.3f}s")
print(f"Stride 8: {time_stride8:.3f}s")
print(f"Stride 1024: {time_stride1024:.3f}s")
# Typical results:
# Stride 1: 0.012s (sequential, optimal)
# Stride 8: 0.015s (slightly worse, still good)
# Stride 1024: 0.045s (3-4x slower due to cache misses)
```

### Spatial Locality

When you access memory address X, you'll likely access X+1, X+2, etc. soon after. Cache lines exploit this by prefetching nearby data.

**Cache-friendly data structures:**

```python
# GOOD: Struct of arrays (SoA) when accessing one field
class Users:
    def __init__(self, n):
        self.user_ids = [0] * n      # Contiguous
        self.ages = [0] * n          # Contiguous
        self.scores = [0] * n        # Contiguous

    def sum_ages(self):
        # Sequential access, excellent cache locality
        return sum(self.ages)

# BAD: Array of structs (AoS) when accessing one field
class User:
    def __init__(self, user_id, age, score):
        self.user_id = user_id  # 8 bytes
        self.age = age          # 8 bytes
        self.score = score      # 8 bytes
        # Total: 24+ bytes per user (Python overhead adds more)

users = [User(i, i % 100, i * 10) for i in range(10_000)]

# To sum ages, must load 24+ bytes but only use 8
# Poor cache utilization
total = sum(u.age for u in users)
```

**Trade-off:** AoS is better when you access all fields together. SoA is better when you access one field across many objects.

---

## False Sharing: The Hidden Killer

False sharing occurs when threads on different cores modify variables that reside in the same cache line, causing cache coherence traffic.

### The Problem

```
Core 0 writes to variable A (bytes 0-7)
Core 1 writes to variable B (bytes 8-15)

Both variables are in the SAME cache line (64 bytes)

┌─────────────────────────────────────────────────────┐
│ Cache Line (64 bytes)                                │
│  A (Core 0) │  B (Core 1) │ unused                   │
└─────────────────────────────────────────────────────┘

Timeline:
1. Core 0 loads cache line, writes A → line marked "dirty" on Core 0
2. Core 1 wants to write B → must invalidate Core 0's cache line
3. Core 0's cache line evicted, sent to Core 1
4. Core 1 loads cache line, writes B → line marked "dirty" on Core 1
5. Core 0 wants to write A again → must invalidate Core 1's cache line
6. Repeat...

Result: Cache line ping-pongs between cores
Throughput: ~100x slower than independent cache lines
```

### Demonstration

```python
import threading
import time
from dataclasses import dataclass

# BROKEN: False sharing
class SharedCounters:
    def __init__(self):
        # Both counters likely in same cache line
        self.counter_a = 0
        self.counter_b = 0

def increment_worker(counters, field_name, iterations):
    for _ in range(iterations):
        current = getattr(counters, field_name)
        setattr(counters, field_name, current + 1)

# FIXED: Padding to separate cache lines
class PaddedCounters:
    def __init__(self):
        self.counter_a = 0
        self._pad1 = [0] * 15  # Force to different cache line
        self.counter_b = 0
        self._pad2 = [0] * 15

# Benchmark false sharing
iterations = 1_000_000

# With false sharing
shared = SharedCounters()
start = time.perf_counter()
t1 = threading.Thread(target=increment_worker, args=(shared, "counter_a", iterations))
t2 = threading.Thread(target=increment_worker, args=(shared, "counter_b", iterations))
t1.start(); t2.start()
t1.join(); t2.join()
false_sharing_time = time.perf_counter() - start

# Without false sharing
padded = PaddedCounters()
start = time.perf_counter()
t1 = threading.Thread(target=increment_worker, args=(padded, "counter_a", iterations))
t2 = threading.Thread(target=increment_worker, args=(padded, "counter_b", iterations))
t1.start(); t2.start()
t1.join(); t2.join()
padded_time = time.perf_counter() - start

print(f"False sharing: {false_sharing_time:.3f}s")
print(f"Padded: {padded_time:.3f}s")
print(f"Speedup: {false_sharing_time / padded_time:.2f}x")
# Typical: 3-10x speedup from eliminating false sharing
```

### Detection and Prevention

**Detection:**
```bash
# perf can detect false sharing
perf c2c record -a -- sleep 10
perf c2c report --stdio

# Look for:
# - High "Shared Data Cache Line Table" entries
# - High cache line contention
```

**Prevention:**
```python
# Align to cache line boundaries (64 bytes)
import ctypes

class CacheAligned:
    """Force each instance to start on cache line boundary"""
    _fields_ = [
        ("value", ctypes.c_long),
        ("_pad", ctypes.c_byte * 56)  # Pad to 64 bytes
    ]

# Or: Use separate arrays, one per thread
class PerThreadCounters:
    def __init__(self, num_threads):
        # Each thread gets its own counter (likely different cache lines)
        self.counters = [0] * num_threads
```

---

## NUMA: Non-Uniform Memory Access

Modern multi-socket servers have NUMA architecture where memory access time depends on the memory's physical location relative to the CPU.

### NUMA Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│   CPU 0 (Socket 0)  │         │   CPU 1 (Socket 1)  │
│   Cores: 0-15       │         │   Cores: 16-31      │
│   ┌───────────┐     │         │   ┌───────────┐     │
│   │ L1/L2/L3  │     │         │   │ L1/L2/L3  │     │
│   └───────────┘     │         │   └───────────┘     │
│         │           │         │         │           │
│         ▼           │         │         ▼           │
│   ┌───────────┐     │         │   ┌───────────┐     │
│   │  Memory   │     │◄───────►│   │  Memory   │     │
│   │   Bank 0  │     │  QPI/   │   │   Bank 1  │     │
│   │   64 GB   │     │ UPI Link│   │   64 GB   │     │
│   └───────────┘     │         │   └───────────┘     │
└─────────────────────┘         └─────────────────────┘

Access Latency:
- Core 0 → Memory Bank 0: ~100ns (local, fast)
- Core 0 → Memory Bank 1: ~140ns (remote, 1.4x slower)
- Core 16 → Memory Bank 0: ~140ns (remote, 1.4x slower)
- Core 16 → Memory Bank 1: ~100ns (local, fast)
```

### Impact on Performance

```python
import numa
import numpy as np
import time

def benchmark_numa():
    """Demonstrate NUMA impact"""

    # Check NUMA nodes
    num_nodes = numa.get_max_node() + 1
    print(f"NUMA nodes: {num_nodes}")

    # Allocate on specific node
    size = 100_000_000

    # Local allocation (same node as CPU)
    numa.set_membind([0])  # Allocate on node 0
    local_array = np.arange(size, dtype=np.int64)

    # Remote allocation (different node)
    numa.set_membind([1])  # Allocate on node 1
    remote_array = np.arange(size, dtype=np.int64)

    # Bind thread to core on node 0
    numa.run_on_nodes([0])

    # Benchmark local access
    start = time.perf_counter()
    local_sum = np.sum(local_array)
    local_time = time.perf_counter() - start

    # Benchmark remote access
    start = time.perf_counter()
    remote_sum = np.sum(remote_array)
    remote_time = time.perf_counter() - start

    print(f"Local access: {local_time:.3f}s")
    print(f"Remote access: {remote_time:.3f}s")
    print(f"Remote penalty: {remote_time / local_time:.2f}x")
    # Typical: 1.3-1.5x slower for remote access
```

### NUMA Best Practices

```bash
# Check NUMA topology
numactl --hardware

# Run process on specific NUMA node
numactl --cpunodebind=0 --membind=0 ./my_server

# Interleave memory across nodes (for global shared data)
numactl --interleave=all ./my_server
```

**Strategy:**
- **Data partitioning:** Each thread processes data allocated on its local NUMA node
- **Worker affinity:** Pin workers to cores on the same NUMA node as their data
- **Shared data:** Interleave across nodes or replicate per-node

---

## Branch Prediction

CPUs use speculative execution to predict which branch will be taken, executing ahead to avoid pipeline stalls.

### Branch Misprediction Cost

```
Pipeline without branch prediction:
┌────┬────┬────┬────┬────┐
│ IF │ ID │ EX │ MEM│ WB │  Instruction 1
└────┴────┴────┴────┴────┘
     ┌────┬────┬────┬────┬────┐
     │ IF │ ID │ EX │ MEM│ WB │  Instruction 2 (branch)
     └────┴────┴────┴────┴────┘
          ┌────┬────┬────┬────┬────┐
          │ IF │ ID │ EX │ MEM│ WB │  Instruction 3
          └────┴────┴────┴────┴────┘

Branch misprediction:
┌────┬────┬────┬────┬────┐
│ IF │ ID │ EX │ MEM│ WB │  Instruction 1
└────┴────┴────┴────┴────┘
     ┌────┬────┬────┬────┬────┐
     │ IF │ ID │ EX │ MEM│ WB │  Instruction 2 (branch)
     └────┴────┴────┴────┴────┘
          ┌────┬────┬────┬────┬────┐
          │ IF │ ID │XXXX│XXXX│XXXX│  Wrong path (flushed)
          └────┴────┴────┴────┴────┘
               STALL STALL STALL STALL
                    ┌────┬────┬────┬────┬────┐
                    │ IF │ ID │ EX │ MEM│ WB │  Correct path
                    └────┴────┴────┴────┴────┘

Misprediction penalty: 10-20 cycles of wasted work
```

### Predictable vs Unpredictable Branches

```python
import random
import time

def branch_benchmark(data, threshold):
    """Count elements greater than threshold"""
    count = 0
    for value in data:
        if value > threshold:  # Branch here
            count += 1
    return count

size = 10_000_000

# Predictable: sorted data (branch always same for long stretches)
sorted_data = sorted([random.randint(0, 100) for _ in range(size)])

start = time.perf_counter()
result = branch_benchmark(sorted_data, 50)
predictable_time = time.perf_counter() - start

# Unpredictable: random data (branch outcome random each time)
random_data = [random.randint(0, 100) for _ in range(size)]

start = time.perf_counter()
result = branch_benchmark(random_data, 50)
unpredictable_time = time.perf_counter() - start

print(f"Predictable branches: {predictable_time:.3f}s")
print(f"Unpredictable branches: {unpredictable_time:.3f}s")
print(f"Slowdown: {unpredictable_time / predictable_time:.2f}x")
# Typical: 2-3x slower with unpredictable branches
```

### Avoiding Branch Mispredictions

**1. Branchless code:**

```python
# With branches
def clamp_branched(value, min_val, max_val):
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value

# Branchless (using min/max)
def clamp_branchless(value, min_val, max_val):
    return max(min_val, min(max_val, value))

# Numpy: vectorized, no branches per element
import numpy as np
def clamp_vectorized(arr, min_val, max_val):
    return np.clip(arr, min_val, max_val)
```

**2. Sort data to make branches predictable:**

```python
# SLOW: Process unsorted data
def process_unsorted(data):
    count = 0
    for value in data:
        if value > 50:  # Unpredictable
            count += 1

# FAST: Sort first, then branches become predictable
def process_sorted(data):
    sorted_data = sorted(data)
    count = 0
    for value in sorted_data:
        if value > 50:  # Predictable after sorting
            count += 1
    # Sorting cost is amortized if processing data multiple times
```

**3. Use lookup tables instead of conditionals:**

```python
# Branchy
def score_letter_branched(letter):
    if letter in 'AEIOU':
        return 1
    elif letter in 'BCDFGHMP':
        return 3
    elif letter in 'JKLNQR':
        return 5
    else:
        return 10

# Branchless (lookup table)
LETTER_SCORES = {
    'A': 1, 'E': 1, 'I': 1, 'O': 1, 'U': 1,
    'B': 3, 'C': 3, 'D': 3, 'F': 3, 'G': 3, 'H': 3, 'M': 3, 'P': 3,
    # ... etc
}

def score_letter_lookup(letter):
    return LETTER_SCORES.get(letter, 10)
```

---

## CPU Pinning and Affinity

Control which cores your threads run on to improve cache locality and reduce context switching.

### Why Pinning Matters

```
Without pinning:
Thread starts on Core 0 → loads cache → OS migrates to Core 3
  → Cache cold on Core 3 → Re-load from memory
  → OS migrates back to Core 0 → Cache cold again

With pinning:
Thread stays on Core 0 → Cache stays warm → Better performance
```

### Python Implementation

```python
import os
import threading
import time

def cpu_intensive_work(duration):
    """Simulate CPU-bound work"""
    end_time = time.time() + duration
    result = 0
    while time.time() < end_time:
        result += 1
    return result

def benchmark_with_affinity():
    """Compare performance with/without CPU affinity"""

    # Without affinity (OS can migrate thread)
    def worker_no_affinity():
        return cpu_intensive_work(1.0)

    start = time.perf_counter()
    thread = threading.Thread(target=worker_no_affinity)
    thread.start()
    thread.join()
    no_affinity_time = time.perf_counter() - start

    # With affinity (pinned to core 0)
    def worker_with_affinity():
        os.sched_setaffinity(0, {0})  # Pin to core 0
        return cpu_intensive_work(1.0)

    start = time.perf_counter()
    thread = threading.Thread(target=worker_with_affinity)
    thread.start()
    thread.join()
    with_affinity_time = time.perf_counter() - start

    print(f"No affinity: {no_affinity_time:.3f}s")
    print(f"With affinity: {with_affinity_time:.3f}s")
    print(f"Improvement: {no_affinity_time / with_affinity_time:.2f}x")

# Production usage
def pin_worker_to_core(core_id):
    """Pin current thread to specific CPU core"""
    os.sched_setaffinity(0, {core_id})

# Pin different workers to different cores
def start_workers(num_workers):
    threads = []
    for i in range(num_workers):
        def worker(core):
            os.sched_setaffinity(0, {core})
            # Do work...
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    return threads
```

### Practical Affinity Strategies

| Strategy | Use Case | Implementation |
|----------|----------|----------------|
| No pinning | I/O-bound, variable load | Default OS scheduler |
| Pin to physical cores | CPU-bound, avoid HT | Pin to cores 0,2,4,... (skip siblings) |
| Pin to NUMA node | Large memory working set | Pin cores + memory to same node |
| Exclusive cores | Real-time, low latency | Isolate cores (isolcpus=), pin workload |

```bash
# Check core topology (physical vs hyperthreading)
lscpu

# Reserve cores 0-3 for OS, 4-31 for application
isolcpus=4-31

# Check thread affinity
taskset -cp <pid>

# Set affinity for process
taskset -c 4-7 ./my_server
```

---

## SIMD and Vectorization

SIMD (Single Instruction, Multiple Data) processes multiple data elements in parallel using vector instructions.

### How SIMD Works

```
Scalar (normal code):
┌─────┐   ┌─────┐   ┌─────┐
│ a₁  │ + │ b₁  │ = │ c₁  │
└─────┘   └─────┘   └─────┘
┌─────┐   ┌─────┐   ┌─────┐
│ a₂  │ + │ b₂  │ = │ c₂  │
└─────┘   └─────┘   └─────┘
┌─────┐   ┌─────┐   ┌─────┐
│ a₃  │ + │ b₃  │ = │ c₃  │
└─────┘   └─────┘   └─────┘
┌─────┐   ┌─────┐   ┌─────┐
│ a₄  │ + │ b₄  │ = │ c₄  │
└─────┘   └─────┘   └─────┘
4 instructions, 4 cycles

SIMD (AVX2 with 256-bit registers):
┌──────────────────────────┐   ┌──────────────────────────┐
│ a₁ │ a₂ │ a₃ │ a₄ │ (256)│ + │ b₁ │ b₂ │ b₃ │ b₄ │      │
└──────────────────────────┘   └──────────────────────────┘
                 ‖
                 ▼
┌──────────────────────────┐
│ c₁ │ c₂ │ c₃ │ c₄ │      │
└──────────────────────────┘
1 instruction, 1 cycle (4x faster)
```

### NumPy SIMD Example

```python
import numpy as np
import time

size = 10_000_000

# Scalar Python (no SIMD)
def add_python_lists(a, b):
    return [a[i] + b[i] for i in range(len(a))]

a_list = list(range(size))
b_list = list(range(size))

start = time.perf_counter()
result = add_python_lists(a_list, b_list)
python_time = time.perf_counter() - start

# NumPy (uses SIMD automatically)
a_np = np.arange(size, dtype=np.int64)
b_np = np.arange(size, dtype=np.int64)

start = time.perf_counter()
result = a_np + b_np  # Vectorized SIMD operation
numpy_time = time.perf_counter() - start

print(f"Python loop: {python_time:.3f}s")
print(f"NumPy SIMD: {numpy_time:.3f}s")
print(f"Speedup: {python_time / numpy_time:.1f}x")
# Typical: 50-100x speedup
```

### Manual Vectorization

```python
import numpy as np

# Scalar: Process one element at a time
def normalize_scalar(data, mean, std):
    result = []
    for value in data:
        result.append((value - mean) / std)
    return result

# Vectorized: Process all elements at once
def normalize_vectorized(data, mean, std):
    return (data - mean) / std

data = np.random.randn(1_000_000)
mean = data.mean()
std = data.std()

# Vectorized is 100x+ faster
result = normalize_vectorized(data, mean, std)
```

### SIMD Instruction Sets

| Instruction Set | Year | Register Width | Int64 Elements | Float32 Elements |
|----------------|------|----------------|----------------|------------------|
| SSE2 | 2001 | 128-bit | 2 | 4 |
| AVX | 2011 | 256-bit | 4 | 8 |
| AVX2 | 2013 | 256-bit | 4 | 8 |
| AVX-512 | 2016 | 512-bit | 8 | 16 |

**When to use:**
- Array operations (mathematical computations)
- Image/video processing
- Machine learning (matrix operations)
- Compression/decompression

**When NOT to use:**
- Irregular access patterns (SIMD requires contiguous data)
- Heavy branching (masks reduce SIMD efficiency)
- Small datasets (overhead not worth it)

---

## Profiling CPU Performance

### perf: Linux Performance Analysis

```bash
# Count CPU events
perf stat ./my_program

# Output:
# Performance counter stats for './my_program':
#   2,847.19 msec task-clock          #    0.997 CPUs utilized
#   9,234,567,890 cycles              #    3.243 GHz
#   6,123,456,789 instructions        #    0.66  insn per cycle
#     234,567,890 cache-misses         #    8.2% of all cache refs
#     890,123,456 branch-misses        #    4.2% of all branches

# Profile where time is spent
perf record -g ./my_program
perf report

# Check cache performance
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses ./my_program

# Monitor specific process
perf top -p <pid>
```

### Interpreting perf Output

```
Key metrics:

Instructions per cycle (IPC):
  > 1.0: Good, CPU executing efficiently
  < 0.5: Poor, CPU stalled (memory bound or branch mispredictions)

Cache miss rate:
  < 1%: Excellent cache locality
  > 10%: Poor cache locality, optimize data structures

Branch miss rate:
  < 2%: Good branch prediction
  > 10%: Unpredictable branches, consider branchless code
```

### Flame Graphs

```bash
# Install flamegraph
git clone https://github.com/brendangregg/FlameGraph
cd FlameGraph

# Record perf data
perf record -F 99 -a -g -- sleep 30

# Generate flame graph
perf script | ./stackcollapse-perf.pl | ./flamegraph.pl > flame.svg

# Open in browser
firefox flame.svg
```

**Reading flame graphs:**
- Width = CPU time (wider = more time)
- Height = call stack depth
- Color = random (for differentiation)
- Look for wide plateaus = hot functions

### Python-Specific Profiling

```python
import cProfile
import pstats

def profile_function():
    # Your code here
    pass

# Profile and save
cProfile.run('profile_function()', 'profile_stats')

# Analyze
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions

# Line-by-line profiling
from line_profiler import LineProfiler

lp = LineProfiler()
lp.add_function(profile_function)
lp.run('profile_function()')
lp.print_stats()
```

---

## Memory Barriers and Ordering

Modern CPUs reorder instructions for performance. Memory barriers enforce ordering guarantees.

### Why Reordering Happens

```
Original code:
1. x = 1
2. flag = True
3. if flag:
4.     print(x)

CPU might execute as:
1. flag = True   (reordered)
2. x = 1
3. if flag:
4.     print(x)  → Might print 0! (if another thread reads)
```

### Memory Ordering

```python
import threading

# Without barrier (broken on some architectures)
class BrokenSingleton:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:  # Check 1 (no lock)
            with cls._lock:
                if cls._instance is None:  # Check 2 (with lock)
                    cls._instance = cls()  # Problem: could be reordered
        return cls._instance

# Fixed with proper barriers (Python's threading primitives include barriers)
class FixedSingleton:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:  # Lock includes memory barrier
                if cls._instance is None:
                    instance = cls()
                    # Memory barrier here ensures all initialization completes
                    cls._instance = instance
        return cls._instance
```

### Types of Memory Barriers

| Barrier Type | Guarantee |
|-------------|-----------|
| Load Barrier | Loads before barrier complete before loads after |
| Store Barrier | Stores before barrier complete before stores after |
| Full Barrier | All memory operations before barrier complete before operations after |

**In Python:** Most synchronization primitives (locks, semaphores) include appropriate memory barriers. You rarely need to worry about this explicitly.

**In C/C++:** Use `std::atomic` with memory_order specifications, or `volatile` (though volatile ≠ atomic).

---

## Comparison: Optimization Techniques

| Technique | Speedup Potential | Complexity | When to Use |
|-----------|-------------------|------------|-------------|
| Cache-friendly data layout | 2-10x | Low | Always consider |
| Avoiding false sharing | 3-10x | Low-Medium | Multi-threaded writes |
| CPU pinning | 1.1-1.5x | Low | CPU-bound, consistent load |
| NUMA awareness | 1.2-1.5x | Medium | Multi-socket servers |
| Branchless code | 1.5-3x | Medium | Unpredictable branches |
| SIMD vectorization | 4-16x | Medium-High | Array operations |
| Algorithm optimization | 10-1000x+ | High | Always start here |

---

## Key Concepts Checklist

- [ ] Explain memory hierarchy (L1/L2/L3/RAM) and latencies
- [ ] Describe cache lines and spatial locality
- [ ] Identify and fix false sharing between threads
- [ ] Understand NUMA impact on multi-socket systems
- [ ] Explain branch prediction and misprediction penalties
- [ ] Use CPU affinity to improve cache locality
- [ ] Recognize opportunities for SIMD vectorization
- [ ] Profile CPU performance with perf and flamegraphs

---

## Practical Insights

**Start with algorithms, not micro-optimizations.** A better algorithm (O(n log n) vs O(n²)) will always beat cache optimizations. Profile first, optimize second. Use `perf stat` to measure before and after—guessing is wrong 90% of the time.

**Cache locality often matters more than algorithmic complexity for real-world data sizes.** A cache-friendly O(n²) algorithm can beat a cache-hostile O(n log n) algorithm for n < 10,000. Measure your actual data sizes and access patterns.

**False sharing is invisible until it's catastrophic.** When adding multi-threading doesn't improve performance linearly, profile for cache coherence traffic. Padding shared data to 64-byte boundaries is cheap and often makes a 5-10x difference.

**NUMA matters at 64GB+ memory and 2+ sockets.** Below that, the complexity isn't worth it. Above that, use `numactl` to bind processes to nodes. For databases and caches, partition data by NUMA node and pin workers accordingly.

**Branch predictors are >95% accurate on regular patterns.** Sorting data before processing often pays for itself by making branches predictable. For random data, consider branchless alternatives or lookup tables instead of if-else chains.

**CPU pinning is a double-edged sword.** It helps when workload is stable and CPU-bound. It hurts when workload is bursty (OS can't load-balance) or when you pin more threads than physical cores (cache thrashing). Test with and without pinning under realistic load.

**Python's NumPy is 50-100x faster than pure Python for array operations** because it uses SIMD and avoids Python interpreter overhead. If you're doing math on arrays, always use NumPy. If you need even more speed, use Numba (JIT compilation) or Cython.
