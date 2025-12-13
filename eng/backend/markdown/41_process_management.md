# Chapter 41: Process Management & IPC

## The Core Problem

Your Flask API server handles 5,000 requests/second. A single long-running request starts computing a complex recommendation algorithm that takes 30 seconds. Because Python's GIL prevents true parallelism in threads, that one request blocks others. Response times spike from 50ms to 5 seconds. Users start complaining.

You switch to a multi-process architecture: 16 worker processes behind a load balancer. Now those workers need to share data - user session state, feature flags, real-time analytics. Without proper inter-process communication (IPC), each worker has its own isolated memory. Cache hit rate drops to 6%. Database load increases 10x.

Understanding process management and IPC is critical for building scalable backend systems that utilize multiple cores while efficiently sharing state.

---

## Process vs Thread: The Fundamental Trade-off

**The Problem:**
You need concurrency. Should you spawn new processes or create threads? The wrong choice wastes memory, creates bottlenecks, or introduces race conditions.

**How It Works:**

```
PROCESSES                          THREADS
┌─────────────────────┐           ┌─────────────────────┐
│ Process A           │           │ Process             │
│ ┌─────────────────┐ │           │ ┌─────┐  ┌─────┐   │
│ │ Code            │ │           │ │Code │  │Code │   │
│ │ Data            │ │           │ │(sh) │  │(sh) │   │
│ │ Heap            │ │           │ ├─────┤  ├─────┤   │
│ │ Stack           │ │           │ │Data │  │Data │   │
│ └─────────────────┘ │           │ │(sh) │  │(sh) │   │
└─────────────────────┘           │ ├─────┤  ├─────┤   │
                                  │ │Stack│  │Stack│   │
┌─────────────────────┐           │ │(priv)  │(priv)   │
│ Process B           │           │ └─────┘  └─────┘   │
│ ┌─────────────────┐ │           │  Thread1  Thread2  │
│ │ Code (separate) │ │           └─────────────────────┘
│ │ Data (separate) │ │
│ │ Heap (separate) │ │           sh = shared
│ │ Stack (separate)│ │           priv = private
│ └─────────────────┘ │
└─────────────────────┘

Complete isolation               Shared memory space
```

**Python Example:**

```python
import os
import threading
import multiprocessing
import time

# Thread example - shared memory
counter = 0
lock = threading.Lock()

def increment_thread():
    global counter
    for _ in range(100000):
        with lock:
            counter += 1

# Process example - isolated memory
def increment_process(shared_counter):
    """
    shared_counter is a multiprocessing.Value object
    Requires explicit shared memory primitives
    """
    for _ in range(100000):
        with shared_counter.get_lock():
            shared_counter.value += 1

# Thread benchmark
threads = [threading.Thread(target=increment_thread) for _ in range(4)]
start = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"Threads: {time.time() - start:.3f}s, counter={counter}")

# Process benchmark
shared_counter = multiprocessing.Value('i', 0)
processes = [
    multiprocessing.Process(target=increment_process, args=(shared_counter,))
    for _ in range(4)
]
start = time.time()
for p in processes:
    p.start()
for p in processes:
    p.join()
print(f"Processes: {time.time() - start:.3f}s, counter={shared_counter.value}")
```

**Trade-offs:**

| Aspect | Processes | Threads |
|--------|-----------|---------|
| **Memory** | ~10-50MB per process | ~8KB per thread (stack only) |
| **Isolation** | Complete - crash doesn't affect others | Crash kills entire process |
| **Sharing data** | Requires IPC (pipes, queues, shared mem) | Direct memory access (need locks) |
| **Context switch** | Slower (~1-10μs) | Faster (~0.1-1μs) |
| **Python GIL** | Bypass - true parallelism | Blocked - only I/O parallelism |
| **Startup time** | Slower (fork overhead) | Faster |

**When to use processes:** CPU-bound work in Python, need isolation, running untrusted code

**When NOT to use processes:** Sharing lots of state, need low latency communication, memory constrained

---

## Process Lifecycle: Fork, Exec, Wait

**The Problem:**
How does a new process get created? How do you run a different program? How do you prevent zombie processes?

**How It Works:**

```
Parent Process Lifecycle
─────────────────────────

┌──────────┐
│  Parent  │
│  PID=100 │
└────┬─────┘
     │ fork()
     ├─────────────────┐
     │                 │
     ▼                 ▼
┌──────────┐      ┌──────────┐
│  Parent  │      │  Child   │
│  PID=100 │      │  PID=101 │
│          │      │ (copy of │
│          │      │  parent) │
└────┬─────┘      └────┬─────┘
     │                 │ exec("other_program")
     │                 ▼
     │            ┌──────────┐
     │            │  Child   │
     │            │  PID=101 │
     │            │ (running │
     │            │  other)  │
     │            └────┬─────┘
     │                 │ exit(0)
     │                 ▼
     │            ┌──────────┐
     │            │  Zombie  │
     │            │  PID=101 │
     │ wait()     │ (waiting)│
     ├────────────┤          │
     │            └──────────┘
     ▼
┌──────────┐      Child reaped
│  Parent  │      Resources freed
│ continues│
└──────────┘
```

**Python Example:**

```python
import os
import sys
import time

def fork_example():
    """Demonstrate fork, exec, wait pattern"""
    print(f"Parent process PID: {os.getpid()}")

    pid = os.fork()

    if pid == 0:
        # Child process
        print(f"Child process PID: {os.getpid()}, Parent PID: {os.getppid()}")

        # Do some work
        time.sleep(2)
        print("Child: Work complete")

        # Replace this process with 'ls' command
        # exec* functions don't return - they replace the process
        os.execvp("ls", ["ls", "-l"])

        # This line never executes if exec succeeds
        print("This won't print")

    else:
        # Parent process
        print(f"Parent: Created child with PID {pid}")

        # Wait for child to complete
        # Without this, child becomes zombie when it exits
        child_pid, exit_status = os.wait()

        print(f"Parent: Child {child_pid} exited with status {exit_status >> 8}")

# More practical example: worker pool
def worker_task(worker_id):
    """Simulated worker task"""
    print(f"Worker {worker_id} (PID {os.getpid()}) starting")
    time.sleep(2)
    return worker_id * 10

def prefork_workers(num_workers):
    """
    Prefork pattern: create worker pool at startup
    Common in Gunicorn, uWSGI, Apache
    """
    worker_pids = []

    for i in range(num_workers):
        pid = os.fork()
        if pid == 0:
            # Child: do work and exit
            result = worker_task(i)
            sys.exit(0)
        else:
            # Parent: track children
            worker_pids.append(pid)

    # Wait for all workers
    for pid in worker_pids:
        os.waitpid(pid, 0)

    print("All workers complete")

if __name__ == "__main__":
    prefork_workers(4)
```

**Key System Calls:**

| Call | Purpose | Returns |
|------|---------|---------|
| `fork()` | Clone current process | 0 in child, child PID in parent |
| `exec*()` | Replace process with new program | Doesn't return on success |
| `wait()` | Wait for any child to exit | Child PID and exit status |
| `waitpid(pid, options)` | Wait for specific child | Child PID and exit status |
| `exit(status)` | Terminate process | Doesn't return |

**When to use:** Building process pools, daemonization, running external commands

**When NOT to use:** High-frequency spawning (use thread pool), Windows (no fork)

---

## Context Switching: The Hidden Cost

**The Problem:**
You have 100 worker processes on an 8-core machine. Every process switch saves/restores registers, flushes TLB, switches page tables. CPU spends more time switching than working.

**How It Works:**

```
Timeline of Context Switch
──────────────────────────

CPU executes Process A
┌─────────────────────────────────────┐
│ ██████████████                      │ Time slice expires
└─────────────────────────────────────┘
                  │
                  ▼
            Context Switch
         ┌──────────────────┐
         │ 1. Save Process A│
         │    - Registers   │
         │    - PC, SP      │
         │    - Page table  │
         │ 2. Load Process B│
         │    - Registers   │
         │    - PC, SP      │
         │    - Page table  │
         │ 3. Flush TLB     │
         └──────────────────┘
         Time: 1-10μs
                  │
                  ▼
CPU executes Process B
┌─────────────────────────────────────┐
│               ██████████████         │
└─────────────────────────────────────┘
```

**Python Measurement:**

```python
import os
import time
import multiprocessing

def cpu_bound_work(n):
    """Pure CPU work to measure overhead"""
    total = 0
    for i in range(n):
        total += i * i
    return total

def measure_context_switch_overhead():
    """
    Compare single process vs many processes
    More processes = more context switches
    """
    iterations = 10_000_000

    # Single process baseline
    start = time.time()
    result = cpu_bound_work(iterations)
    single_time = time.time() - start
    print(f"Single process: {single_time:.3f}s")

    # Many processes competing for CPU
    num_processes = multiprocessing.cpu_count() * 4  # Oversubscribe
    chunk = iterations // num_processes

    start = time.time()
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(cpu_bound_work, [chunk] * num_processes)
    multi_time = time.time() - start
    print(f"{num_processes} processes: {multi_time:.3f}s")

    overhead = (multi_time / single_time - 1) * 100
    print(f"Overhead from context switching: {overhead:.1f}%")

if __name__ == "__main__":
    measure_context_switch_overhead()
```

**Context Switch Costs:**

| Component | Time (approx) | Why |
|-----------|---------------|-----|
| **Save/restore registers** | 50-100 CPU cycles | Direct hardware operation |
| **Switch page table** | 100-500 cycles | Update MMU |
| **TLB flush** | 1000+ cycles | Cache misses on resume |
| **L1/L2 cache pollution** | Variable | New process evicts old data |
| **Total** | 1-10 microseconds | Depends on architecture |

**When to care:** High-frequency task switching, latency-sensitive systems

**When NOT to care:** I/O bound work (context switch time << I/O wait time)

---

## Process Scheduling Algorithms

**The Problem:**
You have 100 runnable processes and 8 CPU cores. Which processes run when? How do you balance throughput, latency, and fairness?

**Common Schedulers:**

```
┌─────────────────────────────────────────────────────────┐
│ ROUND ROBIN (RR)                                        │
├─────────────────────────────────────────────────────────┤
│ Time quantums: 10ms each                                │
│                                                          │
│ P1 ████  P2 ████  P3 ████  P1 ████  P2 ████            │
│    └─┬─┘     └─┬─┘     └─┬─┘                           │
│      10ms      10ms      10ms                           │
│                                                          │
│ Fair, but high context switch overhead                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ COMPLETELY FAIR SCHEDULER (CFS) - Linux default         │
├─────────────────────────────────────────────────────────┤
│ Track "virtual runtime" for each process                │
│ Always run process with lowest virtual runtime          │
│                                                          │
│ Process │ Virtual Runtime │ Priority │ Next to run?    │
│ ─────────┼─────────────────┼──────────┼─────────────   │
│ P1      │ 100ms           │ normal   │                 │
│ P2      │  80ms           │ normal   │ ✓ (lowest)     │
│ P3      │ 120ms           │ normal   │                 │
│ P4      │  90ms           │ high     │                 │
│                                                          │
│ Automatically fair, handles priorities                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ PRIORITY SCHEDULING                                      │
├─────────────────────────────────────────────────────────┤
│ Queue per priority level                                │
│                                                          │
│ Priority 0 (highest):  [P4] ────────► RUN FIRST        │
│ Priority 1:            [P2, P5] ─────► RUN SECOND      │
│ Priority 2:            [P1, P3] ─────► RUN LAST        │
│                                                          │
│ Problem: Starvation (low priority never runs)           │
└─────────────────────────────────────────────────────────┘
```

**Python Example (Process Priority):**

```python
import os
import time
import multiprocessing

def cpu_intensive_task(name, duration):
    """Simulate CPU-bound work"""
    print(f"{name} starting (PID {os.getpid()})")
    end_time = time.time() + duration

    iterations = 0
    while time.time() < end_time:
        # Busy work
        _ = sum(i * i for i in range(1000))
        iterations += 1

    print(f"{name} completed {iterations} iterations")

def set_process_priority(priority):
    """
    Set process nice value (-20 to 19)
    Lower = higher priority
    Requires root for negative values
    """
    try:
        os.nice(priority)
        print(f"Process {os.getpid()} priority set to {priority}")
    except PermissionError:
        print(f"Cannot set priority {priority} (need root)")

def demonstrate_scheduling():
    """Show effect of process priority"""

    def high_priority_worker():
        set_process_priority(-10)  # High priority
        cpu_intensive_task("HIGH PRIORITY", 3)

    def normal_priority_worker():
        set_process_priority(0)  # Normal
        cpu_intensive_task("NORMAL PRIORITY", 3)

    def low_priority_worker():
        set_process_priority(10)  # Low priority
        cpu_intensive_task("LOW PRIORITY", 3)

    # Start all at once
    processes = [
        multiprocessing.Process(target=high_priority_worker),
        multiprocessing.Process(target=normal_priority_worker),
        multiprocessing.Process(target=low_priority_worker),
    ]

    start = time.time()
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print(f"Total time: {time.time() - start:.2f}s")
    print("High priority process should complete most iterations")

if __name__ == "__main__":
    demonstrate_scheduling()
```

**Scheduler Comparison:**

| Algorithm | Pros | Cons | Use Case |
|-----------|------|------|----------|
| **Round Robin** | Simple, fair | High context switch overhead | Time-sharing systems |
| **CFS** | Fair, priority-aware | Complex implementation | General-purpose (Linux default) |
| **Priority** | Important tasks run first | Starvation risk | Real-time systems |
| **FIFO** | No context switches | No fairness | Batch processing |

---

## IPC Mechanisms: Pipes

**The Problem:**
Two processes need to communicate. Process A generates data, Process B consumes it. They can't share memory directly.

**How It Works:**

```
Pipe (Unidirectional)
─────────────────────

┌─────────────┐                    ┌─────────────┐
│  Process A  │                    │  Process B  │
│  (Writer)   │                    │  (Reader)   │
└──────┬──────┘                    └──────▲──────┘
       │                                  │
       │ write()                   read() │
       │                                  │
       └──────►┌──────────────┐──────────┘
               │  Pipe Buffer │
               │  (kernel)    │
               │  Size: 64KB  │
               └──────────────┘

Named Pipe / FIFO (Persists in filesystem)
──────────────────────────────────────────

┌─────────────┐                    ┌─────────────┐
│  Process A  │                    │  Process B  │
│ (unrelated) │                    │ (unrelated) │
└──────┬──────┘                    └──────▲──────┘
       │                                  │
       │ open("/tmp/mypipe", "w")        │
       │ write()                   read() │
       │                                  │
       └──────►┌──────────────┐──────────┘
               │ /tmp/mypipe  │
               │ (named pipe) │
               └──────────────┘
```

**Python Example:**

```python
import os
import multiprocessing
import time

# Anonymous pipe example
def pipe_example():
    """
    Parent and child communicate via pipe
    pipe() returns (read_fd, write_fd)
    """
    # Create pipe before fork
    read_fd, write_fd = os.pipe()

    pid = os.fork()

    if pid == 0:
        # Child: write to pipe
        os.close(read_fd)  # Close unused end

        message = b"Hello from child process!"
        os.write(write_fd, message)
        os.close(write_fd)
        os._exit(0)
    else:
        # Parent: read from pipe
        os.close(write_fd)  # Close unused end

        data = os.read(read_fd, 1024)
        print(f"Parent received: {data.decode()}")

        os.close(read_fd)
        os.wait()

# multiprocessing.Pipe (higher level, bidirectional)
def multiprocessing_pipe_example():
    """
    Cleaner API for Python processes
    Supports both directions
    """
    parent_conn, child_conn = multiprocessing.Pipe()

    def child_process(conn):
        # Receive from parent
        msg = conn.recv()
        print(f"Child received: {msg}")

        # Send to parent
        conn.send({"result": "processed", "data": msg.upper()})
        conn.close()

    p = multiprocessing.Process(target=child_process, args=(child_conn,))
    p.start()

    # Send to child
    parent_conn.send("hello world")

    # Receive from child
    result = parent_conn.recv()
    print(f"Parent received: {result}")

    p.join()

# Named pipe (FIFO)
def named_pipe_example():
    """
    Communicate between unrelated processes
    Persists in filesystem
    """
    fifo_path = "/tmp/my_fifo"

    # Create named pipe
    try:
        os.mkfifo(fifo_path)
    except FileExistsError:
        pass

    pid = os.fork()

    if pid == 0:
        # Child: writer
        time.sleep(1)  # Ensure parent is reading
        with open(fifo_path, 'w') as fifo:
            fifo.write("Data from child\n")
        os._exit(0)
    else:
        # Parent: reader
        # open() blocks until writer connects
        with open(fifo_path, 'r') as fifo:
            data = fifo.read()
            print(f"Parent read: {data.strip()}")

        os.wait()
        os.unlink(fifo_path)  # Clean up

if __name__ == "__main__":
    print("=== Anonymous Pipe ===")
    pipe_example()

    print("\n=== multiprocessing.Pipe ===")
    multiprocessing_pipe_example()

    print("\n=== Named Pipe ===")
    named_pipe_example()
```

**Trade-offs:**

| Type | Pros | Cons | Use Case |
|------|------|------|----------|
| **Anonymous pipe** | Fast, simple | Only parent-child | Process spawning |
| **Named pipe** | Any processes can connect | Slower, filesystem overhead | Daemons, IPC services |
| **multiprocessing.Pipe** | Pythonic, bidirectional | Python-only | Python multiprocessing |

**When to use:** Streaming data between processes, simple producer-consumer

**When NOT to use:** Many-to-many communication, need random access, large data (use shared memory)

---

## IPC Mechanisms: Message Queues

**The Problem:**
Pipes are point-to-point. You need multiple producers, multiple consumers, and message priorities.

**How It Works:**

```
Message Queue
─────────────

┌──────────┐  ┌──────────┐  ┌──────────┐
│Producer 1│  │Producer 2│  │Producer 3│
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     └─────────────┼─────────────┘
                   │ send()
                   ▼
     ┌──────────────────────────────┐
     │     Message Queue (kernel)   │
     ├──────────────────────────────┤
     │ [Msg1, Priority=5]           │
     │ [Msg2, Priority=10]          │
     │ [Msg3, Priority=1]           │
     └──────────────────────────────┘
                   │ receive()
     ┌─────────────┼─────────────┐
     │             │             │
     ▼             ▼             ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│Consumer 1│  │Consumer 2│  │Consumer 3│
└──────────┘  └──────────┘  └──────────┘

Features:
- Messages are discrete units (not byte stream)
- Can prioritize messages
- Multiple readers/writers
- Messages persist until read
```

**Python Example:**

```python
import multiprocessing
import time

def message_queue_example():
    """
    multiprocessing.Queue - thread and process safe
    Built on pipes + locks
    """
    queue = multiprocessing.Queue()

    def producer(queue, producer_id):
        for i in range(5):
            message = {
                "producer_id": producer_id,
                "value": i,
                "timestamp": time.time()
            }
            queue.put(message)
            print(f"Producer {producer_id} sent: {message['value']}")
            time.sleep(0.1)

    def consumer(queue, consumer_id):
        while True:
            try:
                # Timeout prevents hanging if queue is empty
                message = queue.get(timeout=2)
                print(f"Consumer {consumer_id} received: {message}")
                time.sleep(0.2)  # Simulate processing
            except:
                break  # Timeout - no more messages

    # Start multiple producers
    producers = [
        multiprocessing.Process(target=producer, args=(queue, i))
        for i in range(3)
    ]

    # Start multiple consumers
    consumers = [
        multiprocessing.Process(target=consumer, args=(queue, i))
        for i in range(2)
    ]

    for p in producers:
        p.start()
    for c in consumers:
        c.start()

    for p in producers:
        p.join()
    for c in consumers:
        c.join()

# Priority queue example
def priority_queue_example():
    """
    Process urgent messages first
    """
    import queue

    pq = multiprocessing.Queue()

    # Send messages with priorities
    messages = [
        (10, "Low priority task"),
        (1, "URGENT: System failure"),
        (5, "Medium priority update"),
        (2, "High priority alert"),
    ]

    # Note: multiprocessing.Queue doesn't support priorities
    # Use regular queue.PriorityQueue in threads, or
    # implement priority handling in consumer

    def priority_consumer(q):
        # Collect all messages
        buffer = []
        while True:
            try:
                msg = q.get(timeout=1)
                buffer.append(msg)
            except:
                break

        # Sort by priority
        buffer.sort(key=lambda x: x[0])

        # Process in order
        for priority, msg in buffer:
            print(f"Processing (priority {priority}): {msg}")

    for msg in messages:
        pq.put(msg)

    consumer = multiprocessing.Process(target=priority_consumer, args=(pq,))
    consumer.start()
    consumer.join()

if __name__ == "__main__":
    print("=== Basic Message Queue ===")
    message_queue_example()

    print("\n=== Priority Queue ===")
    priority_queue_example()
```

**Message Queue vs Pipe:**

| Feature | Message Queue | Pipe |
|---------|---------------|------|
| **Data unit** | Discrete messages | Byte stream |
| **Multiple readers** | Yes | No (data consumed once) |
| **Priorities** | Yes (with right implementation) | No |
| **Complexity** | Higher | Lower |
| **Performance** | Slower (more overhead) | Faster |

**When to use:** Task distribution, decoupled producers/consumers, priority handling

**When NOT to use:** Streaming large data, need guaranteed order across producers

---

## IPC Mechanisms: Shared Memory

**The Problem:**
Copying data through pipes/queues is slow. You're passing 100MB numpy arrays between processes. Copying that data takes 50ms each time.

**How It Works:**

```
Shared Memory Region
────────────────────

Process A Memory         Shared Region        Process B Memory
┌──────────────┐        ┌──────────────┐      ┌──────────────┐
│ Private      │        │              │      │ Private      │
│ Stack        │        │ SHARED ARRAY │      │ Stack        │
│ Heap         │        │ [1,2,3,...]  │      │ Heap         │
│              │   ┌───►│              │◄───┐ │              │
│              │   │    │ Size: 100MB  │    │ │              │
└──────────────┘   │    └──────────────┘    │ └──────────────┘
                   │                         │
                   │ mmap() same region     │
                   │                         │
                   └─────────┬───────────────┘
                             │
                    Both processes see
                    same physical memory

                    Need synchronization!
                    (locks, semaphores)
```

**Python Example:**

```python
import multiprocessing
import numpy as np
import time
import sys

def shared_memory_array_example():
    """
    Share numpy array between processes
    Zero-copy communication
    """
    # Create shared memory array
    # 'd' = double (float64)
    shared_array = multiprocessing.Array('d', 10_000_000)

    def writer(shared_arr):
        """Write to shared memory"""
        print(f"Writer: Filling array (PID {multiprocessing.current_process().pid})")
        start = time.time()

        # Convert to numpy array (zero copy - uses same memory)
        np_array = np.frombuffer(shared_arr.get_obj())
        np_array[:] = np.random.random(10_000_000)

        print(f"Writer: Done in {time.time() - start:.3f}s")

    def reader(shared_arr):
        """Read from shared memory"""
        time.sleep(0.5)  # Wait for writer
        print(f"Reader: Reading array (PID {multiprocessing.current_process().pid})")
        start = time.time()

        np_array = np.frombuffer(shared_arr.get_obj())
        result = np_array.sum()

        print(f"Reader: Sum = {result:.2f}, done in {time.time() - start:.3f}s")

    processes = [
        multiprocessing.Process(target=writer, args=(shared_array,)),
        multiprocessing.Process(target=reader, args=(shared_array,)),
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print(f"Array size in memory: {sys.getsizeof(shared_array)} bytes (not copied!)")

def shared_memory_with_locks():
    """
    Demonstrate race condition and fix with locks
    """
    # Shared counter
    counter = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()

    def increment_without_lock(counter, n):
        for _ in range(n):
            # RACE CONDITION: read-modify-write not atomic
            counter.value += 1

    def increment_with_lock(counter, lock, n):
        for _ in range(n):
            with lock:
                counter.value += 1

    # Without lock - race condition
    counter.value = 0
    processes = [
        multiprocessing.Process(target=increment_without_lock, args=(counter, 100000))
        for _ in range(4)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print(f"Without lock: {counter.value} (expected 400000) - WRONG!")

    # With lock - correct
    counter.value = 0
    processes = [
        multiprocessing.Process(target=increment_with_lock, args=(counter, lock, 100000))
        for _ in range(4)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print(f"With lock: {counter.value} (expected 400000) - CORRECT!")

def compare_pipe_vs_shared_memory():
    """
    Benchmark: copying via pipe vs shared memory
    """
    size = 10_000_000
    data = np.random.random(size)

    # Pipe approach (copy data)
    print("Testing pipe (copy)...")
    parent_conn, child_conn = multiprocessing.Pipe()

    start = time.time()
    p = multiprocessing.Process(
        target=lambda conn: conn.recv().sum(),
        args=(child_conn,)
    )
    p.start()
    parent_conn.send(data)  # Serializes and copies!
    p.join()
    pipe_time = time.time() - start
    print(f"Pipe: {pipe_time:.3f}s")

    # Shared memory approach (zero copy)
    print("Testing shared memory (zero copy)...")
    shared_arr = multiprocessing.Array('d', size)

    start = time.time()
    np_arr = np.frombuffer(shared_arr.get_obj())
    np_arr[:] = data

    p = multiprocessing.Process(
        target=lambda arr: np.frombuffer(arr.get_obj()).sum(),
        args=(shared_arr,)
    )
    p.start()
    p.join()
    shared_time = time.time() - start
    print(f"Shared memory: {shared_time:.3f}s")

    print(f"Speedup: {pipe_time / shared_time:.1f}x")

if __name__ == "__main__":
    print("=== Shared Memory Array ===")
    shared_memory_array_example()

    print("\n=== Race Condition Demo ===")
    shared_memory_with_locks()

    print("\n=== Performance Comparison ===")
    compare_pipe_vs_shared_memory()
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| **Performance** | Zero-copy, very fast | |
| **Synchronization** | | Need locks/semaphores (complex) |
| **Memory** | Single copy shared | Not isolated - corruption risk |
| **Scalability** | | Contention on locks with many writers |

**When to use:** Large data structures, high-frequency access, performance critical

**When NOT to use:** Simple communication, untrusted processes, need isolation

---

## Worker Process Patterns

**The Problem:**
Your web server receives requests. Creating a new process per request is too slow (fork takes 1-5ms). A single process can't utilize multiple cores.

**Common Patterns:**

```
PREFORK (Apache, Gunicorn)
──────────────────────────

Master Process (doesn't handle requests)
    │
    ├─ fork() ──► Worker 1 ──► handle_request()
    ├─ fork() ──► Worker 2 ──► handle_request()
    ├─ fork() ──► Worker 3 ──► handle_request()
    └─ fork() ──► Worker 4 ──► handle_request()

Pros: Simple, copy-on-write memory sharing, crash isolation
Cons: Fixed worker count, slow to scale

────────────────────────────────────────────

WORKER POOL (multiprocessing.Pool)
──────────────────────────────────

Task Queue: [Task1][Task2][Task3][Task4][Task5]
                │      │      │      │      │
                └──────┼──────┼──────┼──────┘
                       │      │      │
                   ┌───┴──┬───┴──┬───┴──┐
                   │      │      │      │
                   ▼      ▼      ▼      ▼
               Worker1 Worker2 Worker3 Worker4

Tasks distributed dynamically to available workers

Pros: Dynamic distribution, efficient with varying task times
Cons: Queue overhead, coordination complexity

────────────────────────────────────────────

ACTOR MODEL (Celery workers)
────────────────────────────

┌──────────┐     ┌──────────┐     ┌──────────┐
│ Worker 1 │     │ Worker 2 │     │ Worker 3 │
│          │     │          │     │          │
│ Queue 1  │     │ Queue 2  │     │ Queue 3  │
└────▲─────┘     └────▲─────┘     └────▲─────┘
     │                │                │
     └────────────────┼────────────────┘
                      │
                Message Broker (Redis/RabbitMQ)
                      │
                      │
                 Task Producer

Pros: Distributed, scalable, persistent tasks
Cons: Complex setup, external dependencies
```

**Python Implementation:**

```python
import multiprocessing
import time
import os
import signal

class PreforkServer:
    """
    Prefork pattern: create workers at startup
    Similar to Gunicorn
    """
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.workers = []
        self.shutdown = False

    def worker_process(self, worker_id):
        """Worker loop - handles requests"""
        print(f"Worker {worker_id} started (PID {os.getpid()})")

        # Install signal handler for graceful shutdown
        signal.signal(signal.SIGTERM, lambda sig, frame: None)

        while not self.shutdown:
            # Simulate handling a request
            # In real server, this would accept() on shared socket
            time.sleep(0.1)
            self.handle_request(worker_id)

    def handle_request(self, worker_id):
        """Process one request"""
        # Simulate work
        time.sleep(0.01)

    def start(self):
        """Fork worker processes"""
        for i in range(self.num_workers):
            pid = os.fork()
            if pid == 0:
                # Child process
                self.worker_process(i)
                os._exit(0)
            else:
                # Parent process
                self.workers.append(pid)

        print(f"Master process {os.getpid()} started {self.num_workers} workers")

    def shutdown_gracefully(self):
        """Send SIGTERM to all workers, wait for completion"""
        print("Shutting down workers...")
        for pid in self.workers:
            os.kill(pid, signal.SIGTERM)

        for pid in self.workers:
            os.waitpid(pid, 0)
        print("All workers stopped")

class WorkerPool:
    """
    Worker pool pattern with task queue
    Uses multiprocessing.Pool
    """
    def process_task(self, task_id):
        """Worker function"""
        print(f"Worker {os.getpid()} processing task {task_id}")
        time.sleep(0.1)  # Simulate work
        return task_id * 2

    def run(self, num_tasks=20, num_workers=4):
        """Distribute tasks across worker pool"""
        start = time.time()

        with multiprocessing.Pool(num_workers) as pool:
            # map() blocks until all tasks complete
            results = pool.map(self.process_task, range(num_tasks))

        elapsed = time.time() - start
        print(f"Processed {num_tasks} tasks in {elapsed:.2f}s")
        print(f"Results: {results[:5]}...")  # Show first few

def graceful_shutdown_example():
    """
    Handle signals for clean shutdown
    Important for databases, open files, etc.
    """
    shutdown_event = multiprocessing.Event()

    def worker(shutdown_event):
        """Worker that checks for shutdown signal"""
        def signal_handler(sig, frame):
            print(f"Worker {os.getpid()} received signal {sig}")
            shutdown_event.set()

        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        print(f"Worker {os.getpid()} starting")

        while not shutdown_event.is_set():
            # Do work
            time.sleep(0.5)
            print(f"Worker {os.getpid()} working...")

        # Cleanup
        print(f"Worker {os.getpid()} shutting down gracefully")
        time.sleep(0.5)  # Simulate cleanup
        print(f"Worker {os.getpid()} stopped")

    # Start workers
    workers = [
        multiprocessing.Process(target=worker, args=(shutdown_event,))
        for _ in range(3)
    ]

    for w in workers:
        w.start()

    # Let them run
    time.sleep(2)

    # Trigger graceful shutdown
    print("Main: Triggering shutdown")
    shutdown_event.set()

    for w in workers:
        w.join(timeout=5)
        if w.is_alive():
            print(f"Worker {w.pid} didn't stop gracefully, killing")
            w.terminate()

if __name__ == "__main__":
    print("=== Worker Pool Pattern ===")
    pool = WorkerPool()
    pool.run(num_tasks=20, num_workers=4)

    print("\n=== Graceful Shutdown ===")
    graceful_shutdown_example()
```

**Pattern Comparison:**

| Pattern | Best For | Startup Cost | Flexibility |
|---------|----------|--------------|-------------|
| **Prefork** | Stable load, shared resources | High | Low |
| **Worker Pool** | Variable task times | Medium | Medium |
| **On-demand spawn** | Rare events, isolation | Low | High |
| **Actor/Celery** | Distributed systems, async tasks | High | Very High |

---

## Zombie and Orphan Processes

**The Problem:**
A child process exits but parent doesn't call `wait()`. Process becomes zombie - dead but not reaped, consuming PID space. Or parent exits before child - child becomes orphan, reparented to init.

**How It Works:**

```
ZOMBIE PROCESS
──────────────

┌──────────┐
│  Parent  │  Child exits, becomes zombie
│  (alive) │  waiting for parent to wait()
└────┬─────┘
     │
     │ fork()
     ▼
┌──────────┐  exit(0)    ┌──────────────┐
│  Child   │────────────►│ Zombie (Z)   │
│ (running)│             │ PID still    │
└──────────┘             │ allocated    │
                         │ Exit status  │
                         │ saved        │
                         └──────────────┘
                                │
                         Parent never calls wait()
                         Zombie persists!

$ ps aux | grep Z
child     12345  0.0  0.0      0     0 ?   Z   10:23  0:00 [python] <defunct>

ORPHAN PROCESS
──────────────

┌──────────┐
│  Parent  │ exit(0) - dies before child
└────┬─────┘
     │
     │ fork()
     ▼
┌──────────┐             ┌──────────┐
│  Child   │   adopted   │  init    │
│ (running)│────────────►│  PID=1   │
└──────────┘             └──────────┘

Child reparented to init (PID 1)
init will reap child when it exits
Not a problem - handled automatically
```

**Python Example:**

```python
import os
import time
import signal
import subprocess

def create_zombie():
    """
    Demonstrate zombie process creation
    Child exits, parent doesn't wait
    """
    print(f"Parent PID: {os.getpid()}")

    pid = os.fork()

    if pid == 0:
        # Child: exit immediately
        print(f"Child PID {os.getpid()} exiting")
        os._exit(0)
    else:
        # Parent: DON'T call wait() - creates zombie
        print(f"Parent created child {pid}")
        print("Parent sleeping without calling wait()...")

        time.sleep(2)

        # Check for zombie in process list
        result = subprocess.run(
            ["ps", "-o", "pid,stat,command", "-p", str(pid)],
            capture_output=True,
            text=True
        )
        print("Process status:")
        print(result.stdout)
        # Should show 'Z' in STAT column

        # Finally reap the zombie
        print("Reaping zombie...")
        os.waitpid(pid, 0)
        print("Zombie reaped")

def prevent_zombies():
    """
    Proper way: always wait for children
    """
    children = []

    for i in range(3):
        pid = os.fork()
        if pid == 0:
            # Child
            print(f"Child {i} (PID {os.getpid()}) working")
            time.sleep(1)
            os._exit(0)
        else:
            children.append(pid)

    # Wait for all children
    print("Parent waiting for all children...")
    for pid in children:
        finished_pid, status = os.waitpid(pid, 0)
        print(f"Child {finished_pid} exited with status {status >> 8}")

    print("No zombies created!")

def signal_handler_reaping():
    """
    Async reaping with SIGCHLD handler
    Good for servers that spawn many children
    """
    def sigchld_handler(signum, frame):
        """Called when child process exits"""
        # Reap all dead children
        while True:
            try:
                # WNOHANG = don't block if no child ready
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
                print(f"Reaped child {pid} in signal handler")
            except ChildProcessError:
                break  # No more children

    # Install signal handler
    signal.signal(signal.SIGCHLD, sigchld_handler)

    # Spawn children
    for i in range(5):
        pid = os.fork()
        if pid == 0:
            # Child: random exit time
            time.sleep(0.5 + i * 0.2)
            os._exit(i)

    # Parent continues doing other work
    print("Parent doing other work while children run...")
    time.sleep(3)
    print("Parent exiting (children auto-reaped by signal handler)")

def create_orphan():
    """
    Demonstrate orphan process (harmless)
    Parent exits before child
    """
    pid = os.fork()

    if pid == 0:
        # Child: sleep longer than parent
        print(f"Child PID {os.getpid()}, parent PID {os.getppid()}")
        time.sleep(2)
        print(f"Child still running, parent PID now {os.getppid()}")
        print("(Should be 1 or init process - child is orphaned)")
        os._exit(0)
    else:
        # Parent: exit immediately
        print(f"Parent {os.getpid()} exiting before child {pid}")
        # Don't wait - parent exits first
        # Child gets adopted by init

if __name__ == "__main__":
    print("=== Zombie Process Demo ===")
    create_zombie()

    print("\n=== Proper Wait (No Zombies) ===")
    prevent_zombies()

    print("\n=== Signal Handler Reaping ===")
    signal_handler_reaping()

    print("\n=== Orphan Process Demo ===")
    # Fork again so parent can exit
    if os.fork() == 0:
        create_orphan()
    else:
        os.wait()  # Wait for the forked demo
```

**Key Points:**

| Issue | Cause | Fix | Detection |
|-------|-------|-----|-----------|
| **Zombie** | Parent doesn't `wait()` | Call `wait()` or install SIGCHLD handler | `ps aux | grep Z` |
| **Orphan** | Parent exits before child | (Harmless - init reaps them) | Check PPID = 1 |
| **Resource leak** | Many zombies accumulate | Always wait in loops, use signal handlers | Monitor PID exhaustion |

**When to care:** Long-running servers, spawning many processes

**When NOT to care:** Short-lived scripts, using multiprocessing module (handles it)

---

## Key Concepts Checklist

- [ ] Explain process vs thread trade-offs (memory, isolation, GIL)
- [ ] Describe fork-exec-wait lifecycle and prevent zombies
- [ ] Calculate context switching overhead and when it matters
- [ ] Compare IPC mechanisms: pipes, queues, shared memory, sockets
- [ ] Implement worker pool pattern with graceful shutdown
- [ ] Use multiprocessing.Array for zero-copy shared memory
- [ ] Handle signals properly (SIGTERM, SIGCHLD, SIGINT)
- [ ] Debug zombie and orphan processes with ps/top

---

## Practical Insights

**Process vs thread decision:**
In Python, use processes for CPU-bound work (bypass GIL), threads for I/O-bound work (lower overhead). In other languages without GIL, threads are often better unless you need strong isolation. Measure memory: if processes push you into swap, use threads.

**IPC mechanism choice:**
Start with pipes for simple parent-child communication. Use queues when you have multiple producers/consumers. Only reach for shared memory when profiling shows serialization overhead is a bottleneck - usually at 10MB+ data or 1000+ messages/second. Shared memory is fast but debugging race conditions is painful.

**Worker pool sizing:**
For CPU-bound: `num_workers = cpu_count()`. Going higher just adds context switch overhead. For I/O-bound: `num_workers = cpu_count() * 2 to 4`. Too many workers waste memory (10-50MB each), too few waste CPU during I/O waits. Monitor CPU utilization - should be 80%+ for CPU-bound, 20-40% for I/O-bound.

**Graceful shutdown is non-negotiable:**
Always handle SIGTERM for clean shutdown. In-flight requests should complete, database connections should close, temp files should be cleaned up. Kubernetes gives you 30 seconds (default) before SIGKILL. Instagram's shutdown: check health endpoint returns 503, wait for connections to drain (5s), close DB connections, exit. Without this, you get partial writes, corrupted data, leaked connections.

**Zombie prevention patterns:**
In servers, install SIGCHLD handler that calls `waitpid(-1, WNOHANG)` in a loop - reaps all dead children without blocking. In scripts, call `wait()` after every fork. In multiprocessing module, it's handled automatically. I've seen production systems hit PID exhaustion (32768 limit) from zombie accumulation - server stops accepting connections. Monitor zombie count in metrics.

**Shared memory synchronization:**
With 2 writers, contention is low. With 10+ writers, lock contention kills your performance gains from shared memory. Solution: partition the shared memory region - each worker owns a segment, reader aggregates. Or use lock-free data structures (atomic operations) for simple cases like counters. Profile with `perf` to see lock wait time.
