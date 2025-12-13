# Chapter 37: Async Programming & Event Loops

## The Core Problem

Your web server handles 10,000 concurrent connections. Each request waits for:
- Database query: 50ms
- Redis cache lookup: 5ms
- External API call: 200ms

With traditional thread-per-request model:

```
10,000 threads × 8MB stack = 80GB memory just for stacks
Plus: Context switching overhead kills CPU cache
Result: Server handles ~1,000 concurrent connections max
```

With async/await and event loops:

```
1 thread with event loop
Minimal memory per connection (~4KB)
Result: Same server handles 100,000+ concurrent connections
C10K problem? Solved.
```

The catch: You're no longer writing sequential code. You're writing cooperative multitasking where functions voluntarily yield control. Master this paradigm shift, and you unlock massive scalability for I/O-bound workloads. Misunderstand it, and you'll block the event loop, creating mysterious latency spikes that take down your entire service.

---

## Blocking vs Non-Blocking I/O

### The Fundamental Difference

**Blocking I/O:**
```python
# Traditional blocking file read
def handle_request():
    data = file.read()  # Thread SLEEPS here for milliseconds
    # While sleeping:
    # - Thread consumes 8MB stack
    # - CPU time wasted
    # - No other work can happen on this thread
    return process(data)
```

**Non-Blocking I/O:**
```python
# Non-blocking read
def handle_request():
    file.read_async(callback=on_data_ready)
    # Returns IMMEDIATELY
    # OS will call on_data_ready() when data arrives
    # Thread can handle OTHER requests meanwhile

def on_data_ready(data):
    return process(data)
```

### What Actually Happens at OS Level

```
Blocking read():
┌─────────────────────────────────────────────────────┐
│ Thread calls read(fd)                                │
├─────────────────────────────────────────────────────┤
│ 1. Syscall enters kernel                             │
│ 2. Check if data available in kernel buffer          │
│ 3. NO → Thread state = BLOCKED                       │
│ 4. Thread removed from scheduler                     │
│ 5. ... wait for disk I/O ...                        │
│ 6. Disk interrupt → data ready                       │
│ 7. Thread state = RUNNABLE                           │
│ 8. Eventually scheduled, returns from syscall        │
└─────────────────────────────────────────────────────┘
Time: 5-50ms (thread does NOTHING)

Non-blocking read():
┌─────────────────────────────────────────────────────┐
│ Thread calls read(fd, O_NONBLOCK)                    │
├─────────────────────────────────────────────────────┤
│ 1. Syscall enters kernel                             │
│ 2. Check if data available                           │
│ 3. NO → Returns EAGAIN immediately                   │
│ 4. Thread continues executing other code             │
│ 5. Later: poll/select/epoll checks fd readiness      │
│ 6. When ready, call callback                         │
└─────────────────────────────────────────────────────┘
Time: Microseconds (thread stays productive)
```

### I/O Multiplexing: The Enabling Technology

How does one thread monitor thousands of file descriptors?

**select() - The Original (1983):**
```c
// Pseudocode - monitors which FDs are ready
fd_set read_fds;
FD_ZERO(&read_fds);
FD_SET(socket1, &read_fds);
FD_SET(socket2, &read_fds);
// ... add 10,000 sockets

select(max_fd, &read_fds, NULL, NULL, timeout);
// Returns when ANY fd is ready

// Problem: O(n) scan of ALL fds on every call
```

**epoll() - Modern Linux (2002):**
```c
// Register interest once
epoll_ctl(epfd, EPOLL_CTL_ADD, socket1, &event);

// Wait for events - O(1) for ready sockets
int n = epoll_wait(epfd, events, MAX_EVENTS, timeout);
for (int i = 0; i < n; i++) {
    handle_ready_socket(events[i].data.fd);
}
```

| Mechanism | Complexity | Max FDs | Platform |
|-----------|-----------|---------|----------|
| select() | O(n) scan | 1024 (FD_SETSIZE) | All POSIX |
| poll() | O(n) scan | Unlimited | All POSIX |
| epoll() | O(1) | Millions | Linux |
| kqueue() | O(1) | Millions | BSD, macOS |
| IOCP | O(1) | Millions | Windows |

---

## Event Loop: The Heart of Async

An event loop is a while loop that:
1. Waits for events (I/O ready, timers expired)
2. Dispatches callbacks for those events
3. Returns to waiting

```
┌──────────────────────────────────────────────────┐
│                EVENT LOOP                         │
└──────────────────────────────────────────────────┘
              │
              ▼
       ┌────────────┐
       │  Wait for  │ ◄─────────────┐
       │   events   │                │
       └────────────┘                │
              │                      │
              ▼                      │
       ┌────────────┐                │
       │  Dispatch  │                │
       │  callbacks │                │
       └────────────┘                │
              │                      │
              ▼                      │
       ┌────────────┐                │
       │   Execute  │                │
       │  callback  │                │
       └────────────┘                │
              │                      │
              └──────────────────────┘
         (repeat forever)
```

### Minimal Event Loop Implementation

```python
import selectors
import socket

class EventLoop:
    def __init__(self):
        self.selector = selectors.DefaultSelector()  # epoll on Linux
        self.tasks = []

    def register_socket(self, sock, callback):
        """Register interest in socket readability"""
        self.selector.register(sock, selectors.EVENT_READ, callback)

    def schedule_callback(self, callback):
        """Run callback on next iteration"""
        self.tasks.append(callback)

    def run_forever(self):
        while True:
            # Run scheduled tasks
            while self.tasks:
                task = self.tasks.pop(0)
                task()

            # Wait for I/O events (blocks here!)
            events = self.selector.select(timeout=0.1)

            for key, mask in events:
                callback = key.data
                callback(key.fileobj)

# Usage
loop = EventLoop()

def on_client_connect(server_sock):
    client_sock, addr = server_sock.accept()
    client_sock.setblocking(False)
    loop.register_socket(client_sock, on_client_readable)

def on_client_readable(client_sock):
    data = client_sock.recv(1024)
    if data:
        # Process and respond
        client_sock.sendall(b"HTTP/1.1 200 OK\r\n\r\n" + data)
    else:
        client_sock.close()

server = socket.socket()
server.bind(('0.0.0.0', 8080))
server.listen(100)
server.setblocking(False)

loop.register_socket(server, on_client_connect)
loop.run_forever()
```

**Key property:** Single-threaded. Only one callback runs at a time. No race conditions!

---

## Coroutines: Suspendable Functions

A coroutine is a function that can pause execution and resume later, preserving local state.

### Python Generators: The Foundation

```python
def countdown(n):
    while n > 0:
        yield n  # PAUSE here, return n to caller
        n -= 1

# Usage
gen = countdown(3)
print(next(gen))  # 3 (pauses at yield)
print(next(gen))  # 2 (resumes, pauses again)
print(next(gen))  # 1
# next(gen) raises StopIteration
```

**What yield does:**
1. Saves function's local variables and instruction pointer
2. Returns value to caller
3. Function remains "suspended"
4. next() resumes from exactly where it left off

### Generator-Based Coroutines (Pre-async/await)

```python
def fetch_user(user_id):
    # Start database query (non-blocking)
    result = yield db.query(f"SELECT * FROM users WHERE id={user_id}")
    # Execution pauses here until query completes

    # Once resumed with result:
    return result

# Event loop drives this:
coroutine = fetch_user(123)
query = next(coroutine)  # Gets the query object
# ... wait for query to complete ...
try:
    coroutine.send(query_result)  # Resume with result
except StopIteration as e:
    final_result = e.value
```

**The pattern:**
- `yield` = "pause me until this I/O completes"
- Event loop waits for I/O
- `.send()` = "resume with this result"

### Native Coroutines: async/await

Python 3.5+ syntax sugar for coroutines:

```python
# Modern syntax
async def fetch_user(user_id):
    result = await db.query(f"SELECT * FROM users WHERE id={user_id}")
    return result

# Equivalent to generator version, but clearer intent
```

**Key differences from generators:**
- `async def` creates a coroutine function (not generator)
- Can only `await` inside `async def`
- Can only call async functions with `await` or event loop methods

---

## Futures and Promises

A Future/Promise represents a value that will be available later.

```
Timeline:
T0: Create Future (empty box)
T1: Start async operation, return Future immediately
T2: ... operation in progress ...
T3: Operation completes, Future is "resolved" with value
T4: Anyone waiting on Future gets the value
```

### Python asyncio.Future

```python
import asyncio

async def example():
    # Create empty Future
    future = asyncio.Future()

    # Schedule something to fill it later
    async def fill_later():
        await asyncio.sleep(1)
        future.set_result("Hello")

    asyncio.create_task(fill_later())

    # Wait for Future to be filled
    result = await future  # Blocks here until set_result called
    print(result)  # "Hello"

# Future states:
# - Pending: Not yet resolved
# - Fulfilled: Resolved with a value
# - Rejected: Resolved with an error
```

### JavaScript Promise (Comparison)

```javascript
// JavaScript Promise
function fetchUser(userId) {
    return new Promise((resolve, reject) => {
        db.query(`SELECT * FROM users WHERE id=${userId}`, (err, result) => {
            if (err) reject(err);
            else resolve(result);
        });
    });
}

// Usage with async/await
async function example() {
    const user = await fetchUser(123);
    console.log(user);
}
```

**Python vs JavaScript:**

| Aspect | Python | JavaScript |
|--------|--------|------------|
| Syntax | `async`/`await` | `async`/`await` |
| Execution | Requires explicit event loop | Built-in event loop (browser/Node.js) |
| Compatibility | asyncio, Trio, curio | Promise, native |
| Cancellation | Supported (task.cancel()) | No built-in cancellation |

---

## Async/Await: How It Really Works

### Desugaring async/await

```python
# What you write:
async def fetch_and_process():
    data = await fetch_data()
    result = await process(data)
    return result

# What Python compiles to (conceptually):
def fetch_and_process():
    # Create coroutine object
    coro = _fetch_and_process_impl()
    return coro

def _fetch_and_process_impl():
    # State machine:
    state = 0
    data = None
    result = None

    while True:
        if state == 0:
            # Start fetch
            future = fetch_data()
            state = 1
            yield future  # Pause, return Future to event loop

        elif state == 1:
            # Resume with data
            data = (yield)  # Get sent value
            future = process(data)
            state = 2
            yield future

        elif state == 2:
            result = (yield)
            return result
```

### await: The Pause Point

```python
async def example():
    print("Start")
    result = await async_operation()  # ← Execution pauses HERE
    print("Done:", result)            # ← Resumes HERE when done

# When you await:
# 1. Current coroutine pauses
# 2. Event loop runs OTHER coroutines/callbacks
# 3. When async_operation completes, event loop resumes this coroutine
```

**Critical rule:** `await` ONLY pauses the current coroutine, not the entire thread.

```python
async def good_example():
    # These run CONCURRENTLY (not in parallel - still single-threaded):
    task1 = asyncio.create_task(fetch_user(1))
    task2 = asyncio.create_task(fetch_user(2))

    user1 = await task1  # Might already be done!
    user2 = await task2  # Might already be done!

    # Total time: max(fetch1_time, fetch2_time), not sum

async def bad_example():
    # These run SEQUENTIALLY:
    user1 = await fetch_user(1)  # Wait 50ms
    user2 = await fetch_user(2)  # Wait another 50ms
    # Total time: 100ms
```

---

## Callback Hell and How Async/Await Solves It

### The Callback Pyramid of Doom

```javascript
// JavaScript callback hell
function processUser(userId, finalCallback) {
    getUser(userId, (err, user) => {
        if (err) return finalCallback(err);

        getOrders(user.id, (err, orders) => {
            if (err) return finalCallback(err);

            getOrderDetails(orders[0].id, (err, details) => {
                if (err) return finalCallback(err);

                updateInventory(details, (err) => {
                    if (err) return finalCallback(err);

                    sendEmail(user.email, (err) => {
                        if (err) return finalCallback(err);
                        finalCallback(null, "Success");
                    });
                });
            });
        });
    });
}
```

**Problems:**
- Hard to read (rightward drift)
- Error handling duplicated
- Can't use try/catch
- Can't use loops/conditionals naturally

### The async/await Solution

```python
async def process_user(user_id):
    try:
        user = await get_user(user_id)
        orders = await get_orders(user.id)
        details = await get_order_details(orders[0].id)
        await update_inventory(details)
        await send_email(user.email)
        return "Success"
    except Exception as e:
        # Single error handler for entire chain!
        log.error(f"Failed to process user: {e}")
        raise
```

**Benefits:**
- Linear, readable code
- Normal try/catch works
- Can use loops, conditionals naturally
- Debugger can step through

---

## Python asyncio in Depth

### Core Components

```
┌────────────────────────────────────────────────┐
│             asyncio Architecture                │
├────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────────────────────────────┐      │
│  │         Event Loop                    │      │
│  │  - Task scheduler                     │      │
│  │  - I/O multiplexing (epoll/kqueue)    │      │
│  │  - Timer management                   │      │
│  └──────────────────────────────────────┘      │
│              ▲         │                        │
│              │         ▼                        │
│  ┌───────────────┐  ┌──────────────┐           │
│  │    Tasks      │  │   Futures    │           │
│  │  (wrapped     │  │  (results)   │           │
│  │   coroutines) │  │              │           │
│  └───────────────┘  └──────────────┘           │
│              ▲                                  │
│              │                                  │
│  ┌──────────────────────────────────────┐      │
│  │        Your Coroutines                │      │
│  │  async def my_function():             │      │
│  │      await something()                │      │
│  └──────────────────────────────────────┘      │
└────────────────────────────────────────────────┘
```

### Running Async Code

```python
import asyncio

async def main():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

# Method 1: Top-level runner (Python 3.7+)
asyncio.run(main())

# Method 2: Manual event loop
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
```

### Creating and Awaiting Tasks

```python
async def fetch_data(source_id):
    await asyncio.sleep(1)  # Simulate I/O
    return f"Data from {source_id}"

async def main():
    # WRONG: Sequential (3 seconds total)
    data1 = await fetch_data(1)
    data2 = await fetch_data(2)
    data3 = await fetch_data(3)

    # RIGHT: Concurrent (1 second total)
    task1 = asyncio.create_task(fetch_data(1))
    task2 = asyncio.create_task(fetch_data(2))
    task3 = asyncio.create_task(fetch_data(3))

    results = await asyncio.gather(task1, task2, task3)
    # results = ["Data from 1", "Data from 2", "Data from 3"]
```

### asyncio.gather vs asyncio.wait

```python
# gather: Returns results in order
results = await asyncio.gather(coro1(), coro2(), coro3())
# If any raises exception, gather raises it immediately (default)

# gather with exception handling:
results = await asyncio.gather(
    coro1(), coro2(), coro3(),
    return_exceptions=True  # Returns exceptions in results list
)

# wait: More control over completion
done, pending = await asyncio.wait(
    [task1, task2, task3],
    return_when=asyncio.FIRST_COMPLETED  # or ALL_COMPLETED, FIRST_EXCEPTION
)

for task in done:
    result = task.result()  # Get result or raise exception
```

### Async Context Managers

```python
class AsyncDatabaseConnection:
    async def __aenter__(self):
        self.conn = await open_connection()
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        await self.conn.close()

# Usage
async def query_user(user_id):
    async with AsyncDatabaseConnection() as conn:
        return await conn.query(f"SELECT * FROM users WHERE id={user_id}")
    # Connection automatically closed, even on exception
```

### Async Iterators

```python
class AsyncRange:
    def __init__(self, n):
        self.n = n
        self.i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.i >= self.n:
            raise StopAsyncIteration
        await asyncio.sleep(0.1)  # Simulate async operation
        self.i += 1
        return self.i

# Usage
async def main():
    async for num in AsyncRange(5):
        print(num)  # Prints 1, 2, 3, 4, 5 with delays
```

---

## Node.js Event Loop (Comparison)

Node.js has a more complex event loop with phases.

### Event Loop Phases

```
┌───────────────────────────┐
│       timers               │  ← setTimeout/setInterval callbacks
├───────────────────────────┤
│       pending callbacks    │  ← I/O callbacks deferred from previous iteration
├───────────────────────────┤
│       idle, prepare        │  ← Internal use only
├───────────────────────────┤
│       poll                 │  ← Retrieve new I/O events; execute I/O callbacks
│                            │    (blocks here waiting for I/O if no timers)
├───────────────────────────┤
│       check                │  ← setImmediate() callbacks
├───────────────────────────┤
│       close callbacks      │  ← socket.on('close', ...)
└───────────────────────────┘
```

### Microtasks vs Macrotasks

```javascript
// Node.js / Browser JavaScript
console.log('1: Script start');

setTimeout(() => console.log('2: setTimeout'), 0);  // Macrotask

Promise.resolve()
    .then(() => console.log('3: Promise 1'))        // Microtask
    .then(() => console.log('4: Promise 2'));       // Microtask

console.log('5: Script end');

// Output:
// 1: Script start
// 5: Script end
// 3: Promise 1
// 4: Promise 2
// 2: setTimeout

// Rule: Microtasks (Promises) run before next macrotask
```

**Python doesn't have this distinction.** Tasks run in scheduling order.

### process.nextTick (Node.js-specific)

```javascript
// Runs BEFORE any I/O, even before Promises
process.nextTick(() => console.log('nextTick'));
Promise.resolve().then(() => console.log('Promise'));
// Output: nextTick, Promise
```

**Python equivalent:** `loop.call_soon()`

---

## When to Use What: Async vs Threads vs Multiprocessing

```
Decision Tree:

Your workload is:
    │
    ├─ CPU-bound (heavy computation)
    │   └─> Use multiprocessing
    │       - Bypasses GIL
    │       - True parallelism on multiple cores
    │       - Higher overhead (process creation, IPC)
    │
    ├─ I/O-bound (network, disk, database)
    │   ├─ Simple, few connections
    │   │   └─> Use threads
    │   │       - Simple mental model
    │   │       - Works with blocking libraries
    │   │
    │   └─ Many concurrent connections (>1000)
    │       └─> Use async/await
    │           - Low memory per connection
    │           - High scalability
    │           - Requires async libraries
    │
    └─ Mixed workload
        └─> Combine approaches
            - Async for I/O
            - Offload CPU work to thread pool or process pool
```

### Performance Characteristics

| Approach | Memory per Task | Context Switch | Max Concurrency | Use Case |
|----------|----------------|----------------|-----------------|----------|
| Threads | ~8MB (stack) | ~1-10μs | ~1,000 | Simple I/O, blocking libs |
| Async | ~4KB | ~0.1μs (function call) | 100,000+ | High concurrency I/O |
| Multiprocessing | ~50MB (process) | ~10-100μs | ~CPU cores | CPU-bound work |

### Code Examples

**CPU-Bound: Multiprocessing**
```python
from multiprocessing import Pool

def compute_heavy(n):
    return sum(i * i for i in range(n))

def main():
    with Pool(4) as pool:  # 4 processes
        results = pool.map(compute_heavy, [10**7, 10**7, 10**7, 10**7])
    # Uses 4 CPU cores in parallel
```

**I/O-Bound: Async**
```python
import asyncio
import aiohttp

async def fetch_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def main():
    urls = [f"http://example.com/{i}" for i in range(100)]
    tasks = [asyncio.create_task(fetch_url(url)) for url in urls]
    results = await asyncio.gather(*tasks)
    # Handles 100 concurrent requests efficiently
```

**Mixed: Async + Thread Pool for Blocking Code**
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

def blocking_operation(n):
    # Some library without async support
    return expensive_computation(n)

async def main():
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=4)

    # Run blocking code in thread pool
    result = await loop.run_in_executor(executor, blocking_operation, 1000)

    # Meanwhile, event loop remains responsive for async I/O
    await asyncio.sleep(0)  # Other async tasks can run
```

---

## Common Pitfalls

### 1. Blocking the Event Loop

```python
import asyncio
import time

async def bad():
    # BLOCKS the entire event loop for 5 seconds!
    time.sleep(5)  # All other coroutines frozen

async def good():
    # Yields control while sleeping
    await asyncio.sleep(5)  # Other coroutines can run
```

**Detection:**
```python
# Enable debug mode to detect blocking
asyncio.run(main(), debug=True)
# Warns if any callback takes > 100ms
```

### 2. Mixing Sync and Async Code

```python
# WRONG: Can't call async from sync
def sync_function():
    result = fetch_user(123)  # Error! fetch_user is async

# RIGHT: Use asyncio.run or loop.run_until_complete
def sync_function():
    result = asyncio.run(fetch_user(123))

# WRONG: Can't call sync blocking code from async
async def async_function():
    data = requests.get("http://example.com")  # Blocks loop!

# RIGHT: Use async library or run in executor
async def async_function():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://example.com") as resp:
            data = await resp.text()
```

### 3. Forgetting to await

```python
async def fetch_data():
    await asyncio.sleep(1)
    return "data"

async def bad():
    result = fetch_data()  # Forgot await!
    print(result)  # Prints: <coroutine object fetch_data at 0x...>
    # Coroutine NEVER RUNS!

async def good():
    result = await fetch_data()
    print(result)  # Prints: "data"
```

**Python 3.11+** warns about unawaited coroutines.

### 4. Not Handling Exceptions in Tasks

```python
async def failing_task():
    raise ValueError("Oops")

async def bad():
    task = asyncio.create_task(failing_task())
    # Task exception is SWALLOWED if we don't await it
    # No error visible!

async def good():
    task = asyncio.create_task(failing_task())
    try:
        await task
    except ValueError as e:
        print(f"Task failed: {e}")
```

### 5. Race Conditions in Async Code

```python
# Even single-threaded async has race conditions!
class Counter:
    def __init__(self):
        self.value = 0

    async def increment(self):
        temp = self.value
        await asyncio.sleep(0)  # ← Yields control!
        self.value = temp + 1   # ← Another task might have incremented

# Two concurrent increments might both read 0, write 1
# Result: Lost update

# Solution: Use asyncio.Lock
class SafeCounter:
    def __init__(self):
        self.value = 0
        self.lock = asyncio.Lock()

    async def increment(self):
        async with self.lock:
            temp = self.value
            await asyncio.sleep(0)
            self.value = temp + 1
```

---

## Async Best Practices

**Use async libraries for everything:**
- HTTP: aiohttp, httpx
- Database: asyncpg (Postgres), motor (MongoDB), aiosqlite
- Redis: aioredis
- Files: aiofiles

**Limit concurrent operations:**
```python
# Bad: Launch 10,000 concurrent requests (overload)
tasks = [fetch_url(url) for url in urls]
await asyncio.gather(*tasks)

# Good: Use semaphore to limit to 100 concurrent
semaphore = asyncio.Semaphore(100)

async def fetch_limited(url):
    async with semaphore:
        return await fetch_url(url)

tasks = [fetch_limited(url) for url in urls]
await asyncio.gather(*tasks)
```

**Set timeouts:**
```python
try:
    result = await asyncio.wait_for(fetch_data(), timeout=5.0)
except asyncio.TimeoutError:
    print("Operation timed out")
```

**Cancel tasks on shutdown:**
```python
async def main():
    tasks = [asyncio.create_task(worker(i)) for i in range(10)]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        # Cancel all tasks
        for task in tasks:
            task.cancel()

        # Wait for cancellation to complete
        await asyncio.gather(*tasks, return_exceptions=True)
```

---

## Comparison Table

| Aspect | Threads | Async/Await | Multiprocessing |
|--------|---------|-------------|-----------------|
| Concurrency Model | Preemptive | Cooperative | Parallel |
| Parallelism | No (GIL) | No | Yes |
| Memory Overhead | High (8MB/thread) | Low (4KB/coroutine) | Very High (50MB/process) |
| Context Switch | OS scheduler (~μs) | Function call (~ns) | OS scheduler (~μs) |
| Race Conditions | Yes (preemption) | Yes (await points) | No (separate memory) |
| Use Case | Simple I/O | High-concurrency I/O | CPU-bound |
| Debugging | Harder (preemption) | Easier (explicit yields) | Harder (IPC) |
| Library Support | All sync libraries | Only async libraries | All libraries |

---

## Key Concepts Checklist

- [ ] Explain blocking vs non-blocking I/O at OS level
- [ ] Describe how an event loop works internally
- [ ] Understand coroutines and how await pauses execution
- [ ] Choose between async, threads, and multiprocessing for a scenario
- [ ] Identify code that blocks the event loop
- [ ] Use asyncio.gather and asyncio.wait correctly
- [ ] Handle exceptions in async tasks
- [ ] Recognize that async code can still have race conditions

---

## Practical Insights

**Event loop overhead is minimal.** A well-tuned async server can handle 100,000+ concurrent connections on a single core. The bottleneck is usually your backend services (database, cache), not the event loop itself.

**Async is contagious.** Once you have one async function, everything that calls it must also be async. This is by design—you can't accidentally block the event loop. But it means partial async adoption is painful. Go all-in or don't go at all.

**The GIL makes Python async less powerful than Node.js for pure compute.** JavaScript engines can optimize tight async loops better. But Python's async shines for I/O-bound work, where the bottleneck is network/disk, not CPU.

**FastAPI's async is optional but recommended.** You can write `def` (sync) endpoints that run in a thread pool, or `async def` endpoints that run on the event loop. Use `async def` for I/O-bound endpoints (database queries, API calls). Use `def` only for CPU-bound work or when calling blocking libraries.

**Monitor event loop lag.** If the event loop spends >10ms between iterations, you're blocking somewhere. Use `asyncio.run(main(), debug=True)` in development to detect slow callbacks. In production, instrument with Prometheus metrics like `event_loop_lag_seconds`.

**Async doesn't magically make code faster.** It makes code handle more concurrency with less memory. If you have 10 requests and each takes 100ms, async completes in 100ms (all concurrent). Threads also complete in ~100ms (OS parallelizes I/O waits). The difference appears at scale: 10,000 requests where async uses 40MB and threads need 80GB.
