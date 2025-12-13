# Chapter 39: I/O Models & Multiplexing

## The Core Problem

Your web server needs to handle 10,000 concurrent connections. Each connection might be:
- Waiting for a database query
- Waiting for a network response
- Waiting for disk I/O
- Actually sending data

With traditional blocking I/O:

```
1 thread per connection
10,000 connections = 10,000 threads

Each thread:
- 1-8 MB stack memory
- Context switch overhead
- Scheduler contention

Total: 10-80 GB RAM just for stacks
Result: System grinds to a halt
```

This is the **C10K problem**: How do you handle 10,000 concurrent connections without creating 10,000 threads?

The answer lies in understanding I/O models and multiplexing. Master these, and you can build servers like nginx (50,000+ connections per worker) and Redis (100,000+ ops/sec single-threaded). Ignore them, and your server will collapse under moderate load.

---

## Blocking vs Non-Blocking I/O

The fundamental distinction in I/O operations.

### Blocking I/O

```
Application              Kernel              Network
    │                       │                   │
    │──── read(socket) ────►│                   │
    │                       │                   │
    │   BLOCKED             │──── wait ────────►│
    │   (thread sleeps)     │                   │
    │                       │◄──── data ────────│
    │                       │                   │
    │◄──── return data ─────│                   │
    │                       │                   │
    │   (thread resumes)    │                   │
```

**Characteristics:**
- Thread blocks until data is ready
- Simple programming model
- One thread per connection needed
- High memory overhead at scale

### Non-Blocking I/O

```
Application              Kernel              Network
    │                       │                   │
    │──── read(socket) ────►│                   │
    │◄──── EWOULDBLOCK ─────│ (no data ready)   │
    │                       │                   │
    │   (continue work)     │                   │
    │                       │                   │
    │──── read(socket) ────►│                   │
    │◄──── EWOULDBLOCK ─────│ (still no data)   │
    │                       │                   │
    │   (continue work)     │                   │
    │                       │                   │
    │──── read(socket) ────►│                   │
    │◄──── return data ─────│ (data ready!)     │
```

**Characteristics:**
- Returns immediately (even if no data)
- Requires polling or notification
- Enables single thread handling multiple connections
- More complex but scalable

### Python Example: Blocking vs Non-Blocking

```python
import socket

# BLOCKING (default)
sock = socket.socket()
sock.connect(('example.com', 80))
sock.send(b'GET / HTTP/1.1\r\nHost: example.com\r\n\r\n')
data = sock.recv(4096)  # Blocks until data arrives
print(data)

# NON-BLOCKING
sock = socket.socket()
sock.setblocking(False)  # Make socket non-blocking
sock.connect(('example.com', 80))  # Raises BlockingIOError

# Must handle EWOULDBLOCK/EAGAIN
import errno

while True:
    try:
        data = sock.recv(4096)
        if data:
            print(data)
            break
    except BlockingIOError as e:
        if e.errno == errno.EWOULDBLOCK:
            # No data ready, do other work
            continue
        raise
```

---

## Synchronous vs Asynchronous I/O

A different dimension from blocking/non-blocking.

### Synchronous I/O

Application waits for I/O operation to complete (may block or poll).

```
┌────────────────────────────────────────────────┐
│            SYNCHRONOUS I/O                      │
├────────────────────────────────────────────────┤
│  App → Kernel: "Read data"                     │
│  App waits (blocking) or polls (non-blocking)  │
│  Kernel → App: "Here's your data"              │
│  App: Process data                             │
└────────────────────────────────────────────────┘
```

### Asynchronous I/O

Application initiates I/O and continues. Kernel notifies when complete.

```
┌────────────────────────────────────────────────┐
│           ASYNCHRONOUS I/O                      │
├────────────────────────────────────────────────┤
│  App → Kernel: "Read data, notify when done"   │
│  App: Continue other work                      │
│  Kernel: Reads data in background              │
│  Kernel → App: "I/O complete! Here's data"     │
│  App: Process data                             │
└────────────────────────────────────────────────┘
```

### The 2x2 Matrix

| | Blocking | Non-Blocking |
|---|---|---|
| **Synchronous** | Traditional blocking I/O | I/O multiplexing (select/epoll) |
| **Asynchronous** | Doesn't exist | True async I/O (io_uring, IOCP) |

---

## The C10K Problem

Coined by Dan Kegel in 1999: How to handle 10,000 concurrent connections?

### Why 10,000 Was Hard

```
Traditional approach (1 thread per connection):

10,000 connections × 2 MB stack = 20 GB RAM
10,000 threads → scheduler overwhelmed
Context switches: O(n²) with lock contention

System becomes unusable
```

### The Solution: I/O Multiplexing

```
New approach (1 thread handles all connections):

1 thread monitors 10,000 sockets
OS notifies which sockets are ready
Process only ready sockets
No blocking, minimal context switches

Result: Scales to 100K+ connections
```

**Modern successors:**
- C10M problem (10 million connections)
- Requires kernel bypass, user-space networking
- Achieved by specialized systems

---

## I/O Multiplexing: select()

The original I/O multiplexing system call (1983, BSD).

### How select() Works

```
┌────────────────────────────────────────────────┐
│               SELECT OVERVIEW                   │
├────────────────────────────────────────────────┤
│  App: Here are N file descriptors              │
│       Tell me which are ready to read/write    │
│                                                 │
│  Kernel: Checks all N descriptors              │
│          Returns bitmask of ready ones         │
│                                                 │
│  App: Iterate through bitmask, process ready   │
└────────────────────────────────────────────────┘
```

### select() System Call

```c
int select(
    int nfds,              // Highest fd number + 1
    fd_set *readfds,       // Set of fds to check for reading
    fd_set *writefds,      // Set of fds to check for writing
    fd_set *exceptfds,     // Set of fds to check for exceptions
    struct timeval *timeout // Timeout (NULL = block forever)
);
```

### Python Example with select()

```python
import select
import socket

# Create listening socket
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', 8000))
server.listen(100)
server.setblocking(False)

# Track all sockets we're monitoring
sockets_list = [server]

while True:
    # select() blocks until at least one socket is ready
    readable, writable, exceptional = select.select(
        sockets_list,  # Check these for reading
        [],            # Not checking for writing
        sockets_list,  # Check for exceptions
        1.0            # 1 second timeout
    )

    for sock in readable:
        if sock is server:
            # New connection
            client, addr = sock.accept()
            client.setblocking(False)
            sockets_list.append(client)
            print(f"New connection from {addr}")
        else:
            # Data from existing connection
            try:
                data = sock.recv(4096)
                if data:
                    sock.send(b"HTTP/1.1 200 OK\r\n\r\nHello!\r\n")
                else:
                    # Connection closed
                    sockets_list.remove(sock)
                    sock.close()
            except Exception as e:
                sockets_list.remove(sock)
                sock.close()
```

### select() Limitations

| Limitation | Impact |
|------------|--------|
| FD_SETSIZE limit | Usually 1024 max file descriptors |
| O(n) scan | Kernel checks all fds, even if only 1 ready |
| Copy overhead | fd_set copied to/from kernel each call |
| No fd reuse info | Can't tell which fds changed since last call |

**Result:** Works for small number of connections, doesn't scale to C10K.

---

## I/O Multiplexing: poll()

Improvement over select() (1986, System V).

### poll() System Call

```c
int poll(
    struct pollfd *fds,    // Array of file descriptors to monitor
    nfds_t nfds,           // Number of items in fds array
    int timeout            // Timeout in milliseconds
);

struct pollfd {
    int fd;                // File descriptor
    short events;          // Events to monitor (POLLIN, POLLOUT, etc.)
    short revents;         // Events that occurred
};
```

### Improvements Over select()

```
┌────────────────────────────────────────────────┐
│           poll() vs select()                    │
├────────────────────────────────────────────────┤
│  ✓ No FD_SETSIZE limit                         │
│  ✓ Cleaner API (array vs bitmasks)             │
│  ✓ Separate input/output (events/revents)      │
│                                                 │
│  ✗ Still O(n) scan                             │
│  ✗ Still copies all fds to kernel              │
└────────────────────────────────────────────────┘
```

### Python Example with poll()

```python
import select
import socket

server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', 8000))
server.listen(100)
server.setblocking(False)

# Create poller
poller = select.poll()

# Register server socket for read events
poller.register(server.fileno(), select.POLLIN)

# Map fd -> socket object
fd_to_socket = {server.fileno(): server}

while True:
    # poll() returns list of (fd, event) tuples
    events = poller.poll(1000)  # 1 second timeout

    for fd, event in events:
        sock = fd_to_socket[fd]

        if sock is server:
            # New connection
            client, addr = sock.accept()
            client.setblocking(False)

            client_fd = client.fileno()
            poller.register(client_fd, select.POLLIN)
            fd_to_socket[client_fd] = client

        elif event & select.POLLIN:
            # Data available
            try:
                data = sock.recv(4096)
                if data:
                    sock.send(b"HTTP/1.1 200 OK\r\n\r\nHello!\r\n")
                else:
                    # Connection closed
                    poller.unregister(fd)
                    del fd_to_socket[fd]
                    sock.close()
            except Exception:
                poller.unregister(fd)
                del fd_to_socket[fd]
                sock.close()
```

**Still O(n), but better than select() for large fd numbers.**

---

## I/O Multiplexing: epoll (Linux)

The modern solution for Linux (2002, kernel 2.5.44).

### How epoll Works

```
┌────────────────────────────────────────────────┐
│              EPOLL ARCHITECTURE                 │
├────────────────────────────────────────────────┤
│                                                 │
│  1. epoll_create() → returns epoll fd          │
│                                                 │
│  2. epoll_ctl(ADD, fd) → register interest     │
│     Kernel maintains interest list internally  │
│                                                 │
│  3. epoll_wait() → returns ONLY ready fds      │
│     O(1) lookup, no scanning!                  │
│                                                 │
└────────────────────────────────────────────────┘
```

### epoll API

```c
// Create epoll instance
int epoll_create1(int flags);

// Add/modify/remove file descriptors
int epoll_ctl(
    int epfd,              // epoll file descriptor
    int op,                // EPOLL_CTL_ADD, MOD, DEL
    int fd,                // File descriptor to monitor
    struct epoll_event *event
);

// Wait for events
int epoll_wait(
    int epfd,
    struct epoll_event *events,  // Output buffer
    int maxevents,
    int timeout
);
```

### Edge-Triggered vs Level-Triggered

**Level-Triggered (default):**
```
Socket has 1000 bytes available
epoll_wait() → returns fd (data available)
App reads 500 bytes
epoll_wait() → returns fd AGAIN (still 500 bytes left)
App reads 500 bytes
epoll_wait() → blocks (no data)
```

**Edge-Triggered (EPOLLET):**
```
Socket has 1000 bytes available
epoll_wait() → returns fd (state changed: 0 → 1000 bytes)
App reads 500 bytes
epoll_wait() → blocks (no state change!)
Must read until EWOULDBLOCK to avoid missing data
```

### Python Example with epoll

```python
import select
import socket

server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', 8000))
server.listen(100)
server.setblocking(False)

# Create epoll object
epoll = select.epoll()

# Register server for read events
epoll.register(server.fileno(), select.EPOLLIN)

fd_to_socket = {server.fileno(): server}

try:
    while True:
        # epoll.poll() returns only ready file descriptors
        events = epoll.poll(1)  # 1 second timeout

        for fd, event in events:
            sock = fd_to_socket[fd]

            if sock is server:
                # Accept new connection
                client, addr = sock.accept()
                client.setblocking(False)

                client_fd = client.fileno()
                epoll.register(client_fd, select.EPOLLIN | select.EPOLLET)
                fd_to_socket[client_fd] = client

            elif event & select.EPOLLIN:
                # Read data (edge-triggered: read until EWOULDBLOCK)
                try:
                    while True:
                        data = sock.recv(4096)
                        if not data:
                            # Connection closed
                            epoll.unregister(fd)
                            del fd_to_socket[fd]
                            sock.close()
                            break
                        sock.send(b"HTTP/1.1 200 OK\r\n\r\nHello!\r\n")
                except BlockingIOError:
                    # No more data (expected with edge-triggered)
                    pass
                except Exception:
                    epoll.unregister(fd)
                    del fd_to_socket[fd]
                    sock.close()

            elif event & select.EPOLLHUP:
                # Connection hung up
                epoll.unregister(fd)
                del fd_to_socket[fd]
                sock.close()

finally:
    epoll.unregister(server.fileno())
    epoll.close()
    server.close()
```

### epoll Advantages

| Feature | select/poll | epoll |
|---------|-------------|-------|
| Max fds | 1024 (select) or unlimited (poll) | Millions |
| Performance | O(n) scan | O(1) ready list |
| Kernel copy | Every call | Once (at registration) |
| Edge-triggered | No | Yes (optional) |
| Use case | Small servers | High-scale servers |

**This is why nginx can handle 50,000+ connections per worker.**

---

## I/O Multiplexing: kqueue (BSD/macOS)

BSD's answer to epoll (2000, FreeBSD 4.1).

### kqueue Architecture

```
┌────────────────────────────────────────────────┐
│               KQUEUE OVERVIEW                   │
├────────────────────────────────────────────────┤
│  kqueue() → create kernel event queue          │
│                                                 │
│  kevent() → register events AND wait            │
│             (single syscall!)                   │
│                                                 │
│  Supports:                                      │
│  - Socket I/O (like epoll)                     │
│  - File changes (like inotify)                 │
│  - Signals                                      │
│  - Timers                                       │
│  - Process events                               │
└────────────────────────────────────────────────┘
```

### kqueue vs epoll

| Aspect | epoll | kqueue |
|--------|-------|--------|
| Platform | Linux | BSD, macOS |
| Event types | I/O only | I/O, files, signals, timers, processes |
| API | 3 functions | 2 functions |
| Registration | epoll_ctl() | kevent() |
| Waiting | epoll_wait() | kevent() (same function!) |

**kqueue is more general-purpose but less common in production (Linux dominance).**

---

## Python's selectors Module

Cross-platform abstraction over select/poll/epoll/kqueue.

```python
import selectors
import socket

# selectors automatically chooses best available:
# - epoll on Linux
# - kqueue on BSD/macOS
# - poll or select elsewhere

selector = selectors.DefaultSelector()

server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', 8000))
server.listen(100)
server.setblocking(False)

def accept_connection(sock):
    client, addr = sock.accept()
    client.setblocking(False)
    selector.register(client, selectors.EVENT_READ, data=handle_client)
    print(f"Connected: {addr}")

def handle_client(sock):
    try:
        data = sock.recv(4096)
        if data:
            sock.send(b"HTTP/1.1 200 OK\r\n\r\nHello!\r\n")
        else:
            selector.unregister(sock)
            sock.close()
    except Exception:
        selector.unregister(sock)
        sock.close()

# Register server socket
selector.register(server, selectors.EVENT_READ, data=accept_connection)

while True:
    # Wait for events
    events = selector.select(timeout=1)

    for key, mask in events:
        callback = key.data  # Function to call
        callback(key.fileobj)  # key.fileobj is the socket
```

**Portable, high-level, recommended for production Python.**

---

## The Reactor Pattern

Design pattern for I/O multiplexing event loops.

```
┌────────────────────────────────────────────────┐
│              REACTOR PATTERN                    │
├────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────────────────────────┐          │
│  │      Event Loop (Reactor)        │          │
│  │                                  │          │
│  │  while True:                     │          │
│  │      events = wait_for_events()  │          │
│  │      for event in events:        │          │
│  │          dispatch(event)          │          │
│  └──────────────────────────────────┘          │
│                   │                             │
│                   ▼                             │
│         ┌─────────────────┐                     │
│         │   Handlers      │                     │
│         ├─────────────────┤                     │
│         │ on_read()       │                     │
│         │ on_write()      │                     │
│         │ on_connect()    │                     │
│         │ on_close()      │                     │
│         └─────────────────┘                     │
│                                                 │
└────────────────────────────────────────────────┘
```

### Reactor Implementation

```python
import selectors
import socket

class Reactor:
    def __init__(self):
        self.selector = selectors.DefaultSelector()
        self.running = False

    def register(self, sock, events, handler):
        """Register a socket with a handler callback."""
        self.selector.register(sock, events, data=handler)

    def unregister(self, sock):
        """Unregister a socket."""
        self.selector.unregister(sock)

    def run(self):
        """Main event loop."""
        self.running = True
        while self.running:
            events = self.selector.select(timeout=1)
            for key, mask in events:
                handler = key.data
                handler(key.fileobj, mask)

    def stop(self):
        """Stop the event loop."""
        self.running = False

# Usage
reactor = Reactor()

def on_accept(sock, mask):
    client, addr = sock.accept()
    client.setblocking(False)
    reactor.register(client, selectors.EVENT_READ, on_read)

def on_read(sock, mask):
    try:
        data = sock.recv(4096)
        if data:
            sock.send(b"HTTP/1.1 200 OK\r\n\r\nHello!\r\n")
        else:
            reactor.unregister(sock)
            sock.close()
    except Exception:
        reactor.unregister(sock)
        sock.close()

server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', 8000))
server.listen(100)
server.setblocking(False)

reactor.register(server, selectors.EVENT_READ, on_accept)
reactor.run()
```

**Used by:** nginx, Redis, Node.js, Twisted, asyncio

---

## The Proactor Pattern

Asynchronous I/O pattern where kernel performs the operation.

```
┌────────────────────────────────────────────────┐
│            REACTOR vs PROACTOR                  │
├────────────────────────────────────────────────┤
│                                                 │
│  REACTOR:                                       │
│  Kernel → App: "Socket is ready to read"       │
│  App → Kernel: read()                          │
│  App: Process data                             │
│                                                 │
│  PROACTOR:                                      │
│  App → Kernel: "Read and notify when done"     │
│  App: Do other work                            │
│  Kernel → App: "Read complete! Here's data"    │
│  App: Process data                             │
│                                                 │
└────────────────────────────────────────────────┘
```

### Proactor with io_uring (Linux)

```python
# Conceptual example (requires liburing bindings)
import io_uring

ring = io_uring.IoUring(queue_depth=256)

# Submit read operation
buffer = bytearray(4096)
ring.submit_read(
    fd=socket_fd,
    buffer=buffer,
    offset=0,
    user_data=socket_id  # Identify this operation
)

# Kernel performs read asynchronously
# App continues other work

# Later: check for completions
for completion in ring.get_completions():
    socket_id = completion.user_data
    bytes_read = completion.result
    # Data is already in buffer!
    process_data(buffer[:bytes_read])
```

**Used by modern high-performance systems (io_uring is gaining adoption).**

---

## Zero-Copy I/O

Avoid copying data between kernel and user space.

### Traditional I/O (4 copies)

```
┌────────────────────────────────────────────────┐
│          TRADITIONAL FILE SEND                  │
├────────────────────────────────────────────────┤
│                                                 │
│  1. Disk → Kernel buffer (DMA)                 │
│  2. Kernel buffer → User buffer (copy)         │
│  3. User buffer → Socket buffer (copy)         │
│  4. Socket buffer → NIC (DMA)                  │
│                                                 │
│  Total: 2 DMA + 2 CPU copies                   │
│         Data crosses user/kernel boundary 2x   │
└────────────────────────────────────────────────┘
```

### sendfile() (2 copies)

```
┌────────────────────────────────────────────────┐
│            SENDFILE (ZERO-COPY)                 │
├────────────────────────────────────────────────┤
│                                                 │
│  1. Disk → Kernel buffer (DMA)                 │
│  2. Kernel buffer → NIC (DMA, no copy!)        │
│                                                 │
│  Total: 2 DMA, 0 CPU copies                    │
│         No user-space involvement               │
└────────────────────────────────────────────────┘
```

### Python sendfile Example

```python
import os
import socket

# Open file
fd = os.open('large_file.bin', os.O_RDONLY)
file_size = os.fstat(fd).st_size

# Create socket connection
sock = socket.socket()
sock.connect(('example.com', 80))

# Send file with zero-copy
offset = 0
while offset < file_size:
    sent = os.sendfile(
        sock.fileno(),  # Destination socket
        fd,             # Source file
        offset,         # Offset in file
        file_size - offset  # Bytes to send
    )
    offset += sent

os.close(fd)
sock.close()
```

### splice() (Linux, more general)

```python
# Move data between two file descriptors without user-space copy
os.splice(
    fd_in,      # Source fd
    off_in,     # Source offset
    fd_out,     # Destination fd
    off_out,    # Destination offset
    len,        # Bytes to transfer
    flags       # Flags
)
```

**Used by:** nginx, Apache, static file servers

---

## io_uring: Modern Async I/O (Linux 5.1+)

The future of Linux I/O.

### Why io_uring?

```
┌────────────────────────────────────────────────┐
│          PROBLEMS WITH OLD APIS                 │
├────────────────────────────────────────────────┤
│                                                 │
│  epoll:                                         │
│  - Only tells you fd is ready                  │
│  - You still need syscall to read/write        │
│  - One syscall per readiness check             │
│                                                 │
│  AIO (async I/O):                              │
│  - Complex, limited to O_DIRECT                │
│  - Poor performance, rarely used               │
│                                                 │
│  io_uring:                                      │
│  ✓ True async I/O                              │
│  ✓ Batch operations (multiple I/O in 1 syscall)│
│  ✓ Zero-copy                                    │
│  ✓ Poll-based (can avoid syscalls entirely!)  │
└────────────────────────────────────────────────┘
```

### io_uring Architecture

```
┌────────────────────────────────────────────────┐
│              IO_URING DESIGN                    │
├────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────────┐    ┌─────────────────┐   │
│  │ Submission Queue│    │ Completion Queue│   │
│  │      (SQ)       │    │      (CQ)       │   │
│  └─────────────────┘    └─────────────────┘   │
│         │                        ▲             │
│         │                        │             │
│         ▼                        │             │
│    User Space          Kernel Space            │
│  ═══════════════════════════════════════════   │
│         │                        │             │
│         ▼                        │             │
│    [Operations]            [Results]           │
│                                                 │
│  App writes to SQ (shared memory, no syscall)  │
│  Kernel processes operations                   │
│  Kernel writes to CQ (shared memory)           │
│  App reads from CQ (no syscall needed!)        │
│                                                 │
└────────────────────────────────────────────────┘
```

### io_uring Conceptual Example

```python
# Conceptual (requires io_uring Python bindings)

ring = IoUring(entries=256)

# Submit multiple operations in one go
with ring.submission_queue() as sq:
    # Queue a read
    sq.add_read(fd=file_fd, buffer=buf1, size=4096, offset=0, user_data=1)

    # Queue a write
    sq.add_write(fd=socket_fd, buffer=buf2, size=len(buf2), user_data=2)

    # Queue an accept
    sq.add_accept(fd=server_fd, addr=addr_buf, user_data=3)

# Submit all at once (single syscall!)
ring.submit()

# Process completions
for completion in ring.get_completions():
    user_data = completion.user_data
    result = completion.result

    if user_data == 1:
        # Read completed
        handle_read(buf1[:result])
    elif user_data == 2:
        # Write completed
        handle_write_complete(result)
    elif user_data == 3:
        # Accept completed
        handle_new_connection(result)
```

### io_uring Advantages

| Feature | epoll | io_uring |
|---------|-------|----------|
| Syscalls | 1 per ready check + 1 per I/O | Batch many ops in 1 syscall |
| Readiness vs Completion | Readiness notification | I/O completion notification |
| Zero-copy | No | Yes |
| Batching | No | Yes |
| I/O types | Network only | All: files, network, etc. |

**Adoption:** New, but growing. Used by RocksDB, ScyllaDB, QEMU.

---

## How nginx Achieves High Concurrency

Putting it all together.

```
┌────────────────────────────────────────────────┐
│          NGINX ARCHITECTURE                     │
├────────────────────────────────────────────────┤
│                                                 │
│  Master Process                                 │
│    │                                            │
│    ├─► Worker 1 (event loop, 50K connections)  │
│    ├─► Worker 2 (event loop, 50K connections)  │
│    ├─► Worker 3 (event loop, 50K connections)  │
│    └─► Worker 4 (event loop, 50K connections)  │
│                                                 │
│  Each worker:                                   │
│  - Non-blocking I/O                            │
│  - epoll (Linux) or kqueue (BSD)               │
│  - Single-threaded event loop                  │
│  - No context switches                          │
│  - Handles 10K-100K concurrent connections     │
│                                                 │
│  Key techniques:                                │
│  1. Edge-triggered epoll                       │
│  2. Non-blocking sockets                       │
│  3. sendfile() for static files                │
│  4. Connection pooling to upstreams            │
│  5. Asynchronous disk I/O (thread pool)        │
│                                                 │
└────────────────────────────────────────────────┘
```

### Configuration Example

```nginx
# nginx.conf
worker_processes 4;  # One per CPU core

events {
    worker_connections 50000;  # Max connections per worker
    use epoll;                 # Use epoll on Linux
    multi_accept on;           # Accept multiple connections at once
}

http {
    sendfile on;               # Use sendfile() for static files
    tcp_nopush on;             # Optimize packet sending
    tcp_nodelay on;            # Disable Nagle's algorithm

    keepalive_timeout 65;      # Keep connections alive
    keepalive_requests 100;    # Requests per connection

    # Thread pool for blocking operations
    aio threads;
}
```

**Result:** nginx can handle 200K+ concurrent connections on commodity hardware.

---

## How Redis Achieves High Throughput

Single-threaded, yet 100K+ ops/sec.

```
┌────────────────────────────────────────────────┐
│            REDIS I/O MODEL                      │
├────────────────────────────────────────────────┤
│                                                 │
│  Single event loop thread:                      │
│                                                 │
│  while True:                                    │
│      events = epoll.wait()                     │
│      for event in events:                      │
│          if event == NEW_CONNECTION:            │
│              accept()                           │
│          elif event == CLIENT_DATA:             │
│              read_command()                     │
│              execute_command()  # In-memory!    │
│              write_response()                   │
│                                                 │
│  Why single-threaded?                           │
│  - All data in RAM (no blocking I/O)           │
│  - No locks needed                              │
│  - No context switches                          │
│  - CPU cache friendly                           │
│                                                 │
│  Performance:                                   │
│  - 100K ops/sec single instance                │
│  - Bounded by network, not CPU                 │
│                                                 │
└────────────────────────────────────────────────┘
```

### Redis 6.0: I/O Threads

```
┌────────────────────────────────────────────────┐
│        REDIS 6.0+ I/O THREADING                 │
├────────────────────────────────────────────────┤
│                                                 │
│  Main thread (event loop):                      │
│    - epoll/kqueue                               │
│    - Command execution (single-threaded)       │
│    - State management                           │
│                                                 │
│  I/O threads (read/write):                      │
│    - Thread 1: read() from clients 1-1000      │
│    - Thread 2: read() from clients 1001-2000   │
│    - Thread 3: write() to clients 1-1000       │
│    - Thread 4: write() to clients 1001-2000    │
│                                                 │
│  Parallelizes I/O, keeps command exec serial   │
│  No locks on data structures!                  │
│                                                 │
└────────────────────────────────────────────────┘
```

---

## Comparison Table

| Mechanism | Platform | Performance | Complexity | Use Case |
|-----------|----------|-------------|------------|----------|
| select() | All | O(n), max 1024 fds | Low | Legacy, small servers |
| poll() | All | O(n), no fd limit | Low | Portable, medium scale |
| epoll | Linux | O(1), millions of fds | Medium | High-scale Linux servers |
| kqueue | BSD/macOS | O(1), millions of fds | Medium | BSD/macOS servers |
| io_uring | Linux 5.1+ | O(1), true async | High | Cutting-edge performance |
| selectors | All | Best available | Low | Python production apps |

---

## Key Concepts Checklist

- [ ] Explain blocking vs non-blocking I/O
- [ ] Describe the C10K problem and its solution
- [ ] Compare select, poll, and epoll performance characteristics
- [ ] Implement basic event loop with selectors
- [ ] Explain edge-triggered vs level-triggered modes
- [ ] Describe Reactor vs Proactor patterns
- [ ] Explain zero-copy I/O (sendfile)
- [ ] Understand how nginx and Redis achieve high concurrency

---

## Practical Insights

**Use the selectors module in Python.** It automatically picks the best mechanism (epoll on Linux, kqueue on BSD/macOS, poll elsewhere). Don't hand-code select/poll/epoll unless you have specific requirements.

**Edge-triggered epoll is faster but harder to get right.** You must read until EWOULDBLOCK or you'll miss data. Level-triggered is safer for most applications. Profile before optimizing to edge-triggered.

**Single-threaded event loops scale surprisingly well.** nginx and Redis prove that one thread can handle tens of thousands of connections if you avoid blocking. Add threads only for CPU-bound work (hashing, compression) or truly blocking I/O (disk on older kernels).

**io_uring is the future on Linux.** It's more complex than epoll but offers true async I/O with batching and zero-copy. Watch for library support (liburing, Rust's tokio-uring). Not production-ready for all workloads yet, but getting there.

**sendfile() is a huge win for static file servers.** If you're serving files (images, videos, downloads), use sendfile() or similar zero-copy mechanisms. The difference between 2 copies and 0 copies is dramatic at scale.

**Don't mix blocking and non-blocking in the same event loop.** A single blocking call (synchronous database query, file read without AIO) will stall the entire loop, defeating the purpose. Use thread pools or async drivers for I/O operations.
