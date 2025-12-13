# Chapter 42: Socket Programming

## The Core Problem

Your API server handles thousands of concurrent connections. At peak traffic, you notice:

```
Normal: 2,000 requests/second, 50ms latency
Peak: 5,000 requests/second, 2,000ms latency
Monitoring shows: TIME_WAIT states piling up, connection refused errors

One developer added a feature that makes 10 sequential HTTP calls
Result: Connection pool exhausted → New requests fail → 503 errors cascade
```

Without understanding sockets, you can't debug why connections fail, why latency spikes, or why your server runs out of file descriptors. Socket programming is the foundation of all network communication—HTTP, databases, message queues, and RPC all build on sockets.

---

## What is a Socket?

**The Problem:**
Applications need a standard way to communicate over networks. How does your Python code send bytes to a server in another datacenter?

**How It Works:**
A socket is an endpoint for network communication. Think of it as a file descriptor that represents a network connection instead of a file. The operating system provides a socket API that abstracts network hardware.

```
┌─────────────────────────────────────────────────┐
│              Application Layer                   │
│         (HTTP, gRPC, Database Protocol)         │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│              Socket API Layer                    │
│   (send, recv, connect, bind, listen, accept)   │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│            Transport Layer                       │
│              (TCP or UDP)                        │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│              Network Layer                       │
│                  (IP)                            │
└─────────────────────────────────────────────────┘
```

**Socket Types:**

| Type | Protocol | Reliability | Order | Use Case |
|------|----------|-------------|-------|----------|
| SOCK_STREAM | TCP | Guaranteed delivery | Ordered | HTTP, databases, most APIs |
| SOCK_DGRAM | UDP | No guarantee | Unordered | DNS, video streaming, gaming |
| SOCK_RAW | IP | Direct IP access | - | Network tools, packet sniffing |

**When to use:**
- TCP sockets: When you need reliable, ordered delivery (most backend services)
- UDP sockets: When you can tolerate packet loss for lower latency (real-time systems)

**When NOT to use:**
- Don't use raw sockets unless building network tools—SOCK_STREAM or SOCK_DGRAM handle 99% of cases
- Don't implement your own reliability layer over UDP unless you have very specific latency requirements

---

## TCP Socket Lifecycle

**The Problem:**
A TCP connection isn't instant. Both client and server must coordinate setup, data transfer, and teardown. Missing any step causes connection failures.

**How It Works:**

```
Client                                    Server
  │                                         │
  │  socket()                               │  socket()
  │  ─────────                              │  ─────────
  │                                         │
  │                                         │  bind()
  │                                         │  ──────
  │                                         │
  │                                         │  listen()
  │                                         │  ────────
  │                                         │
  │  connect()                              │  accept() [blocks]
  │  ─────────────────────────────────────► │  ────────
  │         SYN                              │
  │ ◄─────────────────────────────────────  │
  │         SYN-ACK                          │
  │  ─────────────────────────────────────► │
  │         ACK                              │  accept() returns
  │                                         │
  │  send() / recv()                        │  recv() / send()
  │ ◄─────────────────────────────────────► │
  │         Data Exchange                    │
  │                                         │
  │  close()                                │  close()
  │  ─────────────────────────────────────► │
  │         FIN                              │
  │ ◄─────────────────────────────────────  │
  │         ACK                              │
```

**Server Code Example:**

```python
import socket

# Step 1: Create a socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Step 2: Bind to an address and port
server_socket.bind(('0.0.0.0', 8080))

# Step 3: Listen for incoming connections (backlog of 128)
server_socket.listen(128)
print("Server listening on port 8080...")

# Step 4: Accept connections in a loop
while True:
    # Accept blocks until a client connects
    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

    # Step 5: Receive data
    data = client_socket.recv(1024)  # Read up to 1024 bytes
    print(f"Received: {data.decode()}")

    # Step 6: Send response
    response = "HTTP/1.1 200 OK\r\n\r\nHello, World!"
    client_socket.send(response.encode())

    # Step 7: Close the connection
    client_socket.close()
```

**Client Code Example:**

```python
import socket

# Step 1: Create a socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Step 2: Connect to the server
client_socket.connect(('localhost', 8080))

# Step 3: Send data
request = "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
client_socket.send(request.encode())

# Step 4: Receive response
response = client_socket.recv(4096)
print(f"Response: {response.decode()}")

# Step 5: Close the connection
client_socket.close()
```

**When to use:** Every TCP-based network application follows this lifecycle.

**When NOT to use:** If you need connectionless communication (UDP), the lifecycle is simpler: socket() → sendto() / recvfrom() → close().

---

## The TCP Three-Way Handshake

**The Problem:**
How do two machines agree to communicate reliably over an unreliable network? They need to synchronize sequence numbers and establish connection state.

**How It Works:**

```
Client                                    Server
  │                                         │
  │  SYN (seq=100)                          │
  │  ─────────────────────────────────────► │
  │                                         │  Creates connection state
  │                                         │  Allocates buffers
  │                                         │
  │         SYN-ACK (seq=200, ack=101)      │
  │ ◄─────────────────────────────────────  │
  │                                         │
  │  Creates connection state               │
  │  Allocates buffers                      │
  │                                         │
  │  ACK (ack=201)                          │
  │  ─────────────────────────────────────► │
  │                                         │
  │         Connection Established          │
  │ ◄─────────────────────────────────────► │
```

**What happens:**
1. **SYN (Synchronize):** Client sends initial sequence number
2. **SYN-ACK:** Server acknowledges and sends its own sequence number
3. **ACK:** Client acknowledges server's sequence number

**Why this matters:**
- Sequence numbers enable ordered, reliable delivery
- Both sides allocate resources (buffers, connection state)
- SYN floods can exhaust server resources (SYN cookies mitigate this)

**Trade-offs:**

| Aspect | TCP (3-way handshake) | UDP (no handshake) |
|--------|----------------------|-------------------|
| Latency | +1 RTT before data transfer | Send data immediately |
| Reliability | Guaranteed delivery, ordering | No guarantees |
| Resource usage | Server allocates per connection | Stateless |
| Attack surface | SYN flood vulnerability | Amplification attacks |

**Connection Establishment Timeout Example:**

```python
import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Set timeout for connection establishment
client_socket.settimeout(5.0)  # 5 seconds

try:
    client_socket.connect(('example.com', 80))
    print("Connected successfully")
except socket.timeout:
    print("Connection timed out during handshake")
except ConnectionRefusedError:
    print("Server refused connection (no listener on port)")
finally:
    client_socket.close()
```

**When to use:** Understanding this helps debug connection timeouts and tune SYN backlog.

**When NOT to use:** Don't try to optimize the handshake itself—focus on keeping connections alive (connection pooling) to avoid repeated handshakes.

---

## Critical Socket Options

**The Problem:**
Default socket behavior isn't optimal for production servers. You'll hit "Address already in use" errors, experience delayed packet sends, or waste resources on dead connections.

### SO_REUSEADDR

**What it does:** Allows binding to a port in TIME_WAIT state.

**The Problem:**
```
$ python server.py
Server listening on port 8080...
^C

$ python server.py
OSError: [Errno 48] Address already in use
```

When a server closes, the socket enters TIME_WAIT (typically 60 seconds). Without SO_REUSEADDR, you can't restart the server immediately.

```python
import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Enable SO_REUSEADDR before binding
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_socket.bind(('0.0.0.0', 8080))
server_socket.listen(128)
```

**When to use:** Always for server sockets.

**When NOT to use:** On client sockets connecting to different servers (not needed).

### SO_REUSEPORT

**What it does:** Allows multiple processes to bind to the same port for load balancing.

```python
import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

server_socket.bind(('0.0.0.0', 8080))
server_socket.listen(128)

# Now you can run multiple server processes on port 8080
# Kernel distributes incoming connections across processes
```

**When to use:** Pre-fork server models (Nginx, Gunicorn with workers).

**When NOT to use:** Single-process servers, or when using a load balancer in front.

### SO_KEEPALIVE

**What it does:** Sends periodic probes to detect dead connections.

**The Problem:**
```
Client crashes without closing socket
Server holds connection open indefinitely
Result: Resource leak, file descriptor exhaustion
```

```python
import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

# Platform-specific tuning (Linux)
if hasattr(socket, 'TCP_KEEPIDLE'):
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)   # Start probes after 60s idle
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)  # Probe every 10s
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 6)     # 6 failed probes = dead

client_socket.connect(('example.com', 80))
```

**When to use:** Long-lived connections (WebSockets, database connections, connection pools).

**When NOT to use:** Short-lived HTTP connections (overhead not worth it).

### TCP_NODELAY

**What it does:** Disables Nagle's algorithm, sending small packets immediately.

**The Problem (Nagle's Algorithm):**
```
Client sends: "H" "e" "l" "l" "o"
Without TCP_NODELAY: Waits 200ms, sends "Hello" (1 packet)
With TCP_NODELAY: Sends 5 packets immediately

Nagle's algorithm batches small writes to reduce packets
But this adds latency for interactive protocols
```

```python
import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Disable Nagle's algorithm
client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

client_socket.connect(('example.com', 80))
client_socket.send(b"GET / HTTP/1.1\r\n")  # Sent immediately
client_socket.send(b"Host: example.com\r\n\r\n")  # Also sent immediately
```

**Trade-offs:**

| Aspect | Nagle Enabled (default) | Nagle Disabled (TCP_NODELAY) |
|--------|------------------------|------------------------------|
| Latency | Higher (up to 200ms) | Lower (immediate send) |
| Bandwidth efficiency | Better (fewer packets) | Worse (more small packets) |
| Use case | Bulk data transfer | Interactive protocols (SSH, games) |

**When to use:** Low-latency requirements (request/response protocols, real-time apps).

**When NOT to use:** Bulk data transfers where throughput matters more than latency.

---

## Connection Backlog and Tuning

**The Problem:**
Your server can only accept connections so fast. What happens to new connections while `accept()` is busy?

**How It Works:**

```
New Connection Arrives
         │
         ▼
   ┌──────────────────────┐
   │   SYN Queue          │  ← Half-open connections (SYN received)
   │   (syn_backlog)      │
   └──────────────────────┘
         │
         ▼ (SYN-ACK sent, ACK received)
   ┌──────────────────────┐
   │   Accept Queue       │  ← Fully established, waiting for accept()
   │   (listen backlog)   │
   └──────────────────────┘
         │
         ▼ accept() called
   ┌──────────────────────┐
   │   Application        │
   └──────────────────────┘
```

**Backlog Parameter:**

```python
import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 8080))

# Backlog = 128 means accept queue can hold 128 established connections
server_socket.listen(128)

# If accept() is slow and 128 connections are waiting,
# new connections get refused or dropped
```

**System-Level Tuning (Linux):**

```bash
# View current limits
sysctl net.core.somaxconn          # Max accept queue (default: 128)
sysctl net.ipv4.tcp_max_syn_backlog  # Max SYN queue (default: 1024)

# Increase for high-traffic servers
sudo sysctl -w net.core.somaxconn=4096
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=8192
```

**Monitoring Backlog Drops:**

```python
import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 8080))
server_socket.listen(4096)

# Monitor with netstat
# netstat -s | grep -i listen
# Output: "123 times the listen queue of a socket overflowed"
```

**When to use:** Set backlog to expected burst connection rate multiplied by accept() latency.

**When NOT to use:** Don't set absurdly high values (e.g., 1 million)—if your accept queue is that full, you have bigger problems.

---

## Non-Blocking Sockets and I/O Multiplexing

**The Problem:**
Blocking I/O wastes threads. If `accept()`, `recv()`, or `send()` block, you need one thread per connection—at 10,000 connections, you need 10,000 threads.

**How It Works:**

```
Blocking Socket (1 thread per connection):
┌──────────┐   ┌──────────┐   ┌──────────┐
│ Thread 1 │   │ Thread 2 │   │ Thread 3 │
│  Client  │   │  Client  │   │  Client  │
│    A     │   │    B     │   │    C     │
└──────────┘   └──────────┘   └──────────┘
     │              │              │
     ▼              ▼              ▼
   recv()         recv()         recv()
   [blocks]       [blocks]       [blocks]

Non-Blocking Socket (1 thread, many connections):
┌─────────────────────────────────────────┐
│           Single Thread                  │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│         select/poll/epoll                │
│   Monitors: Socket A, B, C, D, E, ...    │
└─────────────────────────────────────────┘
     │
     ▼ (Socket C has data)
   recv(C)  ← Only reads from ready sockets
```

**Non-Blocking Socket Example:**

```python
import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 8080))
server_socket.listen(128)

# Make socket non-blocking
server_socket.setblocking(False)

while True:
    try:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        client_socket.setblocking(False)
        # Handle client_socket...
    except BlockingIOError:
        # No connection available, do other work
        pass
```

**Using select() for I/O Multiplexing:**

```python
import socket
import select

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 8080))
server_socket.listen(128)
server_socket.setblocking(False)

# Track all sockets
sockets_list = [server_socket]

while True:
    # Wait until any socket is ready for reading
    readable, _, exceptional = select.select(sockets_list, [], sockets_list, 1.0)

    for notified_socket in readable:
        if notified_socket == server_socket:
            # New connection
            client_socket, addr = server_socket.accept()
            client_socket.setblocking(False)
            sockets_list.append(client_socket)
            print(f"New connection from {addr}")
        else:
            # Data from existing connection
            try:
                data = notified_socket.recv(1024)
                if data:
                    print(f"Received: {data.decode()}")
                    notified_socket.send(b"Echo: " + data)
                else:
                    # Empty data means client closed
                    sockets_list.remove(notified_socket)
                    notified_socket.close()
            except Exception as e:
                print(f"Error: {e}")
                sockets_list.remove(notified_socket)
                notified_socket.close()

    for notified_socket in exceptional:
        sockets_list.remove(notified_socket)
        notified_socket.close()
```

**Trade-offs:**

| Approach | Scalability | Complexity | CPU Usage |
|----------|-------------|------------|-----------|
| Blocking (1 thread/conn) | Low (~1k conns) | Simple | Low per request |
| Non-blocking + select/poll | Medium (~10k conns) | Moderate | O(n) scanning |
| Non-blocking + epoll/kqueue | High (~100k+ conns) | Moderate | O(1) event notification |
| Async frameworks (asyncio) | High | Simple API, complex internals | Efficient |

**When to use:** Servers handling many concurrent connections (web servers, proxies).

**When NOT to use:** Simple scripts or single-connection clients (blocking is simpler).

---

## Connection Pooling

**The Problem:**
Creating a new TCP connection is expensive: DNS lookup, TCP handshake, TLS handshake. For a database query taking 5ms, you waste 50ms on connection setup.

**How It Works:**

```
Without Connection Pooling:
Request 1: Create conn → Query → Close
Request 2: Create conn → Query → Close
Request 3: Create conn → Query → Close
Total time: 3 × (50ms setup + 5ms query) = 165ms

With Connection Pooling:
Startup: Create 10 connections
Request 1: Checkout conn → Query → Return to pool
Request 2: Checkout conn → Query → Return to pool
Request 3: Checkout conn → Query → Return to pool
Total time: 3 × 5ms query = 15ms
```

**Connection Pool Architecture:**

```
┌─────────────────────────────────────────┐
│          Application Threads             │
└─────────────────────────────────────────┘
     │              │              │
     ▼              ▼              ▼
┌─────────────────────────────────────────┐
│         Connection Pool                  │
│                                          │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │Conn1│ │Conn2│ │Conn3│ │Conn4│ (idle)│
│  └─────┘ └─────┘ └─────┘ └─────┘       │
│  ┌─────┐ ┌─────┐                        │
│  │Conn5│ │Conn6│          (in use)      │
│  └─────┘ └─────┘                        │
└─────────────────────────────────────────┘
     │              │              │
     ▼              ▼              ▼
┌─────────────────────────────────────────┐
│          Database Server                 │
└─────────────────────────────────────────┘
```

**Python Connection Pool Example:**

```python
import socket
import queue
import threading

class ConnectionPool:
    def __init__(self, host, port, pool_size=10):
        self.host = host
        self.port = port
        self.pool = queue.Queue(maxsize=pool_size)

        # Pre-create connections
        for _ in range(pool_size):
            conn = self._create_connection()
            self.pool.put(conn)

    def _create_connection(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.connect((self.host, self.port))
        return sock

    def checkout(self, timeout=5.0):
        """Get a connection from the pool."""
        try:
            return self.pool.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("Connection pool exhausted")

    def checkin(self, conn):
        """Return a connection to the pool."""
        try:
            self.pool.put(conn, block=False)
        except queue.Full:
            # Pool is full, close extra connection
            conn.close()

    def close_all(self):
        """Close all pooled connections."""
        while not self.pool.empty():
            conn = self.pool.get()
            conn.close()

# Usage
pool = ConnectionPool('localhost', 6379, pool_size=10)

def make_request():
    conn = pool.checkout()
    try:
        conn.send(b"PING\r\n")
        response = conn.recv(1024)
        print(f"Response: {response.decode()}")
    finally:
        pool.checkin(conn)

# Multiple threads can safely share the pool
threads = [threading.Thread(target=make_request) for _ in range(100)]
for t in threads:
    t.start()
for t in threads:
    t.join()

pool.close_all()
```

**Pool Sizing Calculation:**

```
Pool size = (Request rate × Average request latency) + Buffer

Example:
- 1,000 requests/second
- 10ms average query time
- 20% buffer for spikes

Pool size = (1000 req/s × 0.01s) + (10 × 0.2) = 10 + 2 = 12 connections
```

**When to use:** Any client making repeated connections to the same service (databases, Redis, HTTP clients).

**When NOT to use:** Single-request scripts, or when connecting to many different hosts (pool per host is wasteful).

---

## Keep-Alive and Idle Connections

**The Problem:**
Long-lived idle connections can die silently due to firewalls, NAT timeouts, or crashes. Your application thinks the connection is alive, but writes fail.

**TCP Keep-Alive Settings:**

```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

# Linux-specific tuning
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)    # Start probes after 60s idle
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)   # Send probe every 10s
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 6)      # 6 failed probes = dead

# Total timeout before detecting dead connection:
# 60s (idle) + (10s × 6 probes) = 120 seconds
```

**Application-Level Keep-Alive (Better):**

```python
import socket
import time

def application_keepalive(sock, idle_timeout=30):
    """Send application-level pings to detect dead connections faster."""
    last_activity = time.time()

    while True:
        # Try to receive data (non-blocking)
        sock.setblocking(False)
        try:
            data = sock.recv(1024)
            if data:
                last_activity = time.time()
                # Process data...
            else:
                # Connection closed
                print("Connection closed by peer")
                break
        except BlockingIOError:
            # No data available
            pass

        # Check if we should send a keep-alive ping
        if time.time() - last_activity > idle_timeout:
            try:
                sock.send(b"PING\r\n")
                last_activity = time.time()
            except OSError:
                print("Connection is dead")
                break

        time.sleep(1)
```

**HTTP Keep-Alive (Connection Reuse):**

```python
import socket

# HTTP/1.1 persistent connections
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('example.com', 80))

# First request
request1 = b"GET /page1 HTTP/1.1\r\nHost: example.com\r\nConnection: keep-alive\r\n\r\n"
sock.send(request1)
response1 = sock.recv(4096)

# Second request on the same connection
request2 = b"GET /page2 HTTP/1.1\r\nHost: example.com\r\nConnection: keep-alive\r\n\r\n"
sock.send(request2)
response2 = sock.recv(4096)

sock.close()
```

**Trade-offs:**

| Aspect | TCP Keep-Alive | Application Keep-Alive | No Keep-Alive |
|--------|----------------|----------------------|---------------|
| Detection time | 60-120 seconds (default) | 5-30 seconds (configurable) | Only on write attempt |
| Bandwidth | Minimal (TCP probes) | Protocol-specific pings | None |
| Accuracy | Can miss application hangs | Detects app-level issues | N/A |
| Complexity | Socket option only | Requires protocol support | None |

**When to use:** Connection pools, WebSocket servers, database connections.

**When NOT to use:** Short-lived HTTP requests (overhead not justified).

---

## Building a Production-Grade TCP Server

**The Problem:**
A real server needs more than basic socket code: graceful shutdown, error handling, concurrent connections, and resource limits.

**Complete Example:**

```python
import socket
import select
import signal
import sys
from typing import Dict, Set

class TCPServer:
    def __init__(self, host='0.0.0.0', port=8080, backlog=128):
        self.host = host
        self.port = port
        self.backlog = backlog
        self.server_socket = None
        self.client_sockets: Set[socket.socket] = set()
        self.running = False

        # Graceful shutdown on SIGINT/SIGTERM
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.running = False

    def start(self):
        """Start the server."""
        # Create and configure socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind and listen
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.backlog)
        self.server_socket.setblocking(False)

        print(f"Server listening on {self.host}:{self.port}")
        self.running = True

        self._accept_loop()

    def _accept_loop(self):
        """Main accept loop using select()."""
        while self.running:
            # Build list of sockets to monitor
            sockets_to_read = [self.server_socket] + list(self.client_sockets)

            try:
                readable, _, exceptional = select.select(
                    sockets_to_read, [], sockets_to_read, 1.0
                )
            except ValueError:
                # Socket was closed, rebuild list
                continue

            for sock in readable:
                if sock is self.server_socket:
                    self._accept_client()
                else:
                    self._handle_client(sock)

            for sock in exceptional:
                self._close_client(sock)

        self._shutdown()

    def _accept_client(self):
        """Accept a new client connection."""
        try:
            client_socket, addr = self.server_socket.accept()
            client_socket.setblocking(False)
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            self.client_sockets.add(client_socket)
            print(f"Client connected: {addr}")

            # Send welcome message
            client_socket.send(b"Welcome to TCPServer\r\n")
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"Error accepting client: {e}")

    def _handle_client(self, client_socket):
        """Handle data from a client."""
        try:
            data = client_socket.recv(4096)

            if not data:
                # Client closed connection
                self._close_client(client_socket)
                return

            # Echo server: send data back
            response = b"Echo: " + data
            client_socket.send(response)

        except ConnectionResetError:
            self._close_client(client_socket)
        except Exception as e:
            print(f"Error handling client: {e}")
            self._close_client(client_socket)

    def _close_client(self, client_socket):
        """Close a client connection."""
        if client_socket in self.client_sockets:
            self.client_sockets.remove(client_socket)
            try:
                client_socket.close()
            except Exception:
                pass

    def _shutdown(self):
        """Gracefully shutdown the server."""
        print("Shutting down server...")

        # Close all client connections
        for client_socket in list(self.client_sockets):
            try:
                client_socket.send(b"Server shutting down\r\n")
                client_socket.close()
            except Exception:
                pass

        # Close server socket
        if self.server_socket:
            self.server_socket.close()

        print("Server stopped")

# Run the server
if __name__ == '__main__':
    server = TCPServer(host='0.0.0.0', port=8080)
    server.start()
```

**Testing the Server:**

```python
# test_client.py
import socket

def test_echo_server():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 8080))

    # Receive welcome message
    welcome = client.recv(1024)
    print(f"Server says: {welcome.decode()}")

    # Send test message
    client.send(b"Hello, Server!\r\n")
    response = client.recv(1024)
    print(f"Server echoed: {response.decode()}")

    client.close()

if __name__ == '__main__':
    test_echo_server()
```

**When to use:** Production servers, learning socket programming internals.

**When NOT to use:** Use battle-tested frameworks (asyncio, Twisted, Tornado) for real production—don't reinvent the wheel.

---

## Key Concepts Checklist

- [ ] Understand TCP vs UDP trade-offs and when to use each
- [ ] Explain the TCP three-way handshake and its resource implications
- [ ] Configure critical socket options: SO_REUSEADDR, SO_KEEPALIVE, TCP_NODELAY
- [ ] Tune connection backlog based on accept() latency and connection rate
- [ ] Implement non-blocking I/O with select/poll/epoll for high concurrency
- [ ] Design connection pools with appropriate sizing and timeout handling
- [ ] Distinguish between TCP keep-alive and application-level keep-alive
- [ ] Build a basic TCP server handling concurrent connections and graceful shutdown

---

## Practical Insights

**Socket option layering matters:**
When setting socket options, order can matter. Set SO_REUSEADDR before bind(), set TCP_NODELAY after connect(). On Linux, you can set SO_REUSEPORT only if all processes sharing the port set it. For keep-alive tuning, set SO_KEEPALIVE to 1 first, then tune TCP_KEEPIDLE/KEEPINTVL/KEEPCNT.

**Connection backlog is not a queue size:**
The listen() backlog parameter is a hint, not a hard limit. The actual queue size is min(backlog, net.core.somaxconn). On macOS, the default somaxconn is 128—your listen(4096) won't help unless you tune the kernel. On production Linux servers, set somaxconn to 4096+ and match your listen() backlog. Monitor "listen queue overflows" with `netstat -s | grep -i listen`.

**TIME_WAIT states are a feature, not a bug:**
When you see thousands of TIME_WAIT connections in netstat, don't panic. These are properly closed connections waiting 60 seconds (2×MSL) to handle delayed packets. They consume minimal resources (no buffers, just kernel memory). Only worry if you're exhausting ephemeral ports (check with `sysctl net.ipv4.ip_local_port_range`). If needed, tune `net.ipv4.tcp_tw_reuse` but never use `tcp_tw_recycle` (breaks NAT).

**Keep-alive settings are OS-wide defaults:**
TCP keep-alive settings (TCP_KEEPIDLE, etc.) have system-wide defaults in /proc/sys/net/ipv4/tcp_keepalive_*. Setting them per-socket overrides these defaults. For connection pools, always set per-socket to avoid depending on admin-controlled defaults. Common production values: KEEPIDLE=60, KEEPINTVL=10, KEEPCNT=3 (detects dead connections in 90 seconds).

**File descriptor limits bite in production:**
Each socket consumes one file descriptor. Default limit is often 1024 (ulimit -n). For 10,000 concurrent connections, you need at least 10,000 FDs. Set soft/hard limits in /etc/security/limits.conf or systemd service files (LimitNOFILE=65536). Also tune kernel limits: `fs.file-max` (system-wide) and `fs.nr_open` (per-process max). Monitor with `lsof | wc -l` and `/proc/<pid>/limits`.

**Connection pooling requires idle timeouts:**
Database servers close idle connections (MySQL default: 8 hours, PostgreSQL: infinite with TCP keepalive). If your pool holds connections longer than server timeout, you'll get "connection lost" errors. Set pool idle timeout < server timeout, or rely on keep-alive probes. Better: implement connection validation (ping before checkout) to detect stale connections early.
