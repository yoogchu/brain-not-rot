# Chapter 9: Networking Protocols

## TCP vs UDP: The Fundamental Trade-off

### TCP (Transmission Control Protocol)

**Guarantees:**
- Reliable delivery (retransmits lost packets)
- Ordered delivery (packets arrive in sequence)
- Congestion control (doesn't overwhelm network)
- Flow control (doesn't overwhelm receiver)

**Connection Establishment (3-way handshake):**
```
Client                          Server
   │                               │
   │────── SYN (seq=100) ─────────►│
   │                               │
   │◄──── SYN-ACK (seq=200, ack=101)│
   │                               │
   │────── ACK (ack=201) ─────────►│
   │                               │
   │      Connection Established    │
   │                               │
   │◄──────── Data Flow ──────────►│
```

**Reliability mechanism:**
```
Sender                         Receiver
   │                               │
   │─── Packet 1 (seq=1) ─────────►│
   │                               │
   │─── Packet 2 (seq=2) ────X     │ (lost!)
   │                               │
   │─── Packet 3 (seq=3) ─────────►│
   │                               │
   │◄────── ACK 1 ─────────────────│
   │                               │
   │◄────── ACK 1 (duplicate!) ────│ (missing 2)
   │                               │
   │─── Packet 2 (retransmit) ────►│
   │                               │
   │◄────── ACK 3 ─────────────────│
```

**Overhead:**
- Header: 20 bytes minimum
- Handshake: 1 RTT before data
- Teardown: 4-way handshake

### UDP (User Datagram Protocol)

**No guarantees:**
- Unreliable (packets may be lost)
- Unordered (packets may arrive out of order)
- No congestion control
- No flow control

**Process:**
```
Sender                         Receiver
   │                               │
   │─── Packet 1 ─────────────────►│
   │─── Packet 2 ────X             │ (lost, no retransmit)
   │─── Packet 3 ─────────────────►│
   │                               │
   │     No handshake, no ACKs     │
```

**Overhead:**
- Header: 8 bytes only
- No setup time
- Just send!

### When to Use Each

| Use Case | TCP | UDP | Why |
|----------|-----|-----|-----|
| Web pages | ✓ | | Need every byte, in order |
| REST APIs | ✓ | | Request-response, reliability |
| Database | ✓ | | Data integrity critical |
| File transfer | ✓ | | Complete file required |
| Video streaming | | ✓ | Late frame useless, skip it |
| Voice/video calls | | ✓ | Real-time > perfect |
| Online gaming | | ✓ | Low latency critical |
| DNS queries | | ✓ | Single packet, retry at app layer |
| IoT telemetry | | ✓ | Lossy OK, high volume |

**Real-world note:** Many "UDP" applications (like QUIC, WebRTC) implement their own reliability on top of UDP for specific use cases.

---

## HTTP Versions

### HTTP/1.1 (1997)

```
Client                          Server
   │                               │
   │──── TCP Handshake ───────────►│
   │◄─────────────────────────────│
   │                               │
   │──── GET /page.html ──────────►│
   │◄──── Response ────────────────│
   │                               │
   │──── GET /style.css ──────────►│ (WAIT for previous)
   │◄──── Response ────────────────│
   │                               │
   │──── GET /app.js ─────────────►│ (WAIT again)
   │◄──── Response ────────────────│
```

**Problems:**
- **Head-of-line blocking:** Request 2 waits for Response 1
- **Workaround:** Browsers open 6 parallel connections
- **Workaround:** Domain sharding (assets1.com, assets2.com)

**Connection reuse (Keep-Alive):**
```
HTTP/1.0: New TCP connection per request
HTTP/1.1: Reuse connection (but still sequential)
```

### HTTP/2 (2015)

**Multiplexing over single connection:**
```
┌─────────────────────────────────────────┐
│      Single TCP Connection               │
│                                          │
│  Stream 1: GET /page.html ──► Response   │
│  Stream 2: GET /style.css ──► Response   │ All concurrent!
│  Stream 3: GET /app.js ───── ► Response  │
│                                          │
└─────────────────────────────────────────┘
```

**Key features:**
- **Binary protocol:** More efficient parsing than text
- **Header compression (HPACK):** Headers often redundant
- **Server push:** Send resources before client asks
- **Stream prioritization:** Critical resources first

**Header compression example:**
```
Request 1:
:method: GET
:path: /page.html
:authority: example.com
user-agent: Mozilla/5.0...
accept: text/html
cookie: session=abc123...

Request 2:
:method: GET
:path: /style.css
:authority: example.com      ← Same!
user-agent: Mozilla/5.0...   ← Same!
accept: text/css             ← Different
cookie: session=abc123...    ← Same!

HPACK: Only send differences, reference previous headers
```

**Still has TCP head-of-line blocking:**
```
TCP is a single ordered stream.
If packet 3 is lost:
- Streams 1, 2, 3 all blocked at TCP level
- Even if packet was only for Stream 3
- TCP doesn't know about HTTP streams
```

### HTTP/3 (QUIC) (2022)

**Built on UDP, not TCP:**
```
┌─────────────────────────────────────────┐
│      QUIC Connection (over UDP)          │
│                                          │
│  Stream 1: GET /page.html ──► Response   │
│  Stream 2: GET /style.css ──► Response   │
│  Stream 3: GET /app.js ───── ► Response  │
│               │                          │
│         Packet lost only                 │
│         blocks Stream 3!                 │
└─────────────────────────────────────────┘
```

**Key improvements:**

**1. No head-of-line blocking:**
```
Packet loss on Stream 3:
- Stream 1: continues normally
- Stream 2: continues normally
- Stream 3: waits for retransmit

Each stream has independent loss recovery
```

**2. 0-RTT connection resumption:**
```
First connection: 1-RTT handshake (like TCP+TLS)

Subsequent connection (same server):
Client ─── Encrypted Request + 0-RTT data ───► Server
       (Resume immediately, no handshake wait)
```

**3. Connection migration:**
```
Scenario: Phone switches from WiFi to cellular

TCP: Connection tied to (IP, Port) tuple
     New IP = new connection = restart

QUIC: Connection ID independent of IP
      IP changes, connection continues
      Seamless handoff!
```

**4. Built-in encryption:**
```
TCP + TLS: Encryption is optional, layered
QUIC: TLS 1.3 mandatory, integrated

All QUIC traffic is encrypted, always
```

### HTTP Version Comparison

| Feature | HTTP/1.1 | HTTP/2 | HTTP/3 |
|---------|----------|--------|--------|
| Transport | TCP | TCP | UDP (QUIC) |
| Multiplexing | No | Yes | Yes |
| Header compression | No | HPACK | QPACK |
| Server push | No | Yes | Yes |
| Head-of-line blocking | Yes | TCP level | No |
| 0-RTT | No | No | Yes |
| Connection migration | No | No | Yes |
| Encryption | Optional | Optional | Mandatory |

---

## Real-Time Communication Patterns

### Long Polling

```
Client                          Server
   │                               │
   │──── GET /updates ────────────►│
   │         (waiting...)          │
   │         (waiting...)          │
   │         (waiting...)          │
   │◄──── Response (new data!) ────│
   │                               │
   │──── GET /updates ────────────►│ (immediately reconnect)
   │         (waiting...)          │
   │         ...                   │
```

**Implementation:**
```python
# Server (pseudo-code)
def get_updates(request, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        data = check_for_updates(request.user_id)
        if data:
            return Response(data)
        time.sleep(0.5)
    return Response(status=204)  # No updates, client reconnects
```

**Pros:**
- Works everywhere (just HTTP)
- Simple to implement
- Firewall friendly

**Cons:**
- Connection overhead on each poll
- Server holds many connections
- Latency up to poll interval

### Server-Sent Events (SSE)

```
Client                          Server
   │                               │
   │──── GET /events ─────────────►│
   │◄──── HTTP 200 ────────────────│
   │◄──── data: msg1 ──────────────│
   │◄──── data: msg2 ──────────────│
   │◄──── data: msg3 ──────────────│
   │         (connection stays open)│
   │◄──── data: msg4 ──────────────│
```

**Event format:**
```
event: message
data: {"user": "alice", "text": "Hello"}

event: notification
data: {"type": "alert", "message": "New order"}

event: ping
data: keepalive
```

**Client (JavaScript):**
```javascript
const source = new EventSource('/events');

source.addEventListener('message', (e) => {
    const data = JSON.parse(e.data);
    console.log('Message:', data);
});

source.addEventListener('notification', (e) => {
    const data = JSON.parse(e.data);
    showNotification(data);
});

source.onerror = (e) => {
    // Auto-reconnect built in!
    console.log('Connection lost, reconnecting...');
};
```

**Pros:**
- Simple (just HTTP)
- Auto-reconnect built in
- Event types for routing
- Works with HTTP/2 multiplexing

**Cons:**
- Unidirectional (server → client only)
- Text only (no binary)
- No IE support (polyfill needed)

**Use cases:** Live feeds, notifications, dashboards, stock tickers

### WebSockets

```
Client                          Server
   │                               │
   │──── HTTP Upgrade Request ────►│
   │◄──── 101 Switching Protocols ─│
   │                               │
   │      (WebSocket connection)    │
   │                               │
   │◄──── message ─────────────────│
   │───── message ────────────────►│
   │◄──── message ─────────────────│
   │───── message ────────────────►│
   │◄──── message ─────────────────│
   │                               │
   │    (Bidirectional, full-duplex)│
```

**Upgrade handshake:**
```http
GET /chat HTTP/1.1
Host: example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13

HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
```

**Client (JavaScript):**
```javascript
const ws = new WebSocket('wss://example.com/chat');

ws.onopen = () => {
    ws.send(JSON.stringify({ type: 'join', room: 'general' }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleMessage(data);
};

ws.send(JSON.stringify({ type: 'message', text: 'Hello!' }));
```

**Server (Python with websockets):**
```python
async def handler(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        
        if data['type'] == 'join':
            await join_room(websocket, data['room'])
        elif data['type'] == 'message':
            await broadcast(data['room'], data['text'])
```

**Pros:**
- Bidirectional
- Low latency
- Binary support
- Persistent connection

**Cons:**
- Stateful (harder to scale)
- Load balancing complex (sticky sessions)
- Firewall/proxy issues sometimes
- More complex than SSE

**Use cases:** Chat, gaming, collaborative editing, trading platforms

### Comparison

| Feature | Long Polling | SSE | WebSocket |
|---------|--------------|-----|-----------|
| Direction | Server → Client | Server → Client | Bidirectional |
| Connection | New per message | Persistent | Persistent |
| Protocol | HTTP | HTTP | WS/WSS |
| Binary | Via encoding | No | Yes |
| Auto-reconnect | Manual | Built-in | Manual |
| Browser support | Universal | Good | Good |
| Scaling | Easy | Medium | Hard |

---

## gRPC

### What is gRPC?

**gRPC = Protocol Buffers + HTTP/2**

```
┌─────────────────────────────────────────┐
│            Protocol Buffers              │
│  (Binary serialization, schema-first)    │
└─────────────────────────────────────────┘
                    +
┌─────────────────────────────────────────┐
│              HTTP/2                      │
│  (Multiplexing, streaming, headers)      │
└─────────────────────────────────────────┘
                    =
┌─────────────────────────────────────────┐
│               gRPC                       │
│  (Efficient RPC framework)               │
└─────────────────────────────────────────┘
```

### Protocol Buffer Definition

```protobuf
// user.proto
syntax = "proto3";

package userservice;

service UserService {
    // Unary: one request, one response
    rpc GetUser(GetUserRequest) returns (User);
    
    // Server streaming: one request, stream of responses
    rpc ListUsers(ListUsersRequest) returns (stream User);
    
    // Client streaming: stream of requests, one response
    rpc UploadUsers(stream User) returns (UploadSummary);
    
    // Bidirectional: stream both ways
    rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}

message User {
    int64 id = 1;
    string name = 2;
    string email = 3;
    repeated string roles = 4;
}

message GetUserRequest {
    int64 id = 1;
}

message ListUsersRequest {
    int32 page_size = 1;
    string page_token = 2;
}
```

### Streaming Patterns

**Unary (Simple RPC):**
```
Client ───── GetUser(id=123) ──────► Server
Client ◄───── User{...} ─────────── Server
```

**Server streaming:**
```
Client ───── ListUsers(page_size=100) ──► Server
Client ◄───── User{id=1} ─────────────── Server
Client ◄───── User{id=2} ─────────────── Server
Client ◄───── User{id=3} ─────────────── Server
...
Client ◄───── User{id=100} ────────────── Server
```

**Client streaming:**
```
Client ───── User{...} ──────────────► Server
Client ───── User{...} ──────────────► Server
Client ───── User{...} ──────────────► Server
Client ───── (end stream) ───────────► Server
Client ◄───── UploadSummary{count=3} ─ Server
```

**Bidirectional streaming:**
```
Client ───── ChatMessage ────────────► Server
Client ◄───── ChatMessage ───────────── Server
Client ───── ChatMessage ────────────► Server
Client ───── ChatMessage ────────────► Server
Client ◄───── ChatMessage ───────────── Server
(concurrent in both directions)
```

### gRPC vs REST

| Aspect | gRPC | REST |
|--------|------|------|
| Protocol | HTTP/2 | HTTP/1.1 or 2 |
| Format | Protobuf (binary) | JSON (text) |
| Schema | Required (.proto) | Optional (OpenAPI) |
| Code generation | Automatic | Manual or optional |
| Streaming | Native, all patterns | Limited |
| Browser support | grpc-web (limited) | Universal |
| Performance | Higher (smaller, faster) | Lower |
| Human-readable | No | Yes |
| Debugging | Needs tools | curl/browser |

### When to Use gRPC

**Use gRPC for:**
- Internal microservices
- Performance-critical paths
- Streaming requirements
- Strong typing across services
- Polyglot environments (code generation)

**Use REST for:**
- Public APIs
- Browser clients
- Simple CRUD
- Human debugging needed
- Third-party integration

---

## Key Concepts Checklist

- [ ] Explain TCP vs UDP trade-offs
- [ ] Describe HTTP evolution (1.1 → 2 → 3)
- [ ] Explain head-of-line blocking and how QUIC solves it
- [ ] Compare real-time patterns (Long Poll vs SSE vs WebSocket)
- [ ] Explain gRPC and when to use it over REST
- [ ] Design appropriate protocol for given scenario

---

## Practical Insights

**Protocol selection framework:**
1. Public API? → REST (universality)
2. Internal, high-throughput? → gRPC (efficiency)
3. Real-time, server-push only? → SSE (simplicity)
4. Real-time, bidirectional? → WebSocket (full-duplex)
5. Mobile, unreliable network? → HTTP/3 (connection migration)

**HTTP/3 adoption:**
- Check CDN/LB support
- Fall back to HTTP/2 gracefully
- Monitor QUIC vs TCP performance
- Watch for UDP throttling (some networks)

**WebSocket at scale:**
- Sticky sessions required
- Connection limits per server
- Consider connection pooling services
- Redis Pub/Sub for cross-server messaging

**gRPC considerations:**
- Invest in observability tooling
- Use deadlines (not just timeouts)
- Implement proper error codes
- Consider grpc-web for browser needs
