# Chapter 17: Real-time Systems

## Why Real-time?

Without real-time communication:

```
Stock trader using HTTP polling:
- Poll every 1 second for price updates
- Stock moves in 100ms, trader sees stale data
- Result: Lost trades, angry users

Chat application using polling:
- 10,000 users poll every second for messages
- 10,000 requests/second even when idle
- Result: Server overload, battery drain
```

Real-time communication enables:
- **Instant updates** (chat, notifications, live feeds)
- **Collaborative features** (Google Docs, Figma)
- **Live dashboards** (monitoring, analytics)
- **Gaming** (multiplayer, real-time positioning)

---

## WebSockets

### The Problem

HTTP is request-response. Client asks, server answers. For real-time updates, you need bidirectional persistent connections.

### How It Works

```
Client                           Server
  │                                │
  │──── HTTP GET /chat ────────────│   1. HTTP Upgrade Request
  │     Upgrade: websocket         │
  │     Connection: Upgrade        │
  │                                │
  │◄─── 101 Switching Protocols ──│   2. Protocol Switch
  │     Upgrade: websocket         │
  │                                │
  ╞════════════════════════════════╡   3. WebSocket Connection
  │                                │      (Persistent, Bidirectional)
  │──── Message: "Hello" ──────────►
  │                                │
  │◄─── Message: "Welcome" ────────│
  │                                │
  │──── Ping ──────────────────────►  4. Heartbeat (optional)
  │◄─── Pong ──────────────────────│
  │                                │
```

### WebSocket Handshake

```python
# Client initiates upgrade
GET /chat HTTP/1.1
Host: server.example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13

# Server accepts
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
```

**Sec-WebSocket-Key:** Random nonce to prevent cross-protocol attacks
**Sec-WebSocket-Accept:** Hash of key + magic string, proves server understands WebSocket

### WebSocket Frame Format

```
 0               1               2               3
 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
+-+-+-+-+-------+-+-------------+-------------------------------+
|F|R|R|R| opcode|M| Payload len |    Extended payload length    |
|I|S|S|S|  (4)  |A|     (7)     |             (16/64)           |
|N|V|V|V|       |S|             |                               |
| |1|2|3|       |K|             |                               |
+-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
|     Extended payload length continued, if payload len == 127  |
+ - - - - - - - - - - - - - - - +-------------------------------+
|                               |Masking-key, if MASK set to 1  |
+-------------------------------+-------------------------------+
| Masking-key (continued)       |          Payload Data         |
+-------------------------------- - - - - - - - - - - - - - - - +
:                     Payload Data continued ...                :
+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
|                     Payload Data continued ...                |
+---------------------------------------------------------------+
```

**FIN:** Final fragment
**Opcode:** 0x1 (text), 0x2 (binary), 0x8 (close), 0x9 (ping), 0xA (pong)
**MASK:** Payload masked (required from client)

### Implementation (Python with FastAPI)

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Set
import asyncio

app = FastAPI()

# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: str):
        # Send to all connected clients
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.add(connection)

        # Clean up dead connections
        self.active_connections -= disconnected

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            # Broadcast to all clients
            await manager.broadcast(f"Message: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### Heartbeat/Ping-Pong

Detect dead connections:

```python
async def heartbeat(websocket: WebSocket, timeout=30):
    """Send ping every 30 seconds, expect pong"""
    try:
        while True:
            await asyncio.sleep(timeout)
            try:
                await websocket.send_text('{"type": "ping"}')
                # Wait for pong response
                response = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=10
                )
            except asyncio.TimeoutError:
                # No pong received, connection is dead
                await websocket.close()
                break
    except Exception:
        pass

# Usage
asyncio.create_task(heartbeat(websocket))
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Latency | Very low (ms) | N/A |
| Bidirectional | Yes | More complex |
| Browser support | All modern browsers | Firewall issues |
| Server resources | Persistent connections | High memory per connection |
| Reconnection | Manual handling needed | Complex state management |

**When to use:** Chat, gaming, collaborative editing, live dashboards

**When NOT to use:** Simple one-way updates (use SSE), infrequent updates (use polling)

---

## Server-Sent Events (SSE)

### The Problem

WebSockets are bidirectional, but often you only need server-to-client updates. SSE is simpler for unidirectional streaming.

### How It Works

```
Client                           Server
  │                                │
  │──── GET /events ───────────────│   1. HTTP Request
  │     Accept: text/event-stream  │
  │                                │
  │◄─── 200 OK ────────────────────│   2. Keep connection open
  │     Content-Type:              │
  │       text/event-stream        │
  │                                │
  │◄─── data: {"msg": "Hi"} ───────│   3. Stream events
  │                                │
  │◄─── data: {"msg": "Update"} ───│
  │                                │
```

### SSE Message Format

```
event: message
id: 123
data: {"type": "update", "value": 42}

event: notification
id: 124
data: {"text": "New message"}
```

**event:** Event type (optional, default is "message")
**id:** Event ID for reconnection
**data:** Event payload (can be multi-line)

### Implementation

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

async def event_generator():
    """Generate server-sent events"""
    event_id = 0
    while True:
        # Simulate data updates
        event_id += 1
        data = {"timestamp": time.time(), "value": random.randint(0, 100)}

        # SSE format
        yield f"id: {event_id}\n"
        yield f"event: update\n"
        yield f"data: {json.dumps(data)}\n\n"

        await asyncio.sleep(1)

@app.get("/events")
async def sse_endpoint():
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# Client (JavaScript)
"""
const eventSource = new EventSource('/events');

eventSource.addEventListener('update', (event) => {
    const data = JSON.parse(event.data);
    console.log('Update:', data);
});

eventSource.addEventListener('error', (error) => {
    console.error('Connection error:', error);
    eventSource.close();
});
"""
```

### Automatic Reconnection

SSE has built-in reconnection:

```python
async def event_generator():
    yield "retry: 5000\n\n"  # Reconnect after 5 seconds if disconnected

    event_id = 0
    while True:
        event_id += 1
        yield f"id: {event_id}\n"
        yield f"data: Update {event_id}\n\n"
        await asyncio.sleep(1)
```

Client reconnects automatically using `Last-Event-ID` header:

```
GET /events HTTP/1.1
Last-Event-ID: 123
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Simplicity | Built on HTTP | Unidirectional only |
| Reconnection | Automatic with event ID | N/A |
| Browser support | All modern browsers | No IE support |
| Efficiency | Less overhead than WS | Still persistent connection |

**When to use:** Live feeds, notifications, stock tickers, dashboards

**When NOT to use:** Need bidirectional communication, large binary data

---

## Long Polling

### The Problem

Fallback for environments that block WebSockets/SSE (corporate proxies, old browsers).

### How It Works

```
Client                           Server
  │                                │
  │──── GET /poll ─────────────────│   1. Client requests
  │                                │
  │                          ┌─────┤   2. Server holds request
  │                          │ wait│      until data available
  │                          └─────┤
  │◄─── Response: "update" ────────│   3. Server responds with data
  │                                │
  │──── GET /poll ─────────────────│   4. Client immediately re-polls
  │                                │
  │                          ┌─────┤
  │                          │ wait│
```

Compared to short polling:

```
Short polling (inefficient):
Client polls every 1 second
9/10 requests: "No updates"
Result: Wasted requests

Long polling:
Server holds request until update
Client immediately re-polls
Result: Near real-time with HTTP
```

### Implementation

```python
from fastapi import FastAPI
import asyncio
from collections import defaultdict
import uuid

app = FastAPI()

# In-memory queue for pending updates
pending_updates = defaultdict(asyncio.Queue)

@app.get("/poll/{client_id}")
async def long_poll(client_id: str):
    """
    Hold request until update available or timeout
    """
    queue = pending_updates[client_id]

    try:
        # Wait up to 30 seconds for update
        update = await asyncio.wait_for(queue.get(), timeout=30)
        return {"data": update}
    except asyncio.TimeoutError:
        # No update, return empty response
        return {"data": None}

@app.post("/broadcast")
async def broadcast(message: str):
    """Send message to all clients"""
    for client_id, queue in pending_updates.items():
        await queue.put(message)
    return {"status": "sent"}

# Client pseudo-code
"""
async function poll() {
    while (true) {
        try {
            const response = await fetch('/poll/client-123');
            const data = await response.json();
            if (data.data) {
                handleUpdate(data.data);
            }
            // Immediately poll again
        } catch (error) {
            // Wait before retry on error
            await sleep(5000);
        }
    }
}
"""
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Compatibility | Works everywhere | Less efficient |
| Simplicity | Standard HTTP | Complex state management |
| Latency | Near real-time | Higher than WebSocket |
| Server load | One connection at a time | Frequent reconnections |

**When to use:** Fallback when WebSocket/SSE unavailable

**When NOT to use:** Primary real-time solution (use WS/SSE instead)

---

## Scaling WebSocket Connections

### The Problem

```
Single server: 64,000 connections max (file descriptor limit)
100,000 users online: Need multiple servers

User A on Server 1
User B on Server 2
How do they chat?
```

### Solution 1: Sticky Sessions

Route same user to same server:

```
┌─────────────────────────────────────────────┐
│           Load Balancer                     │
│     (Hash by user_id or cookie)             │
└─────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
    ┌────────┐     ┌────────┐     ┌────────┐
    │Server 1│     │Server 2│     │Server 3│
    │User A,C│     │User B,D│     │User E,F│
    └────────┘     └────────┘     └────────┘
```

**Implementation (nginx):**

```nginx
upstream websocket_backend {
    ip_hash;  # Sticky based on IP
    server ws1.example.com:8000;
    server ws2.example.com:8000;
    server ws3.example.com:8000;
}

server {
    location /ws {
        proxy_pass http://websocket_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;  # 24 hours
    }
}
```

**Limitations:**
- Users on different servers can't communicate directly
- Need pub/sub backend

### Solution 2: Pub/Sub Backend (Redis)

All servers subscribe to Redis, broadcast across servers:

```
    User A                    User B
      │                         │
      ▼                         ▼
┌──────────┐              ┌──────────┐
│ Server 1 │              │ Server 2 │
└──────────┘              └──────────┘
      │                         │
      │      ┌──────────┐       │
      └─────►│  Redis   │◄──────┘
             │ Pub/Sub  │
             └──────────┘

Flow:
1. User A sends message to Server 1
2. Server 1 publishes to Redis channel
3. Server 2 subscribed to channel, receives message
4. Server 2 sends to User B
```

### Implementation with Redis Pub/Sub

```python
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket
import json
import asyncio

app = FastAPI()

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

class PubSubManager:
    def __init__(self):
        self.connections: dict[str, set[WebSocket]] = {}
        self.pubsub = None

    async def connect(self, room_id: str, websocket: WebSocket):
        """Add connection to room"""
        if room_id not in self.connections:
            self.connections[room_id] = set()
        self.connections[room_id].add(websocket)

    async def disconnect(self, room_id: str, websocket: WebSocket):
        """Remove connection from room"""
        self.connections[room_id].discard(websocket)
        if not self.connections[room_id]:
            del self.connections[room_id]

    async def publish(self, room_id: str, message: dict):
        """Publish message to Redis channel"""
        channel = f"room:{room_id}"
        await redis_client.publish(channel, json.dumps(message))

    async def subscribe(self, room_id: str):
        """Subscribe to Redis channel and forward to WebSocket clients"""
        pubsub = redis_client.pubsub()
        channel = f"room:{room_id}"
        await pubsub.subscribe(channel)

        async for message in pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])

                # Send to all WebSocket clients in this room
                if room_id in self.connections:
                    disconnected = set()
                    for ws in self.connections[room_id]:
                        try:
                            await ws.send_json(data)
                        except Exception:
                            disconnected.add(ws)

                    # Clean up dead connections
                    self.connections[room_id] -= disconnected

manager = PubSubManager()

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await websocket.accept()
    await manager.connect(room_id, websocket)

    # Start subscription task for this room
    asyncio.create_task(manager.subscribe(room_id))

    try:
        while True:
            data = await websocket.receive_text()
            message = {"room": room_id, "text": data}

            # Publish to Redis (all servers will receive)
            await manager.publish(room_id, message)
    except WebSocketDisconnect:
        await manager.disconnect(room_id, websocket)
```

### Connection Limits per Server

```python
# Limit connections per server
MAX_CONNECTIONS = 50000

class ConnectionLimiter:
    def __init__(self, max_connections: int):
        self.max_connections = max_connections
        self.current_connections = 0
        self.semaphore = asyncio.Semaphore(max_connections)

    async def acquire(self):
        """Acquire connection slot"""
        acquired = await self.semaphore.acquire()
        if acquired:
            self.current_connections += 1
        return acquired

    def release(self):
        """Release connection slot"""
        self.semaphore.release()
        self.current_connections -= 1

limiter = ConnectionLimiter(MAX_CONNECTIONS)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not await limiter.acquire():
        await websocket.close(code=1008, reason="Server at capacity")
        return

    try:
        await websocket.accept()
        # Handle connection...
    finally:
        limiter.release()
```

---

## Presence and Typing Indicators

### Presence System

Track who's online:

```python
import time

class PresenceManager:
    def __init__(self, redis_client, timeout=300):
        self.redis = redis_client
        self.timeout = timeout  # 5 minutes

    async def mark_online(self, user_id: str):
        """Mark user as online"""
        key = f"presence:{user_id}"
        await self.redis.setex(key, self.timeout, "online")

    async def mark_offline(self, user_id: str):
        """Mark user as offline"""
        key = f"presence:{user_id}"
        await self.redis.delete(key)

    async def is_online(self, user_id: str) -> bool:
        """Check if user is online"""
        key = f"presence:{user_id}"
        return await self.redis.exists(key) > 0

    async def get_online_users(self, user_ids: list[str]) -> list[str]:
        """Get online users from list"""
        pipeline = self.redis.pipeline()
        for user_id in user_ids:
            pipeline.exists(f"presence:{user_id}")
        results = await pipeline.execute()

        return [
            user_id for user_id, online in zip(user_ids, results)
            if online
        ]

    async def heartbeat(self, user_id: str):
        """Extend online status (called periodically)"""
        await self.mark_online(user_id)

# WebSocket handler with presence
presence = PresenceManager(redis_client)

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    await presence.mark_online(user_id)

    # Heartbeat task
    async def heartbeat_task():
        while True:
            await asyncio.sleep(60)  # Every minute
            await presence.heartbeat(user_id)

    task = asyncio.create_task(heartbeat_task())

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await presence.mark_offline(user_id)
        task.cancel()
```

### Typing Indicators

Show "User is typing...":

```python
class TypingManager:
    def __init__(self, redis_client, timeout=5):
        self.redis = redis_client
        self.timeout = timeout  # Typing expires after 5 seconds

    async def start_typing(self, room_id: str, user_id: str):
        """Mark user as typing in room"""
        key = f"typing:{room_id}"
        await self.redis.setex(f"{key}:{user_id}", self.timeout, "1")

    async def stop_typing(self, room_id: str, user_id: str):
        """Mark user as stopped typing"""
        key = f"typing:{room_id}:{user_id}"
        await self.redis.delete(key)

    async def get_typing_users(self, room_id: str) -> list[str]:
        """Get list of users typing in room"""
        pattern = f"typing:{room_id}:*"
        keys = await self.redis.keys(pattern)
        return [key.split(':')[-1] for key in keys]

typing = TypingManager(redis_client)

# Client sends typing events
@app.websocket("/ws/{room_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, user_id: str):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            if data['type'] == 'typing_start':
                await typing.start_typing(room_id, user_id)
                # Broadcast to room
                await manager.publish(room_id, {
                    "type": "user_typing",
                    "user_id": user_id
                })

            elif data['type'] == 'typing_stop':
                await typing.stop_typing(room_id, user_id)
                await manager.publish(room_id, {
                    "type": "user_stopped_typing",
                    "user_id": user_id
                })

            elif data['type'] == 'message':
                # Stop typing when message sent
                await typing.stop_typing(room_id, user_id)
                await manager.publish(room_id, {
                    "type": "message",
                    "user_id": user_id,
                    "text": data['text']
                })
    except WebSocketDisconnect:
        await typing.stop_typing(room_id, user_id)
```

---

## Reconnection Strategies

### Exponential Backoff

```javascript
// Client-side reconnection
class WebSocketClient {
    constructor(url) {
        this.url = url;
        this.reconnectDelay = 1000;  // Start with 1 second
        this.maxDelay = 30000;       // Max 30 seconds
        this.connect();
    }

    connect() {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log('Connected');
            this.reconnectDelay = 1000;  // Reset delay
        };

        this.ws.onclose = () => {
            console.log('Disconnected, reconnecting...');
            setTimeout(() => {
                this.reconnectDelay = Math.min(
                    this.reconnectDelay * 2,
                    this.maxDelay
                );
                this.connect();
            }, this.reconnectDelay);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    send(data) {
        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        } else {
            console.warn('WebSocket not ready');
        }
    }
}
```

### Session Recovery

Restore state after reconnection:

```python
import uuid

class SessionManager:
    def __init__(self):
        self.sessions = {}  # {session_id: {user_id, last_message_id}}

    async def create_session(self, user_id: str) -> str:
        """Create new session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "last_message_id": 0,
            "created_at": time.time()
        }
        return session_id

    async def restore_session(self, session_id: str, last_message_id: int):
        """Get missed messages since last_message_id"""
        # Fetch messages from database/cache
        missed_messages = await get_messages_after(last_message_id)
        return missed_messages

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Client sends session ID if reconnecting
    data = await websocket.receive_json()

    if 'session_id' in data:
        # Reconnection: restore session
        session_id = data['session_id']
        last_message_id = data['last_message_id']
        missed = await session_manager.restore_session(
            session_id, last_message_id
        )
        await websocket.send_json({
            "type": "restore",
            "messages": missed
        })
    else:
        # New connection: create session
        user_id = data['user_id']
        session_id = await session_manager.create_session(user_id)
        await websocket.send_json({
            "type": "session",
            "session_id": session_id
        })
```

---

## Comparison: Real-time Technologies

| Feature | WebSocket | SSE | Long Polling | gRPC Streaming |
|---------|-----------|-----|--------------|----------------|
| Direction | Bidirectional | Server→Client | Request/Response | Bidirectional |
| Protocol | WebSocket | HTTP | HTTP | HTTP/2 |
| Browser support | Excellent | Good (no IE) | Universal | Limited |
| Latency | Very low | Low | Medium | Very low |
| Reconnection | Manual | Automatic | N/A | Manual |
| Binary data | Yes | No | Yes | Yes |
| Firewall friendly | Sometimes blocked | Yes | Yes | Sometimes blocked |
| Complexity | Medium | Low | Medium | High |
| Best for | Chat, gaming | Live feeds, notifications | Fallback | Service-to-service |

---

## Key Concepts Checklist

- [ ] Understand WebSocket handshake and frame format
- [ ] Know when to use WebSocket vs SSE vs Long Polling
- [ ] Implement heartbeat/ping-pong for connection health
- [ ] Design reconnection strategy with exponential backoff
- [ ] Scale WebSockets across servers using Redis Pub/Sub
- [ ] Implement presence system with Redis TTL
- [ ] Handle typing indicators with expiring keys
- [ ] Configure sticky sessions for connection affinity

---

## Practical Insights

**Connection limits:**
- Linux default: 1024 file descriptors per process
- Increase: `ulimit -n 65535` or systemd `LimitNOFILE=65535`
- Rule of thumb: 50,000 connections per server is realistic
- Monitor: open file descriptors (`lsof | wc -l`)

**Memory per connection:**
```
Per WebSocket connection:
- TCP buffer: 16KB - 64KB (send + receive)
- Application buffer: 8KB - 32KB
- Python object overhead: ~5KB
Total: ~50KB - 100KB per connection

50,000 connections = 2.5GB - 5GB RAM
```

**Heartbeat tuning:**
- Too frequent: Wastes bandwidth
- Too infrequent: Slow to detect dead connections
- Sweet spot: 30-60 second ping, 10 second pong timeout
- Load balancers often have 60-300 second idle timeout

**Redis Pub/Sub at scale:**
```python
# Problem: All servers subscribe to all channels
# 1000 rooms × 10 servers = 10,000 subscriptions

# Solution: Shard channels
room_shard = hash(room_id) % num_redis_instances
redis_client = redis_pool[room_shard]

# Now: 1000 rooms ÷ 5 shards = 200 rooms per Redis
```

**Binary vs text frames:**
- Text: UTF-8 encoded, larger overhead
- Binary: Raw bytes, efficient for large data
- Use binary for: images, video, audio, protobuf
- Use text for: JSON messages, simple commands

**Graceful shutdown:**
```python
# Don't abruptly close connections on deploy
async def graceful_shutdown(websocket: WebSocket):
    # Send close frame
    await websocket.send_json({"type": "server_restart"})
    # Wait for client ACK or timeout
    try:
        await asyncio.wait_for(websocket.receive(), timeout=5)
    except asyncio.TimeoutError:
        pass
    # Close connection
    await websocket.close()
```
