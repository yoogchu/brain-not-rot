# Chapter 40: Serialization & Data Formats

## The Core Problem

You're building a microservices architecture serving 100,000 requests per second. Each request involves passing data between services:

```
Without proper serialization:

Service A → JSON string (900 bytes) → Service B
Network: 90 MB/s just for this endpoint
CPU: 15% spent on JSON parsing
Memory: Frequent GC pauses from string allocations

Switch to Protocol Buffers:
Service A → Protobuf bytes (180 bytes) → Service B
Network: 18 MB/s (80% reduction)
CPU: 3% for deserialization (80% reduction)
Memory: Minimal allocations, predictable sizes

One format change: 5x better performance, $20k/month saved in infrastructure
```

The format you choose for serializing data affects performance, maintainability, compatibility, and cost at scale.

---

## What is Serialization?

**The Problem:**
Moving data between processes, services, or persistent storage requires converting in-memory objects to a byte sequence and back.

**How It Works:**

```
┌─────────────────┐                    ┌─────────────────┐
│  Python Object  │                    │  Python Object  │
│                 │                    │                 │
│  user = {       │                    │  user = {       │
│    "id": 42,    │                    │    "id": 42,    │
│    "name": "X"  │                    │    "name": "X"  │
│  }              │                    │  }              │
└─────────────────┘                    └─────────────────┘
         │                                      ▲
         │ Serialize                            │ Deserialize
         ▼                                      │
┌──────────────────────────────────────────────┴────┐
│         Byte Sequence (on wire/disk)               │
│  {"id":42,"name":"X"}  ← JSON (text)               │
│  or                                                │
│  0x08 0x2A 0x12 0x01 0x58  ← Protobuf (binary)     │
└────────────────────────────────────────────────────┘
```

**Key Considerations:**

| Aspect | Text Formats | Binary Formats |
|--------|--------------|----------------|
| Human Readable | Yes (JSON, XML) | No (Protobuf, Avro) |
| Size | Larger (verbose) | Smaller (compact) |
| Speed | Slower (parsing) | Faster (direct mapping) |
| Schema Required | No (self-describing) | Yes (usually) |
| Language Support | Universal | Library-dependent |
| Debugging | Easy (curl, logs) | Hard (need tools) |

**When to use text:** APIs, configs, logs, human-in-the-loop systems
**When to use binary:** High-throughput inter-service calls, data pipelines, mobile apps

---

## JSON: The Universal Format

**The Problem:**
You need a format that works everywhere, is easy to debug, and doesn't require code generation.

**How It Works:**

```python
import json
import time

# Serialization
user = {
    "id": 42,
    "name": "Alice",
    "email": "alice@example.com",
    "created_at": "2025-01-15T10:30:00Z",
    "tags": ["premium", "verified"]
}

# To JSON string
start = time.perf_counter()
json_bytes = json.dumps(user).encode('utf-8')
serialize_time = time.perf_counter() - start

print(f"JSON size: {len(json_bytes)} bytes")
print(f"Serialize time: {serialize_time*1000:.3f} ms")
print(f"Content: {json_bytes}")

# From JSON string
start = time.perf_counter()
decoded = json.loads(json_bytes.decode('utf-8'))
deserialize_time = time.perf_counter() - start

print(f"Deserialize time: {deserialize_time*1000:.3f} ms")
print(f"Decoded: {decoded}")
```

**Output:**
```
JSON size: 121 bytes
Serialize time: 0.015 ms
Content: b'{"id":42,"name":"Alice","email":"alice@example.com",...}'
Deserialize time: 0.012 ms
Decoded: {'id': 42, 'name': 'Alice', ...}
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Compatibility | Works everywhere | No real cons here |
| Speed | Fast enough for most use cases | 5-10x slower than binary |
| Size | Readable and debuggable | 2-5x larger than binary |
| Schema | No code generation needed | No compile-time validation |
| Types | Strings, numbers, bools, arrays, objects | No date, no binary, no int64 |

**Performance Tips:**

```python
import orjson  # Faster JSON library

# 2-3x faster than standard library
json_bytes = orjson.dumps(user)
decoded = orjson.loads(json_bytes)

# For streaming large arrays
import ijson

with open('large_data.json', 'rb') as f:
    # Parse without loading entire file into memory
    for item in ijson.items(f, 'users.item'):
        process(item)
```

**When to use:** REST APIs, configuration files, webhooks, admin tools, anything human-readable
**When NOT to use:** High-frequency inter-service calls (>10k RPS), mobile apps on metered connections, when every millisecond counts

---

## Protocol Buffers: The Efficiency Champion

**The Problem:**
Your microservices exchange millions of messages per second. JSON parsing consumes 30% of CPU and bandwidth costs are escalating.

**How It Works:**

```
┌─────────────────────────────────────────────────┐
│              Protocol Buffers                    │
└─────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
   ┌──────────┐              ┌──────────┐
   │  .proto  │              │ Generated│
   │  Schema  │─── protoc ──►│   Code   │
   └──────────┘              └──────────┘
                                    │
                                    ▼
                           ┌────────────────┐
                           │  Binary Data   │
                           │  (compact)     │
                           └────────────────┘
```

**Schema Definition (.proto file):**

```protobuf
syntax = "proto3";

message User {
  int32 id = 1;
  string name = 2;
  string email = 3;
  int64 created_at = 4;
  repeated string tags = 5;
}
```

**Python Usage:**

```python
# Install: pip install protobuf
# Generate: protoc --python_out=. user.proto

import time
from user_pb2 import User

# Create message
user = User()
user.id = 42
user.name = "Alice"
user.email = "alice@example.com"
user.created_at = 1736936400
user.tags.extend(["premium", "verified"])

# Serialize to bytes
start = time.perf_counter()
protobuf_bytes = user.SerializeToString()
serialize_time = time.perf_counter() - start

print(f"Protobuf size: {len(protobuf_bytes)} bytes")
print(f"Serialize time: {serialize_time*1000:.3f} ms")

# Deserialize
start = time.perf_counter()
decoded_user = User()
decoded_user.ParseFromString(protobuf_bytes)
deserialize_time = time.perf_counter() - start

print(f"Deserialize time: {deserialize_time*1000:.3f} ms")
print(f"Decoded name: {decoded_user.name}")
```

**Output:**
```
Protobuf size: 45 bytes  (vs 121 bytes JSON)
Serialize time: 0.003 ms  (vs 0.015 ms JSON)
Deserialize time: 0.002 ms  (vs 0.012 ms JSON)
```

**Schema Evolution (Backwards Compatibility):**

```protobuf
// Version 1
message User {
  int32 id = 1;
  string name = 2;
}

// Version 2 - Add fields (safe)
message User {
  int32 id = 1;
  string name = 2;
  string email = 3;        // New field - old code ignores it
  repeated string tags = 4;  // New field - defaults to empty
}

// Version 3 - Mark deprecated (safe)
message User {
  int32 id = 1;
  string name = 2;
  string email = 3;
  repeated string tags = 4;
  int32 age = 5 [deprecated = true];  // Signal removal intent
}

// UNSAFE: Never reuse field numbers
// UNSAFE: Never change field types
// UNSAFE: Never change field names if using JSON mapping
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Size | 60-80% smaller than JSON | Not human readable |
| Speed | 5-10x faster than JSON | Requires code generation |
| Schema | Strong typing, validation | Schema changes need coordination |
| Evolution | Forward/backward compatible | Must follow strict rules |
| Languages | 15+ languages supported | Generated code adds complexity |

**When to use:** Internal microservices, mobile APIs, data pipelines, anywhere performance matters
**When NOT to use:** Public APIs (not debuggable), one-off scripts, prototyping (overhead not worth it)

---

## Apache Avro: Schema Evolution Master

**The Problem:**
You have a data pipeline with hundreds of consumers. Schema changes break everything because protobuf field numbers are scattered across 50 repos.

**How It Works:**

```
┌─────────────────────────────────────────────────┐
│                 Apache Avro                      │
└─────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
   ┌──────────┐              ┌──────────────┐
   │  Schema  │              │  Binary Data │
   │  (JSON)  │◄─────────────┤  + Schema ID │
   └──────────┘              └──────────────┘
         │                          │
         ▼                          ▼
 ┌────────────────┐         ┌────────────┐
 │ Schema Registry│         │  Consumer  │
 │  (central)     │────────►│  (reads)   │
 └────────────────┘         └────────────┘
```

**Key Difference from Protobuf:**
- Protobuf: Field numbers in schema (must never change)
- Avro: Field names in schema (can add/remove with defaults)

**Schema Definition (JSON):**

```python
import avro.schema
import avro.io
import io
import time

# Define schema
schema_str = '''
{
  "type": "record",
  "name": "User",
  "fields": [
    {"name": "id", "type": "int"},
    {"name": "name", "type": "string"},
    {"name": "email", "type": "string"},
    {"name": "created_at", "type": "long"},
    {"name": "tags", "type": {"type": "array", "items": "string"}}
  ]
}
'''

schema = avro.schema.parse(schema_str)

# Serialize
user_data = {
    "id": 42,
    "name": "Alice",
    "email": "alice@example.com",
    "created_at": 1736936400,
    "tags": ["premium", "verified"]
}

start = time.perf_counter()
bytes_writer = io.BytesIO()
encoder = avro.io.BinaryEncoder(bytes_writer)
writer = avro.io.DatumWriter(schema)
writer.write(user_data, encoder)
avro_bytes = bytes_writer.getvalue()
serialize_time = time.perf_counter() - start

print(f"Avro size: {len(avro_bytes)} bytes")
print(f"Serialize time: {serialize_time*1000:.3f} ms")

# Deserialize
start = time.perf_counter()
bytes_reader = io.BytesIO(avro_bytes)
decoder = avro.io.BinaryDecoder(bytes_reader)
reader = avro.io.DatumReader(schema)
decoded = reader.read(decoder)
deserialize_time = time.perf_counter() - start

print(f"Deserialize time: {deserialize_time*1000:.3f} ms")
print(f"Decoded: {decoded}")
```

**Schema Evolution Example:**

```python
# Original schema
schema_v1 = '''
{
  "type": "record",
  "name": "User",
  "fields": [
    {"name": "id", "type": "int"},
    {"name": "name", "type": "string"}
  ]
}
'''

# Evolved schema - add field with default
schema_v2 = '''
{
  "type": "record",
  "name": "User",
  "fields": [
    {"name": "id", "type": "int"},
    {"name": "name", "type": "string"},
    {"name": "email", "type": "string", "default": ""}
  ]
}
'''

# Evolved schema - remove field (mark as optional first)
schema_v3 = '''
{
  "type": "record",
  "name": "User",
  "fields": [
    {"name": "id", "type": "int"},
    {"name": "email", "type": "string", "default": ""}
  ]
}
'''

# Old data with v1 schema can be read with v2 schema
# Missing "email" field gets default value ""
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Evolution | Best-in-class schema evolution | Requires schema registry |
| Size | Compact (similar to Protobuf) | Slightly larger than Protobuf |
| Speed | Fast (slower than Protobuf) | Not as fast as Protobuf |
| Dynamic | No code generation needed | Runtime schema resolution |
| Ecosystem | Kafka integration | Smaller ecosystem than Protobuf |

**When to use:** Kafka data pipelines, data lakes, systems with frequent schema changes
**When NOT to use:** Low-latency services (Protobuf faster), simple request/response (JSON easier)

---

## MessagePack: JSON's Faster Sibling

**The Problem:**
You love JSON's simplicity but need better performance. You don't want to deal with schemas and code generation.

**How It Works:**

```
JSON:     {"id":42,"name":"Alice"}  →  26 bytes (text)
                     ↓ MessagePack converts
MsgPack:  [binary representation]   →  18 bytes (binary)

Same structure, binary encoding, no schema needed
```

**Python Usage:**

```python
import msgpack
import json
import time

user = {
    "id": 42,
    "name": "Alice",
    "email": "alice@example.com",
    "created_at": 1736936400,
    "tags": ["premium", "verified"]
}

# MessagePack
start = time.perf_counter()
msgpack_bytes = msgpack.packb(user)
msgpack_time = time.perf_counter() - start

# JSON for comparison
start = time.perf_counter()
json_bytes = json.dumps(user).encode('utf-8')
json_time = time.perf_counter() - start

print(f"MessagePack: {len(msgpack_bytes)} bytes in {msgpack_time*1000:.3f} ms")
print(f"JSON:        {len(json_bytes)} bytes in {json_time*1000:.3f} ms")
print(f"Size savings: {(1 - len(msgpack_bytes)/len(json_bytes))*100:.1f}%")

# Deserialize
decoded = msgpack.unpackb(msgpack_bytes, raw=False)
print(f"Decoded: {decoded}")

# Streaming large data
with open('data.msgpack', 'wb') as f:
    packer = msgpack.Packer()
    for item in large_dataset:
        f.write(packer.pack(item))

with open('data.msgpack', 'rb') as f:
    unpacker = msgpack.Unpacker(f, raw=False)
    for item in unpacker:
        process(item)
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Simplicity | No schema, drop-in JSON replacement | No type safety |
| Size | 20-30% smaller than JSON | Larger than Protobuf/Avro |
| Speed | 2-3x faster than JSON | Slower than Protobuf |
| Compatibility | Many languages | Less universal than JSON |
| Debug | Tools exist but not curl-friendly | Not human readable |

**When to use:** Internal APIs, caching layer, WebSocket messages, Redis storage
**When NOT to use:** Public APIs, when JSON is already fast enough, highly structured data (use Protobuf)

---

## Apache Thrift: The Facebook Legacy

**The Problem:**
You need RPC framework + serialization in one package with multi-language support.

**How It Works:**

```
┌─────────────────────────────────────────────────┐
│              Apache Thrift                       │
│  (Serialization + RPC Framework)                 │
└─────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
   ┌──────────┐              ┌──────────────┐
   │  .thrift │              │   Generated  │
   │  IDL     │─── thrift ──►│ Client/Server│
   └──────────┘              └──────────────┘
```

**Thrift IDL:**

```thrift
struct User {
  1: i32 id,
  2: string name,
  3: string email,
  4: i64 created_at,
  5: list<string> tags
}

service UserService {
  User getUser(1: i32 userId),
  void createUser(1: User user)
}
```

**Python Usage:**

```python
from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport
import time

# Generated code: user_types import User

user = User(
    id=42,
    name="Alice",
    email="alice@example.com",
    created_at=1736936400,
    tags=["premium", "verified"]
)

# Serialize
transport = TTransport.TMemoryBuffer()
protocol = TBinaryProtocol.TBinaryProtocol(transport)

start = time.perf_counter()
user.write(protocol)
thrift_bytes = transport.getvalue()
serialize_time = time.perf_counter() - start

print(f"Thrift size: {len(thrift_bytes)} bytes")
print(f"Serialize time: {serialize_time*1000:.3f} ms")

# Deserialize
transport = TTransport.TMemoryBuffer(thrift_bytes)
protocol = TBinaryProtocol.TBinaryProtocol(transport)

start = time.perf_counter()
decoded_user = User()
decoded_user.read(protocol)
deserialize_time = time.perf_counter() - start

print(f"Deserialize time: {deserialize_time*1000:.3f} ms")
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Integration | Serialization + RPC together | Heavyweight (most just need serialization) |
| Size | Similar to Protobuf | Similar to Protobuf |
| Speed | Similar to Protobuf | Similar to Protobuf |
| Ecosystem | Mature, battle-tested | Less momentum than gRPC |
| Language Support | 20+ languages | Generated code quality varies |

**When to use:** Legacy Facebook/Twitter-style architectures, need tight RPC integration
**When NOT to use:** New projects (use gRPC + Protobuf), simple serialization (overkill)

---

## Python struct: Low-Level Binary Control

**The Problem:**
You need to parse a binary network protocol or file format with exact byte layout control.

**How It Works:**

```python
import struct
import socket

# Pack data into binary format
# Format: "!I" = network byte order (!), unsigned int (I)

user_id = 42
name = b"Alice"

# Format string: ! = network (big-endian), I = uint32, 10s = 10 char string
data = struct.pack("!I10s", user_id, name)
print(f"Packed: {data.hex()}")  # '0000002a416c6963650000000000'
print(f"Size: {len(data)} bytes")

# Unpack
unpacked_id, unpacked_name = struct.unpack("!I10s", data)
print(f"Unpacked: id={unpacked_id}, name={unpacked_name.rstrip(b'\x00')}")

# Real-world example: TCP header parsing
tcp_header = b'\x00\x50\x1f\x90\x00\x00\x00\x01\x00\x00\x00\x00\x50\x02\x20\x00'

# TCP header format
fields = struct.unpack("!HHLLBBHHH", tcp_header)
src_port, dst_port, seq_num, ack_num, offset_reserved, flags, window, checksum, urgent = fields

print(f"Source Port: {src_port}")
print(f"Dest Port: {dst_port}")
print(f"Sequence: {seq_num}")
print(f"Window: {window}")

# Writing binary file with exact layout
with open("data.bin", "wb") as f:
    # Write header: magic number, version, record count
    header = struct.pack("!4sHI", b"MYDB", 1, 1000)
    f.write(header)

    # Write records
    for i in range(1000):
        record = struct.pack("!I50sQ", i, f"user_{i}".encode().ljust(50, b'\x00'), i * 1000)
        f.write(record)

# Reading binary file
with open("data.bin", "rb") as f:
    # Read header
    magic, version, count = struct.unpack("!4sHI", f.read(10))
    print(f"Magic: {magic}, Version: {version}, Records: {count}")

    # Read first record
    user_id, username, timestamp = struct.unpack("!I50sQ", f.read(62))
    print(f"User: {user_id}, {username.rstrip(b'\x00').decode()}, {timestamp}")
```

**Format Characters:**

```python
# Format string components:
# Byte order:
#   @ = native, = = native (standard size), < = little-endian, > = big-endian, ! = network (big-endian)
#
# Types:
#   x = pad byte
#   c = char (1 byte)
#   b = signed char (1 byte)
#   B = unsigned char (1 byte)
#   ? = bool (1 byte)
#   h = short (2 bytes)
#   H = unsigned short (2 bytes)
#   i = int (4 bytes)
#   I = unsigned int (4 bytes)
#   l = long (4 bytes)
#   L = unsigned long (4 bytes)
#   q = long long (8 bytes)
#   Q = unsigned long long (8 bytes)
#   f = float (4 bytes)
#   d = double (8 bytes)
#   s = char[] (string)
#   p = pascal string

# Calculate size
size = struct.calcsize("!IHH")  # Returns 8 bytes
```

**When to use:** Network protocol parsing, binary file formats, embedded systems, interfacing with C libraries
**When NOT to use:** General serialization (use Protobuf/JSON), when you don't need exact byte control

---

## Endianness: Byte Order Matters

**The Problem:**
Your service runs on x86 (little-endian). You receive binary data from a network device (big-endian). Numbers are garbled.

**How It Works:**

```
Number: 42 (decimal) = 0x0000002A (hex)

Big-Endian (Network Byte Order):
┌────┬────┬────┬────┐
│ 00 │ 00 │ 00 │ 2A │  Most significant byte first
└────┴────┴────┴────┘
  0    1    2    3

Little-Endian (x86):
┌────┬────┬────┬────┐
│ 2A │ 00 │ 00 │ 00 │  Least significant byte first
└────┴────┴────┴────┘
  0    1    2    3
```

**Python Example:**

```python
import struct

value = 42

# Different byte orders
big_endian = struct.pack(">I", value)     # Network byte order
little_endian = struct.pack("<I", value)  # x86 byte order
native = struct.pack("@I", value)         # Platform native

print(f"Big-endian:    {big_endian.hex()}")    # 0000002a
print(f"Little-endian: {little_endian.hex()}")  # 2a000000
print(f"Native:        {native.hex()}")         # Depends on your CPU

# Reading network data (always big-endian)
network_bytes = b'\x00\x00\x00\x2a'
value = struct.unpack("!I", network_bytes)[0]  # ! = network = big-endian
print(f"Network value: {value}")  # 42

# Common mistake: using native byte order for network data
wrong_value = struct.unpack("@I", network_bytes)[0]
print(f"Wrong (native): {wrong_value}")  # 704643072 on little-endian machine!

# Byte swapping
import socket

# Convert to network byte order (big-endian)
network_bytes = socket.htonl(42)  # Host TO Network Long
print(f"htonl(42): {network_bytes:08x}")

# Convert from network byte order
host_value = socket.ntohl(network_bytes)  # Network TO Host Long
print(f"ntohl: {host_value}")
```

**When to use:**
- Always use network byte order (big-endian, "!") for network protocols
- Use little-endian ("<") when reading x86 binary files
- Use native ("@") only for local file formats on same architecture

**When NOT to use:**
- Don't use native byte order for anything crossing machine boundaries
- Modern serialization formats (Protobuf, Avro) handle this for you

---

## Performance Comparison

**Benchmark Setup:**

```python
import json
import msgpack
import time
from user_pb2 import User  # Protobuf generated code

# Test data
user_dict = {
    "id": 42,
    "name": "Alice Smith",
    "email": "alice@example.com",
    "created_at": 1736936400,
    "tags": ["premium", "verified", "admin"],
    "metadata": {
        "last_login": 1736950000,
        "login_count": 1523,
        "country": "US"
    }
}

iterations = 100000

def benchmark(name, serialize_fn, deserialize_fn, data):
    # Serialize
    start = time.perf_counter()
    for _ in range(iterations):
        serialized = serialize_fn(data)
    serialize_time = time.perf_counter() - start

    # Deserialize
    start = time.perf_counter()
    for _ in range(iterations):
        deserialize_fn(serialized)
    deserialize_time = time.perf_counter() - start

    print(f"{name:20} | Size: {len(serialized):4} bytes | "
          f"Serialize: {serialize_time:.3f}s | Deserialize: {deserialize_time:.3f}s")

    return len(serialized), serialize_time, deserialize_time

# JSON
benchmark(
    "JSON (stdlib)",
    lambda d: json.dumps(d).encode(),
    lambda b: json.loads(b.decode()),
    user_dict
)

# JSON (orjson - optimized)
import orjson
benchmark(
    "JSON (orjson)",
    orjson.dumps,
    orjson.loads,
    user_dict
)

# MessagePack
benchmark(
    "MessagePack",
    msgpack.packb,
    lambda b: msgpack.unpackb(b, raw=False),
    user_dict
)

# Protobuf
user_pb = User()
user_pb.id = user_dict["id"]
user_pb.name = user_dict["name"]
user_pb.email = user_dict["email"]
user_pb.created_at = user_dict["created_at"]
user_pb.tags.extend(user_dict["tags"])

benchmark(
    "Protobuf",
    lambda u: u.SerializeToString(),
    lambda b: User().ParseFromString(b),
    user_pb
)
```

**Typical Results:**

| Format | Size (bytes) | Serialize (100k) | Deserialize (100k) | Best For |
|--------|--------------|------------------|--------------------| ---------|
| JSON (stdlib) | 156 | 1.2s | 0.9s | Public APIs, debugging |
| JSON (orjson) | 156 | 0.4s | 0.3s | High-perf JSON needs |
| MessagePack | 108 | 0.5s | 0.4s | Internal APIs, caching |
| Protobuf | 62 | 0.2s | 0.15s | Microservices, mobile |
| Avro | 68 | 0.8s | 0.6s | Data pipelines |

**Real-World Impact:**

```
Service handling 50,000 RPS:

JSON (stdlib):
- CPU: 18% serialization/parsing
- Bandwidth: 7.8 GB/hour
- Monthly cost: $450 (bandwidth) + $200 (CPU)

Switch to Protobuf:
- CPU: 3% serialization/parsing (6x reduction)
- Bandwidth: 3.1 GB/hour (2.5x reduction)
- Monthly cost: $180 (bandwidth) + $35 (CPU)

Savings: $435/month, lower latency, better user experience
```

---

## Schema Evolution Strategies

**The Problem:**
You deploy a schema change to 100 services. Half deploy successfully, half fail. Old and new versions must coexist for 48 hours.

**Forward Compatibility:**
Old code can read data written by new code

**Backward Compatibility:**
New code can read data written by old code

**Both are required for zero-downtime deployments**

```
┌─────────────┐         ┌─────────────┐
│  Service A  │         │  Service B  │
│  (v1 code)  │◄───────►│  (v2 code)  │
└─────────────┘         └─────────────┘
       │                       │
       └───────────┬───────────┘
                   ▼
           Must both work!
```

**Safe Changes:**

```python
# Protobuf - Safe changes
message User {
  int32 id = 1;
  string name = 2;

  // ADD new fields (old code ignores them)
  string email = 3;  // ✓ Safe

  // Mark deprecated (signal removal intent)
  int32 age = 4 [deprecated = true];  // ✓ Safe
}

# Avro - Safe changes with defaults
{
  "type": "record",
  "name": "User",
  "fields": [
    {"name": "id", "type": "int"},
    {"name": "name", "type": "string"},
    {"name": "email", "type": "string", "default": ""}  // ✓ Safe
  ]
}
```

**Unsafe Changes:**

```python
# Protobuf - UNSAFE changes
message User {
  // NEVER change field number
  int32 id = 1;  // Was 2 before ✗ BREAKS EVERYTHING

  // NEVER change field type
  string id = 1;  // Was int32 ✗ BREAKS

  // NEVER remove required field without migration
  // (deleted: int32 id = 1)  ✗ BREAKS old readers
}

# Avro - UNSAFE changes
{
  "fields": [
    // Remove field without default ✗ BREAKS old writers
    // (deleted: {"name": "id", "type": "int"})

    // Change type without union ✗ BREAKS
    {"name": "id", "type": "string"}  // Was int
  ]
}
```

**Migration Strategy:**

```python
# Step 1: Add new field as optional (v1 → v2)
message User {
  int32 id = 1;
  string name = 2;
  string email = 3;  // New, optional
}

# Deploy v2 code everywhere (can read v1 and v2 data)
# Wait for all services to upgrade

# Step 2: Start writing new field (v2 → v3)
# Update all writers to populate "email"

# Step 3: Make field required in validation (v3 → v4)
# Application validates email is present
# But schema still marks as optional for compatibility

# Step 4: After months, can update schema to required (v4 → v5)
# Only if absolutely certain no old data exists
```

---

## Choosing the Right Format

**Decision Tree:**

```
Start: Need to serialize data
│
├─ Public API?
│  └─ YES → JSON (human readable, universal)
│
├─ High throughput (>10k RPS)?
│  ├─ YES → Need schema validation?
│  │        ├─ YES → Protobuf (fastest, typed)
│  │        └─ NO → MessagePack (fast, schemaless)
│  └─ NO → JSON (simplicity wins)
│
├─ Frequent schema changes?
│  └─ YES → Using Kafka?
│           ├─ YES → Avro (best evolution)
│           └─ NO → JSON or MessagePack
│
├─ Mobile app / metered bandwidth?
│  └─ YES → Protobuf (smallest size)
│
├─ Need RPC framework?
│  └─ YES → gRPC + Protobuf
│
└─ Parsing binary protocol?
   └─ YES → Python struct
```

**Comparison Table:**

| Use Case | Best Format | Why |
|----------|-------------|-----|
| REST API | JSON | Universal, debuggable, no tooling |
| gRPC services | Protobuf | Integrated, fastest, typed |
| Kafka pipeline | Avro | Schema registry, evolution |
| Redis cache | MessagePack | Compact, fast, schemaless |
| WebSocket | MessagePack or JSON | Depends on payload size |
| Mobile API | Protobuf | Bandwidth savings matter |
| Config files | JSON or YAML | Human editable |
| Logs | JSON | Structured, parseable |
| Network protocol | struct | Exact byte control |
| Data lake | Avro or Parquet | Schema evolution, columnar |

---

## Key Concepts Checklist

- [ ] Explain binary vs text serialization trade-offs (size, speed, debuggability)
- [ ] Know when to use JSON vs Protobuf vs Avro vs MessagePack
- [ ] Understand schema evolution (forward and backward compatibility)
- [ ] Describe Protobuf field numbering and why you can't change them
- [ ] Explain Avro's schema registry pattern and evolution model
- [ ] Implement serialization/deserialization in Python for each format
- [ ] Understand endianness and network byte order
- [ ] Calculate bandwidth and CPU savings from format migration

---

## Practical Insights

**Schema evolution discipline:**
In a large microservices environment, schema changes cause more outages than code bugs. Enforce strict review process: all schema changes require backward AND forward compatibility tests. At Google, Protobuf field number reuse is caught by presubmit checks - a deleted field's number is permanently reserved. Implement similar guardrails or you will have production incidents.

**Performance testing in production:**
Serialization benchmarks are worthless without production traffic patterns. JSON might be 5x slower in benchmarks but if you're spending 2% of CPU on serialization, the optimization to Protobuf saves 1.6% CPU - not worth the complexity. Profile first: `py-spy record --pid <PID> --format speedscope` shows actual serialization hotspots. Only optimize if it's >10% of CPU.

**MessagePack sweet spot:**
MessagePack shines for internal APIs where you want JSON-like flexibility but better performance. Use it for: service-to-service calls that aren't ultra-high throughput, caching layer serialization, WebSocket payloads, Redis value storage. Don't use it for: public APIs (not debuggable), ultra-high perf needs (Protobuf faster), or highly structured data (lose type safety).

**Avro schema registry is critical:**
If using Avro without a schema registry (Confluent, AWS Glue), you're doing it wrong. The registry ensures: writers and readers agree on schema versions, safe evolution is enforced, schema IDs are embedded in messages (tiny overhead). Set up registry before writing any Avro producers. Use compatibility modes: `BACKWARD` (most common), `FORWARD`, `FULL`, or `NONE` (dangerous).

**Protobuf anti-patterns:**
Don't use Protobuf for everything. Bad use cases: configuration files (use JSON/YAML), admin tools (debugging pain), prototyping (overhead not worth it), small services (<5k RPS where JSON is fine). Do use Protobuf for: service mesh inter-service calls, mobile APIs, high-throughput data pipelines, when you need strong typing.

**Migration strategy:**
Never do big-bang serialization format migrations. Rollout plan: (1) Add new format as optional, dual-write both formats, (2) Deploy readers that prefer new format but fall back to old, (3) Monitor for 1 week, (4) Switch readers to new format only, (5) Remove old format writers. Keep old format readers for 1 month as safety net. Tag all data with format version - `{"_format": "protobuf-v2", ...}` saves you during incidents.
