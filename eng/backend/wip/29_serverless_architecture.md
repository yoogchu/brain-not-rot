# Chapter 29: Serverless Architecture

## Why Serverless?

Your e-commerce site handles 1,000 requests/day during normal times.

```
Black Friday arrives: 50,000 requests/hour
Traditional approach:
- Provision 50 servers in advance (expensive, mostly idle)
- Or: Underprovision → Site crashes → Lost sales

Serverless approach:
- Pay only for actual requests
- Automatic scaling from 1 to 1000 concurrent executions
- Zero capacity planning
```

But there's a catch. The first request to a cold function takes 3 seconds to respond. Your checkout page just lost a customer.

Serverless isn't magic. It's a trade-off: operational complexity for cold start latency.

---

## Function as a Service (FaaS) Fundamentals

### The Problem:
Traditional servers require capacity planning, patching, monitoring, and scaling orchestration. You spend more time managing infrastructure than writing business logic.

### How It Works:

```
┌─────────────────────────────────────────────────────┐
│                   Event Source                       │
│   (API Gateway, S3, SQS, EventBridge, etc.)         │
└─────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Function Invocation                     │
│                                                      │
│  Cold Start Path        Hot Path                    │
│  ┌──────────────┐      ┌──────────────┐            │
│  │ Download     │      │ Use existing │            │
│  │ code         │      │ container    │            │
│  │ ↓            │      │ ↓            │            │
│  │ Start        │      │ Execute      │            │
│  │ container    │      │ handler      │            │
│  │ ↓            │      └──────────────┘            │
│  │ Initialize   │       ▲                          │
│  │ runtime      │       │ 10-100ms                 │
│  │ ↓            │       │                          │
│  │ Load code    │                                  │
│  │ ↓            │                                  │
│  │ Execute      │                                  │
│  │ handler      │                                  │
│  └──────────────┘                                  │
│   ▲                                                │
│   │ 500ms - 10s                                    │
│                                                    │
└────────────────────────────────────────────────────┘
                      │
                      ▼
                  Response
```

### Basic Lambda Function

```python
import json
import boto3
from datetime import datetime

# Initialized ONCE per container (outside handler)
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('orders')

def lambda_handler(event, context):
    """
    Handler function - called for each invocation.

    event: Request data (API Gateway, S3 event, etc.)
    context: Runtime info (request_id, memory_limit, etc.)
    """
    # Parse input
    body = json.loads(event.get('body', '{}'))
    user_id = body.get('user_id')

    # Business logic
    order = {
        'order_id': context.request_id,
        'user_id': user_id,
        'timestamp': datetime.utcnow().isoformat(),
        'items': body.get('items', [])
    }

    # Write to database
    table.put_item(Item=order)

    # Return response (API Gateway format)
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({'order_id': order['order_id']})
    }
```

**Deployment configuration:**

```yaml
# serverless.yml or SAM template
CreateOrder:
  Type: AWS::Serverless::Function
  Properties:
    Handler: order.lambda_handler
    Runtime: python3.11
    MemorySize: 512    # More memory = more CPU
    Timeout: 30        # Max execution time
    Environment:
      Variables:
        TABLE_NAME: orders
    Events:
      CreateOrderAPI:
        Type: Api
        Properties:
          Path: /orders
          Method: POST
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Scaling | Automatic, instant | Cold starts on scale-up |
| Cost | Pay-per-invocation | Expensive at high volume |
| Operations | No servers to manage | Harder to debug/monitor |
| State | Forces stateless design | Can't keep in-memory cache |

**When to use:** Event-driven workloads, unpredictable traffic, rapid prototyping

**When NOT to use:** Long-running tasks (>15min), consistent low latency requirements (<50ms), high throughput workloads (>10M requests/day may be cheaper on containers)

---

## Cold Starts: The Hidden Tax

### The Problem:
When a function hasn't been invoked recently, the cloud provider must provision a new execution environment. This initialization overhead can add seconds to your response time.

### Cold Start Anatomy:

```
Cold Start Breakdown (AWS Lambda Python):

┌───────────────────────────────────────────┐
│ Download code package      │ 100-500ms    │
├───────────────────────────────────────────┤
│ Start micro-VM/container   │ 200-800ms    │
├───────────────────────────────────────────┤
│ Initialize runtime         │ 100-300ms    │
├───────────────────────────────────────────┤
│ Import dependencies        │ 50-5000ms    │  ← Biggest variable
├───────────────────────────────────────────┤
│ Execute global scope code  │ 50-2000ms    │  ← Your control
├───────────────────────────────────────────┤
│ Execute handler            │ 10-1000ms    │
└───────────────────────────────────────────┘

Total: 500ms - 10 seconds
```

### Mitigation Strategies

#### 1. Minimize Package Size

```python
# BAD: Import entire library
import pandas as pd  # 100MB+, 2-3 second import

def lambda_handler(event, context):
    # Only using one function
    return pd.to_datetime(event['date'])

# GOOD: Use lightweight alternatives
from datetime import datetime

def lambda_handler(event, context):
    return datetime.fromisoformat(event['date'])

# GOOD: Use Lambda Layers for shared dependencies
# Layer: /opt/python/lib/python3.11/site-packages/
# Cached separately, faster loading
```

#### 2. Lazy Initialization

```python
# BAD: Initialize everything globally
import boto3
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')
secrets = boto3.client('secretsmanager')

# GOOD: Initialize on-demand
_dynamodb = None
_s3 = None

def get_dynamodb():
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource('dynamodb')
    return _dynamodb

def lambda_handler(event, context):
    # Only pay for what you use
    db = get_dynamodb()
    # s3 not initialized if not needed
```

#### 3. Provisioned Concurrency

```python
# Keep N instances always warm
# AWS Lambda configuration:
AutoScalingConfiguration:
  MinimumProvisionedConcurrency: 5
  MaximumProvisionedConcurrency: 100
  UtilizationTarget: 0.7

# Trade-off:
# - No cold starts for provisioned instances
# - Pay for idle time (like traditional servers)
# - Use for latency-critical paths
```

#### 4. Predictive Warming

```python
# Scheduled pre-warming before known traffic spikes
import boto3

lambda_client = boto3.client('lambda')

def warm_functions():
    """Run before daily 9 AM traffic spike"""
    for i in range(10):
        lambda_client.invoke(
            FunctionName='critical-api',
            InvocationType='RequestResponse',
            Payload=json.dumps({'warmup': True})
        )

# In the function:
def lambda_handler(event, context):
    if event.get('warmup'):
        return {'status': 'warmed'}
    # Normal logic
```

**Cold Start Comparison by Runtime:**

| Runtime | Typical Cold Start | Notes |
|---------|-------------------|-------|
| Python 3.11 | 200-500ms | Fast, but imports add time |
| Node.js 20 | 150-400ms | Fastest for small packages |
| Java 17 | 1-3 seconds | JVM initialization overhead |
| Go 1.x | 100-300ms | Compiled, very fast |
| .NET 6 | 500ms-2s | Better than Java, worse than Python |

**Runtime memory impact:**

```
128 MB memory:  Slow CPU, long cold start
256 MB memory:  2x faster (2x cost)
512 MB memory:  4x faster (4x cost)
1024 MB memory: 8x faster (8x cost)

Price/performance sweet spot: Usually 512-1024 MB
```

---

## Event-Driven Patterns

### S3 Triggered Processing

**The Problem:**
You need to process uploaded files (images, videos, CSVs) without a constantly-running server polling for new files.

**How It Works:**

```
┌──────────────┐
│ User uploads │
│ image to S3  │
└──────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ S3 Bucket: user-uploads          │
│ Event: s3:ObjectCreated:*        │
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ Lambda: process-image            │
│ 1. Download from S3              │
│ 2. Resize/compress               │
│ 3. Upload to processed bucket    │
│ 4. Update database               │
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ S3 Bucket: processed-images      │
└──────────────────────────────────┘
```

**Implementation:**

```python
import boto3
import os
from PIL import Image
from io import BytesIO

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # S3 event structure
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        # Download original
        response = s3.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()

        # Process
        image = Image.open(BytesIO(image_data))

        # Create thumbnail
        thumbnail = image.copy()
        thumbnail.thumbnail((300, 300))

        # Upload processed
        buffer = BytesIO()
        thumbnail.save(buffer, format=image.format)
        buffer.seek(0)

        output_key = f"thumbnails/{key}"
        s3.put_object(
            Bucket='processed-images',
            Key=output_key,
            Body=buffer,
            ContentType=f'image/{image.format.lower()}'
        )

        return {'statusCode': 200, 'thumbnailKey': output_key}
```

### API Gateway Integration

**The Problem:**
Need a scalable REST API without managing load balancers, auto-scaling groups, or container orchestration.

**How It Works:**

```
┌──────────────┐
│ HTTP Request │
│ POST /users  │
└──────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ API Gateway                             │
│ - Authentication (API keys, Cognito)    │
│ - Request validation                    │
│ - Rate limiting                         │
│ - Request transformation                │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ Lambda: create-user                     │
│ - Receives transformed event            │
│ - Executes business logic               │
│ - Returns structured response           │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ API Gateway                             │
│ - Response transformation               │
│ - CORS headers                          │
│ - Cache (optional)                      │
└─────────────────────────────────────────┘
       │
       ▼
  HTTP Response
```

**Implementation:**

```python
import json
import uuid
from typing import Dict, Any

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    API Gateway Lambda Proxy Integration
    """
    # Extract HTTP details
    http_method = event['httpMethod']
    path = event['path']
    headers = event['headers']
    query_params = event.get('queryStringParameters', {})
    body = json.loads(event.get('body', '{}'))

    # Authentication context (if using API Gateway authorizer)
    user_id = event['requestContext'].get('authorizer', {}).get('user_id')

    # Business logic
    if http_method == 'POST' and path == '/users':
        new_user = {
            'user_id': str(uuid.uuid4()),
            'email': body['email'],
            'name': body['name']
        }
        # Save to database...

        return {
            'statusCode': 201,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(new_user)
        }

    return {
        'statusCode': 404,
        'body': json.dumps({'error': 'Not found'})
    }
```

### SQS Queue Processing

**The Problem:**
Decouple producers and consumers, handle spiky workloads, ensure at-least-once processing.

**How It Works:**

```
┌────────────────┐        ┌────────────────┐
│ Order Service  │───────►│   SQS Queue    │
│ (Producer)     │        │ order-events   │
└────────────────┘        └────────────────┘
                                  │
                          Polls every 5s
                                  ▼
                          ┌────────────────┐
                          │ Lambda Process │
                          │ Batch size: 10 │
                          └────────────────┘
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
            ┌─────────────────┐      ┌─────────────────┐
            │ Success: Delete │      │ Failure: Return │
            │ from queue      │      │ to queue        │
            └─────────────────┘      └─────────────────┘
                                              │
                                     After 3 retries
                                              ▼
                                    ┌──────────────────┐
                                    │ Dead Letter Queue│
                                    └──────────────────┘
```

**Implementation:**

```python
import json
import boto3

sqs = boto3.client('sqs')

def lambda_handler(event, context):
    """
    Process SQS messages in batch

    Lambda polls SQS automatically.
    event['Records'] contains up to 10 messages.
    """
    for record in event['Records']:
        message_id = record['messageId']
        body = json.loads(record['body'])

        try:
            # Process message
            process_order(body)

            # Success - Lambda automatically deletes from queue
            print(f"Processed message {message_id}")

        except Exception as e:
            # Failure - Lambda returns message to queue
            # After max retries, goes to DLQ
            print(f"Failed to process {message_id}: {e}")
            raise  # Re-raise to trigger retry

    return {'statusCode': 200}

def process_order(order_data):
    # Business logic
    order_id = order_data['order_id']
    # Save to database, call external API, etc.
    pass
```

**Configuration:**

```yaml
OrderProcessor:
  Type: AWS::Serverless::Function
  Properties:
    Handler: process.lambda_handler
    Events:
      SQSEvent:
        Type: SQS
        Properties:
          Queue: !GetAtt OrderQueue.Arn
          BatchSize: 10           # Process 10 messages at once
          MaximumBatchingWindowInSeconds: 5  # Wait up to 5s to fill batch

OrderQueue:
  Type: AWS::SQS::Queue
  Properties:
    VisibilityTimeout: 300        # 5 minutes (> Lambda timeout)
    RedrivePolicy:
      deadLetterTargetArn: !GetAtt OrderDLQ.Arn
      maxReceiveCount: 3          # Retry 3 times before DLQ
```

---

## Serverless Databases

### DynamoDB: Serverless NoSQL

**The Problem:**
Traditional databases require provisioning instances, managing capacity, handling replication, and patching. You want a database that scales like your functions.

**How It Works:**

```
┌──────────────────────────────────────────────────┐
│ DynamoDB Table: users                            │
│                                                  │
│ Partition Key: user_id (hash)                   │
│ Sort Key: timestamp (range) [optional]          │
│                                                  │
│ Capacity Modes:                                  │
│ ┌─────────────────┐    ┌──────────────────┐    │
│ │ Provisioned     │    │ On-Demand        │    │
│ │ - Set RCU/WCU   │    │ - Pay per request│    │
│ │ - Cheaper if    │    │ - Auto-scale     │    │
│ │   predictable   │    │ - No planning    │    │
│ └─────────────────┘    └──────────────────┘    │
└──────────────────────────────────────────────────┘
```

**Lambda + DynamoDB:**

```python
import boto3
from decimal import Decimal
from boto3.dynamodb.conditions import Key

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('users')

def get_user(user_id: str):
    """Point read - very fast (single-digit milliseconds)"""
    response = table.get_item(Key={'user_id': user_id})
    return response.get('Item')

def query_user_events(user_id: str, start_time: str, end_time: str):
    """
    Query using partition key + sort key range
    Efficient - reads only relevant partition
    """
    response = table.query(
        KeyConditionExpression=
            Key('user_id').eq(user_id) &
            Key('timestamp').between(start_time, end_time)
    )
    return response['Items']

def scan_all_premium_users():
    """
    AVOID: Scan reads entire table
    Expensive and slow for large tables
    """
    response = table.scan(
        FilterExpression=Key('tier').eq('premium')
    )
    return response['Items']

def batch_write_users(users):
    """Write up to 25 items atomically"""
    with table.batch_writer() as batch:
        for user in users:
            batch.put_item(Item=user)

def atomic_increment_login_count(user_id: str):
    """Atomic counter - no race conditions"""
    table.update_item(
        Key={'user_id': user_id},
        UpdateExpression='SET login_count = login_count + :inc',
        ExpressionAttributeValues={':inc': 1}
    )
```

**DynamoDB Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Scaling | Automatic, unlimited | Hot partition throttling |
| Cost | On-demand = no waste | Expensive for scans |
| Performance | Single-digit ms reads | No complex joins/aggregations |
| Operations | Zero maintenance | Schema design critical |

**When to use:** Key-value lookups, session storage, user profiles, event logging

**When NOT to use:** Complex queries, ad-hoc analytics, relational data with many joins

### Aurora Serverless: Serverless RDBMS

**The Problem:**
You need SQL (joins, transactions, foreign keys) but don't want to manage database instances.

**How It Works:**

```
┌──────────────────────────────────────────────────┐
│ Aurora Serverless v2                             │
│                                                  │
│ ┌──────────────────────────────────────────┐   │
│ │ Capacity Units (ACUs)                    │   │
│ │ Min: 0.5 ACU  Max: 128 ACUs              │   │
│ │                                          │   │
│ │  Low traffic: 0.5 ACU  ($0.06/hour)     │   │
│ │  High traffic: 10 ACUs ($1.20/hour)     │   │
│ │                                          │   │
│ │  Auto-scales in seconds                 │   │
│ └──────────────────────────────────────────┘   │
│                                                  │
│ Storage: Auto-scaling, pay for what you use     │
└──────────────────────────────────────────────────┘
```

**Lambda + Aurora:**

```python
import json
import boto3
import pymysql

# Store DB credentials in Secrets Manager
secrets_client = boto3.client('secretsmanager')

def get_db_connection():
    # Retrieve credentials
    secret = secrets_client.get_secret_value(SecretId='db-credentials')
    creds = json.loads(secret['SecretString'])

    # Connect via proxy (RDS Proxy for connection pooling)
    connection = pymysql.connect(
        host=creds['proxy_endpoint'],
        user=creds['username'],
        password=creds['password'],
        database='app_db',
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection

def lambda_handler(event, context):
    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            # SQL queries just work
            cursor.execute("""
                SELECT u.name, COUNT(o.id) as order_count
                FROM users u
                LEFT JOIN orders o ON u.id = o.user_id
                WHERE u.created_at > %s
                GROUP BY u.id
            """, (event['since_date'],))

            results = cursor.fetchall()

        conn.commit()
        return {
            'statusCode': 200,
            'body': json.dumps(results)
        }
    finally:
        conn.close()
```

**Critical: RDS Proxy for Connection Pooling**

```
Without RDS Proxy:
Lambda spikes to 100 concurrent → 100 DB connections
Database max_connections = 150 → Database overload!

With RDS Proxy:
Lambda spikes to 100 concurrent → RDS Proxy maintains 10-20 connections
Proxy multiplexes → Database stable
```

**Aurora Serverless Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| SQL | Full PostgreSQL/MySQL | More expensive than DynamoDB |
| Scaling | Auto-scales capacity | Cold start on v1 (paused DB) |
| Cost | Pay for actual usage | Minimum capacity charge |
| Compatibility | Drop-in replacement | Connection pooling required |

---

## Step Functions: Workflow Orchestration

### The Problem:
You need to coordinate multiple Lambda functions into a workflow with retries, error handling, and conditional logic. Embedding this in code creates spaghetti.

### How It Works:

```
┌──────────────────────────────────────────────────┐
│ Step Functions State Machine                     │
│                                                  │
│  Start                                          │
│    │                                            │
│    ▼                                            │
│  ┌──────────────┐                              │
│  │ Validate     │                              │
│  │ Order        │                              │
│  └──────────────┘                              │
│    │                                            │
│    ▼                                            │
│  ┌──────────────┐   Success  ┌──────────────┐ │
│  │ Check        │──────────►  │ Charge       │ │
│  │ Inventory    │             │ Payment      │ │
│  └──────────────┘             └──────────────┘ │
│    │                                │          │
│    │ Out of stock                   │ Success  │
│    ▼                                ▼          │
│  ┌──────────────┐             ┌──────────────┐ │
│  │ Send         │             │ Ship Order   │ │
│  │ Notification │             └──────────────┘ │
│  └──────────────┘                    │         │
│    │                                 ▼         │
│    ▼                          ┌──────────────┐ │
│  End (Failed)                 │ Send Receipt │ │
│                               └──────────────┘ │
│                                      │         │
│                                      ▼         │
│                               End (Success)    │
└──────────────────────────────────────────────────┘
```

**State Machine Definition (ASL - Amazon States Language):**

```json
{
  "Comment": "Order processing workflow",
  "StartAt": "ValidateOrder",
  "States": {
    "ValidateOrder": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:validate-order",
      "Next": "CheckInventory",
      "Catch": [{
        "ErrorEquals": ["ValidationError"],
        "Next": "OrderFailed"
      }],
      "Retry": [{
        "ErrorEquals": ["States.TaskFailed"],
        "IntervalSeconds": 2,
        "MaxAttempts": 3,
        "BackoffRate": 2.0
      }]
    },
    "CheckInventory": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:check-inventory",
      "Next": "InventoryAvailable?"
    },
    "InventoryAvailable?": {
      "Type": "Choice",
      "Choices": [{
        "Variable": "$.inventory_available",
        "BooleanEquals": true,
        "Next": "ChargePayment"
      }],
      "Default": "OutOfStock"
    },
    "ChargePayment": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:charge-payment",
      "Next": "ShipOrder",
      "Catch": [{
        "ErrorEquals": ["PaymentFailed"],
        "ResultPath": "$.error",
        "Next": "PaymentFailed"
      }]
    },
    "ShipOrder": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:ship-order",
      "Next": "SendReceipt"
    },
    "SendReceipt": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:send-receipt",
      "End": true
    },
    "OutOfStock": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:notify-out-of-stock",
      "End": true
    },
    "PaymentFailed": {
      "Type": "Fail",
      "Error": "PaymentProcessingFailed",
      "Cause": "Payment could not be processed"
    },
    "OrderFailed": {
      "Type": "Fail",
      "Error": "OrderValidationFailed",
      "Cause": "Order validation failed"
    }
  }
}
```

**Lambda functions stay simple:**

```python
# validate_order.py
def lambda_handler(event, context):
    order = event['order']

    if not order.get('items'):
        raise ValidationError("No items in order")

    if order['total'] <= 0:
        raise ValidationError("Invalid total")

    # Return data flows to next state
    return {
        'order_id': order['id'],
        'validated': True
    }

# check_inventory.py
def lambda_handler(event, context):
    items = event['order']['items']

    available = check_stock(items)

    return {
        'inventory_available': available,
        'order_id': event['order_id']
    }
```

**Step Functions Features:**

- **Built-in retry logic**: Exponential backoff, max attempts
- **Error handling**: Catch specific errors, route to recovery states
- **Wait states**: Delay execution (e.g., wait for human approval)
- **Parallel execution**: Run multiple branches concurrently
- **Visibility**: See exactly where workflow is, full execution history
- **Long-running**: Up to 1 year (vs Lambda's 15 min limit)

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Orchestration | Visual, declarative | JSON config can get large |
| Durability | Survives failures | State transitions cost money |
| Debugging | Full execution history | Learning curve for ASL |
| Long-running | Up to 1 year | Expensive for high-frequency workflows |

**When to use:** Multi-step workflows, saga patterns, human-in-the-loop processes

**When NOT to use:** Simple sequential tasks (just call Lambda from Lambda), high-frequency micro-workflows (millions/day gets expensive)

---

## Vendor Lock-in and Portability

### The Reality Check:

Serverless is deeply integrated with cloud provider services. "Portable serverless" is largely a myth.

**AWS Lambda dependencies:**

```python
import boto3  # AWS SDK
from aws_lambda_powertools import Logger  # AWS-specific

# Direct coupling to AWS services
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
secrets = boto3.client('secretsmanager')
```

**Mitigation strategies:**

#### 1. Abstraction Layer

```python
# storage_interface.py
from abc import ABC, abstractmethod

class StorageInterface(ABC):
    @abstractmethod
    def get(self, key: str) -> bytes:
        pass

    @abstractmethod
    def put(self, key: str, data: bytes) -> None:
        pass

# aws_storage.py
import boto3

class S3Storage(StorageInterface):
    def __init__(self, bucket: str):
        self.s3 = boto3.client('s3')
        self.bucket = bucket

    def get(self, key: str) -> bytes:
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        return response['Body'].read()

    def put(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)

# gcp_storage.py
from google.cloud import storage

class GCSStorage(StorageInterface):
    def __init__(self, bucket: str):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket)

    def get(self, key: str) -> bytes:
        blob = self.bucket.blob(key)
        return blob.download_as_bytes()

    def put(self, key: str, data: bytes) -> None:
        blob = self.bucket.blob(key)
        blob.upload_from_string(data)

# handler.py
def lambda_handler(event, context):
    # Inject storage implementation
    storage = get_storage_impl()  # Factory based on env
    data = storage.get('user-data.json')
    # Business logic is portable
```

**Trade-off:** Abstraction adds complexity, performance overhead. Only worth it if multi-cloud is a real requirement (it usually isn't).

#### 2. Framework Abstraction (Serverless Framework, SAM)

```yaml
# serverless.yml - Multi-cloud deployment config
service: my-app

provider:
  name: aws  # or: google, azure
  runtime: python3.11

functions:
  hello:
    handler: handler.hello
    events:
      - http:
          path: /hello
          method: get

# Deploy to AWS:    serverless deploy --provider aws
# Deploy to Google: serverless deploy --provider google
```

**Reality:** Event formats differ, features differ. You'll still write provider-specific code.

---

## Cost Model: Pay-Per-Invocation vs Always-On

### Serverless Pricing (AWS Lambda):

```
Request charges: $0.20 per 1M requests

Compute charges: $0.0000166667 per GB-second
- 512 MB memory, 100ms execution = $0.0000008333
- 1024 MB memory, 100ms execution = $0.0000016667

Example:
1M requests/month
512 MB memory
Average 200ms duration

Cost = (1M * $0.20 / 1M) + (1M * 0.2s * 0.5GB * $0.0000166667)
     = $0.20 + $1.67
     = $1.87/month
```

### vs. EC2/Container (Always-On):

```
t3.medium instance:
- 2 vCPU, 4 GB RAM
- $0.0416/hour = $30.37/month
- Handles ~1000 requests/second

Break-even calculation:
Serverless: $1.87 for 1M requests/month (12 req/sec average)
Container: $30.37 for any amount

Serverless cheaper if: < 10M requests/month
Container cheaper if: > 10M requests/month (consistent traffic)
```

**Cost optimizations:**

```python
# 1. Right-size memory (more memory = faster = cheaper overall)
# Test different configurations:
# 512 MB, 500ms = $0.0000042
# 1024 MB, 250ms = $0.0000042  # Same cost, 2x faster!

# 2. Batch processing
def lambda_handler(event, context):
    # Bad: Process 1 item per invocation
    # 1M items = 1M invocations = $0.20

    # Good: Process 100 items per invocation
    # 1M items = 10K invocations = $0.002
    for item in event['Records']:  # SQS batch
        process(item)

# 3. Keep functions warm for critical paths
# Provisioned Concurrency: $0.000003472 per GB-second
# Always ready, no cold starts
# Use for <10% of traffic on critical endpoints
```

---

## Serverless Provider Comparison

| Feature | AWS Lambda | Google Cloud Functions | Azure Functions | Cloudflare Workers |
|---------|------------|----------------------|-----------------|-------------------|
| **Max execution time** | 15 minutes | 9 minutes (gen2) | Unlimited (premium) | 30 seconds (free), 15 min (paid) |
| **Max memory** | 10 GB | 32 GB | 14 GB | 128 MB |
| **Cold start (Python)** | 200-500ms | 300-800ms | 500ms-2s | <1ms |
| **Pricing (compute)** | $0.0000166667/GB-s | $0.0000025/GB-s | $0.000016/GB-s | $0.50/1M requests |
| **Free tier** | 1M requests/month | 2M requests/month | 1M requests/month | 100K requests/day |
| **Edge locations** | Lambda@Edge (limited) | No | No | Yes (global edge) |
| **Supported runtimes** | Many (Python, Node, Java, Go, .NET, Ruby) | Many | Many | JS/Wasm only |
| **VPC integration** | Yes | Yes | Yes | No (edge network) |
| **Best for** | AWS ecosystem | GCP ecosystem | Azure ecosystem | Edge compute, global latency |

**Cloudflare Workers unique model:**

```javascript
// Runs on CDN edge (200+ locations)
// <1ms cold start (V8 isolates, not containers)

addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  // Executes close to user, anywhere in world
  const ip = request.headers.get('CF-Connecting-IP')
  const location = request.cf.city

  return new Response(`Hello from ${location}!`)
}
```

**When to use each:**

- **AWS Lambda:** Already on AWS, need AWS service integration, mature ecosystem
- **Google Cloud Functions:** On GCP, need BigQuery/Pub-Sub integration
- **Azure Functions:** On Azure, .NET workloads, Durable Functions (workflow orchestration)
- **Cloudflare Workers:** Global edge compute, sub-10ms latency requirements, simple workloads

---

## Key Concepts Checklist

- [ ] Understand cold start causes and mitigation (package size, provisioned concurrency, runtime choice)
- [ ] Design event-driven architectures (S3, SQS, API Gateway, EventBridge)
- [ ] Choose appropriate serverless database (DynamoDB for key-value, Aurora for SQL)
- [ ] Orchestrate complex workflows with Step Functions (retries, error handling, visibility)
- [ ] Calculate cost break-even point (serverless vs containers/VMs)
- [ ] Implement connection pooling for databases (RDS Proxy)
- [ ] Know when NOT to use serverless (consistent high throughput, strict latency SLAs, long-running tasks)
- [ ] Monitor and debug distributed serverless applications (X-Ray, CloudWatch)

---

## Practical Insights

**Cold start reality:**
Most workloads never notice cold starts. If 99% of your requests are <100ms and 1% are 2 seconds (cold starts), your p99 is still 2 seconds - but that might be fine. Measure actual user impact before over-optimizing. Provisioned concurrency is expensive; use it sparingly for truly latency-sensitive paths (checkout, payment).

**The 15-minute wall:**
Lambda's 15-minute timeout is a hard limit. For video encoding, large data processing, or ML inference that runs longer, use containers (ECS/Fargate) or batch processing (AWS Batch). Don't try to work around this with Step Functions just to stay "serverless" - use the right tool.

**Database connections are your enemy:**
```
100 concurrent Lambdas = 100 DB connections
Most databases max out at 100-500 connections
Result: Connection pool exhaustion, errors

Solutions:
1. RDS Proxy (AWS): Connection multiplexing
2. HTTP-based databases (DynamoDB, FaunaDB, Planetscale)
3. Connection pooling libraries with TTL
```

**Monitoring is harder, not easier:**
Serverless doesn't mean "no ops." You still need:
- Distributed tracing (AWS X-Ray, Datadog APM)
- Structured logging with correlation IDs
- Custom metrics for business logic
- Alerting on error rates, duration p99, throttling

The difference: Instead of monitoring 5 servers, you're monitoring 50 functions across 20 event sources. Complexity shifts from infrastructure to architecture.

**The lock-in calculation:**
```
Switching cost = (Lines of provider-specific code) * (Engineer hours to rewrite)

If you have:
- 50K lines of code
- 20% AWS-specific (S3, DynamoDB, SQS, Step Functions)
- 10K lines to rewrite
- 1 engineer can port 500 lines/day
- 20 days * 8 hours = 160 engineer hours
- At $200/hour fully loaded = $32,000

Compare to:
- AWS bill: $5,000/month
- Potential savings on GCP: $1,000/month
- Break-even: 32 months

Usually not worth it. Lock-in is a business decision, not a technical one.
```

**Serverless for startups vs enterprises:**

*Startups:* Serverless is often the right default. Zero servers to manage, scales automatically, pay only for usage. Ship features faster. Accept vendor lock-in as a reasonable trade-off.

*Enterprises:* Serverless for specific use cases (event processing, APIs with spiky traffic, glue code). Keep core business logic in containers/VMs for cost predictability and control at scale. Use serverless for 20% of workloads, containers for 80%.

**The cold start distribution:**
```
Cold starts happen when:
1. First invocation ever: 100% cold
2. Traffic spike: ~10-30% cold (new containers)
3. Deployment: 100% cold for first requests
4. After idle period: 100% cold (AWS reclaims after ~15 min)

If you serve 1M requests/day evenly:
- ~12 requests/second average
- Probably 2-5 warm containers
- Cold start rate: ~1-5%

If you serve 1M requests/day with spikes:
- 100 requests/second peak
- Cold start rate: ~10-20% during spikes
```

