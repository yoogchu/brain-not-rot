# Chapter 23: Chaos Engineering & Fault Tolerance

## Why Break Things On Purpose?

Your production system runs perfectly for 6 months. Then:

```
3:17 AM Saturday: DNS server crashes
3:18 AM: Auto-failover... doesn't work (never tested)
3:19 AM: Database connections pile up (no timeout configured)
3:22 AM: Cascading failure across all services
3:45 AM: Complete outage, 2M users affected

Root cause: DNS failover config had typo, discovered during real incident
```

**The problem:** Most systems work fine until they don't. You discover your failure modes during incidents, not before.

**Chaos engineering:** Intentionally inject failures into production to:
- Find weaknesses BEFORE they cause outages
- Validate assumptions about system resilience
- Build confidence in fault tolerance mechanisms
- Train teams to respond to incidents

---

## Chaos Engineering Principles

### The Netflix Origin Story

Netflix created the "Simian Army" to test AWS resilience:

```
Chaos Monkey: Randomly terminates instances
Chaos Gorilla: Takes down entire AWS availability zone
Latency Monkey: Introduces artificial delays
Chaos Kong: Simulates region failure

Philosophy: If your system can't survive a monkey randomly killing
instances, it's not production-ready.
```

### Core Methodology

```
┌─────────────────────────────────────────────────┐
│ 1. Define Steady State                          │
│    "Users can search and view products"         │
│    "99th percentile latency < 200ms"            │
└─────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ 2. Hypothesize: Steady state continues during   │
│    failure (e.g., "killing 1 instance doesn't   │
│    affect user experience")                     │
└─────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ 3. Inject Real-World Failure                    │
│    - Terminate instances                        │
│    - Inject latency                             │
│    - Corrupt network packets                    │
└─────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│ 4. Measure: Did steady state hold?              │
│    YES → System is resilient                    │
│    NO  → Found weakness, fix it                 │
└─────────────────────────────────────────────────┘
```

**Key insight:** Don't test in staging. Staging is too different from production. Test in production with controlled blast radius.

---

## Fault Injection Types

### 1. Resource Exhaustion

**The Problem:**
Your app works fine at 20% CPU, but what happens at 95%? Does it gracefully degrade or crash?

**How it works:**

```python
# CPU stress test
import multiprocessing
import time

def stress_cpu(duration_seconds):
    """Consume 100% of one CPU core"""
    end_time = time.time() + duration_seconds
    while time.time() < end_time:
        _ = sum(i * i for i in range(10000))

# Stress all cores for 60 seconds
def stress_test():
    processes = []
    for _ in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=stress_cpu, args=(60,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

# Memory exhaustion
def stress_memory(mb_to_consume):
    """Allocate memory until exhaustion"""
    data = []
    chunk_size = 1024 * 1024  # 1MB
    for _ in range(mb_to_consume):
        data.append(b'0' * chunk_size)
        time.sleep(0.1)  # Gradual increase

# Disk I/O stress
def stress_disk(file_path, duration_seconds):
    """Generate heavy disk I/O"""
    end_time = time.time() + duration_seconds
    while time.time() < end_time:
        with open(file_path, 'wb') as f:
            f.write(b'0' * (10 * 1024 * 1024))  # 10MB writes
        with open(file_path, 'rb') as f:
            _ = f.read()
```

**When to use:** Test autoscaling, resource limits, graceful degradation
**When NOT to use:** Systems with manual intervention requirements, no monitoring in place

### 2. Network Failures

**The Problem:**
Your microservices architecture assumes network calls always work. They don't.

**Types of network chaos:**

```
┌──────────────────────────────────────────────────┐
│                                                  │
│  Service A          Network Fault      Service B │
│                                                  │
│     │                                      │     │
│     ├─── Packet loss (5%) ────────────────┤     │
│     │                                      │     │
│     ├─── Latency (+500ms) ────────────────┤     │
│     │                                      │     │
│     ├─── Bandwidth limit (1Mbps) ─────────┤     │
│     │                                      │     │
│     ├─── Connection drop (reset) ─────────┤     │
│     │                                      │     │
│     ├─── DNS failure (NXDOMAIN) ──────────┤     │
│                                                  │
└──────────────────────────────────────────────────┘
```

**Implementation using tc (Traffic Control on Linux):**

```python
import subprocess
import time
from contextlib import contextmanager

class NetworkChaos:
    def __init__(self, interface="eth0"):
        self.interface = interface

    def add_latency(self, delay_ms, jitter_ms=0):
        """Add network latency"""
        cmd = f"tc qdisc add dev {self.interface} root netem delay {delay_ms}ms {jitter_ms}ms"
        subprocess.run(cmd, shell=True, check=True)

    def add_packet_loss(self, loss_percent):
        """Simulate packet loss"""
        cmd = f"tc qdisc add dev {self.interface} root netem loss {loss_percent}%"
        subprocess.run(cmd, shell=True, check=True)

    def add_bandwidth_limit(self, rate_kbps):
        """Limit bandwidth"""
        cmd = f"tc qdisc add dev {self.interface} root tbf rate {rate_kbps}kbit burst 32kbit latency 400ms"
        subprocess.run(cmd, shell=True, check=True)

    def clear(self):
        """Remove all network chaos"""
        subprocess.run(f"tc qdisc del dev {self.interface} root",
                      shell=True,
                      stderr=subprocess.DEVNULL)

@contextmanager
def inject_latency(delay_ms=100, duration_seconds=60):
    """Context manager for temporary latency injection"""
    chaos = NetworkChaos()
    try:
        print(f"Injecting {delay_ms}ms latency for {duration_seconds}s")
        chaos.add_latency(delay_ms)
        yield
        time.sleep(duration_seconds)
    finally:
        chaos.clear()
        print("Network chaos cleared")

# Usage
with inject_latency(delay_ms=500, duration_seconds=120):
    # Your application runs with 500ms latency
    run_load_test()
```

### 3. Process Failures

**The Problem:**
Processes crash. Kubernetes restarts them, but does your app handle the downtime?

```python
import random
import signal
import os

class ChaosMonkey:
    def __init__(self, kill_probability=0.01):
        """
        Kill probability: chance of termination per check
        0.01 = 1% chance per minute = ~9 kills/day
        """
        self.kill_probability = kill_probability

    def maybe_kill_self(self):
        """Randomly terminate current process"""
        if random.random() < self.kill_probability:
            print("Chaos Monkey strikes! Terminating...")
            os.kill(os.getpid(), signal.SIGKILL)

    def maybe_kill_process(self, pid):
        """Randomly terminate specific process"""
        if random.random() < self.kill_probability:
            print(f"Chaos Monkey killing process {pid}")
            os.kill(pid, signal.SIGKILL)

# Integration in your service
import time
import threading

def chaos_monkey_thread(interval_seconds=60):
    """Background thread that randomly kills the service"""
    monkey = ChaosMonkey(kill_probability=0.01)
    while True:
        time.sleep(interval_seconds)
        monkey.maybe_kill_self()

# Start chaos monkey in production (with feature flag!)
if os.getenv("CHAOS_MONKEY_ENABLED") == "true":
    threading.Thread(target=chaos_monkey_thread, daemon=True).start()
```

**Trade-offs:**
| Aspect | Pros | Cons |
|--------|------|------|
| Realism | Tests actual crash scenarios | Can cause real outages |
| Simplicity | Easy to implement | Hard to predict timing |
| Coverage | Finds restart bugs | May miss complex states |

**When to use:** Testing container orchestration, health checks, stateless services
**When NOT to use:** Stateful databases without replication, single points of failure

---

## Chaos Engineering Tools

### Tool Comparison

| Tool | Best For | Fault Types | Environment | Complexity |
|------|----------|-------------|-------------|------------|
| **Chaos Monkey** | AWS EC2 instance termination | Process, compute | AWS, Netflix OSS | Low |
| **Gremlin** | Enterprise with GUI, attack library | All types | Any cloud, on-prem | Medium |
| **LitmusChaos** | Kubernetes-native chaos | Container, network, storage | Kubernetes | Medium-High |
| **AWS FIS** | AWS-managed chaos experiments | AWS service failures | AWS only | Low |
| **Toxiproxy** | Network-level proxy failures | Network latency, timeouts | Any | Low |

### Chaos Monkey

**Philosophy:** Kill instances randomly to ensure auto-healing works

```yaml
# chaos-monkey-config.yml
enabled: true
schedule:
  - cron: "0 10-16 * * MON-FRI"  # Weekdays, business hours
probability: 0.2  # 20% of instances eligible for termination
blacklist:
  - database-*
  - auth-service
whitelist:
  - web-server-*
  - api-gateway-*
```

**Pros:**
- Forces you to build resilient infrastructure
- Simple concept, battle-tested at Netflix
- Proves auto-scaling works

**Cons:**
- Only does one thing (instance termination)
- No network or resource chaos
- Requires mature infrastructure

### Gremlin

**Philosophy:** Comprehensive attack library with safety controls

```python
# Example Gremlin API usage
import gremlin

# CPU attack
attack = gremlin.attacks.ResourceAttack(
    type="cpu",
    cores=2,
    percent=80,
    length=300  # 5 minutes
)

# Target specific containers
target = gremlin.targets.Container(
    labels={"app": "payment-service"}
)

# Run with safety controls
experiment = gremlin.Experiment(
    attack=attack,
    target=target,
    hypothesis="Payment service handles high CPU gracefully"
)

# Halt conditions (automatic rollback)
experiment.add_halt_condition(
    metric="error_rate",
    threshold=0.05,  # Stop if errors > 5%
    duration=60      # Over 60 seconds
)

experiment.run()
```

**Pros:**
- Comprehensive attack library
- Excellent GUI and reporting
- Built-in safety controls and blast radius limiting
- Enterprise support

**Cons:**
- Commercial product (paid)
- Can be overkill for simple use cases

### LitmusChaos

**Philosophy:** Kubernetes-native chaos engineering

```yaml
# pod-delete-experiment.yaml
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: payment-chaos
spec:
  appinfo:
    appns: production
    applabel: "app=payment-service"
  engineState: active
  chaosServiceAccount: litmus-admin
  experiments:
    - name: pod-delete
      spec:
        components:
          env:
            - name: TOTAL_CHAOS_DURATION
              value: "60"
            - name: CHAOS_INTERVAL
              value: "10"  # Delete pod every 10s
            - name: FORCE
              value: "false"  # Graceful termination
```

**Network chaos in Kubernetes:**

```yaml
# network-latency-experiment.yaml
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: network-chaos
spec:
  appinfo:
    appns: production
    applabel: "app=api-gateway"
  experiments:
    - name: pod-network-latency
      spec:
        components:
          env:
            - name: NETWORK_LATENCY
              value: "2000"  # 2 second delay
            - name: JITTER
              value: "500"   # +/- 500ms
            - name: TARGET_CONTAINER
              value: "api-gateway"
```

**Pros:**
- Native Kubernetes integration
- Declarative chaos experiments (GitOps)
- Open source, active community
- Chaos workflows for complex scenarios

**Cons:**
- Kubernetes-only
- Steeper learning curve
- Requires CRD installation

### AWS Fault Injection Simulator (FIS)

**Philosophy:** Managed chaos engineering for AWS services

```yaml
# aws-fis-template.json
{
  "description": "Test RDS failover",
  "targets": {
    "myDB": {
      "resourceType": "aws:rds:db",
      "resourceTags": {
        "Environment": "production"
      },
      "selectionMode": "ALL"
    }
  },
  "actions": {
    "RDSFailover": {
      "actionId": "aws:rds:reboot-db-instances",
      "parameters": {
        "forceFailover": "true"
      },
      "targets": {
        "DBInstances": "myDB"
      }
    }
  },
  "stopConditions": [
    {
      "source": "aws:cloudwatch:alarm",
      "value": "arn:aws:cloudwatch:...:alarm:HighErrorRate"
    }
  ]
}
```

**Pros:**
- Managed service, no infrastructure
- Deep AWS integration (RDS, ECS, EC2, etc.)
- Built-in CloudWatch integration for stop conditions
- IAM-based access control

**Cons:**
- AWS-only
- Limited to AWS service failures
- Less control than open-source tools

---

## Game Days

**The Problem:**
Having chaos tools is not enough. Teams need practice responding to incidents.

### What is a Game Day?

A scheduled event where you intentionally break production (with safeguards) and practice incident response.

```
┌─────────────────────────────────────────────────┐
│                 GAME DAY AGENDA                  │
├─────────────────────────────────────────────────┤
│ 09:00 - Brief team on scenario                  │
│ 09:15 - Inject failure (unknown to responders)  │
│ 09:20 - Oncall receives alert                   │
│ 09:25 - Team starts investigation                │
│ 10:00 - Mitigation completed                    │
│ 10:15 - Debrief: What went well/poorly          │
└─────────────────────────────────────────────────┘
```

### Game Day Scenarios

**Scenario 1: Database failover**
```
Inject: Kill primary database instance
Expected: Automatic promotion of replica
Reality: Often finds gaps in monitoring, runbooks
```

**Scenario 2: Dependency failure**
```
Inject: Block traffic to payment provider
Expected: Graceful degradation, retry logic works
Reality: Timeouts cascade, no circuit breaker
```

**Scenario 3: Regional outage**
```
Inject: Block all traffic to us-east-1
Expected: Traffic routes to us-west-2
Reality: DNS changes take 15 minutes, cache issues
```

### Game Day Checklist

```python
# game-day-checklist.py
from dataclasses import dataclass
from typing import List
from datetime import datetime

@dataclass
class GameDay:
    scenario: str
    hypothesis: str
    blast_radius: str
    participants: List[str]
    rollback_plan: str
    success_criteria: List[str]

    def pre_flight_check(self):
        """Run before starting game day"""
        checks = [
            "Rollback plan documented and tested",
            "All participants briefed",
            "Monitoring dashboards ready",
            "Incident channel created",
            "Executive stakeholders notified",
            "Customer support team on standby"
        ]
        print("Pre-flight checklist:")
        for check in checks:
            print(f"  [ ] {check}")

    def run(self):
        print(f"\n=== GAME DAY: {self.scenario} ===")
        print(f"Hypothesis: {self.hypothesis}")
        print(f"Blast radius: {self.blast_radius}")
        print(f"Starting at: {datetime.now()}")

        # Inject failure here
        print("\n[CHAOS INJECTION]")

    def debrief(self, mttr_minutes, root_cause, improvements):
        """Post-game day analysis"""
        print(f"\n=== DEBRIEF ===")
        print(f"MTTR (Mean Time To Recovery): {mttr_minutes} minutes")
        print(f"Root cause: {root_cause}")
        print("\nImprovements identified:")
        for i, improvement in enumerate(improvements, 1):
            print(f"  {i}. {improvement}")

# Example game day
game_day = GameDay(
    scenario="Primary database failure",
    hypothesis="System survives primary DB failure with <1min downtime",
    blast_radius="10% of user traffic (canary region)",
    participants=["oncall-engineer", "SRE-lead", "product-owner"],
    rollback_plan="Promote replica to primary using script: ./failover.sh",
    success_criteria=[
        "Automatic failover within 60 seconds",
        "No data loss",
        "User-facing errors < 0.1%"
    ]
)

game_day.pre_flight_check()
```

---

## Blast Radius Control

**The Problem:**
Chaos experiments can cause real outages if uncontrolled.

### Progressive Rollout

```
Step 1: Dev environment (safe to break)
  │
  ▼
Step 2: Staging (catch obvious issues)
  │
  ▼
Step 3: Production canary (1% traffic)
  │
  ▼ If metrics stable
Step 4: Production (10% traffic)
  │
  ▼ If metrics stable
Step 5: Production (50% traffic)
  │
  ▼ If metrics stable
Step 6: Production (100% traffic)
```

### Implementation

```python
import random
from enum import Enum

class BlastRadius(Enum):
    SINGLE_INSTANCE = "single"
    CANARY = "canary"  # 1-5%
    PARTIAL = "partial"  # 10-25%
    FULL = "full"  # 100%

class SafetyChaos:
    def __init__(self, total_instances, blast_radius: BlastRadius):
        self.total_instances = total_instances
        self.blast_radius = blast_radius

    def select_targets(self):
        """Select instances based on blast radius"""
        if self.blast_radius == BlastRadius.SINGLE_INSTANCE:
            count = 1
        elif self.blast_radius == BlastRadius.CANARY:
            count = max(1, int(self.total_instances * 0.05))
        elif self.blast_radius == BlastRadius.PARTIAL:
            count = int(self.total_instances * 0.25)
        else:  # FULL
            count = self.total_instances

        # Randomly select instances
        all_instances = list(range(self.total_instances))
        return random.sample(all_instances, count)

    def should_halt(self, error_rate, latency_p99):
        """Automatic halt conditions"""
        return (
            error_rate > 0.05 or      # 5% error rate
            latency_p99 > 5000        # 5 second p99
        )

# Usage
chaos = SafetyChaos(total_instances=100, blast_radius=BlastRadius.CANARY)
targets = chaos.select_targets()  # [3, 47, 89, 12, 55] - 5 random instances

# Monitor during experiment
if chaos.should_halt(error_rate=0.03, latency_p99=1200):
    rollback_experiment()
```

### Steady State Hypothesis

Before starting chaos, define what "normal" looks like:

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class SteadyState:
    metric_name: str
    check: Callable[[], float]
    min_value: float
    max_value: float

    def is_steady(self):
        """Check if metric is within expected range"""
        value = self.check()
        return self.min_value <= value <= self.max_value

# Define steady state
steady_states = [
    SteadyState(
        metric_name="Request success rate",
        check=lambda: get_success_rate(),
        min_value=0.999,  # 99.9%
        max_value=1.0
    ),
    SteadyState(
        metric_name="P99 latency",
        check=lambda: get_p99_latency(),
        min_value=0,
        max_value=500  # 500ms
    ),
    SteadyState(
        metric_name="Active connections",
        check=lambda: get_active_connections(),
        min_value=100,
        max_value=10000
    )
]

# Check before and during experiment
def verify_steady_state():
    for state in steady_states:
        if not state.is_steady():
            print(f"ALERT: {state.metric_name} out of range")
            return False
    return True

# Before chaos
assert verify_steady_state(), "System not in steady state, aborting"

# Inject chaos
inject_failure()

# During chaos (continuous monitoring)
import time
while experiment_running:
    if not verify_steady_state():
        rollback()
        break
    time.sleep(10)
```

---

## Resilience Patterns

### 1. Bulkheads

**The Problem:**
One failing component exhausts shared resources (thread pool, connections), cascading to healthy components.

**How it works:**

```
Without Bulkheads:
┌─────────────────────────────────────┐
│     Shared Thread Pool (10)         │
│                                     │
│  ████████ (8) → Slow Service A      │
│  ██ (2) → Fast Service B (starved)  │
└─────────────────────────────────────┘

With Bulkheads:
┌──────────────────┐  ┌──────────────┐
│ Pool A (5)       │  │ Pool B (5)   │
│  ████ → Service A│  │  ██ → Service B
│  Isolated        │  │  Protected   │
└──────────────────┘  └──────────────┘
```

**Implementation:**

```python
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

class Bulkhead:
    """Isolate resources per service"""
    def __init__(self, name, max_workers):
        self.name = name
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"bulkhead-{name}"
        )

    def execute(self, func, *args, **kwargs):
        """Execute function in isolated thread pool"""
        return self.executor.submit(func, *args, **kwargs)

# Create separate bulkheads for each service
bulkheads = {
    "user-service": Bulkhead("user-service", max_workers=10),
    "payment-service": Bulkhead("payment-service", max_workers=5),
    "recommendation": Bulkhead("recommendation", max_workers=20)
}

def call_service(service_name, func, *args):
    bulkhead = bulkheads[service_name]
    future = bulkhead.execute(func, *args)
    return future.result(timeout=5)

# Usage
try:
    # Even if payment-service is slow, user-service unaffected
    user_data = call_service("user-service", fetch_user, user_id)
    payment_data = call_service("payment-service", fetch_payment, user_id)
except TimeoutError:
    # Handle slow service without affecting others
    payment_data = None
```

**When to use:** Microservices architecture, shared resource pools
**When NOT to use:** Single-purpose services, resource constraints

### 2. Timeouts

**The Problem:**
Waiting indefinitely for slow service response locks up resources.

```python
import time
import signal
from contextlib import contextmanager

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager for function timeout"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")

    # Set signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore old handler and cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# Usage
try:
    with timeout(5):
        result = slow_external_api_call()
except TimeoutError:
    # Fall back or return cached data
    result = get_cached_result()
```

**Timeout values by layer:**
```
Client timeout: 10 seconds
├── Load balancer: 9 seconds
    ├── API Gateway: 8 seconds
        ├── Service A: 5 seconds
            ├── Database: 3 seconds
```

**Rule:** Each layer should have progressively shorter timeout.

### 3. Retries with Exponential Backoff

**The Problem:**
Immediate retries during outage hammer already-struggling service.

```python
import time
import random
from functools import wraps

def retry_with_backoff(
    max_retries=3,
    base_delay=1,
    max_delay=60,
    exponential_base=2,
    jitter=True
):
    """Decorator for retry with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise  # Final attempt failed

                    # Calculate delay: base * (exponential_base ^ attempt)
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)

        return wrapper
    return decorator

# Usage
@retry_with_backoff(max_retries=5, base_delay=1, max_delay=60)
def fetch_from_api(url):
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()

# Retry attempts:
# Attempt 1: Immediate
# Attempt 2: Wait ~1 second
# Attempt 3: Wait ~2 seconds
# Attempt 4: Wait ~4 seconds
# Attempt 5: Wait ~8 seconds
```

**When to use:** Transient failures (network blips, rate limits)
**When NOT to use:** Client errors (4xx), validation errors

### 4. Graceful Degradation

**The Problem:**
When dependency fails, entire feature breaks instead of partial functionality.

```python
class FeatureFlags:
    """Feature degradation based on dependency health"""
    def __init__(self):
        self.features = {
            "personalized_recommendations": True,
            "real_time_inventory": True,
            "high_res_images": True,
            "product_reviews": True
        }

    def degrade_features(self, failing_service):
        """Disable features that depend on failing service"""
        degradation_map = {
            "recommendation-service": ["personalized_recommendations"],
            "inventory-service": ["real_time_inventory"],
            "cdn": ["high_res_images"],
            "review-service": ["product_reviews"]
        }

        for feature in degradation_map.get(failing_service, []):
            self.features[feature] = False
            print(f"Degraded: {feature}")

    def is_enabled(self, feature_name):
        return self.features.get(feature_name, False)

# Product page rendering
flags = FeatureFlags()

def render_product_page(product_id):
    page = {"product": get_product_basic_info(product_id)}

    # Core feature: Always show
    page["price"] = get_price(product_id)

    # Degradable: Personalized recommendations
    if flags.is_enabled("personalized_recommendations"):
        try:
            page["recommendations"] = get_recommendations(product_id)
        except ServiceError:
            flags.degrade_features("recommendation-service")
            page["recommendations"] = get_popular_products()  # Fallback

    # Degradable: High-res images
    if flags.is_enabled("high_res_images"):
        page["images"] = get_high_res_images(product_id)
    else:
        page["images"] = get_thumbnail_images(product_id)

    return page
```

**Degradation levels:**
```
Level 0: Full functionality
  ↓ Recommendation service fails
Level 1: Show popular products instead of personalized
  ↓ Inventory service fails
Level 2: Show cached inventory (stale but functional)
  ↓ Payment service fails
Level 3: Accept orders, process later (write to queue)
  ↓ Database read replicas fail
Level 4: Read from primary only (slower but works)
```

---

## Chaos Engineering Comparison

| Aspect | Chaos Monkey | Gremlin | LitmusChaos | AWS FIS |
|--------|--------------|---------|-------------|---------|
| **Scope** | Instance termination | All fault types | Kubernetes chaos | AWS services |
| **Ease of use** | Very simple | GUI-friendly | Moderate | Simple |
| **Cost** | Free (OSS) | Paid | Free (OSS) | Pay per experiment |
| **Best for** | AWS EC2 basic chaos | Enterprise teams | K8s-native apps | AWS-heavy workloads |
| **Learning curve** | 1 hour | 1 day | 1 week | 2 hours |
| **Safety features** | Limited | Excellent | Good | Good |

---

## Key Concepts Checklist

- [ ] Explain steady-state hypothesis and why it matters
- [ ] Design chaos experiment with controlled blast radius
- [ ] Implement resilience patterns (bulkheads, timeouts, retries)
- [ ] Choose appropriate chaos tool for environment
- [ ] Run Game Days to practice incident response
- [ ] Know when NOT to run chaos (during incidents, deployments, holidays)
- [ ] Set up automatic halt conditions for experiments
- [ ] Build organizational buy-in through incremental adoption

---

## Practical Insights

**Start small, build confidence:**
Start in dev environment, then staging, then production canary. Don't jump straight to production-wide chaos. Early wins build organizational buy-in.

**Chaos during business hours:**
Run experiments when engineers are available to respond. Netflix runs Chaos Monkey only during business hours on weekdays. 3 AM Saturday chaos experiments are resume-generating events.

**Feature flag your chaos:**
```python
if os.getenv("CHAOS_ENABLED") == "true" and is_canary_instance():
    inject_chaos()
```
Never hardcode chaos. Always gate behind feature flags for instant killswitch.

**Measure MTTR, not MTBF:**
Mean Time To Recovery matters more than Mean Time Between Failures. Failures will happen. Can you recover in 5 minutes or 5 hours?

**Blameless postmortems:**
When chaos experiments find issues, celebrate the discovery. If you blame engineers for writing "bad" code, they'll resist chaos engineering. Focus on systemic improvements, not individual fault.

**Automate verification:**
```python
# Don't manually check dashboards
assert error_rate < 0.01, "Error rate too high"
assert latency_p99 < 500, "Latency too high"
```
Automated checks let you scale chaos experiments without human monitoring.

**Know when NOT to run chaos:**
- During active incidents (you're already in chaos)
- During major deployments (too many variables)
- During holidays or low-staffing periods
- When blast radius can't be controlled
- When rollback plan doesn't exist

**Document runbooks from Game Days:**
Every Game Day should produce updated runbooks. "We learned the failover script doesn't work" → Update the script and document the new procedure.

**Integration with CI/CD:**
```yaml
# .github/workflows/chaos.yml
name: Chaos Testing
on:
  schedule:
    - cron: '0 10 * * TUE'  # Every Tuesday 10 AM
jobs:
  chaos:
    runs-on: ubuntu-latest
    steps:
      - name: Run pod-delete chaos
        run: kubectl apply -f chaos-experiments/pod-delete.yaml
      - name: Verify steady state
        run: ./scripts/verify-metrics.sh
```
Automated chaos in CI/CD catches regressions in resilience.
