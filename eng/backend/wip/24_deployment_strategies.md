# Chapter 24: Deployment Strategies

## Why Deployment Strategy Matters

Without a deployment strategy:

```
Your team deploys new code at 2 PM
Bug in production: Users can't checkout
Rollback attempt: "Which version was running before?"
Meanwhile: $50,000 revenue lost per minute
Outcome: 45 minutes to identify, rollback, and restore service
```

The deployment is the most dangerous moment in a service's lifecycle. A bad deployment strategy means:
- **User-facing outages** during deploys
- **Inability to rollback** quickly when things go wrong
- **All-or-nothing releases** coupling deployment with feature releases
- **Database migration failures** causing data corruption

Good deployment strategies minimize risk, enable fast rollback, and decouple deployment from feature releases.

---

## Rolling Deployment

### The Problem

Replacing all servers at once means downtime. You need a way to gradually replace old servers with new ones while maintaining service availability.

### How It Works

```
Start:     [v1] [v1] [v1] [v1]  ← All running version 1
           └──┘
Step 1:    [v2] [v1] [v1] [v1]  ← Deploy to first server
            ✓   └──┘
Step 2:    [v2] [v2] [v1] [v1]  ← Deploy to second server
            ✓    ✓   └──┘
Step 3:    [v2] [v2] [v2] [v1]  ← Deploy to third server
            ✓    ✓    ✓   └──┘
Done:      [v2] [v2] [v2] [v2]  ← All running version 2
```

### Implementation

```python
from typing import List
import time
import subprocess

class RollingDeployer:
    def __init__(self, servers: List[str], health_check_url: str):
        self.servers = servers
        self.health_check_url = health_check_url
        self.batch_size = max(1, len(servers) // 4)  # 25% at a time
        self.health_check_timeout = 30

    def deploy(self, version: str):
        """Deploy new version using rolling strategy"""
        for i in range(0, len(self.servers), self.batch_size):
            batch = self.servers[i:i + self.batch_size]
            print(f"Deploying batch {i//self.batch_size + 1}: {batch}")

            for server in batch:
                # Remove from load balancer
                self._drain_server(server)

                # Deploy new version
                success = self._deploy_to_server(server, version)
                if not success:
                    raise DeploymentError(f"Failed to deploy to {server}")

                # Wait for health checks
                if not self._wait_for_healthy(server):
                    # Rollback this server
                    self._rollback_server(server)
                    raise DeploymentError(f"Health check failed for {server}")

                # Add back to load balancer
                self._enable_server(server)

            # Wait between batches
            time.sleep(10)

    def _drain_server(self, server: str):
        """Remove server from load balancer rotation"""
        subprocess.run([
            "aws", "elbv2", "deregister-targets",
            "--target-group-arn", self.target_group_arn,
            "--targets", f"Id={server}"
        ], check=True)

        # Wait for existing connections to drain
        time.sleep(30)

    def _deploy_to_server(self, server: str, version: str) -> bool:
        """Deploy new version to server"""
        try:
            # SSH and run deployment script
            subprocess.run([
                "ssh", f"deploy@{server}",
                f"docker pull myapp:{version} && "
                f"docker stop myapp && "
                f"docker run -d --name myapp myapp:{version}"
            ], check=True, timeout=300)
            return True
        except subprocess.CalledProcessError:
            return False

    def _wait_for_healthy(self, server: str) -> bool:
        """Wait for server to pass health checks"""
        import requests
        url = f"http://{server}{self.health_check_url}"
        deadline = time.time() + self.health_check_timeout

        while time.time() < deadline:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    # Check several times to be sure
                    time.sleep(2)
                    if requests.get(url, timeout=5).status_code == 200:
                        return True
            except requests.RequestException:
                pass
            time.sleep(5)

        return False

    def _enable_server(self, server: str):
        """Add server back to load balancer"""
        subprocess.run([
            "aws", "elbv2", "register-targets",
            "--target-group-arn", self.target_group_arn,
            "--targets", f"Id={server}"
        ], check=True)

# Usage
deployer = RollingDeployer(
    servers=["10.0.1.10", "10.0.1.11", "10.0.1.12", "10.0.1.13"],
    health_check_url="/health"
)
deployer.deploy(version="v2.3.1")
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Risk | Gradual rollout limits blast radius | Mixed versions running simultaneously |
| Rollback | Can stop mid-deployment | Must roll forward or backward across servers |
| Resources | No extra capacity needed | Deployment takes longer |
| Compatibility | Must support two versions running | Requires backward compatibility |

**When to use:** Standard deployments with stateless services, when you can't afford extra infrastructure.

**When NOT to use:** Breaking database changes, services with complex state synchronization, urgent hotfixes (too slow).

---

## Blue-Green Deployment

### The Problem

Rolling deployments are slow and involve running mixed versions. You want instant switchover and instant rollback capability.

### How It Works

```
1. Current State (Blue is live)
┌─────────────────┐    ┌─────────────────┐
│  Blue Env (v1)  │◄───│  Load Balancer  │◄── Users
│  ████████████   │    │   Points to     │
│  [Live Traffic] │    │     BLUE        │
└─────────────────┘    └─────────────────┘

┌─────────────────┐
│  Green Env (v1) │
│  (Idle)         │
└─────────────────┘

2. Deploy to Green
┌─────────────────┐    ┌─────────────────┐
│  Blue Env (v1)  │◄───│  Load Balancer  │◄── Users
│  ████████████   │    │   Points to     │
│  [Live Traffic] │    │     BLUE        │
└─────────────────┘    └─────────────────┘

┌─────────────────┐
│  Green Env (v2) │ ← Deploy & test here
│  [Testing]      │
└─────────────────┘

3. Switch Traffic (instantaneous)
┌─────────────────┐    ┌─────────────────┐
│  Blue Env (v1)  │    │  Load Balancer  │◄── Users
│  (Idle)         │    │   Points to     │
└─────────────────┘    │     GREEN       │
                       └─────────────────┘
┌─────────────────┐               │
│  Green Env (v2) │◄──────────────┘
│  ████████████   │
│  [Live Traffic] │
└─────────────────┘
```

### Implementation

```python
class BlueGreenDeployer:
    def __init__(self, blue_target_group: str, green_target_group: str,
                 listener_arn: str):
        self.blue_tg = blue_target_group
        self.green_tg = green_target_group
        self.listener_arn = listener_arn
        self.current_live = self._get_current_live()

    def deploy(self, version: str):
        """Deploy using blue-green strategy"""
        # Determine which environment is currently live
        if self.current_live == "blue":
            deploy_to = "green"
            deploy_tg = self.green_tg
            switch_to_tg = self.green_tg
        else:
            deploy_to = "blue"
            deploy_tg = self.blue_tg
            switch_to_tg = self.blue_tg

        print(f"Deploying {version} to {deploy_to} environment")

        # Deploy to inactive environment
        servers = self._get_target_group_servers(deploy_tg)
        for server in servers:
            self._deploy_to_server(server, version)

        # Run smoke tests on new environment
        if not self._run_smoke_tests(deploy_tg):
            raise DeploymentError("Smoke tests failed on new environment")

        # Switch traffic
        print(f"Switching traffic to {deploy_to}")
        self._switch_traffic(switch_to_tg)

        # Monitor for errors
        time.sleep(60)  # Warm-up period
        if not self._check_metrics():
            print("Metrics show problems, rolling back")
            self.rollback()
            raise DeploymentError("Deployment failed metrics check")

        print("Deployment successful")
        self.current_live = deploy_to

    def rollback(self):
        """Instant rollback by switching traffic back"""
        rollback_tg = self.blue_tg if self.current_live == "green" else self.green_tg
        print(f"Rolling back to {rollback_tg}")
        self._switch_traffic(rollback_tg)

    def _switch_traffic(self, target_group_arn: str):
        """Atomic traffic switch at load balancer"""
        import boto3
        elbv2 = boto3.client('elbv2')

        elbv2.modify_listener(
            ListenerArn=self.listener_arn,
            DefaultActions=[{
                'Type': 'forward',
                'TargetGroupArn': target_group_arn
            }]
        )

    def _run_smoke_tests(self, target_group_arn: str) -> bool:
        """Run automated tests against new environment"""
        servers = self._get_target_group_servers(target_group_arn)
        test_url = f"http://{servers[0]}"

        tests = [
            ("/health", 200),
            ("/api/version", 200),
            ("/api/users/1", 200),
        ]

        for path, expected_status in tests:
            response = requests.get(f"{test_url}{path}")
            if response.status_code != expected_status:
                print(f"Test failed: {path} returned {response.status_code}")
                return False

        return True

    def _check_metrics(self) -> bool:
        """Check error rates, latency after switch"""
        # Query CloudWatch or Datadog
        error_rate = self._get_metric("error_rate")
        latency_p99 = self._get_metric("latency_p99")

        return error_rate < 0.01 and latency_p99 < 500  # 1% errors, 500ms p99
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Switchover | Instant (seconds) | Requires 2x infrastructure |
| Rollback | Instant - just switch back | Must maintain both environments |
| Testing | Test in production-like env before switch | Database state can diverge |
| Risk | Very low - test before users see it | Database migrations are complex |

**When to use:** Critical services, when instant rollback is essential, when you can afford 2x capacity.

**When NOT to use:** Cost-constrained environments, stateful services with complex data, frequent deployments (expensive to maintain dual env).

---

## Canary Deployment

### The Problem

You want to test new code with real production traffic, but only expose a small percentage of users to risk.

### How It Works

```
1. Deploy canary (5% traffic)
┌─────────────────────────────────────────────┐
│         Load Balancer / Traffic Split       │
└─────────────────────────────────────────────┘
         │                           │
    5% traffic                  95% traffic
         │                           │
         ▼                           ▼
┌─────────────────┐      ┌─────────────────────────┐
│  Canary (v2)    │      │   Stable (v1)           │
│  █              │      │   ███████████████████   │
│  [1 server]     │      │   [19 servers]          │
└─────────────────┘      └─────────────────────────┘

2. Monitor metrics (error rate, latency, business metrics)

3. Gradually increase (if healthy)
   5% → 10% → 25% → 50% → 100%

4. Rollback if issues detected
```

### Implementation

```python
class CanaryDeployer:
    def __init__(self, stable_servers: List[str], canary_servers: List[str]):
        self.stable_servers = stable_servers
        self.canary_servers = canary_servers
        self.traffic_percentages = [5, 10, 25, 50, 100]
        self.bake_time_minutes = 15  # Monitor each stage

    def deploy(self, version: str):
        """Deploy using canary strategy"""
        # Deploy to canary servers
        print(f"Deploying {version} to canary servers")
        for server in self.canary_servers:
            self._deploy_to_server(server, version)

        # Gradually increase traffic
        for percentage in self.traffic_percentages:
            print(f"Routing {percentage}% traffic to canary")
            self._set_traffic_split(canary_percentage=percentage)

            # Bake and monitor
            if not self._monitor_and_wait(percentage):
                print(f"Canary failed at {percentage}% traffic")
                self.rollback()
                raise DeploymentError("Canary deployment failed")

            if percentage == 100:
                # Deploy to all stable servers
                print("Canary successful, deploying to all servers")
                for server in self.stable_servers:
                    self._deploy_to_server(server, version)

    def _set_traffic_split(self, canary_percentage: int):
        """Configure load balancer traffic split"""
        import boto3
        elbv2 = boto3.client('elbv2')

        stable_weight = 100 - canary_percentage

        elbv2.modify_listener(
            ListenerArn=self.listener_arn,
            DefaultActions=[{
                'Type': 'forward',
                'ForwardConfig': {
                    'TargetGroups': [
                        {
                            'TargetGroupArn': self.canary_tg_arn,
                            'Weight': canary_percentage
                        },
                        {
                            'TargetGroupArn': self.stable_tg_arn,
                            'Weight': stable_weight
                        }
                    ]
                }
            }]
        )

    def _monitor_and_wait(self, percentage: int) -> bool:
        """Monitor metrics during bake period"""
        import time
        deadline = time.time() + (self.bake_time_minutes * 60)

        while time.time() < deadline:
            # Compare canary metrics vs stable
            canary_metrics = self._get_metrics("canary")
            stable_metrics = self._get_metrics("stable")

            # Check for statistically significant differences
            if not self._metrics_healthy(canary_metrics, stable_metrics):
                return False

            time.sleep(30)  # Check every 30 seconds

        return True

    def _metrics_healthy(self, canary: dict, stable: dict) -> bool:
        """Compare canary metrics to stable baseline"""
        # Error rate should not be significantly higher
        if canary['error_rate'] > stable['error_rate'] * 1.5:
            print(f"Error rate too high: {canary['error_rate']} vs {stable['error_rate']}")
            return False

        # Latency should not be significantly higher
        if canary['latency_p99'] > stable['latency_p99'] * 1.3:
            print(f"Latency too high: {canary['latency_p99']} vs {stable['latency_p99']}")
            return False

        # Business metrics (conversion, etc)
        if canary.get('conversion_rate', 1) < stable.get('conversion_rate', 1) * 0.95:
            print(f"Conversion rate dropped: {canary.get('conversion_rate')} vs {stable.get('conversion_rate')}")
            return False

        return True

    def rollback(self):
        """Rollback by setting canary traffic to 0%"""
        print("Rolling back canary")
        self._set_traffic_split(canary_percentage=0)

# Usage with automated analysis
deployer = CanaryDeployer(
    stable_servers=["server1", "server2", "server3"],
    canary_servers=["canary1"]
)
deployer.deploy(version="v2.4.0")
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Risk | Minimal - only affects small % | Complex traffic routing needed |
| Detection | Real production data | Requires good metrics/monitoring |
| Rollback | Fast - reduce traffic to 0% | Some users saw broken version |
| Duration | Slow - gradual rollout | Prolonged mixed-version state |

**When to use:** High-risk changes, user-facing services, when you have good metrics and monitoring.

**When NOT to use:** Backend jobs (no traffic routing), when you lack metrics to detect issues, time-sensitive deployments.

---

## Feature Flags

### The Problem

Deployment and feature release are coupled. You want to deploy code to production but control when features become visible to users.

### How It Works

```
Code deployed to production
         │
         ▼
┌─────────────────────────────────────┐
│  if feature_flag("new_checkout"):   │
│      show_new_checkout()             │
│  else:                               │
│      show_old_checkout()             │
└─────────────────────────────────────┘
         │
         ▼
   Feature Flag Service
┌─────────────────────────────────┐
│  new_checkout:                  │
│    enabled: true                │
│    rollout: 10%                 │
│    rules:                       │
│      - user.plan == "beta"      │
│      - user.id % 100 < 10       │
└─────────────────────────────────┘
```

### Implementation

```python
from typing import Dict, Any, Callable
import hashlib

class FeatureFlags:
    def __init__(self):
        self.flags: Dict[str, FlagConfig] = {}
        self._load_flags_from_config()

    def is_enabled(self, flag_name: str, context: Dict[str, Any]) -> bool:
        """Check if feature flag is enabled for this context"""
        flag = self.flags.get(flag_name)
        if not flag:
            return False  # Default to disabled

        if not flag.enabled:
            return False

        # Check targeting rules
        if flag.rules:
            for rule in flag.rules:
                if self._evaluate_rule(rule, context):
                    return True
            return False  # No rules matched

        # Check percentage rollout
        if flag.rollout_percentage < 100:
            return self._in_rollout(flag_name, context, flag.rollout_percentage)

        return True

    def _evaluate_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate targeting rule"""
        # user.plan == "beta"
        if rule['type'] == 'equals':
            return context.get(rule['field']) == rule['value']

        # user.id in [1, 2, 3]
        if rule['type'] == 'in':
            return context.get(rule['field']) in rule['values']

        # user.created_at > "2024-01-01"
        if rule['type'] == 'greater_than':
            return context.get(rule['field']) > rule['value']

        return False

    def _in_rollout(self, flag_name: str, context: Dict[str, Any],
                    percentage: int) -> bool:
        """Consistent hash-based percentage rollout"""
        # Deterministic: same user always gets same result
        user_id = context.get('user_id', context.get('session_id', ''))
        hash_input = f"{flag_name}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = hash_value % 100
        return bucket < percentage

class FlagConfig:
    def __init__(self, name: str, enabled: bool = False,
                 rollout_percentage: int = 0, rules: list = None):
        self.name = name
        self.enabled = enabled
        self.rollout_percentage = rollout_percentage
        self.rules = rules or []

# Usage in application
flags = FeatureFlags()

def process_checkout(user):
    context = {
        'user_id': user.id,
        'plan': user.plan,
        'country': user.country
    }

    if flags.is_enabled('new_checkout_flow', context):
        return new_checkout_flow(user)
    else:
        return old_checkout_flow(user)

# Configuration (loaded from database or config service)
flags.flags['new_checkout_flow'] = FlagConfig(
    name='new_checkout_flow',
    enabled=True,
    rollout_percentage=10,  # 10% of users
    rules=[
        {'type': 'equals', 'field': 'plan', 'value': 'beta'},
        {'type': 'in', 'field': 'user_id', 'values': [1, 2, 3]}
    ]
)
```

### Feature Flag Patterns

```python
# 1. Kill Switch (emergency disable)
if not flags.is_enabled('recommendations_service', context):
    return []  # Disable broken feature

# 2. Gradual Rollout
# Day 1: 5% internal users
# Day 2: 10% all users
# Day 3: 25%
# Day 4: 50%
# Day 5: 100%

# 3. A/B Test
variant = flags.get_variant('checkout_button_color', context)
if variant == 'red':
    button_color = '#FF0000'
elif variant == 'green':
    button_color = '#00FF00'
else:
    button_color = '#0000FF'  # control

# 4. Ops Flag (operational control)
if flags.is_enabled('use_new_database', context):
    db = new_database
else:
    db = old_database
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Decoupling | Deploy != Release | Code complexity increases |
| Rollback | Instant (flip flag) | Technical debt (old code paths) |
| Testing | Test in production safely | Both paths must work |
| Targeting | User-specific rollouts | Flag management overhead |

**When to use:** Risky features, gradual rollouts, A/B testing, operational controls.

**When NOT to use:** Simple bug fixes, when code paths can't coexist, short-lived changes (use branches instead).

**Anti-pattern: Long-lived flags**
```python
# BAD - flags live for months/years
if flags.is_enabled('feature_from_2020'):
    # Dead code path, never removed
```

Remove flags after full rollout to reduce complexity.

---

## Database Migration Strategies

### The Problem

Code and database changes are interdependent. Deploying incompatible versions causes errors.

### Strategy 1: Expand-Contract

Three-phase migration for backwards compatibility.

```
Phase 1: EXPAND (Add new schema, keep old)
┌──────────────────────────────────────┐
│  Users Table                         │
├──────────────────────────────────────┤
│  id | name | phone | phone_number   │ ← NEW column
└──────────────────────────────────────┘
      Old code writes "phone"
      New code writes both "phone" and "phone_number"

Phase 2: MIGRATE DATA
Background job copies phone → phone_number

Phase 3: CONTRACT (Remove old schema)
┌──────────────────────────────────────┐
│  Users Table                         │
├──────────────────────────────────────┤
│  id | name | phone_number           │ ← old "phone" dropped
└──────────────────────────────────────┘
      All code uses "phone_number"
```

```python
# Migration 001: Expand - Add new column
def upgrade():
    op.add_column('users', sa.Column('phone_number', sa.String(20), nullable=True))

# Application code v2: Dual writes
class User(Base):
    id = Column(Integer, primary_key=True)
    name = Column(String)
    phone = Column(String)  # Old column
    phone_number = Column(String)  # New column

    def set_phone(self, value):
        self.phone = value  # Old code can still read
        self.phone_number = value  # New column populated

# Migration 002: Migrate data
def upgrade():
    op.execute("UPDATE users SET phone_number = phone WHERE phone_number IS NULL")

# Migration 003: Contract - Drop old column
def upgrade():
    op.drop_column('users', 'phone')
```

### Strategy 2: Dual Writes

Write to both old and new systems during migration.

```
┌──────────────────────────────────────────┐
│         Application Code                 │
└──────────────────────────────────────────┘
         │                 │
    Write to old      Write to new
         │                 │
         ▼                 ▼
┌─────────────┐    ┌─────────────┐
│  Old Table  │    │  New Table  │
│  (MySQL)    │    │  (Postgres) │
└─────────────┘    └─────────────┘
         │                 │
    Read from old     Compare/Verify
```

```python
class DualWriteRepository:
    def __init__(self, old_db, new_db):
        self.old_db = old_db
        self.new_db = new_db
        self.compare_reads = True  # Shadow mode

    def save_user(self, user):
        # Write to old system (source of truth)
        old_result = self.old_db.save(user)

        # Write to new system (shadow)
        try:
            new_result = self.new_db.save(user)

            # Compare results in background
            if self.compare_reads:
                self._compare_async(old_result, new_result)
        except Exception as e:
            # Log but don't fail - new system is not critical yet
            logger.error(f"New system write failed: {e}")

        return old_result

    def get_user(self, user_id):
        # Read from old system (source of truth)
        old_user = self.old_db.get(user_id)

        # Shadow read from new system
        try:
            new_user = self.new_db.get(user_id)

            # Compare in background
            if self.compare_reads:
                self._compare_async(old_user, new_user)
        except Exception as e:
            logger.error(f"New system read failed: {e}")

        return old_user

    def _compare_async(self, old_data, new_data):
        """Async comparison to detect discrepancies"""
        # Queue for background processing
        comparison_queue.enqueue({
            'old': old_data,
            'new': new_data,
            'timestamp': time.time()
        })
```

**Migration phases:**
1. Dual write: Write to both, read from old
2. Dual read: Write to both, read from both (compare)
3. Switch: Write to both, read from new
4. Cleanup: Write to new only, drop old

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Risk | Can validate new system before switch | Double write cost |
| Rollback | Easy - just read from old | Data inconsistency possible |
| Validation | Compare results before full switch | Complex application code |

---

## Rollback Strategies

### Automated Rollback

```python
class AutoRollback:
    def __init__(self, deployer, monitors):
        self.deployer = deployer
        self.monitors = monitors
        self.rollback_threshold = {
            'error_rate': 0.05,  # 5%
            'latency_p99': 1000,  # 1 second
            'cpu_usage': 0.90,    # 90%
        }

    def deploy_with_auto_rollback(self, version: str):
        """Deploy with automatic rollback on issues"""
        previous_version = self.deployer.current_version

        try:
            # Deploy new version
            self.deployer.deploy(version)

            # Monitor for issues
            print("Monitoring deployment for 5 minutes...")
            for minute in range(5):
                time.sleep(60)

                metrics = self.monitors.get_current_metrics()

                if not self._metrics_healthy(metrics):
                    print(f"Unhealthy metrics detected: {metrics}")
                    raise UnhealthyDeployment("Metrics exceeded threshold")

                print(f"Minute {minute + 1}: OK")

            print("Deployment successful and stable")

        except Exception as e:
            print(f"Deployment failed: {e}")
            print(f"Auto-rolling back to {previous_version}")
            self.deployer.rollback(previous_version)
            raise

    def _metrics_healthy(self, metrics: dict) -> bool:
        """Check if metrics are within thresholds"""
        for metric, threshold in self.rollback_threshold.items():
            if metrics.get(metric, 0) > threshold:
                return False
        return True

# Advanced: Circuit breaker for rollback
class DeploymentCircuitBreaker:
    def __init__(self):
        self.consecutive_failures = 0
        self.failure_threshold = 3
        self.blocked_until = None

    def can_deploy(self) -> bool:
        """Check if deployments are blocked due to failures"""
        if self.blocked_until and time.time() < self.blocked_until:
            return False
        return True

    def record_success(self):
        self.consecutive_failures = 0
        self.blocked_until = None

    def record_failure(self):
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            # Block deployments for 1 hour
            self.blocked_until = time.time() + 3600
            alert_ops_team("Deployments blocked due to repeated failures")
```

---

## Deployment Strategy Comparison

| Strategy | Switchover Speed | Rollback Speed | Extra Resources | Risk Level | Best For |
|----------|------------------|----------------|-----------------|------------|----------|
| Rolling | Gradual (10-60min) | Medium (re-roll) | None | Medium | Standard deploys |
| Blue-Green | Instant (seconds) | Instant | 2x capacity | Low | Critical services |
| Canary | Gradual (hours) | Fast (reduce %) | ~10% extra | Very Low | High-risk changes |
| Feature Flags | N/A | Instant | None | Very Low | Decoupled releases |

### Decision Tree

```
Do you need instant rollback?
├─ Yes: Blue-Green
└─ No
   └─ Is this a high-risk change?
      ├─ Yes
      │  └─ Can you monitor user impact?
      │     ├─ Yes: Canary
      │     └─ No: Blue-Green
      └─ No
         └─ Can you afford 2x capacity?
            ├─ Yes: Blue-Green
            └─ No: Rolling
```

For most cases: Use **Rolling** for deployments + **Feature Flags** for releases.

---

## Key Concepts Checklist

- [ ] Choose deployment strategy based on risk tolerance and resources
- [ ] Implement proper health checks that validate application functionality
- [ ] Design database migrations to be backwards compatible (expand-contract)
- [ ] Set up automated rollback based on metrics (error rate, latency)
- [ ] Decouple deployment from feature release using feature flags
- [ ] Test rollback procedures regularly (don't wait for production incident)
- [ ] Monitor key metrics during deployment (errors, latency, business KPIs)
- [ ] Handle stateful services specially (sessions, WebSockets, long-running jobs)

---

## Practical Insights

**Health checks must validate real functionality:**
Don't just check if the process is running. Health checks should validate database connectivity, downstream service availability, and core functionality. A service that returns 200 but can't access the database will cause an outage.

```python
# BAD
@app.get("/health")
def health():
    return {"status": "ok"}

# GOOD
@app.get("/health")
def health():
    # Check database
    db.execute("SELECT 1")
    # Check cache
    redis.ping()
    # Check critical dependency
    requests.get(downstream_service + "/health", timeout=2)
    return {"status": "ok"}
```

**Deployment frequency matters more than strategy:**
Companies deploying 10x/day have fundamentally different needs than those deploying weekly. High-frequency deployments favor simpler strategies (rolling) with extensive automation. Low-frequency deployments can afford more complex strategies (blue-green) with manual validation steps.

**Database migrations are the hard part:**
Most deployment strategy complexity comes from database changes. Rule of thumb: any schema change requires at least 2 deployments (expand, then contract). Never deploy breaking database changes atomically with code. The expand-contract pattern isn't elegant, but it's the only safe way to maintain zero-downtime.

**Rollback is a deployment:**
Rollback isn't a magic undo button - it's deploying the previous version, which means it has all the risks of a deployment. This is why instant rollback strategies (blue-green, feature flags) are valuable: they're not actually deploying anything, just changing routing.

**Feature flags accumulate as technical debt:**
Every feature flag doubles your code paths (flag on, flag off). After 6 months, you have 2^n combinations to test. Be disciplined about removing flags after rollout completes. Set calendar reminders to clean up flags 2 weeks after 100% rollout. Otherwise, you'll have a codebase full of `if feature_flag('experiment_from_2020')` that nobody dares remove.

**Canary analysis requires statistical rigor:**
Don't just eyeball metrics. A 10% canary means small sample size, which means high variance. Use statistical significance tests or wait for enough samples. False positives (rolling back good deployments) are expensive. False negatives (not catching bad deployments) are catastrophic. Err on the side of false positives.

**State makes everything harder:**
WebSocket connections, in-progress uploads, active sessions - all make switchover complex. Blue-green with instant switchover will drop WebSocket connections. Rolling deployment means some users switch mid-session. Solutions: session affinity (sticky sessions), graceful connection draining (30-60 second wait), or state externalization (Redis for sessions).
