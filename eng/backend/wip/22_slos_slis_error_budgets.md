# Chapter 22: SLOs, SLIs, and Error Budgets

## Why Measuring Reliability Matters

Your payment service has been "working fine" for months:

```
Scenario 1: No SLOs
Dev: "The service is up!"
On-call: *paged at 2 AM for 99.8% availability*
PM: "Why are we investing in reliability?"

Scenario 2: With SLOs
SLO: 99.9% availability = 43 minutes downtime/month
Error budget: 43 minutes remaining
Last incident: 15 minutes → 28 minutes left

Now you can answer:
- Should we roll back? (Check error budget)
- Can we deploy Friday? (Check error budget)
- Invest in reliability or features? (Data-driven decision)
```

Without SLOs:
- **Subjective arguments** about reliability vs velocity
- **Arbitrary alerts** (CPU > 80%? Why 80%?)
- **No data** to justify reliability investments
- **Burnout** from alert fatigue on meaningless thresholds

With SLOs:
- **Objective reliability targets** tied to user experience
- **Quantified risk** for each deployment
- **Clear communication** between eng and product
- **Focused alerts** on what actually impacts users

---

## SLI (Service Level Indicator)

### The Problem

You can't improve what you can't measure. But what should you measure?

```
Bad metrics:
- CPU usage 45% ← Doesn't tell you if users are happy
- Database connections: 127 ← Internal implementation detail
- Memory: 8.2 GB ← Meaningless without user impact

Good SLIs:
- Request success rate: 99.95% ← User gets correct response
- Request latency P99: 245ms ← User experiences fast response
- Data freshness: < 5 minutes old ← User sees current data
```

**SLI = Quantitative measure of service level provided to users**

### How It Works

```
User makes request
         │
         ▼
┌─────────────────────────────────────────────────┐
│               Your Service                       │
│                                                  │
│  ┌──────────┐    ┌──────────┐   ┌──────────┐  │
│  │Measure   │    │Measure   │   │Measure   │  │
│  │Success/  │    │Latency   │   │Complete  │  │
│  │Failure   │    │          │   │ness      │  │
│  └──────────┘    └──────────┘   └──────────┘  │
└─────────────────────────────────────────────────┘
         │                │              │
         ▼                ▼              ▼
     SLI: 99.95%    SLI: 250ms     SLI: 100%
```

### Implementation

```python
from dataclasses import dataclass
from typing import List
import time

@dataclass
class RequestMetrics:
    timestamp: float
    success: bool
    latency_ms: float
    status_code: int

class SLITracker:
    def __init__(self):
        self.metrics: List[RequestMetrics] = []

    def record_request(self, success: bool, latency_ms: float, status_code: int):
        self.metrics.append(RequestMetrics(
            timestamp=time.time(),
            success=success,
            latency_ms=latency_ms,
            status_code=status_code
        ))

    def calculate_availability_sli(self, window_seconds=3600):
        """Calculate percentage of successful requests in window"""
        now = time.time()
        window_start = now - window_seconds

        recent = [m for m in self.metrics if m.timestamp > window_start]
        if not recent:
            return 1.0

        successful = sum(1 for m in recent if m.success)
        return successful / len(recent)

    def calculate_latency_sli(self, percentile=99, window_seconds=3600):
        """Calculate Nth percentile latency in window"""
        now = time.time()
        window_start = now - window_seconds

        recent = [m for m in self.metrics if m.timestamp > window_start]
        if not recent:
            return 0.0

        latencies = sorted([m.latency_ms for m in recent])
        index = int(len(latencies) * percentile / 100)
        return latencies[min(index, len(latencies) - 1)]

    def calculate_success_rate_by_status(self, window_seconds=3600):
        """Break down success rate by status code"""
        now = time.time()
        window_start = now - window_seconds

        recent = [m for m in self.metrics if m.timestamp > window_start]
        status_counts = {}

        for metric in recent:
            status_counts[metric.status_code] = status_counts.get(metric.status_code, 0) + 1

        total = len(recent)
        return {
            status: (count / total * 100)
            for status, count in status_counts.items()
        }

# Usage
tracker = SLITracker()

# Record requests
tracker.record_request(success=True, latency_ms=120, status_code=200)
tracker.record_request(success=False, latency_ms=5000, status_code=500)
tracker.record_request(success=True, latency_ms=95, status_code=200)

# Calculate SLIs
availability = tracker.calculate_availability_sli(window_seconds=3600)
p99_latency = tracker.calculate_latency_sli(percentile=99, window_seconds=3600)

print(f"Availability SLI: {availability * 100:.2f}%")
print(f"P99 Latency SLI: {p99_latency:.2f}ms")
```

### Common SLI Types

| SLI Type | Definition | Example | Good For |
|----------|------------|---------|----------|
| **Availability** | % requests that succeed | 99.95% of requests return 2xx/3xx | User-facing APIs |
| **Latency** | % requests faster than threshold | 99% of requests < 300ms | Interactive services |
| **Throughput** | Requests processed per second | 10,000 QPS sustained | Batch processing |
| **Correctness** | % results that are accurate | 99.99% of reads return correct data | Data consistency |
| **Freshness** | Data age | 95% of data < 5 minutes old | Real-time systems |
| **Durability** | % data retained | 99.999999999% objects survive | Storage systems |

**When to use availability SLIs:** User-facing services where "works or doesn't work" is clear

**When NOT to use availability SLIs:** Background jobs, batch processing (use freshness/completeness instead)

---

## SLO (Service Level Objective)

### The Problem

You've measured SLIs. Now what? What's "good enough"?

```
Without SLO:
Engineer: "We had 3 minutes of downtime"
PM: "Is that bad?"
Engineer: "...depends?"

With SLO:
SLO: 99.9% availability = 43.2 minutes downtime/month allowed
Actual: 3 minutes downtime
Status: ✓ Within SLO, 40.2 minutes budget remaining
```

**SLO = Target value or range for an SLI**

### How It Works

```
SLI Measurement          SLO Target           Status
─────────────────────────────────────────────────────

Availability: 99.97%  →  SLO: ≥ 99.9%    →  ✓ GOOD
                                              (budget remaining)

Latency P99: 450ms    →  SLO: ≤ 300ms    →  ✗ VIOLATION
                                              (burned budget)

Freshness: 2 min      →  SLO: ≤ 5 min    →  ✓ GOOD
                                              (budget remaining)
```

### Implementation

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class SLOStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    VIOLATED = "violated"

@dataclass
class SLO:
    name: str
    description: str
    target: float  # e.g., 0.999 for 99.9%
    window_seconds: int  # Measurement window

    def check_compliance(self, sli_value: float) -> tuple[SLOStatus, dict]:
        """
        Check if current SLI meets SLO target

        Returns:
            (status, details)
        """
        # Calculate how much we're over/under target
        difference = sli_value - self.target

        # Define warning threshold (within 10% of error budget)
        warning_threshold = self.target * 0.9

        if sli_value >= self.target:
            status = SLOStatus.HEALTHY
        elif sli_value >= warning_threshold:
            status = SLOStatus.WARNING
        else:
            status = SLOStatus.VIOLATED

        details = {
            "target": self.target,
            "actual": sli_value,
            "difference": difference,
            "percentage_of_target": (sli_value / self.target * 100),
        }

        return status, details

    def calculate_error_budget(self, sli_value: float) -> dict:
        """Calculate remaining error budget"""
        # Error budget = 1 - target
        # e.g., 99.9% target = 0.1% error budget
        total_error_budget = 1 - self.target

        # How much budget have we used?
        errors = 1 - sli_value
        budget_consumed = errors / total_error_budget if total_error_budget > 0 else 0
        budget_remaining = 1 - budget_consumed

        return {
            "total_budget": total_error_budget,
            "consumed": budget_consumed,
            "remaining": budget_remaining,
            "remaining_percentage": budget_remaining * 100,
        }

# Example SLOs
availability_slo = SLO(
    name="API Availability",
    description="Percentage of successful API requests",
    target=0.999,  # 99.9%
    window_seconds=30 * 24 * 3600  # 30 days
)

latency_slo = SLO(
    name="API Latency P99",
    description="99th percentile latency under 300ms",
    target=300.0,  # 300ms - note: this is threshold-based, not percentage
    window_seconds=24 * 3600  # 24 hours
)

# Check compliance
current_availability = 0.9997  # 99.97%
status, details = availability_slo.check_compliance(current_availability)
budget = availability_slo.calculate_error_budget(current_availability)

print(f"Status: {status.value}")
print(f"Target: {details['target']*100:.2f}%")
print(f"Actual: {details['actual']*100:.2f}%")
print(f"Error budget consumed: {budget['consumed']*100:.1f}%")
print(f"Error budget remaining: {budget['remaining_percentage']:.1f}%")
```

### Setting Realistic SLOs

```python
def calculate_required_availability(downtime_minutes_per_month: float) -> float:
    """
    Calculate availability % from acceptable downtime

    Args:
        downtime_minutes_per_month: How much downtime users can tolerate

    Returns:
        Required availability as decimal (e.g., 0.999 for 99.9%)
    """
    minutes_per_month = 30 * 24 * 60  # ~43,200 minutes
    uptime_required = (minutes_per_month - downtime_minutes_per_month) / minutes_per_month
    return uptime_required

# Examples
print(f"99.9%  = {calculate_required_availability(43.2):.4f}")  # 43.2 min/month
print(f"99.95% = {calculate_required_availability(21.6):.5f}")  # 21.6 min/month
print(f"99.99% = {calculate_required_availability(4.32):.5f}")  # 4.32 min/month
```

**The Nines:**

| Availability | Downtime/Year | Downtime/Month | Downtime/Week |
|--------------|---------------|----------------|---------------|
| 90% (one nine) | 36.5 days | 3 days | 16.8 hours |
| 99% (two nines) | 3.65 days | 7.2 hours | 1.68 hours |
| 99.9% (three nines) | 8.76 hours | 43.2 minutes | 10.1 minutes |
| 99.95% | 4.38 hours | 21.6 minutes | 5.04 minutes |
| 99.99% (four nines) | 52.6 minutes | 4.32 minutes | 1.01 minutes |
| 99.999% (five nines) | 5.26 minutes | 25.9 seconds | 6.05 seconds |

**When to use 99.9%:** Most user-facing services, balances reliability and velocity

**When NOT to use 99.99%+:** Unless contractually required or life-critical. Cost grows exponentially, velocity drops.

---

## SLA (Service Level Agreement)

### The Problem

SLOs are internal targets. SLAs are external commitments with consequences.

```
SLO (Internal):
"We aim for 99.9% availability"
Miss it → Learn, improve, no penalty

SLA (External Contract):
"We guarantee 99.9% availability"
Miss it → Refund 10% of monthly fees
         → Potential customer churn
         → Legal/reputational risk
```

**SLA = Contractual commitment with penalties for violations**

### How It Works

```
┌─────────────────────────────────────────────────┐
│              SLA Structure                       │
│                                                  │
│  SLI: Availability                              │
│  SLO: ≥ 99.9%                                   │
│  SLA: ≥ 99.9% or customer gets credit          │
│                                                  │
│  Penalties:                                      │
│  ├─ 99.0-99.9%: 10% monthly credit              │
│  ├─ 95.0-99.0%: 25% monthly credit              │
│  └─ < 95.0%:    100% monthly credit             │
└─────────────────────────────────────────────────┘
```

### Implementation

```python
from dataclasses import dataclass
from typing import List

@dataclass
class SLAPenaltyTier:
    min_availability: float
    max_availability: float
    credit_percentage: float
    description: str

class SLA:
    def __init__(self, name: str, slo_target: float, penalty_tiers: List[SLAPenaltyTier]):
        self.name = name
        self.slo_target = slo_target
        self.penalty_tiers = sorted(penalty_tiers, key=lambda t: t.min_availability)

    def calculate_penalty(self, actual_availability: float, monthly_revenue: float) -> dict:
        """
        Calculate SLA penalty/credit based on actual performance

        Args:
            actual_availability: Measured availability (0.0 to 1.0)
            monthly_revenue: Monthly revenue from customer

        Returns:
            Dictionary with penalty details
        """
        # If we met SLA, no penalty
        if actual_availability >= self.slo_target:
            return {
                "penalty_triggered": False,
                "credit_percentage": 0,
                "credit_amount": 0,
                "tier": "Met SLA",
            }

        # Find applicable penalty tier
        for tier in self.penalty_tiers:
            if tier.min_availability <= actual_availability < tier.max_availability:
                credit_amount = monthly_revenue * tier.credit_percentage
                return {
                    "penalty_triggered": True,
                    "credit_percentage": tier.credit_percentage * 100,
                    "credit_amount": credit_amount,
                    "tier": tier.description,
                }

        # If below all tiers, apply maximum penalty
        max_tier = self.penalty_tiers[-1]
        return {
            "penalty_triggered": True,
            "credit_percentage": max_tier.credit_percentage * 100,
            "credit_amount": monthly_revenue * max_tier.credit_percentage,
            "tier": max_tier.description,
        }

# Example SLA
api_sla = SLA(
    name="API Availability SLA",
    slo_target=0.999,
    penalty_tiers=[
        SLAPenaltyTier(0.990, 0.999, 0.10, "Minor degradation"),
        SLAPenaltyTier(0.950, 0.990, 0.25, "Significant degradation"),
        SLAPenaltyTier(0.000, 0.950, 1.00, "Severe degradation"),
    ]
)

# Scenario: Actual availability was 99.7%
actual = 0.997
monthly_revenue = 10000

penalty = api_sla.calculate_penalty(actual, monthly_revenue)
print(f"Penalty triggered: {penalty['penalty_triggered']}")
print(f"Tier: {penalty['tier']}")
print(f"Credit: {penalty['credit_percentage']}% = ${penalty['credit_amount']:.2f}")
```

**SLO vs SLA Buffer:**

```
Best practice: Set SLA looser than SLO

SLO (Internal): 99.95%  ← What we aim for
SLA (External): 99.9%   ← What we promise

Buffer: 0.05% → Protects against edge cases
                → Room for experimentation
                → Avoids SLA violations from normal variance
```

**When to use SLAs:** Customer contracts, enterprise customers, regulated industries

**When NOT to use SLAs:** Internal services, early-stage products (use SLOs only)

---

## Error Budgets

### The Problem

Perfect reliability is impossible and counterproductive:

```
Scenario 1: No error budget
Every outage triggers:
- Emergency response
- Blame culture
- Fear of deploying
- Slow innovation

Scenario 2: With error budget
43 minutes/month budget
- Planned maintenance: 10 minutes
- Deployments (small risks): 15 minutes
- Unplanned incidents: 8 minutes
Total: 33 minutes → 10 minutes remaining
Status: Healthy, keep shipping
```

**Error Budget = Allowed unreliability = 1 - SLO**

### How It Works

```
SLO: 99.9% availability
Error Budget: 100% - 99.9% = 0.1%

In a 30-day month:
Total time: 43,200 minutes
Error budget: 43,200 × 0.001 = 43.2 minutes of downtime allowed

┌────────────────────────────────────────────────┐
│         Error Budget Tracking                  │
│                                                │
│  Budget: 43.2 minutes/month                   │
│                                                │
│  Consumed:                                     │
│  ├─ Incident on 3/1:  15 min                  │
│  ├─ Incident on 3/8:  8 min                   │
│  └─ Deploy on 3/15:   5 min                   │
│                                                │
│  Total consumed: 28 minutes (64.8%)           │
│  Remaining: 15.2 minutes (35.2%)              │
│                                                │
│  Status: ⚠ WARNING - Slow down deploys        │
└────────────────────────────────────────────────┘
```

### Implementation

```python
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ErrorBudgetEvent:
    timestamp: datetime
    duration_minutes: float
    description: str
    event_type: str  # "incident", "planned", "deploy"

class ErrorBudget:
    def __init__(self, slo_target: float, window_days: int = 30):
        """
        Args:
            slo_target: Target availability (e.g., 0.999 for 99.9%)
            window_days: Rolling window for budget calculation
        """
        self.slo_target = slo_target
        self.window_days = window_days
        self.events: List[ErrorBudgetEvent] = []

        # Calculate total budget
        total_minutes = window_days * 24 * 60
        error_rate = 1 - slo_target
        self.total_budget_minutes = total_minutes * error_rate

    def add_event(self, event: ErrorBudgetEvent):
        """Record an event that consumed error budget"""
        self.events.append(event)

    def get_consumed_budget(self, as_of: Optional[datetime] = None) -> dict:
        """Calculate error budget consumption in rolling window"""
        if as_of is None:
            as_of = datetime.now()

        window_start = as_of - timedelta(days=self.window_days)

        # Sum events in window
        consumed_minutes = sum(
            event.duration_minutes
            for event in self.events
            if window_start <= event.timestamp <= as_of
        )

        consumed_percentage = (consumed_minutes / self.total_budget_minutes) * 100
        remaining_minutes = self.total_budget_minutes - consumed_minutes
        remaining_percentage = 100 - consumed_percentage

        return {
            "total_budget_minutes": self.total_budget_minutes,
            "consumed_minutes": consumed_minutes,
            "consumed_percentage": consumed_percentage,
            "remaining_minutes": remaining_minutes,
            "remaining_percentage": remaining_percentage,
        }

    def get_policy_action(self) -> str:
        """Determine action based on budget remaining"""
        budget = self.get_consumed_budget()
        remaining_pct = budget["remaining_percentage"]

        if remaining_pct > 50:
            return "PROCEED: Plenty of budget, ship features aggressively"
        elif remaining_pct > 25:
            return "CAUTION: Moderate budget remaining, balanced approach"
        elif remaining_pct > 10:
            return "SLOW DOWN: Low budget, focus on reliability"
        else:
            return "FREEZE: Critical budget exhaustion, emergency reliability work only"

    def can_afford_risk(self, estimated_failure_minutes: float) -> bool:
        """Check if we can afford a risky change (e.g., deploy)"""
        budget = self.get_consumed_budget()
        return estimated_failure_minutes <= budget["remaining_minutes"]

# Usage
budget = ErrorBudget(slo_target=0.999, window_days=30)

# Record incidents
budget.add_event(ErrorBudgetEvent(
    timestamp=datetime.now() - timedelta(days=10),
    duration_minutes=15,
    description="Database failover",
    event_type="incident"
))

budget.add_event(ErrorBudgetEvent(
    timestamp=datetime.now() - timedelta(days=3),
    duration_minutes=8,
    description="API rate limit breach",
    event_type="incident"
))

# Check status
status = budget.get_consumed_budget()
print(f"Total budget: {status['total_budget_minutes']:.1f} minutes")
print(f"Consumed: {status['consumed_minutes']:.1f} minutes ({status['consumed_percentage']:.1f}%)")
print(f"Remaining: {status['remaining_minutes']:.1f} minutes ({status['remaining_percentage']:.1f}%)")
print(f"\nPolicy: {budget.get_policy_action()}")

# Can we afford a risky deploy?
risky_deploy_estimate = 5  # minutes
can_deploy = budget.can_afford_risk(risky_deploy_estimate)
print(f"\nCan deploy (est. {risky_deploy_estimate}min risk)? {can_deploy}")
```

**Error Budget Policy:**

| Remaining Budget | Action | Example |
|------------------|--------|---------|
| > 75% | Ship aggressively | Try experimental features, frequent deploys |
| 50-75% | Normal velocity | Standard deploy cadence |
| 25-50% | Focus on safety | More testing, smaller changes |
| 10-25% | Reliability freeze | Only critical fixes, no features |
| < 10% | Emergency mode | All hands on reliability, no deploys |

---

## Burn Rate Alerts

### The Problem

Traditional threshold alerts are noisy and slow:

```
Bad alert: "Error rate > 1%"
- Spikes for 10 seconds → Alert fires → Resolves itself
- Slow degradation over days → No alert until catastrophic

Good alert: "Burning error budget 10x faster than sustainable"
- Short spikes → Ignored (self-healing)
- Sustained issues → Alert within minutes
- Tied to actual user impact (SLO violation)
```

**Burn Rate = How fast you're consuming error budget**

### How It Works

```
Normal burn rate: 1x
- Consuming budget at exactly sustainable rate
- Will hit 0% budget at end of window

Fast burn rate: 10x
- Consuming budget 10x faster than sustainable
- Will exhaust budget in 1/10th of window
- Need to alert and fix ASAP

Calculation:
Current error rate: 2% (98% availability)
SLO target: 99.9% (0.1% error budget)

Burn rate = Current error rate / Error budget
          = 2% / 0.1%
          = 20x

At this rate, will burn entire month's budget in:
30 days / 20 = 1.5 days
```

### Implementation

```python
from dataclasses import dataclass
import time

@dataclass
class BurnRateAlert:
    name: str
    window_minutes: int  # How far back to look
    burn_rate_threshold: float  # Alert if burning budget X times faster
    notification_window: int  # How long until budget exhausted at this rate

class BurnRateMonitor:
    def __init__(self, slo_target: float, window_days: int = 30):
        self.slo_target = slo_target
        self.window_days = window_days
        self.error_budget = 1 - slo_target

    def calculate_burn_rate(self, current_error_rate: float) -> float:
        """
        Calculate how fast we're burning error budget

        Args:
            current_error_rate: Current observed error rate

        Returns:
            Burn rate multiplier (1.0 = sustainable, >1 = too fast)
        """
        if self.error_budget == 0:
            return float('inf')

        burn_rate = current_error_rate / self.error_budget
        return burn_rate

    def time_to_exhaustion(self, burn_rate: float) -> float:
        """
        Calculate hours until error budget exhausted at current burn rate

        Args:
            burn_rate: Current burn rate multiplier

        Returns:
            Hours until budget exhausted
        """
        if burn_rate <= 0:
            return float('inf')

        total_hours = self.window_days * 24
        hours_to_exhaustion = total_hours / burn_rate
        return hours_to_exhaustion

    def check_alert(self, current_error_rate: float, alert: BurnRateAlert) -> dict:
        """
        Check if we should alert based on burn rate

        Args:
            current_error_rate: Observed error rate in alert window
            alert: Alert configuration

        Returns:
            Alert details
        """
        burn_rate = self.calculate_burn_rate(current_error_rate)
        hours_to_exhaustion = self.time_to_exhaustion(burn_rate)

        should_alert = burn_rate >= alert.burn_rate_threshold

        return {
            "alert_name": alert.name,
            "should_alert": should_alert,
            "burn_rate": burn_rate,
            "threshold": alert.burn_rate_threshold,
            "hours_to_exhaustion": hours_to_exhaustion,
            "severity": self._get_severity(burn_rate),
        }

    def _get_severity(self, burn_rate: float) -> str:
        """Map burn rate to severity"""
        if burn_rate >= 20:
            return "CRITICAL"
        elif burn_rate >= 10:
            return "HIGH"
        elif burn_rate >= 5:
            return "MEDIUM"
        elif burn_rate >= 2:
            return "LOW"
        else:
            return "NORMAL"

# Multi-window alerting strategy (Google SRE approach)
monitor = BurnRateMonitor(slo_target=0.999, window_days=30)

alerts = [
    # Fast burn, short window: Catch severe incidents quickly
    BurnRateAlert(
        name="Critical: Fast burn",
        window_minutes=5,
        burn_rate_threshold=14.4,  # Will exhaust budget in 2.08 hours
        notification_window=60
    ),

    # Medium burn, medium window: Catch ongoing issues
    BurnRateAlert(
        name="High: Sustained burn",
        window_minutes=60,
        burn_rate_threshold=6,  # Will exhaust budget in 5 days
        notification_window=360
    ),

    # Slow burn, long window: Catch slow degradation
    BurnRateAlert(
        name="Warning: Slow burn",
        window_minutes=1440,  # 24 hours
        burn_rate_threshold=3,  # Will exhaust budget in 10 days
        notification_window=1440
    ),
]

# Simulate current error rate
current_error_rate = 0.02  # 2% errors (98% success)

for alert in alerts:
    result = monitor.check_alert(current_error_rate, alert)
    if result["should_alert"]:
        print(f"🚨 {result['alert_name']}")
        print(f"   Burn rate: {result['burn_rate']:.1f}x (threshold: {result['threshold']}x)")
        print(f"   Time to exhaustion: {result['hours_to_exhaustion']:.1f} hours")
        print(f"   Severity: {result['severity']}")
```

**Multi-Window Strategy:**

| Window | Burn Rate Threshold | Catches | Time to Exhaust |
|--------|---------------------|---------|-----------------|
| 5 min | 14.4x | Severe incidents | 2 hours |
| 1 hour | 6x | Sustained issues | 5 days |
| 24 hours | 3x | Slow degradation | 10 days |

---

## Comparison: SLO Types

| SLO Type | What It Measures | Example | Pros | Cons | Best For |
|----------|------------------|---------|------|------|----------|
| **Availability** | % successful requests | 99.9% of requests return 2xx/3xx | Simple, universal | Doesn't capture latency | User-facing APIs |
| **Latency** | % requests under threshold | 99% < 300ms | Captures UX | Choosing threshold is hard | Interactive apps |
| **Correctness** | % accurate results | 99.99% reads are correct | Critical for data systems | Hard to measure | Databases, financial systems |
| **Freshness** | Data age | 95% data < 5min old | Good for async systems | Not always applicable | Analytics, caches |
| **Durability** | % data retained | 99.999999999% | Critical for storage | Only relevant for storage | Object stores, backups |

---

## Key Concepts Checklist

- [ ] Choose appropriate SLIs (availability, latency, correctness)
- [ ] Set realistic SLOs based on user needs, not aspirations
- [ ] Calculate error budgets (1 - SLO)
- [ ] Implement error budget policies (freeze when low)
- [ ] Set up burn rate alerts (multi-window strategy)
- [ ] Distinguish SLOs (internal) from SLAs (contractual)
- [ ] Buffer SLAs looser than SLOs (e.g., 99.9% SLA, 99.95% SLO)
- [ ] Track error budget consumption by incident
- [ ] Use error budgets to negotiate feature vs reliability tradeoffs

---

## Practical Insights

**Start simple, iterate:**
- First SLO: Just availability. Don't over-engineer.
- Use historical data: What did you actually achieve last quarter?
- Set initial SLO = Historical performance - 10% buffer
- Tighten after 3-6 months of hitting targets consistently

**Error budgets enable velocity:**
```
Without error budget:
PM: "Ship faster!"
Eng: "But reliability!"
Result: Endless arguments

With error budget:
Error budget: 70% remaining
Decision: Ship the risky feature, we can afford it
          If budget drops to 20%, we'll slow down
Result: Data-driven decisions, no arguments
```

**Alert on burn rate, not thresholds:**
```
Bad alert: "Error rate > 1%"
- 1.5% for 10 seconds → Pages on-call → False alarm
- 0.9% for 3 days → No alert → Exhausted budget

Good alert: "Burning budget 10x too fast"
- Short spike → Ignored (self-healing)
- Sustained 0.9% → Alerts after a few hours
- Tied to actual SLO impact
```

**Multiple windows catch different issues:**
```
5-minute window, 14x burn rate → Catch severe incidents (outages)
1-hour window, 6x burn rate → Catch sustained issues (memory leak)
24-hour window, 3x burn rate → Catch slow degradation (database)

Don't use just one window. You'll either page too much or too little.
```

**SLO ≠ SLA buffer is critical:**
```
Common mistake:
SLO: 99.99%
SLA: 99.99%
Result: Any incident triggers SLA violation → Refunds → Pain

Better:
SLO: 99.99% (internal target)
SLA: 99.9% (customer promise)
Buffer: 0.09% → Protection against normal variance
```

**Error budget policy must have teeth:**
```
Weak policy:
"When budget is low, consider focusing on reliability"
Result: Ignored, features shipped anyway

Strong policy:
Budget < 20% → Automated deploy freeze
             → All hands on reliability
             → VP approval required for any feature work
Result: Respected, reliability improves
```

**Track budget by source:**
```python
Budget consumed:
- 40% from planned maintenance (acceptable)
- 35% from deployments (room to optimize)
- 25% from incidents (need to reduce)

Action: Focus on reducing deployment risk and incident frequency
```
