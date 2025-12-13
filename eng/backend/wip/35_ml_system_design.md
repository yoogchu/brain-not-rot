# Chapter 35: ML System Design Patterns

## Why ML Systems Are Different

Your recommendation model works perfectly in Jupyter notebooks. You deploy it to production.

```
Day 1: 95% accuracy, 50ms p99 latency
Day 30: 87% accuracy, 200ms p99 latency
Day 60: Model serving OOM crashes

What changed? Everything:
- User behavior shifted (data drift)
- Model predictions affected user behavior (feedback loops)
- Feature computation slowed (training-serving skew)
- No rollback strategy when accuracy dropped
```

ML systems fail in ways traditional software doesn't. You need infrastructure that treats models as first-class citizens, handles data drift, and supports experimentation at scale.

---

## Feature Stores

### The Problem

Your training pipeline computes features from raw data. Six months later, your serving code reimplements the same logic. The implementations diverge. Training uses `total_purchases / days_active` but serving uses `total_purchases / days_since_signup`. Accuracy tanks in production.

**Training-serving skew is the #1 killer of ML systems.**

### How Feature Stores Work

```
┌────────────────────────────────────────────────────────┐
│                   Feature Store                         │
│  ┌──────────────────────┐  ┌──────────────────────┐   │
│  │  Offline Features    │  │  Online Features     │   │
│  │  (Batch/Historical)  │  │  (Real-time)         │   │
│  │                      │  │                      │   │
│  │  Storage: S3/GCS     │  │  Storage: Redis/     │   │
│  │           Snowflake  │  │           DynamoDB   │   │
│  │  Latency: Minutes    │  │  Latency: <10ms      │   │
│  │  Use: Training       │  │  Use: Serving        │   │
│  └──────────────────────┘  └──────────────────────┘   │
│           ▲                          ▲                 │
│           │                          │                 │
│           └──────────┬───────────────┘                 │
│                      │                                 │
│           ┌──────────────────────┐                     │
│           │  Feature Definitions │                     │
│           │  (Python/SQL)        │                     │
│           └──────────────────────┘                     │
└────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌──────────────────┐
│  Training Job    │         │  Model Serving   │
│  (Batch)         │         │  (Real-time)     │
└──────────────────┘         └──────────────────┘
```

**Key insight:** Define features once, materialize for both training and serving.

### Implementation Example

```python
# Feature definition (shared between training and serving)
from feast import Entity, Feature, FeatureView, Field
from feast.types import Float64, Int64
from datetime import timedelta

# Define entity (what features are about)
user = Entity(name="user", join_keys=["user_id"])

# Define feature view (how to compute features)
user_features = FeatureView(
    name="user_activity_features",
    entities=[user],
    schema=[
        Field(name="total_purchases", dtype=Int64),
        Field(name="avg_purchase_amount", dtype=Float64),
        Field(name="days_since_last_purchase", dtype=Int64),
    ],
    source="user_activity_source",  # Points to data source
    ttl=timedelta(days=1),  # Feature freshness requirement
)

# Training: Get historical features
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Get point-in-time correct features for training
training_df = store.get_historical_features(
    entity_df=entity_df,  # DataFrame with user_ids and timestamps
    features=[
        "user_activity_features:total_purchases",
        "user_activity_features:avg_purchase_amount",
        "user_activity_features:days_since_last_purchase",
    ],
).to_df()

# Serving: Get real-time features
online_features = store.get_online_features(
    features=[
        "user_activity_features:total_purchases",
        "user_activity_features:avg_purchase_amount",
        "user_activity_features:days_since_last_purchase",
    ],
    entity_rows=[{"user_id": 12345}],
).to_dict()
```

### Offline vs Online Features

| Aspect | Offline Features | Online Features |
|--------|------------------|-----------------|
| Storage | Data warehouse (Snowflake, BigQuery) | Key-value store (Redis, DynamoDB) |
| Latency | Minutes to hours | <10ms p99 |
| Data volume | Full history (TB-PB) | Recent snapshot (GB) |
| Use case | Training, backtesting | Real-time serving |
| Cost | Storage optimized | Latency optimized |

**When to use feature stores:**
- Multiple models sharing features
- Need point-in-time correctness for training
- Training-serving skew is a problem
- Team has >3 ML engineers

**When NOT to use:**
- Single model, simple features
- Features are model-specific
- Small team, early stage
- Real-time feature computation (use streaming instead)

---

## Model Serving Patterns

### Pattern 1: Batch Prediction

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Input Data  │───►│    Model     │───►│  Write to    │
│  (S3/HDFS)   │    │  (Spark job) │    │   Database   │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │  Application │
                                        │  reads cache │
                                        └──────────────┘

Schedule: Every hour/day
Latency: Predictions ready in minutes/hours
```

**Implementation:**

```python
# Batch prediction with PySpark
from pyspark.sql import SparkSession
import mlflow.pyfunc

# Load model
model = mlflow.pyfunc.load_model("models:/recommendation/production")

# Read input data
spark = SparkSession.builder.getOrCreate()
user_features = spark.read.parquet("s3://data/user_features/")

# Apply model to all users
predictions = model.predict(user_features)

# Write predictions to database
predictions.write \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://db/predictions") \
    .option("dbtable", "user_recommendations") \
    .mode("overwrite") \
    .save()
```

**When to use:** Pre-compute predictions for known entities (all users, all products)
**When NOT to use:** Need predictions for new/unknown entities in real-time

### Pattern 2: Real-time Prediction

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  HTTP        │───►│  Model API   │───►│   Response   │
│  Request     │    │  (FastAPI)   │    │  (JSON)      │
└──────────────┘    └──────────────┘    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Model in    │
                    │  Memory      │
                    └──────────────┘

Latency: <100ms p99
Throughput: Thousands of QPS
```

**Implementation:**

```python
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np

app = FastAPI()

# Load model once at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = mlflow.pyfunc.load_model("models:/fraud_detection/production")

class PredictionRequest(BaseModel):
    transaction_amount: float
    merchant_id: str
    user_history_features: list[float]

class PredictionResponse(BaseModel):
    fraud_probability: float
    decision: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Prepare features
    features = np.array([[
        request.transaction_amount,
        hash(request.merchant_id) % 10000,  # Feature engineering
        *request.user_history_features
    ]])

    # Get prediction
    fraud_prob = model.predict(features)[0]

    # Business logic
    decision = "reject" if fraud_prob > 0.8 else "approve"

    return PredictionResponse(
        fraud_probability=fraud_prob,
        decision=decision
    )
```

**Scaling considerations:**

```python
# Use model server with batching for higher throughput
from ray import serve

@serve.deployment(num_replicas=4, max_concurrent_queries=100)
class ModelServer:
    def __init__(self):
        self.model = mlflow.pyfunc.load_model("models:/fraud_detection/production")
        self.batch_size = 32

    async def __call__(self, request):
        # Ray Serve handles batching automatically
        return self.model.predict(request.features)
```

**When to use:** User-facing predictions, dynamic inputs, fraud detection
**When NOT to use:** Can pre-compute, >1s latency acceptable

### Pattern 3: Streaming Prediction

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Kafka      │───►│  Flink Job   │───►│   Kafka      │
│   Input      │    │  + Model     │    │   Output     │
└──────────────┘    └──────────────┘    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  State       │
                    │  (Features)  │
                    └──────────────┘

Latency: <1s
Throughput: Millions of events/day
```

**Implementation with Kafka Streams:**

```python
from kafka import KafkaConsumer, KafkaProducer
import json
import mlflow.pyfunc

# Load model
model = mlflow.pyfunc.load_model("models:/click_prediction/production")

# Kafka setup
consumer = KafkaConsumer('user_events', bootstrap_servers=['localhost:9092'])
producer = KafkaProducer(bootstrap_servers=['localhost:9092'])

# Stateful processing
user_state = {}  # In production: use Flink state or Redis

for message in consumer:
    event = json.loads(message.value)
    user_id = event['user_id']

    # Update user state (windowed features)
    if user_id not in user_state:
        user_state[user_id] = {'clicks': 0, 'impressions': 0}

    user_state[user_id]['impressions'] += 1

    # Compute features
    click_rate = user_state[user_id]['clicks'] / max(user_state[user_id]['impressions'], 1)
    features = [click_rate, event['item_id'], event['time_of_day']]

    # Predict
    prediction = model.predict([features])[0]

    # Emit prediction
    output = {
        'user_id': user_id,
        'item_id': event['item_id'],
        'click_probability': prediction
    }
    producer.send('predictions', json.dumps(output).encode())
```

**When to use:** Event-driven predictions, need stateful features, IoT/time-series
**When NOT to use:** Simple request-response, no event stream infrastructure

---

## A/B Testing for ML Models

### The Problem

You trained a new model with 2% higher offline accuracy. You deploy it to all users. Engagement drops 5%. Why?

**Offline metrics don't always correlate with business metrics.**

### Multi-Armed Bandit Pattern

```
┌─────────────────────────────────────────────────────┐
│  Traffic Splitter (Thompson Sampling)              │
└─────────────────────────────────────────────────────┘
         │             │              │
         ▼             ▼              ▼
    ┌────────┐    ┌────────┐    ┌────────┐
    │Model A │    │Model B │    │Model C │
    │ 50%    │    │ 30%    │    │ 20%    │
    └────────┘    └────────┘    └────────┘
         │             │              │
         └─────────────┴──────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  Collect Rewards       │
         │  (Click, Purchase)     │
         └────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  Update Traffic Split  │
         │  (Bayesian update)     │
         └────────────────────────┘
```

**Implementation:**

```python
import numpy as np
from scipy.stats import beta

class ThompsonSamplingABTest:
    def __init__(self, model_names):
        # Beta distribution parameters for each model
        self.models = {name: {'alpha': 1, 'beta': 1} for name in model_names}
        self.model_instances = {}

    def select_model(self):
        # Sample from each model's Beta distribution
        samples = {
            name: np.random.beta(params['alpha'], params['beta'])
            for name, params in self.models.items()
        }

        # Choose model with highest sample
        return max(samples, key=samples.get)

    def update(self, model_name, reward):
        """Update based on reward (1 = success, 0 = failure)"""
        if reward:
            self.models[model_name]['alpha'] += 1
        else:
            self.models[model_name]['beta'] += 1

    def get_stats(self):
        """Get current performance estimates"""
        stats = {}
        for name, params in self.models.items():
            a, b = params['alpha'], params['beta']
            mean = a / (a + b)
            var = (a * b) / ((a + b) ** 2 * (a + b + 1))
            stats[name] = {
                'mean': mean,
                'std': np.sqrt(var),
                'samples': a + b - 2
            }
        return stats

# Usage
ab_test = ThompsonSamplingABTest(['model_v1', 'model_v2', 'model_v3'])

# For each request
model_name = ab_test.select_model()
prediction = models[model_name].predict(features)

# Track reward (e.g., did user click?)
reward = user_clicked  # 1 or 0
ab_test.update(model_name, reward)

# Check stats
print(ab_test.get_stats())
# {'model_v1': {'mean': 0.45, 'std': 0.02, 'samples': 1000},
#  'model_v2': {'mean': 0.52, 'std': 0.02, 'samples': 900},
#  'model_v3': {'mean': 0.40, 'std': 0.03, 'samples': 600}}
```

### Shadow Mode Testing

```
Request
   │
   ▼
┌──────────────────┐
│ Production Model │──► Return to user
│ (Model v1)       │
└──────────────────┘
   │
   │ (also send to)
   ▼
┌──────────────────┐
│  Shadow Model    │──► Log predictions, don't return
│  (Model v2)      │
└──────────────────┘
   │
   ▼
┌──────────────────┐
│  Compare         │──► Metrics dashboard
│  Predictions     │
└──────────────────┘
```

**When to use shadow mode:** Validate new model before affecting users
**When to use A/B test:** Ready to test impact on business metrics

---

## Model Versioning and Registry

### The Problem

```
Engineer 1: "Which model is in production?"
Engineer 2: "The one from last week? Or the retrained one?"
Engineer 3: "I can't reproduce the v2 results..."

No source of truth for model versions
No way to rollback when v3 fails
Can't reproduce training runs
```

### Model Registry Architecture

```
┌───────────────────────────────────────────────────┐
│              Model Registry                        │
│  ┌─────────────────────────────────────────────┐ │
│  │  Model: fraud_detection                     │ │
│  │                                             │ │
│  │  Versions:                                  │ │
│  │  ┌────────────────────────────────────┐    │ │
│  │  │ v1: Staging    (accuracy: 0.92)    │    │ │
│  │  │ v2: Production (accuracy: 0.94)    │    │ │
│  │  │ v3: Archived   (accuracy: 0.91)    │    │ │
│  │  └────────────────────────────────────┘    │ │
│  │                                             │ │
│  │  Metadata:                                  │ │
│  │  - Training run ID                          │ │
│  │  - Hyperparameters                          │ │
│  │  - Training data version                    │ │
│  │  - Metrics                                  │ │
│  │  - Model artifacts (weights, config)        │ │
│  └─────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────┘
```

**Implementation with MLflow:**

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Start a training run
with mlflow.start_run(run_name="fraud_detection_v3") as run:
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("data_version", "2024-01-15")

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision_score(y_test, y_pred))

    # Log model
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="fraud_detection"
    )

# Promote model to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="fraud_detection",
    version=3,
    stage="Production"
)

# Load production model for serving
model = mlflow.pyfunc.load_model("models:/fraud_detection/Production")
```

**Model stages:**

| Stage | Purpose | Who can write |
|-------|---------|---------------|
| None | Newly registered | Training jobs |
| Staging | Testing/validation | ML engineers |
| Production | Live serving | Deployment automation |
| Archived | Deprecated | Anyone |

---

## Training Pipelines

### The Problem

Training a model involves: data extraction, validation, preprocessing, feature engineering, training, evaluation, registration. Doing this manually is error-prone. You need reproducible, scheduled pipelines.

### Pipeline Architecture

```
┌──────────────────────────────────────────────────────┐
│  Training Pipeline (DAG)                             │
│                                                       │
│  ┌─────────────┐   ┌─────────────┐   ┌────────────┐│
│  │Extract Data │──►│Validate Data│──►│  Feature   ││
│  │(BigQuery)   │   │(Great Expec)│   │Engineering ││
│  └─────────────┘   └─────────────┘   └────────────┘│
│                                           │          │
│                                           ▼          │
│  ┌─────────────┐   ┌─────────────┐   ┌────────────┐│
│  │  Register   │◄──│  Evaluate   │◄──│   Train    ││
│  │  Model      │   │   Model     │   │   Model    ││
│  └─────────────┘   └─────────────┘   └────────────┘│
└──────────────────────────────────────────────────────┘
```

**Kubeflow Pipelines Implementation:**

```python
from kfp import dsl
from kfp.v2.dsl import component, Output, Model, Metrics

@component(base_image="python:3.9", packages_to_install=["pandas", "scikit-learn"])
def extract_data(output_path: Output[str]):
    import pandas as pd
    # Extract from data warehouse
    df = pd.read_gbq("SELECT * FROM project.dataset.training_data")
    df.to_parquet(output_path)

@component(base_image="python:3.9", packages_to_install=["great_expectations"])
def validate_data(input_path: str) -> bool:
    import great_expectations as ge
    df = ge.read_parquet(input_path)

    # Define expectations
    df.expect_column_values_to_not_be_null("user_id")
    df.expect_column_values_to_be_between("age", 0, 120)

    results = df.validate()
    return results.success

@component(base_image="python:3.9", packages_to_install=["scikit-learn"])
def train_model(
    data_path: str,
    model_output: Output[Model],
    metrics_output: Output[Metrics]
):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    # Load data
    df = pd.read_parquet(data_path)
    X, y = df.drop("target", axis=1), df["target"]

    # Train
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    # Save model
    joblib.dump(model, model_output.path)

    # Log metrics
    accuracy = model.score(X, y)
    metrics_output.log_metric("accuracy", accuracy)

@dsl.pipeline(name="fraud-detection-training")
def training_pipeline():
    extract_task = extract_data()
    validate_task = validate_data(input_path=extract_task.output)

    with dsl.Condition(validate_task.output == True):
        train_task = train_model(data_path=extract_task.output)
```

**When to use Kubeflow:** Kubernetes-native, complex pipelines, team has K8s expertise
**When to use MLflow Projects:** Simpler, git-based, less infrastructure overhead

---

## Data Pipelines for ML

### Feature Engineering at Scale

```
Raw Events (Kafka)
       │
       ▼
┌────────────────────┐
│  Stream Processing │
│  (Flink/Spark)     │
│                    │
│  - Windowing       │
│  - Aggregations    │
│  - Joins           │
└────────────────────┘
       │
       ▼
┌────────────────────┐
│  Feature Store     │
│  (Offline/Online)  │
└────────────────────┘
       │
       ├──────────────┬────────────────┐
       ▼              ▼                ▼
   Training      Batch Predict   Real-time Serve
```

**Spark Feature Engineering:**

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()

# Read raw events
events = spark.read.parquet("s3://events/user_activity/")

# Define window for aggregations
user_window = Window.partitionBy("user_id").orderBy("timestamp").rangeBetween(-86400, 0)

# Compute features
features = events.groupBy("user_id").agg(
    # Count features
    F.count("event_id").alias("total_events"),
    F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("total_purchases"),

    # Time-based features
    F.datediff(F.current_date(), F.max("timestamp")).alias("days_since_last_event"),

    # Windowed aggregations
    F.avg(F.col("purchase_amount")).over(user_window).alias("avg_purchase_amount_30d"),

    # Category features
    F.collect_set("product_category").alias("purchased_categories")
)

# Write to feature store
features.write \
    .format("feast") \
    .option("feature_view", "user_activity_features") \
    .save()
```

---

## Monitoring ML Models

### The Problem

```
Week 1: Model accuracy: 94%
Week 4: Model accuracy: 89%
Week 8: Model accuracy: 76%

What happened?
- User behavior changed (data drift)
- Feature distributions shifted
- Bugs in feature computation
- Feedback loops corrupted training data
```

### Types of Drift

**Data Drift: Input distribution changes**

```python
import numpy as np
from scipy.stats import ks_2samp

def detect_data_drift(training_data, production_data, feature_name, threshold=0.05):
    """Kolmogorov-Smirnov test for distribution shift"""
    statistic, p_value = ks_2samp(
        training_data[feature_name],
        production_data[feature_name]
    )

    if p_value < threshold:
        return True, f"Drift detected in {feature_name}: p={p_value}"
    return False, None

# Check all features
for feature in features:
    drift_detected, message = detect_data_drift(train_df, prod_df, feature)
    if drift_detected:
        alert(message)
```

**Model Drift: Model performance degrades**

```python
def monitor_model_performance(predictions, ground_truth):
    """Track model metrics over time"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    metrics = {
        'accuracy': accuracy_score(ground_truth, predictions),
        'precision': precision_score(ground_truth, predictions),
        'recall': recall_score(ground_truth, predictions),
        'timestamp': datetime.now()
    }

    # Write to monitoring system
    write_to_prometheus(metrics)

    # Alert if metrics drop
    if metrics['accuracy'] < 0.85:  # Threshold
        alert(f"Model accuracy dropped to {metrics['accuracy']}")
```

**Prediction Drift: Output distribution changes**

```
Training: 30% predicted positive class
Week 1: 32% predicted positive class (normal)
Week 4: 68% predicted positive class (DRIFT!)

Causes:
- Input data changed
- Model bug
- Feature engineering bug
```

### Monitoring Dashboard

| Metric | Alert Threshold | Current Value | Status |
|--------|----------------|---------------|--------|
| Accuracy | < 0.90 | 0.93 | OK |
| Data drift (age) | p < 0.05 | p = 0.12 | OK |
| Data drift (income) | p < 0.05 | p = 0.02 | ALERT |
| Prediction positive rate | > 0.40 | 0.35 | OK |
| Latency p99 | > 200ms | 145ms | OK |

---

## ML Platform Comparison

| Platform | Best For | Strengths | Weaknesses | Cost |
|----------|----------|-----------|------------|------|
| **MLflow** | Small-medium teams, OSS | Simple, flexible, model registry | No managed training infra | Free (OSS) |
| **Kubeflow** | K8s-native teams | Kubernetes integration, pipelines | Complex setup, steep learning curve | Infra cost only |
| **SageMaker** | AWS shops | Managed infrastructure, AutoML | Vendor lock-in, expensive | $$$$ |
| **Vertex AI** | GCP shops | Integrated with GCP, AutoML | Vendor lock-in | $$$ |

**MLflow:**
```python
# Simple, local development
mlflow.start_run()
mlflow.log_param("alpha", 0.01)
model = train_model()
mlflow.sklearn.log_model(model, "model")
```

**Kubeflow:**
```python
# Production pipelines, K8s
@dsl.pipeline
def training_pipeline():
    train_op = train_model_op()
    deploy_op = deploy_model_op(train_op.outputs['model'])
```

**SageMaker:**
```python
# Managed training, AWS
estimator = sagemaker.estimator.Estimator(
    image_uri="my-training-image",
    role="SageMakerRole",
    instance_type="ml.p3.2xlarge"
)
estimator.fit({"training": s3_data_path})
```

---

## Key Concepts Checklist

- [ ] Design feature store for training-serving consistency
- [ ] Choose serving pattern (batch, real-time, streaming) based on latency needs
- [ ] Implement A/B testing or shadow mode for model validation
- [ ] Use model registry for versioning and rollback capability
- [ ] Build reproducible training pipelines
- [ ] Monitor for data drift, model drift, and prediction drift
- [ ] Calculate infrastructure costs (training vs serving)
- [ ] Plan rollback strategy for model failures

---

## Practical Insights

**Feature store or not:**
- Single model with simple features? Don't build a feature store yet.
- Multiple models, multiple teams, training-serving skew problems? Invest in feature store.
- Rule of thumb: If you have >3 ML engineers, feature store ROI is positive.

**Training-serving skew is inevitable:**
```python
# Training code (Python/Pandas)
features['avg_purchase'] = df.groupby('user_id')['amount'].mean()

# Serving code (SQL, different engineer, 6 months later)
SELECT user_id, SUM(amount) / COUNT(*) as avg_purchase  -- Bug: includes nulls differently

# Prevention: Generate serving code from training code
# Use tools like Feast, Tecton, or shared feature computation libraries
```

**Model rollback strategy:**
```python
# Always keep previous model version warm
MODELS = {
    'current': load_model('v3'),
    'previous': load_model('v2'),
    'fallback': load_model('v1')  # Known stable
}

# Feature flag for instant rollback
if feature_flag('use_model_v3'):
    prediction = MODELS['current'].predict(features)
else:
    prediction = MODELS['previous'].predict(features)
```

**ML infrastructure costs:**
```
Training: $500/month (periodic, GPU instances)
Serving: $5,000/month (continuous, high QPS)

Optimize serving first!
- Batch predictions where possible (100x cheaper)
- Model quantization (4x-10x speedup)
- Caching (avoid redundant predictions)
```

**When to build vs buy:**

Build if:
- Unique domain (your data moat)
- Specific performance requirements
- Want full control and customization

Buy if:
- Commodity problem (fraud, recommendations)
- Small team (<5 ML engineers)
- Need to move fast
- Don't want to maintain infrastructure

**Data quality > Model complexity:**
```
Bad data + complex model = Bad predictions
Good data + simple model = Good predictions

Invest in:
- Data validation (Great Expectations)
- Feature monitoring
- Training data versioning
- Labeling quality
```

**Shadow mode before A/B test:**
```
1. Shadow mode (1-2 weeks)
   - Run new model in parallel
   - Log predictions, don't serve
   - Validate: no errors, latency OK, predictions reasonable

2. A/B test (2-4 weeks)
   - 5% traffic to new model
   - Monitor business metrics
   - Gradually increase to 50%

3. Full rollout
   - 100% traffic to new model
   - Keep old model for rollback
```

**Monitoring alert fatigue:**
- Too many alerts = ignored alerts
- Start with critical metrics only:
  - Model accuracy/precision (if ground truth available)
  - Severe data drift (p < 0.01, not 0.05)
  - Serving errors/latency
- Add alerts incrementally based on incidents
