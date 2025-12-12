# Chapter 28: Container Orchestration (Kubernetes)

## Why Kubernetes?

```
Without orchestration:
- Deploy to 50 servers manually
- Server dies → manually restart service elsewhere
- Scale up → provision new servers, deploy, configure
- Rolling update → pray nothing breaks

With Kubernetes:
- "Run 10 replicas of my app" → K8s figures out where
- Pod dies → K8s restarts it automatically
- Scale up → kubectl scale deployment/app --replicas=20
- Rolling update → Zero-downtime by default
```

---

## Core Concepts

### Pods

```
Pod = smallest deployable unit
    = one or more containers that share:
      - Network namespace (same IP)
      - Storage volumes
      - Lifecycle

┌─────────────────────────────────────┐
│              Pod                     │
│  ┌─────────────┐ ┌─────────────┐    │
│  │   App       │ │   Sidecar   │    │
│  │ Container   │ │ Container   │    │
│  │             │ │ (logging)   │    │
│  └─────────────┘ └─────────────┘    │
│         │              │            │
│         └──── localhost ───┘        │
│                  │                  │
│            Shared Volume            │
└─────────────────────────────────────┘
```

**Pod spec:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app
  labels:
    app: my-app
spec:
  containers:
  - name: app
    image: my-app:v1.2.3
    ports:
    - containerPort: 8080
    resources:
      requests:
        memory: "128Mi"
        cpu: "100m"
      limits:
        memory: "256Mi"
        cpu: "500m"
```

### Deployments

```
Deployment manages ReplicaSets manages Pods

┌────────────────────────────────────────────┐
│              Deployment                     │
│                                             │
│  ┌────────────────────────────────────┐    │
│  │          ReplicaSet (v2)            │    │
│  │  ┌─────┐ ┌─────┐ ┌─────┐           │    │
│  │  │Pod 1│ │Pod 2│ │Pod 3│  replicas=3│   │
│  │  └─────┘ └─────┘ └─────┘           │    │
│  └────────────────────────────────────┘    │
│                                             │
│  ┌────────────────────────────────────┐    │
│  │   ReplicaSet (v1) - scaling down   │    │
│  │  ┌─────┐                 replicas=1│    │
│  │  │Pod  │ (draining)                │    │
│  │  └─────┘                           │    │
│  └────────────────────────────────────┘    │
│                                             │
└────────────────────────────────────────────┘
```

**Deployment spec:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: app
        image: my-app:v1.2.3
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "500m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 20
```

### Services

```
Problem: Pods have dynamic IPs, come and go
Solution: Service provides stable endpoint

┌─────────────────────────────────────────────────┐
│                    Service                       │
│              my-app-service                      │
│              ClusterIP: 10.96.0.1                │
│              Port: 80                            │
└─────────────────────────────────────────────────┘
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
     ┌───────┐  ┌───────┐  ┌───────┐
     │ Pod 1 │  │ Pod 2 │  │ Pod 3 │
     │:8080  │  │:8080  │  │:8080  │
     └───────┘  └───────┘  └───────┘

Internal DNS: my-app-service.default.svc.cluster.local
```

**Service types:**

| Type | Description | Use Case |
|------|-------------|----------|
| ClusterIP | Internal only | Service-to-service |
| NodePort | Expose on each node's IP | Development, simple access |
| LoadBalancer | Cloud LB provisioned | Production external access |
| ExternalName | DNS alias | External service reference |

**Service spec:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  type: ClusterIP
  selector:
    app: my-app  # Matches pods with this label
  ports:
  - port: 80           # Service port
    targetPort: 8080   # Pod port
```

### Ingress

```
Ingress = HTTP routing rules

                    ┌─────────────┐
      Internet ────►│   Ingress   │
                    │ Controller  │
                    └─────────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
    /api/* route    /web/* route    / route
           │              │              │
           ▼              ▼              ▼
    ┌───────────┐  ┌───────────┐  ┌───────────┐
    │ API       │  │ Web       │  │ Default   │
    │ Service   │  │ Service   │  │ Service   │
    └───────────┘  └───────────┘  └───────────┘
```

**Ingress spec:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - example.com
    secretName: tls-secret
  rules:
  - host: example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 80
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80
```

### ConfigMaps and Secrets

```yaml
# ConfigMap for non-sensitive config
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DATABASE_HOST: "postgres.default.svc"
  LOG_LEVEL: "info"
  config.yaml: |
    server:
      port: 8080
    features:
      newUI: true

---
# Secret for sensitive data
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  DATABASE_PASSWORD: cGFzc3dvcmQxMjM=  # base64 encoded
  API_KEY: c2VjcmV0LWFwaS1rZXk=
```

**Using in pods:**
```yaml
spec:
  containers:
  - name: app
    env:
    - name: DATABASE_HOST
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: DATABASE_HOST
    - name: DATABASE_PASSWORD
      valueFrom:
        secretKeyRef:
          name: app-secrets
          key: DATABASE_PASSWORD
    volumeMounts:
    - name: config-volume
      mountPath: /etc/config
  volumes:
  - name: config-volume
    configMap:
      name: app-config
```

---

## Resource Management

### Requests vs Limits

```yaml
resources:
  requests:    # Guaranteed resources (for scheduling)
    memory: "128Mi"
    cpu: "100m"    # 100 millicores = 0.1 CPU
  limits:      # Maximum allowed (hard cap)
    memory: "256Mi"
    cpu: "500m"
```

**Behavior:**
```
CPU:
- Request: Guaranteed CPU time
- Limit: Throttled if exceeded (not killed)

Memory:
- Request: Guaranteed memory
- Limit: OOMKilled if exceeded!
```

### Quality of Service (QoS)

```
Guaranteed:
  requests == limits (for all containers)
  → Last to be evicted

Burstable:
  requests < limits (for any container)
  → Evicted after BestEffort

BestEffort:
  No requests or limits
  → First to be evicted
```

---

## Deployment Strategies

### Rolling Update (Default)

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%        # Max pods above desired
      maxUnavailable: 25%  # Max pods below desired
```

```
Deployment: 4 replicas, updating v1 → v2

Step 1: [v1] [v1] [v1] [v1]           (start)
Step 2: [v1] [v1] [v1] [v1] [v2]      (surge: +1 v2)
Step 3: [v1] [v1] [v1] [v2] [v2]      (terminate 1 v1, start 1 v2)
Step 4: [v1] [v1] [v2] [v2] [v2]      
Step 5: [v1] [v2] [v2] [v2] [v2]      
Step 6: [v2] [v2] [v2] [v2]           (complete)
```

### Blue-Green

```
Blue (current): [v1] [v1] [v1] [v1]  ← Service points here
Green (new):    [v2] [v2] [v2] [v2]  ← Deploy, test

Switch:
Blue (old):     [v1] [v1] [v1] [v1]
Green (current):[v2] [v2] [v2] [v2]  ← Service points here now

Rollback: Just switch service back to blue
```

**Implementation:**
```yaml
# Blue deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-blue
  labels:
    version: blue

---
# Green deployment  
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-green
  labels:
    version: green

---
# Service - switch selector to change
apiVersion: v1
kind: Service
metadata:
  name: app
spec:
  selector:
    app: my-app
    version: green  # Change to "blue" for rollback
```

### Canary

```
Canary: Route small % of traffic to new version

[v1] [v1] [v1] [v1] [v1] [v1] [v1] [v1] [v1] [v2]
         90% of traffic                       10%

Monitor error rates, latency
If OK: Gradually increase v2 %
If not: Route 100% back to v1
```

**With Istio:**
```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
spec:
  http:
  - route:
    - destination:
        host: my-app
        subset: v1
      weight: 90
    - destination:
        host: my-app
        subset: v2
      weight: 10
```

---

## Health Checks

### Liveness Probe

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
  initialDelaySeconds: 15  # Wait before first check
  periodSeconds: 20        # Check every 20s
  failureThreshold: 3      # 3 failures = restart

# If fails: Container is RESTARTED
```

### Readiness Probe

```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
  failureThreshold: 3

# If fails: Pod removed from Service endpoints
# (no traffic, but not restarted)
```

### Startup Probe

```yaml
startupProbe:
  httpGet:
    path: /healthz
    port: 8080
  failureThreshold: 30
  periodSeconds: 10
  
# For slow-starting containers
# Liveness/readiness probes wait until startup succeeds
```

---

## Scaling

### Horizontal Pod Autoscaler (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  minReplicas: 3
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

```
Current: 3 pods, 90% CPU average
Target: 70% CPU
Calculation: ceil(3 * 90/70) = ceil(3.86) = 4 pods
Action: Scale up to 4 pods
```

### Vertical Pod Autoscaler (VPA)

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: my-app-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  updatePolicy:
    updateMode: "Auto"  # Recreate pods with new resources
```

---

## Networking

### Network Policies

```yaml
# Allow only specific traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-allow-frontend
spec:
  podSelector:
    matchLabels:
      app: api
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - port: 8080
```

---

## Interview Checklist

- [ ] Explain Pod, Deployment, Service, Ingress
- [ ] Describe resource requests vs limits
- [ ] Compare deployment strategies (rolling, blue-green, canary)
- [ ] Explain liveness vs readiness probes
- [ ] Describe HPA scaling behavior
- [ ] Know when to use each Service type

---

## Staff+ Insights

**Resource sizing:**
- Start with requests based on load testing
- Set memory limit = 2x request (OOM buffer)
- CPU limits optional (throttling vs guaranteed)
- Monitor actual usage, adjust quarterly

**Pod disruption budget:**
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
spec:
  minAvailable: 2  # Or maxUnavailable: 1
  selector:
    matchLabels:
      app: my-app

# Prevents cluster operations from killing too many pods
```

**Production checklist:**
- Resource requests/limits set
- Liveness and readiness probes
- Pod disruption budgets
- Network policies (zero trust)
- Pod security policies
- Proper labels for observability

**Anti-patterns:**
- Running as root
- No resource limits (noisy neighbor)
- No health checks (zombie pods)
- Storing state in pods (use PVCs or external)
- Hardcoded configs (use ConfigMaps)
