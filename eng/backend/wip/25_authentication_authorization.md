# Chapter 25: Authentication & Authorization

## Authentication vs Authorization

```
Authentication (AuthN): WHO are you?
- "I am Alice, here's my password"
- Verify identity

Authorization (AuthZ): WHAT can you do?
- "Alice can read documents, but not delete them"
- Verify permissions

Order matters:
1. First authenticate (who?)
2. Then authorize (allowed?)
```

---

## Authentication Methods

### Password-Based

```
┌─────────┐     username/password     ┌─────────┐
│  User   │─────────────────────────►│ Server  │
└─────────┘                           └─────────┘
                                           │
                                     Hash password
                                     Compare to stored hash
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │  Database   │
                                    │ users table │
                                    │ (hashed pw) │
                                    └─────────────┘
```

**Password storage:**
```python
# NEVER store plaintext passwords!

# Good: Use bcrypt (or argon2)
import bcrypt

def hash_password(password: str) -> bytes:
    salt = bcrypt.gensalt(rounds=12)  # Cost factor
    return bcrypt.hashpw(password.encode(), salt)

def verify_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), hashed)

# bcrypt includes salt in the hash
# $2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.V0HBNpOiWS6LJe
#  │  │   │                    │
#  │  │   └─ Salt              └─ Hash
#  │  └─ Cost factor (2^12 iterations)
#  └─ Algorithm version
```

### Token-Based (JWT)

```
Login Flow:
┌─────────┐   username/password    ┌─────────┐
│  User   │───────────────────────►│ Server  │
└─────────┘                        └─────────┘
     ▲                                  │
     │                            Verify credentials
     │                            Generate JWT
     │                                  │
     └──────────── JWT Token ───────────┘

Subsequent Requests:
┌─────────┐   Authorization: Bearer <JWT>   ┌─────────┐
│  User   │────────────────────────────────►│ Server  │
└─────────┘                                 └─────────┘
                                                 │
                                           Verify JWT signature
                                           Extract claims
                                           No database lookup!
```

**JWT Structure:**
```
eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsIm5hbWUiOiJBbGljZSIsInJvbGVzIjpbImFkbWluIl0sImlhdCI6MTcwNTMwMDAwMCwiZXhwIjoxNzA1MzAzNjAwfQ.signature

Header (base64):
{
  "alg": "RS256",
  "typ": "JWT"
}

Payload (base64):
{
  "sub": "user-123",        // Subject (user ID)
  "name": "Alice",
  "roles": ["admin"],
  "iat": 1705300000,        // Issued at
  "exp": 1705303600         // Expires (1 hour later)
}

Signature:
RS256(base64(header) + "." + base64(payload), private_key)
```

**JWT Verification:**
```python
import jwt

def verify_jwt(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            PUBLIC_KEY,
            algorithms=["RS256"],
            options={"require": ["exp", "sub"]}
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthError("Token expired")
    except jwt.InvalidTokenError:
        raise AuthError("Invalid token")
```

### JWT vs Sessions

| Aspect | JWT | Sessions |
|--------|-----|----------|
| Storage | Client-side (token) | Server-side (session store) |
| Scalability | Stateless, easy | Requires shared session store |
| Revocation | Hard (wait for expiry) | Instant (delete from store) |
| Size | Larger (carries claims) | Small (just session ID) |
| Security | Token theft = full access | Session ID theft = full access |

**When to use JWT:**
- Stateless microservices
- Cross-domain authentication
- Short-lived access tokens

**When to use sessions:**
- Single monolith
- Need instant revocation
- Traditional web apps

### Refresh Tokens

```
Problem: Short-lived tokens (15 min) = frequent re-login
Solution: Refresh tokens

┌─────────┐                              ┌─────────┐
│  User   │                              │ Server  │
└─────────┘                              └─────────┘
     │                                        │
     │──── Login ────────────────────────────►│
     │                                        │
     │◄─── Access Token (15 min) ─────────────│
     │◄─── Refresh Token (7 days) ────────────│
     │                                        │
     │                                        │
     │  (15 minutes later, access token expired)
     │                                        │
     │──── Refresh Token ────────────────────►│
     │                                        │
     │◄─── New Access Token (15 min) ─────────│
     │                                        │
```

**Refresh token rotation:**
```python
def refresh_access_token(refresh_token: str):
    # Verify refresh token
    payload = verify_refresh_token(refresh_token)
    
    # Invalidate old refresh token (one-time use)
    invalidate_refresh_token(refresh_token)
    
    # Issue new tokens
    new_access_token = create_access_token(payload["sub"])
    new_refresh_token = create_refresh_token(payload["sub"])
    
    return new_access_token, new_refresh_token
```

### OAuth 2.0 / OpenID Connect

```
"Login with Google" flow:

┌─────────┐    ┌─────────┐    ┌─────────────┐
│  User   │    │ Your App│    │   Google    │
└─────────┘    └─────────┘    └─────────────┘
     │              │                │
     │─── Click "Login with Google" ─►
     │              │                │
     │◄──── Redirect to Google ──────│
     │              │                │
     │─────────────────── Login at Google ───►
     │              │                │
     │◄──────────── Redirect with code ──────│
     │              │                │
     │              │── Exchange code ──────►│
     │              │                │
     │              │◄─── Access Token ──────│
     │              │                │
     │              │── Get user info ──────►│
     │              │                │
     │              │◄─── User profile ──────│
     │              │                │
     │◄─── Logged in ─│                │
```

---

## Authorization Models

### Role-Based Access Control (RBAC)

```
Roles:
├── admin:    [read, write, delete, manage_users]
├── editor:   [read, write]
├── viewer:   [read]
└── guest:    []

Users:
├── Alice: admin
├── Bob: editor
└── Carol: viewer

Check: Can Bob delete documents?
→ Bob has role "editor"
→ "editor" has permissions [read, write]
→ "delete" not in [read, write]
→ DENIED
```

```python
ROLE_PERMISSIONS = {
    "admin": ["read", "write", "delete", "manage_users"],
    "editor": ["read", "write"],
    "viewer": ["read"],
}

def check_permission(user: User, permission: str) -> bool:
    user_permissions = ROLE_PERMISSIONS.get(user.role, [])
    return permission in user_permissions

# Usage
@require_permission("delete")
def delete_document(doc_id: str):
    ...
```

### Attribute-Based Access Control (ABAC)

```
Policy: "Users can edit documents in their department"

Attributes:
├── User: {id: "alice", department: "engineering", role: "editor"}
├── Document: {id: "doc-123", department: "engineering", owner: "bob"}
└── Action: "edit"

Evaluation:
user.department == document.department AND user.role == "editor"
→ "engineering" == "engineering" AND "editor" == "editor"
→ ALLOWED
```

```python
def can_edit_document(user: User, document: Document) -> bool:
    # Must be in same department
    if user.department != document.department:
        return False
    
    # Must have editor role or be owner
    if user.role in ["editor", "admin"] or document.owner == user.id:
        return True
    
    return False
```

### Relationship-Based Access Control (ReBAC)

```
Google Zanzibar model (used by Google Docs, Drive):

Relationships stored as tuples:
(object, relation, user)

Examples:
(doc:123, owner, user:alice)
(doc:123, viewer, user:bob)
(folder:456, parent, doc:123)
(folder:456, editor, group:engineering)
(group:engineering, member, user:carol)

Query: Can Carol edit doc:123?

Check path:
carol ─member─► group:engineering ─editor─► folder:456 ─parent─► doc:123
                                                         │
                                            editor on parent = editor on child
→ ALLOWED
```

### Permission Inheritance

```
Folder hierarchy:
/company (owner: admin)
  └── /engineering (editor: eng-team)
        └── /projects (viewer: all-staff)
              └── /secret-project (editor: alice only)

Inherited permissions for /secret-project:
- admin: owner (inherited from /company)
- eng-team: editor (inherited from /engineering)
- all-staff: viewer (inherited from /projects)
- alice: editor (direct)

Effective permission = union of direct + inherited
```

---

## API Authentication

### API Keys

```http
GET /api/data HTTP/1.1
Host: api.example.com
X-API-Key: sk_live_abc123def456
```

```python
def authenticate_api_key(request):
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise AuthError("Missing API key")
    
    # Lookup key (store hashed, like passwords!)
    key_hash = hash_api_key(api_key)
    key_record = db.get_api_key(key_hash)
    
    if not key_record or key_record.revoked:
        raise AuthError("Invalid API key")
    
    return key_record.owner
```

**API Key best practices:**
- Prefix keys: `sk_live_` (live), `sk_test_` (test)
- Hash before storing
- Support key rotation
- Scope keys (read-only, specific resources)
- Set expiration

### Service-to-Service Auth (mTLS)

```
Service A                              Service B
    │                                      │
    │──── Present certificate ────────────►│
    │                                      │
    │◄─── Present certificate ─────────────│
    │                                      │
    │     Both verify against CA           │
    │     Both authenticated!              │
    │                                      │
    │◄────── Encrypted traffic ───────────►│
```

---

## Security Best Practices

### Password Requirements

```python
def validate_password(password: str) -> bool:
    if len(password) < 12:
        return False
    
    # Check against common passwords
    if password.lower() in COMMON_PASSWORDS:
        return False
    
    # Check against breach databases (Have I Been Pwned)
    if is_breached_password(password):
        return False
    
    return True

# Don't require special characters!
# Length + uniqueness > complexity
# "correct horse battery staple" > "P@ssw0rd!"
```

### Rate Limiting Auth Endpoints

```python
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=5, period=60)  # 5 attempts per minute
def login(username: str, password: str):
    ...

# Also implement:
# - Account lockout after N failures
# - CAPTCHA after suspicious activity
# - IP-based blocking for brute force
```

### Secure Token Storage (Client-side)

```javascript
// Web: Use httpOnly cookies for refresh tokens
// Access tokens can be in memory (not localStorage!)

// Mobile: Use secure storage
// iOS: Keychain
// Android: EncryptedSharedPreferences

// Never store tokens in:
// - localStorage (XSS vulnerable)
// - URL parameters (logged, cached)
// - Plain cookies (CSRF vulnerable without protections)
```

---

## Key Concepts Checklist

- [ ] Explain authentication vs authorization
- [ ] Describe JWT structure and verification
- [ ] Compare JWT vs sessions trade-offs
- [ ] Explain OAuth 2.0 flow
- [ ] Describe RBAC vs ABAC vs ReBAC
- [ ] Know API authentication patterns
- [ ] Discuss password storage best practices

---

## Practical Insights

**Token strategy:**
- Access tokens: 15-60 minutes
- Refresh tokens: 7-30 days
- Rotate refresh tokens on use
- Store refresh tokens server-side for revocation

**Authorization at scale:**
- Cache permission checks
- Denormalize for hot paths
- Consider policy engines (Open Policy Agent)
- Audit all access decisions

**Security layers:**
```
1. Network: mTLS, VPN, firewall
2. Application: AuthN, AuthZ, input validation
3. Data: Encryption at rest, field-level encryption
4. Audit: Log all access, anomaly detection
```

**Common pitfalls:**
- JWT in localStorage (XSS)
- No rate limiting on login
- Predictable session IDs
- Missing CSRF protection
- Over-permissive default roles
