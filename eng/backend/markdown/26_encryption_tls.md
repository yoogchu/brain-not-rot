# Chapter 26: Encryption & TLS

## Why Encryption Matters

Without encryption:

```
2:34 AM: Network tap installed on datacenter switch
2:35 AM: Attacker captures:
  - User passwords (plaintext)
  - Credit card numbers (plaintext)
  - Session tokens (plaintext)
  - API keys (plaintext)
  - Customer PII (plaintext)

6:00 AM: 2.3 million accounts compromised
8:00 AM: Company stock down 35%
Week 2: $847M class action lawsuit filed
```

Encryption prevents:
- **Man-in-the-middle attacks** (network eavesdropping)
- **Data breaches** (stolen hard drives, database dumps)
- **Insider threats** (rogue employees, contractors)
- **Compliance violations** (GDPR, PCI-DSS, HIPAA)

---

## Symmetric vs Asymmetric Encryption

### Symmetric Encryption

**The Problem:**
You need to encrypt large amounts of data quickly. Asymmetric encryption is too slow for bulk data.

**How It Works:**

```
Same key for encryption AND decryption

┌──────────────┐                    ┌──────────────┐
│   Plaintext  │                    │   Plaintext  │
│   "Secret"   │                    │   "Secret"   │
└──────────────┘                    └──────────────┘
       │                                    ▲
       ▼                                    │
    Encrypt                              Decrypt
     (AES)                                (AES)
       │                                    │
       ▼                                    │
┌──────────────┐    Network/Disk    ┌──────────────┐
│  Ciphertext  │ ───────────────────►│  Ciphertext  │
│  "a7d9f3e1"  │                    │  "a7d9f3e1"  │
└──────────────┘                    └──────────────┘

     Same Key: K                         Same Key: K
```

**Implementation (AES-256-GCM):**

```python
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

def encrypt_symmetric(plaintext: bytes, key: bytes) -> dict:
    """
    Encrypt data using AES-256-GCM (authenticated encryption)

    Returns dict with:
    - ciphertext: encrypted data
    - nonce: initialization vector (must be unique per encryption)
    - tag: authentication tag (included in GCM ciphertext)
    """
    aesgcm = AESGCM(key)  # key must be 32 bytes for AES-256
    nonce = os.urandom(12)  # 96 bits for GCM

    # GCM mode provides both encryption AND authentication
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)

    return {
        "ciphertext": ciphertext,
        "nonce": nonce
    }

def decrypt_symmetric(ciphertext: bytes, nonce: bytes, key: bytes) -> bytes:
    """Decrypt AES-GCM ciphertext"""
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    return plaintext

# Usage
key = AESGCM.generate_key(bit_length=256)  # 32 bytes
message = b"User credit card: 4532-1234-5678-9010"

encrypted = encrypt_symmetric(message, key)
decrypted = decrypt_symmetric(
    encrypted["ciphertext"],
    encrypted["nonce"],
    key
)

assert decrypted == message
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Speed | Very fast (hardware accelerated) | N/A |
| Key distribution | Simple once shared | How do you share the key securely? |
| Use case | Bulk data encryption | Not suitable for initial key exchange |
| Performance | ~1-10 GB/s throughput | N/A |

**When to use:** Encrypting data at rest, encrypting large payloads after key exchange

**When NOT to use:** Initial communication setup (how do both sides get the same key?)

### Asymmetric Encryption

**The Problem:**
How do you establish a shared secret over an untrusted network? If you send the symmetric key in plaintext, an attacker can intercept it.

**How It Works:**

```
Different keys: Public key encrypts, Private key decrypts

┌──────────┐                              ┌──────────┐
│  Alice   │                              │   Bob    │
│          │                              │          │
│ Private  │                              │ Public   │
│   Key    │                              │   Key    │
└──────────┘                              └──────────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │  Plaintext   │
                                          │  "Secret"    │
                                          └──────────────┘
                                                 │
                                                 ▼
                                              Encrypt
                                             (RSA/ECDH)
                                                 │
                                                 ▼
┌──────────┐                              ┌──────────────┐
│  Alice   │◄─────── Network ─────────────│  Ciphertext  │
│          │                              │  "x9f2a1b5"  │
└──────────┘                              └──────────────┘
     │
     ▼
  Decrypt
  (Private)
     │
     ▼
┌──────────────┐
│  Plaintext   │
│  "Secret"    │
└──────────────┘
```

**Implementation (RSA):**

```python
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization

# Generate key pair (do this once, store private key securely)
def generate_keypair():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048  # 2048 bits minimum, 4096 for high security
    )
    public_key = private_key.public_key()
    return private_key, public_key

# Encrypt with public key
def encrypt_asymmetric(plaintext: bytes, public_key) -> bytes:
    ciphertext = public_key.encrypt(
        plaintext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return ciphertext

# Decrypt with private key
def decrypt_asymmetric(ciphertext: bytes, private_key) -> bytes:
    plaintext = private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return plaintext

# Usage
private_key, public_key = generate_keypair()

# Bob encrypts with Alice's public key
message = b"Shared secret key for AES"
ciphertext = encrypt_asymmetric(message, public_key)

# Only Alice can decrypt with her private key
decrypted = decrypt_asymmetric(ciphertext, private_key)
assert decrypted == message
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Speed | N/A | 1000x slower than symmetric |
| Key distribution | Public key can be shared openly | Need PKI to verify authenticity |
| Use case | Key exchange, digital signatures | Not for bulk data |
| Max message size | Limited (key size - padding) | RSA-2048 can encrypt ~190 bytes |

**When to use:** Establishing initial secure channel, digital signatures, small sensitive data

**When NOT to use:** Encrypting large files (use hybrid approach instead)

### Hybrid Encryption (Real-World Pattern)

**How TLS Actually Works:**

```
1. Use asymmetric to exchange a symmetric key
2. Use symmetric for all subsequent data

┌─────────────────────────────────────────────────┐
│  Client                           Server        │
│                                                  │
│  Generate random AES key                        │
│  Encrypt AES key with server's public key       │
│  ────────────────────────────────────────────►  │
│                                                  │
│                    Decrypt with private key     │
│                    Now both have same AES key   │
│                                                  │
│  ◄────────────────────────────────────────────  │
│           All data encrypted with AES           │
│  ────────────────────────────────────────────►  │
│  ◄────────────────────────────────────────────  │
│                                                  │
└─────────────────────────────────────────────────┘

Fast symmetric encryption for data
Asymmetric only for initial key exchange
```

---

## TLS Handshake Deep Dive

### TLS 1.2 Handshake (6 Steps)

```
Client                                              Server
  │                                                    │
  │  1. ClientHello                                    │
  │     - TLS version: 1.2                             │
  │     - Cipher suites: [TLS_RSA_WITH_AES_128_GCM...] │
  │     - Random bytes (32)                            │
  ├───────────────────────────────────────────────────►│
  │                                                    │
  │                                  2. ServerHello    │
  │                    - Chosen cipher: AES_128_GCM    │
  │                    - Server random (32 bytes)      │
  │                    - Certificate (public key)      │
  │                    - ServerHelloDone               │
  │◄───────────────────────────────────────────────────┤
  │                                                    │
  │  3. ClientKeyExchange                              │
  │     - Pre-master secret (encrypted with server pub)│
  ├───────────────────────────────────────────────────►│
  │                                                    │
  │  Both compute:                                     │
  │  Master Secret = PRF(pre-master, client_random,    │
  │                      server_random)                │
  │                                                    │
  │  4. ChangeCipherSpec + Finished                    │
  │     - Encrypted with derived keys                  │
  ├───────────────────────────────────────────────────►│
  │                                                    │
  │                    5. ChangeCipherSpec + Finished  │
  │                       - Encrypted confirmation     │
  │◄───────────────────────────────────────────────────┤
  │                                                    │
  │  6. Application Data (encrypted)                   │
  │◄──────────────────────────────────────────────────►│
  │                                                    │

Total: 2 round trips before data transfer
```

**Key Derivation (TLS 1.2):**

```python
def derive_keys_tls12(pre_master_secret, client_random, server_random):
    """
    Derive encryption keys from handshake parameters
    TLS 1.2 uses PRF (Pseudorandom Function) based on HMAC-SHA256
    """
    # Master secret = PRF(pre-master, "master secret", random)
    master_secret = prf(
        pre_master_secret,
        b"master secret",
        client_random + server_random,
        48  # 48 bytes
    )

    # Key material = PRF(master, "key expansion", random)
    key_material = prf(
        master_secret,
        b"key expansion",
        server_random + client_random,
        104  # Enough for all keys
    )

    # Split key material into:
    # - Client write MAC key
    # - Server write MAC key
    # - Client write encryption key
    # - Server write encryption key
    # - Client write IV
    # - Server write IV

    return parse_key_material(key_material)
```

### TLS 1.3 Handshake (Improved)

```
Client                                              Server
  │                                                    │
  │  1. ClientHello                                    │
  │     - Supported groups: [x25519, secp256r1]        │
  │     - Key share: [pre-generated DH key]            │
  │     - Cipher suites: [TLS_AES_128_GCM_SHA256]      │
  ├───────────────────────────────────────────────────►│
  │                                                    │
  │                                  2. ServerHello    │
  │                    - Key share (DH key)            │
  │                    {Certificate}*                  │
  │                    {Finished}*                     │
  │◄───────────────────────────────────────────────────┤
  │                                                    │
  │  * = Encrypted with derived handshake keys         │
  │                                                    │
  │  3. {Finished}                                     │
  ├───────────────────────────────────────────────────►│
  │                                                    │
  │  4. Application Data (encrypted)                   │
  │◄──────────────────────────────────────────────────►│
  │                                                    │

Total: 1 round trip before data transfer (50% faster!)
Certificate encrypted (privacy improvement)
```

**Major TLS 1.3 Improvements:**

```python
# 1. Forward Secrecy (Ephemeral Diffie-Hellman)
# Even if private key compromised, past sessions remain secure

# 2. Simplified cipher suites
TLS_1_2_SUITES = [
    "TLS_RSA_WITH_AES_128_CBC_SHA",
    "TLS_RSA_WITH_AES_256_CBC_SHA256",
    "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
    # 37 total cipher suites...
]

TLS_1_3_SUITES = [
    "TLS_AES_128_GCM_SHA256",
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256",
    # Only 5 cipher suites (all secure)
]

# 3. Removed insecure features
# - No more RSA key exchange (no forward secrecy)
# - No more CBC mode (padding oracle attacks)
# - No more SHA-1, MD5
# - No more compression (CRIME attack)
```

---

## Certificate Chains & PKI

### The Trust Problem

**How do you know the server's public key is legitimate?**

```
Without certificates:

Attacker intercepts connection
Sends own public key pretending to be bank.com
You encrypt password with attacker's public key
Attacker decrypts, reads password, re-encrypts with real bank key
You never know you were compromised
```

### Certificate Chain Verification

```
┌─────────────────────────────────────────────────┐
│  Root CA (Certificate Authority)                │
│  CN: GlobalSign Root CA                         │
│  Self-signed (everyone trusts this)             │
│  Validity: 20 years                             │
│  ┌───────────────────────────────────────────┐  │
│  │ Public Key: [root_public_key]             │  │
│  │ Signature: Self-signed                    │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                    │ Signs
                    ▼
┌─────────────────────────────────────────────────┐
│  Intermediate CA                                │
│  CN: GlobalSign Domain Validation CA           │
│  Signed by Root CA                              │
│  Validity: 5 years                              │
│  ┌───────────────────────────────────────────┐  │
│  │ Public Key: [intermediate_public_key]     │  │
│  │ Signature: Signed by Root CA              │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                    │ Signs
                    ▼
┌─────────────────────────────────────────────────┐
│  Server Certificate                             │
│  CN: api.example.com                            │
│  Signed by Intermediate CA                      │
│  Validity: 90 days (Let's Encrypt)              │
│  ┌───────────────────────────────────────────┐  │
│  │ Public Key: [server_public_key]           │  │
│  │ Signature: Signed by Intermediate CA      │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

**Verification Algorithm:**

```python
def verify_certificate_chain(server_cert, intermediate_cert, root_cert):
    """
    Verify certificate chain from server up to trusted root
    """
    # 1. Verify server cert is signed by intermediate
    intermediate_public_key = extract_public_key(intermediate_cert)
    if not verify_signature(server_cert, intermediate_public_key):
        raise CertificateError("Server cert not signed by intermediate")

    # 2. Verify intermediate cert is signed by root
    root_public_key = extract_public_key(root_cert)
    if not verify_signature(intermediate_cert, root_public_key):
        raise CertificateError("Intermediate cert not signed by root")

    # 3. Verify root cert is in trust store
    if root_cert not in TRUSTED_ROOT_STORE:
        raise CertificateError("Root CA not trusted")

    # 4. Check validity dates
    now = datetime.utcnow()
    for cert in [server_cert, intermediate_cert]:
        if not (cert.not_before <= now <= cert.not_after):
            raise CertificateError("Certificate expired or not yet valid")

    # 5. Check domain name matches
    if not check_hostname(server_cert, "api.example.com"):
        raise CertificateError("Certificate hostname mismatch")

    # 6. Check revocation status (OCSP or CRL)
    if is_revoked(server_cert):
        raise CertificateError("Certificate has been revoked")

    return True
```

### Certificate Pinning

**Problem:** Even valid certificates can be compromised (rogue CAs, government pressure)

**Solution:** Pin expected certificate or public key in client

```python
EXPECTED_PUBLIC_KEY_HASH = "sha256/x9e1b2c3d4..."

def verify_pinned_certificate(server_cert):
    """
    Certificate pinning: Only accept specific certificate
    """
    public_key = extract_public_key(server_cert)
    key_hash = hashlib.sha256(public_key).digest()

    if base64.b64encode(key_hash) != EXPECTED_PUBLIC_KEY_HASH:
        raise PinningError("Certificate does not match pinned value")

    return True
```

**Trade-off:** More secure but requires client updates when cert rotates

---

## Mutual TLS (mTLS)

### The Problem

Standard TLS: Server proves identity to client
Mutual TLS: Both sides prove identity

**Use case:** Service-to-service communication in microservices

### mTLS Handshake

```
Client                                              Server
  │                                                    │
  │  ClientHello                                       │
  ├───────────────────────────────────────────────────►│
  │                                                    │
  │                                  ServerHello       │
  │                                  Certificate       │
  │                                  CertificateRequest│
  │◄───────────────────────────────────────────────────┤
  │                                                    │
  │  Certificate (client cert)                         │
  │  CertificateVerify (proof of private key)          │
  │  Finished                                          │
  ├───────────────────────────────────────────────────►│
  │                                                    │
  │  Server verifies client certificate                │
  │                                                    │
  │                                  Finished          │
  │◄───────────────────────────────────────────────────┤
  │                                                    │
  │  Encrypted application data                        │
  │◄──────────────────────────────────────────────────►│
```

**Implementation (Python with client cert):**

```python
import ssl
import requests

# Server side: Require client certificates
def create_mtls_server():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    # Load server certificate
    context.load_cert_chain(
        certfile="/path/to/server.crt",
        keyfile="/path/to/server.key"
    )

    # Require client certificates
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile="/path/to/client-ca.crt")

    return context

# Client side: Present client certificate
def call_mtls_service(url):
    response = requests.get(
        url,
        cert=(
            "/path/to/client.crt",  # Client certificate
            "/path/to/client.key"   # Client private key
        ),
        verify="/path/to/server-ca.crt"  # Verify server cert
    )
    return response.json()
```

**mTLS at Scale (Service Mesh):**

```yaml
# Istio automatically injects mTLS between services
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
spec:
  mtls:
    mode: STRICT  # Require mTLS for all traffic

# Certificate rotation handled automatically (every 24h default)
```

**Trade-offs:**

| Aspect | Standard TLS | mTLS |
|--------|--------------|------|
| Security | Server authenticated | Both sides authenticated |
| Complexity | Simple | Complex cert management |
| Use case | Client-server (browser) | Service-to-service |
| Certificate rotation | Server cert only | Both client and server certs |
| Performance overhead | Low | Slightly higher (2 cert verifications) |

---

## Encryption at Rest

### Database Encryption

**Problem:** Database files stolen via disk theft, cloud snapshot, or insider access

**Transparent Data Encryption (TDE):**

```
┌─────────────────────────────────────────────────┐
│  Application                                    │
│  SELECT * FROM users WHERE id = 123             │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  Database Engine                                │
│  Query Execution (plaintext in memory)          │
└─────────────────────────────────────────────────┘
                    │
                    ▼
         Encryption Layer (TDE)
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  Disk Storage                                   │
│  Encrypted Pages: x7a9f2e1c8b4...              │
└─────────────────────────────────────────────────┘

Key stored in: HSM, KMS, or separate key file
```

**PostgreSQL TDE Example:**

```sql
-- Enable encryption (requires pgcrypto extension)
CREATE EXTENSION pgcrypto;

-- Column-level encryption
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email TEXT,
    ssn BYTEA  -- Encrypted column
);

-- Insert encrypted data
INSERT INTO users (email, ssn)
VALUES (
    'user@example.com',
    pgp_sym_encrypt('123-45-6789', 'encryption-key')
);

-- Query encrypted data
SELECT
    email,
    pgp_sym_decrypt(ssn, 'encryption-key') AS ssn_decrypted
FROM users
WHERE id = 1;
```

**MongoDB Encryption at Rest:**

```yaml
# mongod.conf
security:
  enableEncryption: true
  encryptionKeyFile: /path/to/keyfile

# Key rotation
db.adminCommand({
  rotateEncryptionKey: 1
})
```

### Application-Level Encryption

**When to use:** Need to encrypt before data reaches database (zero-trust)

```python
from cryptography.fernet import Fernet

class EncryptedField:
    """
    Encrypt sensitive fields at application layer
    Database sees only ciphertext
    """
    def __init__(self, encryption_key: bytes):
        self.fernet = Fernet(encryption_key)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt and return base64-encoded ciphertext"""
        ciphertext = self.fernet.encrypt(plaintext.encode())
        return ciphertext.decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt base64-encoded ciphertext"""
        plaintext = self.fernet.decrypt(ciphertext.encode())
        return plaintext.decode()

# Usage in ORM model
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    email = Column(String)
    _ssn_encrypted = Column("ssn", String)  # Stored encrypted

    _encryptor = EncryptedField(ENCRYPTION_KEY)

    @property
    def ssn(self) -> str:
        """Decrypt when accessed"""
        return self._encryptor.decrypt(self._ssn_encrypted)

    @ssn.setter
    def ssn(self, value: str):
        """Encrypt when set"""
        self._ssn_encrypted = self._encryptor.encrypt(value)

# Application code
user = User(email="user@example.com", ssn="123-45-6789")
session.add(user)
session.commit()

# Database contains only encrypted value
# Even DBA cannot see SSN without encryption key
```

---

## Key Management & Rotation

### The Key Management Problem

```
You encrypted everything. Great!
But now you have a new problem:

How do you store the encryption keys securely?

Bad solutions:
❌ Hard-code in source code (exposed in git)
❌ Environment variable (visible in process list, logs)
❌ Config file on disk (stolen with database dump)

The Paradox:
Encryption keys need to be encrypted...
But what encrypts those keys?
```

### Key Hierarchy (Envelope Encryption)

```
┌─────────────────────────────────────────────────┐
│  Master Key (Root Key)                          │
│  Stored in: HSM, KMS                            │
│  Never leaves secure boundary                   │
│  Rotated: Rarely (years)                        │
└─────────────────────────────────────────────────┘
                    │ Encrypts
                    ▼
┌─────────────────────────────────────────────────┐
│  Data Encryption Key (DEK)                      │
│  Generated per: table, file, tenant             │
│  Stored encrypted in database/disk              │
│  Rotated: Frequently (days/months)              │
└─────────────────────────────────────────────────┘
                    │ Encrypts
                    ▼
┌─────────────────────────────────────────────────┐
│  Actual Data                                    │
│  User records, files, etc.                      │
└─────────────────────────────────────────────────┘
```

**Envelope Encryption Implementation:**

```python
import boto3
from cryptography.fernet import Fernet

class EnvelopeEncryption:
    """
    Use AWS KMS master key to encrypt data encryption keys
    """
    def __init__(self, kms_key_id: str):
        self.kms = boto3.client('kms')
        self.kms_key_id = kms_key_id

    def encrypt_data(self, plaintext: bytes) -> dict:
        """
        1. Generate random DEK
        2. Encrypt data with DEK
        3. Encrypt DEK with KMS master key
        4. Return encrypted data + encrypted DEK
        """
        # Generate data encryption key
        dek = Fernet.generate_key()

        # Encrypt data with DEK
        fernet = Fernet(dek)
        ciphertext = fernet.encrypt(plaintext)

        # Encrypt DEK with KMS master key
        response = self.kms.encrypt(
            KeyId=self.kms_key_id,
            Plaintext=dek
        )
        encrypted_dek = response['CiphertextBlob']

        return {
            'ciphertext': ciphertext,
            'encrypted_key': encrypted_dek
        }

    def decrypt_data(self, ciphertext: bytes, encrypted_dek: bytes) -> bytes:
        """
        1. Decrypt DEK using KMS master key
        2. Decrypt data using decrypted DEK
        """
        # Decrypt DEK with KMS
        response = self.kms.decrypt(
            CiphertextBlob=encrypted_dek
        )
        dek = response['Plaintext']

        # Decrypt data with DEK
        fernet = Fernet(dek)
        plaintext = fernet.decrypt(ciphertext)

        return plaintext

# Usage
encryptor = EnvelopeEncryption(kms_key_id='arn:aws:kms:...')

# Encrypt
data = b"Sensitive customer data"
encrypted = encryptor.encrypt_data(data)

# Store in database
db.save({
    'data': encrypted['ciphertext'],
    'key': encrypted['encrypted_key']
})

# Decrypt
record = db.load(id=123)
plaintext = encryptor.decrypt_data(
    record['data'],
    record['key']
)
```

### Key Rotation Strategies

**Why rotate keys?**
- Limit blast radius if key compromised
- Compliance requirements (PCI-DSS: annual rotation)
- Reduce cryptanalysis risk

**Strategy 1: Re-encrypt all data**

```python
def rotate_encryption_key_full(old_key: bytes, new_key: bytes):
    """
    Decrypt all data with old key, re-encrypt with new key
    Downside: Expensive, requires downtime or read-write lock
    """
    records = db.query("SELECT id, encrypted_data FROM sensitive_table")

    old_fernet = Fernet(old_key)
    new_fernet = Fernet(new_key)

    for record in records:
        # Decrypt with old key
        plaintext = old_fernet.decrypt(record['encrypted_data'])

        # Re-encrypt with new key
        new_ciphertext = new_fernet.encrypt(plaintext)

        # Update database
        db.execute(
            "UPDATE sensitive_table SET encrypted_data = ? WHERE id = ?",
            new_ciphertext, record['id']
        )
```

**Strategy 2: Versioned keys (gradual rotation)**

```python
class VersionedEncryption:
    """
    Store key version with each encrypted value
    Gradually re-encrypt on read/write
    """
    def __init__(self):
        self.keys = {
            1: Fernet(KEY_V1),
            2: Fernet(KEY_V2),
            3: Fernet(KEY_V3)  # Current key
        }
        self.current_version = 3

    def encrypt(self, plaintext: bytes) -> bytes:
        """Always encrypt with current version"""
        version_byte = self.current_version.to_bytes(1, 'big')
        ciphertext = self.keys[self.current_version].encrypt(plaintext)
        return version_byte + ciphertext

    def decrypt(self, versioned_ciphertext: bytes) -> bytes:
        """Decrypt using version prefix"""
        version = int.from_bytes(versioned_ciphertext[:1], 'big')
        ciphertext = versioned_ciphertext[1:]

        if version not in self.keys:
            raise ValueError(f"Unknown key version: {version}")

        return self.keys[version].decrypt(ciphertext)

    def should_reencrypt(self, versioned_ciphertext: bytes) -> bool:
        """Check if using old key version"""
        version = int.from_bytes(versioned_ciphertext[:1], 'big')
        return version != self.current_version

# Gradual migration on read
def get_user(user_id):
    user = db.get(user_id)

    if encryptor.should_reencrypt(user.encrypted_ssn):
        # Decrypt with old key
        plaintext = encryptor.decrypt(user.encrypted_ssn)

        # Re-encrypt with new key
        user.encrypted_ssn = encryptor.encrypt(plaintext)
        db.save(user)

    return user
```

---

## Hardware Security Modules (HSMs)

### The Problem

Software-based key storage is vulnerable:
- Memory dumps can expose keys
- OS vulnerabilities can leak keys
- Privileged users can access key files

### What is an HSM?

```
┌─────────────────────────────────────────────────┐
│  Hardware Security Module (HSM)                 │
│  ┌───────────────────────────────────────────┐  │
│  │  Tamper-resistant hardware chip           │  │
│  │  - Keys never leave device                │  │
│  │  - Cryptographic operations in hardware   │  │
│  │  - Physical destruction if tampered       │  │
│  │                                            │  │
│  │  ┌──────────────────────────────────┐     │  │
│  │  │  Master Keys (stored in chip)    │     │  │
│  │  │  - Cannot be exported            │     │  │
│  │  │  - Cannot be read                │     │  │
│  │  └──────────────────────────────────┘     │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  API Interface:                                  │
│  - Encrypt(plaintext) → ciphertext               │
│  - Decrypt(ciphertext) → plaintext               │
│  - Sign(data) → signature                        │
│  - GenerateKey() → key_id                        │
└─────────────────────────────────────────────────┘
```

**Usage (AWS CloudHSM):**

```python
import cloudhsm_client

# Connect to CloudHSM
hsm = cloudhsm_client.CloudHsmClient()

# Generate key in HSM (never leaves device)
key_handle = hsm.generate_aes_key(key_size=256)

# Encrypt using key in HSM
ciphertext = hsm.encrypt(
    key_handle=key_handle,
    plaintext=b"Secret data"
)

# Decrypt using key in HSM
plaintext = hsm.decrypt(
    key_handle=key_handle,
    ciphertext=ciphertext
)

# Key never exposed to application
# All crypto operations happen inside HSM
```

**HSM vs KMS:**

| Aspect | KMS (Key Management Service) | HSM (Hardware Security Module) |
|--------|------------------------------|--------------------------------|
| Hardware | Shared multi-tenant | Dedicated hardware (or single-tenant) |
| Performance | High latency (API calls) | Low latency (dedicated) |
| Cost | $1/key/month | $1000-5000/month |
| Compliance | Most use cases | FIPS 140-2 Level 3, PCI-DSS |
| Use case | General purpose | High security, compliance-critical |

**When to use HSM:** Payment processing, certificate authorities, crypto key signing

**When NOT to use HSM:** General web applications (KMS is sufficient and cheaper)

---

## Comparison: TLS 1.2 vs TLS 1.3

| Aspect | TLS 1.2 | TLS 1.3 |
|--------|---------|---------|
| Handshake latency | 2 round trips | 1 round trip (0-RTT possible) |
| Cipher suites | 37 options (many insecure) | 5 options (all secure) |
| Forward secrecy | Optional (DHE/ECDHE) | Mandatory (all suites) |
| Certificate encryption | No (sent plaintext) | Yes (encrypted after ServerHello) |
| Session resumption | Session IDs, tickets | Pre-shared keys (PSK) |
| RSA key exchange | Supported | Removed (no forward secrecy) |
| CBC cipher mode | Supported | Removed (padding oracle attacks) |
| Performance | Baseline | 20-30% faster |
| Browser support | Universal | 95%+ (IE 11 unsupported) |

---

## Key Concepts Checklist

- [ ] Explain symmetric vs asymmetric encryption and when to use each
- [ ] Describe TLS 1.2 vs 1.3 handshake differences
- [ ] Verify certificate chains from leaf to root CA
- [ ] Implement mTLS for service-to-service authentication
- [ ] Design encryption at rest strategy (TDE vs application-level)
- [ ] Implement envelope encryption with key hierarchy
- [ ] Plan key rotation strategy (full re-encrypt vs versioned keys)
- [ ] Know when to use HSM vs KMS

---

## Practical Insights

**TLS configuration mistakes:**
```nginx
# BAD: Allows insecure TLS 1.0/1.1
ssl_protocols TLSv1 TLSv1.1 TLSv1.2;

# GOOD: Modern TLS only
ssl_protocols TLSv1.2 TLSv1.3;

# BAD: Weak cipher suites enabled
ssl_ciphers HIGH:!aNULL:!MD5;

# GOOD: Specific strong ciphers
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
ssl_prefer_server_ciphers on;

# Essential security headers
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
```

**Certificate lifecycle at scale:**
- Automate renewal (Let's Encrypt + certbot, AWS Certificate Manager)
- Monitor expiration (alert 30 days before)
- Use short-lived certificates (90 days) to limit compromise window
- Centralize cert management (HashiCorp Vault, cert-manager in Kubernetes)

**Performance impact of encryption:**
```
Benchmark: Processing 1 GB of data

Plaintext:              100 ms
AES-256-GCM:            150 ms  (50% overhead)
RSA-2048 encryption:    25,000 ms  (250x slower!)

Takeaway: Use symmetric for data, asymmetric only for key exchange
Hardware AES acceleration: Reduces overhead to <10%
```

**Key rotation frequency:**
```
Master keys (HSM/KMS):    1-2 years (or on compromise)
Data encryption keys:     30-90 days
TLS certificates:         90 days (Let's Encrypt standard)
API keys/tokens:          On user request or compromise

Compliance requirements:
- PCI-DSS: Annual key rotation
- HIPAA: Reasonable and appropriate (often interpreted as annual)
- SOC 2: Document and follow rotation policy
```

**mTLS at scale (service mesh benefits):**
- Automatic certificate provisioning (no manual cert distribution)
- Short-lived certificates (1-24 hour TTL, auto-renewed)
- Zero-trust networking (every connection authenticated)
- Centralized policy (Istio, Linkerd, Consul Connect)
- Observability (see which services communicate, detect anomalies)

**Common pitfalls:**
```python
# DON'T: Use ECB mode (patterns visible in ciphertext)
cipher = AES.new(key, AES.MODE_ECB)

# DO: Use GCM mode (authenticated encryption)
cipher = AES.new(key, AES.MODE_GCM)

# DON'T: Reuse IV/nonce for same key
nonce = b"00000000"  # Same every time → BREAKS SECURITY

# DO: Generate random nonce per encryption
nonce = os.urandom(12)  # Unique every time

# DON'T: Store encryption key with encrypted data
db.save({"data": encrypted_data, "key": encryption_key})

# DO: Use envelope encryption with separate key storage
db.save({"data": encrypted_data, "encrypted_dek": kms_encrypted_key})
```

**When to use application-level encryption:**
- Zero-trust: Don't trust database administrators
- Multi-tenancy: Per-tenant encryption keys
- Compliance: Data must be encrypted before leaving application boundary
- Selective encryption: Only specific fields need encryption (not full database)

**Cost considerations:**
```
AWS KMS:
- $1/key/month
- $0.03 per 10,000 requests
- Cost at 100 RPS: ~$8/month

AWS CloudHSM:
- $1.60/hour ($1,168/month minimum)
- Use case: Payment processing, certificate authorities

Rule of thumb:
- <1000 encryptions/sec: KMS
- >1000 encryptions/sec + compliance: CloudHSM
- >10,000 encryptions/sec: Application-level with KMS envelope encryption
```
