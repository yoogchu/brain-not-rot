# Chapter 7: CDNs & Edge Computing

## The Speed of Light Problem

Physics imposes hard limits on latency:

```
Speed of light in fiber: ~200,000 km/s
NYC to Singapore distance: ~15,000 km
Minimum round-trip time: ~150ms (before any server processing!)

If your server is in NYC:
- NYC user: 10ms RTT
- London user: 70ms RTT
- Singapore user: 150ms+ RTT
- Sydney user: 200ms+ RTT
```

Users perceive **100ms** as "sluggish" and **1 second** as "broken."

Singapore users hitting a NYC server will ALWAYS have a terrible experience. No amount of server optimization can fix physics.

**CDN Solution:** Cache content at edge locations worldwide.

```
Origin Server (NYC)
      │
      └──────────────────────────────────────┐
      │                                      │
Edge PoP (NYC)    Edge PoP (Frankfurt)    Edge PoP (Singapore)
      │                   │                     │
      ▼                   ▼                     ▼
  US Users            EU Users              APAC Users
  (10ms)              (20ms)                (20ms)
```

Instead of everyone going to NYC, users hit nearby edge servers.

---

## CDN Architecture

### Points of Presence (PoPs)

PoPs are data centers at network edges worldwide:

```
Major CDN PoP counts:
- Cloudflare: 310+ cities
- AWS CloudFront: 410+ PoPs  
- Fastly: 90+ PoPs (focused on major metros)
- Akamai: 4000+ PoPs (most extensive)
```

More PoPs = closer to more users = lower latency.

### Request Flow

```
1. User requests: cdn.example.com/image.jpg
   
2. DNS Resolution:
   cdn.example.com → CDN's DNS
   CDN DNS returns IP of nearest PoP (geolocation)
   
3. User connects to nearest PoP

4. Edge Server checks local cache:
   
   CACHE HIT (fast path):
   └─> Return cached content immediately
       Total latency: ~20ms
   
   CACHE MISS (slow path):
   └─> Fetch from origin server
       Cache the response
       Return to user
       Total latency: ~200ms (first request)
       
5. Subsequent requests: CACHE HIT (~20ms)
```

### Cache Hierarchy

Modern CDNs have multiple cache layers:

```
┌─────────────────────────────────────────────────┐
│                  Origin Server                   │
│                  (Your server)                   │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│              Origin Shield (Optional)            │
│         (Single cache in front of origin)        │
│         Reduces origin load significantly        │
└─────────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Regional   │ │  Regional   │ │  Regional   │
│   Cache     │ │   Cache     │ │   Cache     │
│  (US-East)  │ │  (EU-West)  │ │   (APAC)    │
└─────────────┘ └─────────────┘ └─────────────┘
        │             │             │
    ┌───┴───┐     ┌───┴───┐     ┌───┴───┐
    ▼       ▼     ▼       ▼     ▼       ▼
┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
│Edge 1││Edge 2││Edge 3││Edge 4││Edge 5││Edge 6│
│(NYC) ││(DC)  ││(LON) ││(FRA) ││(SIN) ││(TYO) │
└──────┘└──────┘└──────┘└──────┘└──────┘└──────┘
```

Cache miss at Edge → check Regional → check Origin Shield → fetch Origin

---

## Caching Strategies

### Pull CDN (On-Demand)

Most common. Content cached on first request.

```
First request for /image.jpg:
1. Edge: MISS
2. Regional: MISS  
3. Origin Shield: MISS
4. Fetch from origin
5. Cache at all layers
6. Return to user (slow: ~500ms)

Second request for /image.jpg:
1. Edge: HIT
2. Return immediately (fast: ~20ms)
```

**Pros:**
- Simple setup (just point DNS to CDN)
- Only caches requested content (efficient)
- Automatic cache management

**Cons:**
- First request is slow (cold cache)
- Origin still gets traffic on misses
- Cache misses during traffic spikes

### Push CDN (Pre-population)

You upload content directly to CDN:

```
Deploy process:
1. Generate static assets
2. Upload to CDN storage (S3-like)
3. CDN distributes to all edges proactively
4. Content available everywhere immediately

No "first request" penalty!
```

**Pros:**
- No cold cache problem
- Origin never hit for cached content
- Predictable performance

**Cons:**
- Manual management of content
- Higher storage costs
- Must handle invalidation explicitly
- Only works for known, static content

**Use case:** Large files (videos, software downloads, game assets)

### Hybrid Approach

```
Static assets: Push to CDN
  /static/app.js
  /static/logo.png
  /videos/*.mp4

Dynamic content: Pull through CDN
  /api/products (cache 5 minutes)
  /api/user (no cache, but still benefit from CDN network)
```

---

## Cache Control Headers

### Cache-Control Header

```http
# Public cache for 1 hour
Cache-Control: public, max-age=3600

# Cache for 1 day, stale OK while revalidating
Cache-Control: public, max-age=86400, stale-while-revalidate=3600

# No caching at all (dynamic/personalized content)
Cache-Control: no-store

# Cache but always revalidate with origin
Cache-Control: no-cache

# CDN can cache, but browser cannot
Cache-Control: s-maxage=3600, max-age=0

# Immutable content (versioned URLs)
Cache-Control: public, max-age=31536000, immutable
```

**Key directives:**

| Directive | Meaning |
|-----------|---------|
| `public` | Any cache can store (CDN, browser) |
| `private` | Only browser can cache (personalized content) |
| `max-age=N` | Cache for N seconds |
| `s-maxage=N` | CDN-specific max-age (overrides max-age for CDN) |
| `no-cache` | Must revalidate before using cached copy |
| `no-store` | Never cache |
| `immutable` | Content will never change (skip revalidation) |
| `stale-while-revalidate=N` | Serve stale for N seconds while fetching fresh |

### ETag (Validation)

```http
# Origin response:
HTTP/1.1 200 OK
ETag: "abc123def456"
Content-Type: image/jpeg

[image data]

# Later request with validation:
GET /image.jpg HTTP/1.1
If-None-Match: "abc123def456"

# If unchanged:
HTTP/1.1 304 Not Modified
(no body - use cached version)

# If changed:
HTTP/1.1 200 OK
ETag: "xyz789"
[new image data]
```

ETags save bandwidth when content hasn't changed.

### Vary Header

Cache different versions for different request characteristics:

```http
# Cache different versions per Accept-Language
Vary: Accept-Language

# Cache different versions per Accept-Encoding
Vary: Accept-Encoding

# Be careful with Vary!
Vary: Cookie  # BAD: Every user gets unique cache = no sharing
Vary: User-Agent  # BAD: Thousands of user agents = cache explosion
```

---

## Cache Invalidation

**"There are only two hard things in Computer Science: cache invalidation and naming things."**

Content changed at origin, but CDN still has old version. Now what?

### Strategy 1: TTL Expiration

```http
Cache-Control: max-age=60
```

Content automatically expires after 60 seconds.

**Pros:** Simple, automatic
**Cons:** Content stale for up to TTL duration

**Trade-off:**
```
Short TTL (60s): Fresher content, more origin traffic
Long TTL (1 day): Stale content, less origin traffic
```

### Strategy 2: Purge API

Explicitly tell CDN to remove cached content:

```python
# Cloudflare
cloudflare.purge_cache(
    zone_id="abc123",
    files=["https://example.com/image.jpg"]
)

# AWS CloudFront
cloudfront.create_invalidation(
    DistributionId="EDFDVBD6EXAMPLE",
    InvalidationBatch={
        "Paths": {"Quantity": 1, "Items": ["/image.jpg"]},
        "CallerReference": "unique-id"
    }
)

# Fastly
fastly.purge_key("product-123")  # Surrogate key purging
```

**Pros:** Immediate invalidation
**Cons:** API calls needed, propagation delay (seconds to minutes), costs money at scale

### Strategy 3: Versioned URLs (Best Practice)

```
Old URL: /static/style.css
New URL: /static/style.v2.css
   - or -
New URL: /static/style.css?v=abc123

HTML references new URL:
<link rel="stylesheet" href="/static/style.v2.css">
```

**Benefits:**
- Instant "invalidation" (new URL = cache miss = fetch fresh)
- Old and new versions coexist (graceful rollout)
- Rollback by reverting to old URL
- Can use very long TTL (immutable content)
- No purge API calls needed

**Implementation:**
```python
# Build process adds content hash to filename
style.css → style.a1b2c3d4.css

# Or query string version
style.css?v=a1b2c3d4

# Reference in HTML with hash
<link href="/static/style.{{ file_hash }}.css">
```

**This is how all major sites handle static assets.**

### Strategy 4: Surrogate Keys (Advanced)

Tag cached content, purge by tag:

```python
# Response includes surrogate key
X-Surrogate-Key: product-123 category-electronics

# Later, purge all content for product-123
fastly.purge_key("product-123")
# Purges: /products/123, /products/123/reviews, /products/123/images
```

**Use case:** Complex content relationships (product page, related images, reviews)

---

## CDN Security Features

### DDoS Protection

```
Attack traffic: 100 Gbps aimed at your server
Your server: 1 Gbps capacity = instant death

With CDN:
Attack traffic distributed across 300+ PoPs
Each PoP absorbs small fraction
CDN's total capacity: 100+ Tbps
Your origin protected, never sees attack traffic
```

CDNs are the first line of defense against DDoS.

### WAF (Web Application Firewall)

Rules at edge block malicious requests:

```
Block SQL injection:
  /search?q='; DROP TABLE users; --
  → Blocked at edge, never reaches origin

Block XSS:
  /comment?text=<script>evil()</script>
  → Blocked at edge

Block bad bots:
  User-Agent matches known scraper patterns
  → Blocked or challenged

Rate limiting:
  > 100 requests/second from same IP
  → Throttled or blocked
```

### Bot Management

```
Good bots (allow):
- Googlebot, Bingbot (search indexing)
- Uptimerobot (monitoring)

Bad bots (block/challenge):
- Scrapers stealing content
- Credential stuffing attacks
- Inventory hoarding bots

Detection methods:
- User-Agent analysis
- JavaScript challenges (bots can't execute JS)
- CAPTCHA challenges
- Behavioral analysis (request patterns)
- Browser fingerprinting
```

---

## Edge Computing

### Beyond Caching: Run Code at the Edge

Traditional CDN: Cache and serve static content
Edge Computing: Execute code at edge locations

```
Traditional:
User → Edge (cache) → Origin (logic)

Edge Computing:
User → Edge (cache + logic)
```

### Cloudflare Workers Example

```javascript
// Runs at 300+ edge locations worldwide
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  const url = new URL(request.url)
  
  // A/B testing at edge (no origin round-trip!)
  const variant = Math.random() < 0.5 ? 'A' : 'B'
  
  // Modify request to appropriate backend
  url.pathname = `/experiment-${variant}${url.pathname}`
  
  const response = await fetch(url, request)
  
  // Add experiment tracking header
  const newResponse = new Response(response.body, response)
  newResponse.headers.set('X-Experiment', variant)
  
  return newResponse
}
```

### Edge Computing Use Cases

| Use Case | Benefit |
|----------|---------|
| A/B testing | No origin round-trip for variant selection |
| Authentication | Validate JWT at edge, reject bad requests early |
| Personalization | Inject user-specific content at edge |
| Image optimization | Resize/compress images on-the-fly |
| Geolocation | Route/customize based on user location |
| API response transformation | Modify JSON responses |
| Security | Block attacks before they reach origin |

### Edge Computing Platforms

| Platform | Runtime | Key Feature |
|----------|---------|-------------|
| Cloudflare Workers | V8 isolates | Fastest cold start (~0ms) |
| AWS Lambda@Edge | Node.js | CloudFront integration |
| Fastly Compute@Edge | WebAssembly | Sub-millisecond startup |
| Vercel Edge Functions | V8 | Next.js integration |
| Deno Deploy | V8/Deno | TypeScript native |

---

## CDN Selection Criteria

| Factor | What to Evaluate |
|--------|------------------|
| PoP locations | Coverage in your user geographies |
| Performance | Actual latency (not just PoP count) |
| Cache hit ratio | How effectively they cache |
| Origin shield | Reduce origin load |
| Edge compute | Need to run code at edge? |
| WAF/Security | DDoS protection, bot management |
| Pricing | Bandwidth, requests, features |
| Analytics | Real-time visibility |
| API/Automation | Purge API, deploy automation |
| Support | Enterprise support quality |

### Quick Comparison

| CDN | Best For |
|-----|----------|
| Cloudflare | General purpose, great free tier, Workers |
| AWS CloudFront | AWS ecosystem integration |
| Fastly | Real-time purging, edge compute |
| Akamai | Enterprise, maximum coverage |
| Bunny CDN | Budget-friendly, good performance |

---

## Interview Checklist

- [ ] Identify cacheable vs dynamic content
- [ ] Choose cache invalidation strategy (prefer versioned URLs)
- [ ] Design cache hierarchy (edge → regional → origin)
- [ ] Set appropriate Cache-Control headers
- [ ] Consider security features needed (WAF, DDoS)
- [ ] Calculate cost (bandwidth pricing varies wildly)
- [ ] Address failover if CDN fails
- [ ] Consider edge computing for logic at the edge

---

## Staff+ Insights

**Cache hit ratio is everything:**
- 90% hit ratio: Origin handles 10% of traffic
- 99% hit ratio: Origin handles 1% of traffic (10x reduction!)
- Monitor and optimize for hit ratio

**Versioned URLs > Purge API:**
- Purge API is expensive at scale
- Purge propagation takes time
- Versioned URLs are instant and free

**Origin shield saves money:**
- Without: Each PoP cache miss hits origin
- With: All PoPs share one cache → origin traffic reduced 90%+
- Worth the small extra latency

**CDN failures happen:**
- Have fallback to origin
- Monitor CDN health
- Consider multi-CDN for critical services

**Edge computing trade-offs:**
- Adds complexity
- Debugging is harder (distributed logs)
- Cold starts can add latency
- Use for: High-value computations that benefit from proximity
- Don't use for: Simple pass-through or rarely-accessed endpoints
