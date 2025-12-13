# Chapter 33: LLM Serving & Optimization

## Why Optimize LLM Serving?

Without optimization:

```
User sends 100-token prompt
Model: Llama 70B on A100 80GB GPU

Without optimization:
- First token: 500ms (prompt processing)
- Subsequent tokens: 80ms each (generation)
- GPU Memory: 150GB needed → Out of Memory
- Cost: $3.50/hour GPU idle while waiting

With optimization:
- First token: 100ms (KV cache)
- Subsequent tokens: 30ms each (batching)
- GPU Memory: 45GB (quantization + paging)
- Cost: Same GPU serves 10x more requests
```

LLM inference is fundamentally different from training:
- **Memory-bound, not compute-bound** (70% of GPU memory is KV cache)
- **Latency-sensitive** (users notice 100ms delays)
- **Cost-intensive** ($1-10 per million tokens on hosted APIs)

---

## Autoregressive Generation Fundamentals

### The Problem

LLMs generate text one token at a time, but each token needs context from ALL previous tokens.

```
Prompt: "Write a poem about"
Token 1: " the" → needs prompt context
Token 2: " ocean" → needs prompt + token 1
Token 3: "'s" → needs prompt + tokens 1-2
...
```

Each token requires running the full transformer forward pass. This gets expensive fast.

### How It Works

```
┌──────────────────────────────────────────────────────┐
│              Autoregressive Generation                │
└──────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Prefill    │  │  Decode 1   │  │  Decode 2   │
│  (prompt)   │  │  (token 1)  │  │  (token 2)  │
├─────────────┤  ├─────────────┤  ├─────────────┤
│ Input: Full │  │ Input: tok1 │  │ Input: tok2 │
│   prompt    │  │ + KV cache  │  │ + KV cache  │
│ Output: KV  │  │ Output: KV  │  │ Output: KV  │
│   cache +   │  │   update +  │  │   update +  │
│   token 1   │  │   token 2   │  │   token 3   │
└─────────────┘  └─────────────┘  └─────────────┘
    Compute          Memory          Memory
    intensive        intensive       intensive
```

**Prefill Phase (Compute-bound):**
- Process entire prompt in parallel
- Generate KV cache for all prompt tokens
- Latency: ~500ms for 1000-token prompt on A100

**Decode Phase (Memory-bound):**
- Generate one token at a time
- Read KV cache from memory for attention
- Latency: ~50-100ms per token

**Memory breakdown for Llama 70B:**

```python
# Model parameters: 70B * 2 bytes (FP16) = 140GB
model_memory = 70e9 * 2

# KV cache per token:
# - 80 layers * 2 (key + value) * 8 heads * 128 dim * 2 bytes
kv_per_token = 80 * 2 * 8 * 128 * 2  # = 327,680 bytes ≈ 0.33MB

# For 2048-token context:
kv_memory = kv_per_token * 2048  # ≈ 671MB per request

# Batch size 32:
total_kv = kv_memory * 32  # ≈ 21GB just for KV cache!
```

**Why this matters:**
- KV cache grows with sequence length and batch size
- Memory bottleneck limits throughput
- Cannot serve multiple requests without smart memory management

---

## KV Cache & Memory Management

### The Problem

KV cache stores attention keys and values for all previous tokens. For long contexts or large batches, this explodes:

```
Without KV cache: Recompute attention for all tokens every step
- 100 tokens generated = 100 full forward passes = 5050 token computations

With KV cache: Store previous computations
- 100 tokens generated = 100 incremental computations = 100 token computations
- 50x speedup!
- But: Memory grows linearly with sequence length
```

### Memory Explosion Example

```
Model: GPT-3 style (96 layers, 96 heads, 128 dim per head)
Context length: 4096 tokens
Batch size: 16

KV cache size:
= batch * seq_len * layers * 2 * heads * dim * bytes
= 16 * 4096 * 96 * 2 * 96 * 128 * 2
= 24GB of KV cache alone!

This is more than the model weights on an A100 40GB.
```

### Traditional Approach: Static Allocation

```python
class NaiveKVCache:
    def __init__(self, batch_size, max_seq_len, num_layers, num_heads, head_dim):
        # Pre-allocate worst case
        self.cache = torch.zeros(
            batch_size,
            max_seq_len,  # Often padded to 4096 or 8192
            num_layers,
            2,  # key and value
            num_heads,
            head_dim,
            dtype=torch.float16
        )

    def get_cache(self, layer_idx, batch_idx):
        return self.cache[batch_idx, :, layer_idx]
```

**Problems:**
- Wastes memory on short sequences
- Fixed batch size (cannot mix different lengths)
- Fragmentation as sequences finish

---

## PagedAttention & vLLM

### The Problem

Traditional KV cache allocates contiguous memory blocks. Sequences of different lengths waste space:

```
┌────────────────────────────────────────────────┐
│ Seq 1: "Hi" (2 tokens, allocated 2048)        │
│ [KV][KV][  wasted space (2046 slots)       ]  │
├────────────────────────────────────────────────┤
│ Seq 2: "Write essay" (1500 tokens)            │
│ [KV][KV]...[KV][  wasted (548 slots)       ]  │
└────────────────────────────────────────────────┘

30-40% memory wasted on fragmentation!
```

### How PagedAttention Works

Inspired by virtual memory in operating systems. Break KV cache into fixed-size pages:

```
┌──────────────────────────────────────────────────────┐
│           Virtual Memory (Page Table)                 │
└──────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐        ┌─────────────────┐
│   Sequence 1    │        │   Sequence 2    │
│  Logical Pages  │        │  Logical Pages  │
│  [0][1][2]      │        │  [0][1][2][3]   │
└─────────────────┘        └─────────────────┘
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────────────────┐
│         Physical Memory (KV Cache Blocks)             │
│  [Page][Page][Page][Page][Page][Page][Page][Page]    │
│    0     1     2     3     4     5     6     7       │
└──────────────────────────────────────────────────────┘

Page size: 16 tokens
Sequence 1 (40 tokens): Uses pages 0, 1, 2 (partially)
Sequence 2 (70 tokens): Uses pages 3, 4, 5, 6 (partially)

No fragmentation! Only last page of each sequence is partial.
```

**Implementation concept:**

```python
class PagedKVCache:
    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_dim):
        # Physical memory: pool of blocks
        self.blocks = torch.zeros(
            num_blocks,
            block_size,  # e.g., 16 tokens per block
            num_layers,
            2,  # key, value
            num_heads,
            head_dim,
            dtype=torch.float16
        )

        # Free block pool
        self.free_blocks = list(range(num_blocks))

        # Page table: sequence_id -> [block_ids]
        self.page_table = {}

    def allocate_sequence(self, sequence_id):
        """Allocate first block for new sequence."""
        if not self.free_blocks:
            raise OutOfMemoryError("No free blocks")

        block_id = self.free_blocks.pop(0)
        self.page_table[sequence_id] = [block_id]
        return block_id

    def append_token(self, sequence_id, token_kv):
        """Add token's KV to sequence, allocate new block if needed."""
        blocks = self.page_table[sequence_id]
        current_block = blocks[-1]
        block_offset = self.get_block_offset(sequence_id)

        # Check if current block is full
        if block_offset >= self.block_size:
            # Allocate new block
            if not self.free_blocks:
                raise OutOfMemoryError("No free blocks")
            new_block = self.free_blocks.pop(0)
            blocks.append(new_block)
            current_block = new_block
            block_offset = 0

        # Write KV to block
        self.blocks[current_block][block_offset] = token_kv

    def free_sequence(self, sequence_id):
        """Return all blocks to free pool."""
        blocks = self.page_table.pop(sequence_id)
        self.free_blocks.extend(blocks)
```

**vLLM architecture:**

```
┌──────────────────────────────────────────────────────┐
│                    vLLM Engine                        │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌────────────────┐        ┌────────────────┐       │
│  │   Scheduler    │───────►│ Continuous     │       │
│  │                │        │ Batching       │       │
│  └────────────────┘        └────────────────┘       │
│         │                          │                 │
│         │                          ▼                 │
│         │                  ┌────────────────┐       │
│         └─────────────────►│ PagedAttention │       │
│                            │   KV Cache     │       │
│                            └────────────────┘       │
│                                    │                 │
│                                    ▼                 │
│                            ┌────────────────┐       │
│                            │  GPU Kernels   │       │
│                            └────────────────┘       │
└──────────────────────────────────────────────────────┘
```

**Trade-offs:**

| Aspect | PagedAttention | Static Allocation |
|--------|----------------|-------------------|
| Memory efficiency | 90%+ utilization | 50-60% (fragmentation) |
| Complexity | Higher (page table management) | Simple |
| Throughput | 2-4x higher | Baseline |
| Latency | Same | Same |
| Implementation | Requires custom CUDA kernels | Standard PyTorch |

**When to use:** Production serving with variable-length requests
**When NOT to use:** Single-user inference, research prototyping

---

## Continuous Batching

### The Problem

Traditional batching waits for all sequences in batch to complete:

```
Batch size: 4
Seq lengths: [10, 50, 200, 500] tokens

┌────────────────────────────────────────────────────┐
│ Seq 1: ████████████ (done at step 10)             │
│        ░░░░░░░░░░░░░░░░░░░░░░░░░ (waiting)         │
├────────────────────────────────────────────────────┤
│ Seq 2: ████████████████████████████████████ (50)  │
│        ░░░░░░░░░░░░░░░ (waiting)                   │
├────────────────────────────────────────────────────┤
│ Seq 3: ████████████████████████████████████████   │
│        ████████████████████████████ (200)          │
├────────────────────────────────────────────────────┤
│ Seq 4: ████████████████████████████████████████   │
│        ████████████████████████████████████████   │
│        ████████████ (500)                          │
└────────────────────────────────────────────────────┘

GPU idle while waiting for longest sequence!
Throughput = 4 sequences / 500 steps = 0.008 seq/step
```

### How Continuous Batching Works

Remove completed sequences and add new ones during generation:

```
Step 0: [Seq1, Seq2, Seq3, Seq4] (batch=4)
Step 10: Seq1 done → [Seq2, Seq3, Seq4, Seq5] (add Seq5)
Step 50: Seq2 done → [Seq3, Seq4, Seq5, Seq6] (add Seq6)
Step 200: Seq3 done → [Seq4, Seq5, Seq6, Seq7] (add Seq7)
...

GPU always busy! No waiting for batch to complete.
```

**Flow diagram:**

```
┌─────────────┐
│ Request     │
│   Queue     │
│ [R1][R2][R3]│
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│      Scheduler (each step)          │
│                                     │
│  1. Check active batch              │
│  2. Remove finished sequences       │
│  3. Add new sequences from queue    │
│  4. Ensure GPU memory fits          │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│    Active Batch                      │
│  [Seq1][Seq2][Seq3][Seq4]           │
│   (variable size each step)          │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│   Generate 1 token for each         │
└─────────────────────────────────────┘
```

**Implementation sketch:**

```python
class ContinuousBatchingEngine:
    def __init__(self, max_batch_size, max_seq_len):
        self.max_batch_size = max_batch_size
        self.request_queue = []
        self.active_sequences = {}
        self.kv_cache = PagedKVCache(...)

    def add_request(self, prompt, max_tokens):
        request_id = generate_id()
        self.request_queue.append({
            'id': request_id,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'generated': 0
        })
        return request_id

    def step(self):
        # Remove finished sequences
        finished = [
            seq_id for seq_id, seq in self.active_sequences.items()
            if seq['generated'] >= seq['max_tokens'] or seq['done']
        ]
        for seq_id in finished:
            self.kv_cache.free_sequence(seq_id)
            del self.active_sequences[seq_id]
            yield seq_id, seq['tokens']  # Return completed

        # Add new sequences from queue
        while (len(self.active_sequences) < self.max_batch_size
               and self.request_queue
               and self.kv_cache.has_free_blocks()):

            req = self.request_queue.pop(0)
            self.active_sequences[req['id']] = req
            self.kv_cache.allocate_sequence(req['id'])

        # Generate one token for all active sequences
        if self.active_sequences:
            tokens = self.model.generate_batch(
                [seq['prompt'] for seq in self.active_sequences.values()]
            )

            for seq_id, token in zip(self.active_sequences.keys(), tokens):
                self.active_sequences[seq_id]['tokens'].append(token)
                self.active_sequences[seq_id]['generated'] += 1
```

**Performance impact:**

```
Traditional static batching:
- Batch size 8, avg 200 tokens, longest 800 tokens
- Throughput: 8 sequences / 800 steps = 0.01 seq/step

Continuous batching:
- Avg batch size 6-8 (dynamic)
- Throughput: ~0.04 seq/step (4x improvement!)
```

**When to use:** High-throughput serving, variable request lengths
**When NOT to use:** Guaranteed latency SLAs (unpredictable per-request timing)

---

## Quantization

### The Problem

Model weights consume massive memory:
- Llama 70B in FP16: 140GB
- GPT-3 175B in FP16: 350GB

Most consumer/server GPUs: 24-80GB. Need multiple GPUs or reduce precision.

### How It Works

Reduce precision of weights and activations:

```
┌────────────────────────────────────────────────────┐
│              Precision Formats                      │
├────────────┬──────────┬────────────┬───────────────┤
│  Format    │ Bits     │ Range      │ Memory (70B)  │
├────────────┼──────────┼────────────┼───────────────┤
│ FP32       │ 32       │ ±3.4e38    │ 280GB         │
│ FP16       │ 16       │ ±65,504    │ 140GB         │
│ BF16       │ 16       │ ±3.4e38    │ 140GB         │
│ INT8       │ 8        │ -128..127  │ 70GB          │
│ INT4       │ 4        │ -8..7      │ 35GB          │
└────────────┴──────────┴────────────┴───────────────┘
```

**Quantization methods:**

1. **Post-Training Quantization (PTQ):** Convert trained model
2. **Quantization-Aware Training (QAT):** Train with quantization in mind

### INT8 Quantization

```python
def quantize_int8(weight_fp16):
    """Symmetric quantization to INT8."""
    # Find scale factor
    max_val = torch.max(torch.abs(weight_fp16))
    scale = max_val / 127.0

    # Quantize
    weight_int8 = torch.round(weight_fp16 / scale).to(torch.int8)

    return weight_int8, scale

def dequantize_int8(weight_int8, scale):
    """Convert back to FP16 for computation."""
    return weight_int8.to(torch.float16) * scale

# Usage
weight_fp16 = model.layer.weight  # [4096, 4096]
weight_int8, scale = quantize_int8(weight_fp16)

# During inference
x_fp16 = input
weight_dequant = dequantize_int8(weight_int8, scale)
output = torch.matmul(x_fp16, weight_dequant)
```

**Memory savings:**
```
FP16: 70B * 2 bytes = 140GB
INT8: 70B * 1 byte + scales = 70GB (2x reduction)
INT4: 70B * 0.5 bytes + scales = 35GB (4x reduction)
```

### GPTQ (Gradient-based Post-Training Quantization)

Minimizes quantization error using second-order information:

```python
# Conceptual algorithm (simplified)
def gptq_quantize_layer(weight, calibration_data):
    """
    GPTQ quantizes weights to minimize output error.
    Uses Hessian (second derivative) information.
    """
    # 1. Compute Hessian of loss w.r.t. weights
    hessian = compute_hessian(weight, calibration_data)

    # 2. Quantize weights in order of importance
    quantized_weight = weight.clone()
    for i in range(weight.shape[0]):
        # Quantize row i
        quant_error = quantize_row(quantized_weight[i])

        # Compensate error in remaining rows using Hessian
        remaining_rows = quantized_weight[i+1:]
        compensation = -hessian[i, i+1:] / hessian[i, i] * quant_error
        remaining_rows += compensation

    return quantized_weight
```

**Accuracy comparison:**

```
Llama 70B on benchmarks (perplexity, lower is better):

FP16:        10.2 (baseline)
INT8 naive:  10.8 (+0.6, 6% degradation)
GPTQ INT4:   10.4 (+0.2, 2% degradation)
AWQ INT4:    10.3 (+0.1, 1% degradation)

GPTQ maintains quality even at 4-bit!
```

### AWQ (Activation-aware Weight Quantization)

Protects important weights based on activation magnitudes:

```
Key insight: Not all weights are equally important.
Weights with larger activations contribute more to output.

┌────────────────────────────────────────────────┐
│  Activation magnitude distribution             │
│                                                │
│      │                                         │
│  Freq│     █                                   │
│      │    ███                                  │
│      │   █████                                 │
│      │  ███████                                │
│      │ █████████                               │
│      └──────────────────                       │
│      0%  20%  40%  60%  80%  100%              │
│           Weight importance                    │
│                                                │
│  Strategy: Keep top 1% weights in FP16,       │
│            quantize rest to INT4               │
└────────────────────────────────────────────────┘
```

**Trade-offs:**

| Method | Accuracy | Speed | Memory | Setup Complexity |
|--------|----------|-------|--------|------------------|
| FP16   | Baseline | 1x    | 1x     | None |
| INT8   | -1-3%    | 1.5x  | 0.5x   | Low |
| GPTQ INT4 | -1-2% | 2x    | 0.25x  | Medium (needs calibration) |
| AWQ INT4  | -0.5-1% | 2.5x | 0.25x  | Medium (needs calibration) |

**When to use:** Always quantize for production (2-4x throughput gain)
**When NOT to use:** Research tasks requiring exact FP16 accuracy

---

## Model Parallelism

### The Problem

Single GPU cannot fit large models:
- A100 80GB: Can fit Llama 70B (140GB) → NO
- H100 80GB: Can fit GPT-4 scale (1T+ params) → NO

Need to split model across multiple GPUs.

### Tensor Parallelism

Split individual layers across GPUs:

```
┌──────────────────────────────────────────────────────┐
│          Single GPU (won't fit)                       │
│                                                       │
│  ┌────────────────────────────────────────────┐     │
│  │  Layer: Linear(4096 → 16384)               │     │
│  │  Weight: [16384, 4096] = 64M params        │     │
│  └────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│        Tensor Parallel (2 GPUs)                       │
│                                                       │
│  GPU 0                      GPU 1                    │
│  ┌────────────────────┐    ┌────────────────────┐   │
│  │ Linear(4096→8192)  │    │ Linear(4096→8192)  │   │
│  │ Weight: [8192,4096]│    │ Weight: [8192,4096]│   │
│  └────────┬───────────┘    └────────┬───────────┘   │
│           │                         │               │
│           └──────────┬──────────────┘               │
│                      ▼                               │
│              Concatenate outputs                     │
└──────────────────────────────────────────────────────┘
```

**Implementation (simplified):**

```python
class TensorParallelLinear:
    def __init__(self, in_features, out_features, world_size, rank):
        self.world_size = world_size
        self.rank = rank

        # Each GPU gets 1/world_size of output dimension
        self.out_features_per_gpu = out_features // world_size

        # This GPU's weight shard
        self.weight = torch.randn(
            self.out_features_per_gpu,
            in_features
        ).cuda(rank)

    def forward(self, x):
        # Each GPU computes its output shard
        local_output = torch.matmul(x, self.weight.T)

        # All-gather to combine results
        output_shards = [torch.zeros_like(local_output)
                        for _ in range(self.world_size)]
        dist.all_gather(output_shards, local_output)

        # Concatenate
        return torch.cat(output_shards, dim=-1)
```

**Communication pattern:**

```
Forward pass:
  Input → Broadcast → All GPUs compute → All-reduce → Output

Backward pass (for training):
  Gradient → Broadcast → All GPUs compute → All-reduce → Weight updates
```

### Pipeline Parallelism

Split model into stages across GPUs:

```
┌──────────────────────────────────────────────────────┐
│              Pipeline Parallel (4 GPUs)               │
│                                                       │
│  GPU 0           GPU 1           GPU 2       GPU 3   │
│  ┌────────┐    ┌────────┐    ┌────────┐  ┌────────┐│
│  │Layers  │───►│Layers  │───►│Layers  │─►│Layers  ││
│  │ 0-20   │    │ 21-40  │    │ 41-60  │  │ 61-80  ││
│  └────────┘    └────────┘    └────────┘  └────────┘│
│                                                       │
│  Microbatching to keep GPUs busy:                    │
│                                                       │
│  Time 0: [Batch1] ───►                               │
│  Time 1: [Batch2]───► [Batch1]───►                   │
│  Time 2: [Batch3]───► [Batch2]───► [Batch1]───►      │
│  Time 3: [Batch4]───► [Batch3]───► [Batch2]───► [B1] │
└──────────────────────────────────────────────────────┘
```

**Trade-offs:**

| Aspect | Tensor Parallel | Pipeline Parallel |
|--------|----------------|-------------------|
| Communication | High (every layer) | Low (only boundaries) |
| Bubble time | None | 10-30% (waiting for microbatches) |
| Implementation | Complex | Moderate |
| Best for | High-bandwidth interconnect (NVLink) | Multiple nodes |
| Memory per GPU | Split evenly | Uneven (first layers larger) |

### Hybrid: Tensor + Pipeline

Real deployments often combine both:

```
┌────────────────────────────────────────────────────────┐
│         8 GPUs: Tensor Parallel=2, Pipeline=4          │
│                                                         │
│  Stage 0 (Layers 0-20):   GPU0 ◄─► GPU1               │
│                           [Tensor Parallel Group]      │
│                                                         │
│  Stage 1 (Layers 21-40):  GPU2 ◄─► GPU3               │
│                           [Tensor Parallel Group]      │
│                                                         │
│  Stage 2 (Layers 41-60):  GPU4 ◄─► GPU5               │
│                           [Tensor Parallel Group]      │
│                                                         │
│  Stage 3 (Layers 61-80):  GPU6 ◄─► GPU7               │
│                           [Tensor Parallel Group]      │
└────────────────────────────────────────────────────────┘
```

**When to use:**
- Tensor Parallel: Model doesn't fit on 1 GPU, have fast interconnect
- Pipeline Parallel: Serving long sequences, multi-node deployment
- Hybrid: Large models (70B+) on multi-node clusters

**When NOT to use:**
- Model fits on single GPU (overhead not worth it)
- High latency requirements (parallelism adds communication delay)

---

## Streaming Responses

### The Problem

Users want to see output as it's generated (ChatGPT-style), not wait for full completion:

```
Without streaming:
  User sends prompt → Wait 30s → Receive full response

With streaming:
  User sends prompt → 0.5s → First word → 0.5s → Next word → ...

Perceived latency: 0.5s vs 30s!
```

### How It Works

Use Server-Sent Events (SSE) or WebSocket to stream tokens:

```
┌──────────────────────────────────────────────────────┐
│                 Streaming Architecture                │
└──────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────┐      ┌──────────────┐      ┌─────────┐
│   Client     │─────►│  FastAPI     │─────►│  vLLM   │
│  (Browser)   │ HTTP │   Server     │ Async│ Engine  │
└──────────────┘      └──────────────┘      └─────────┘
         ▲                    │                   │
         │                    │                   │
         │                    ▼                   │
         │            ┌──────────────┐            │
         └────────────│  SSE Stream  │◄───────────┘
          Token-by-   └──────────────┘  Generate
           token                          tokens
```

**Server implementation:**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from vllm import LLM, SamplingParams

app = FastAPI()
llm = LLM(model="meta-llama/Llama-2-70b-hf")

@app.post("/generate/stream")
async def generate_stream(prompt: str, max_tokens: int = 512):
    """Stream generated tokens using Server-Sent Events."""

    async def token_generator():
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True  # Enable streaming
        )

        # vLLM returns async generator
        async for output in llm.generate_async(prompt, sampling_params):
            if output.outputs:
                token = output.outputs[0].text

                # SSE format
                yield f"data: {json.dumps({'token': token})}\n\n"

        # Signal completion
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream"
    )
```

**Client implementation:**

```python
import requests
import json

def stream_response(prompt):
    """Client that consumes streaming response."""
    response = requests.post(
        "http://localhost:8000/generate/stream",
        json={"prompt": prompt},
        stream=True
    )

    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = json.loads(line[6:])

                if data.get('done'):
                    break

                token = data['token']
                print(token, end='', flush=True)

    print()  # Newline at end

# Usage
stream_response("Write a poem about the ocean")
# Output appears token-by-token: "The" "ocean" "'s" "waves"...
```

**Buffering strategies:**

```python
class TokenBuffer:
    """Buffer tokens before sending to reduce overhead."""

    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.buffer = []

    def add_token(self, token):
        self.buffer.append(token)

        # Flush if buffer full
        if len(self.buffer) >= self.buffer_size:
            return self.flush()
        return None

    def flush(self):
        """Return buffered tokens and clear."""
        if self.buffer:
            tokens = ''.join(self.buffer)
            self.buffer = []
            return tokens
        return None

# Usage: Send every 5 tokens instead of every token
# Reduces HTTP overhead by 80% while maintaining responsiveness
```

**When to use:** User-facing chat interfaces, long-form generation
**When NOT to use:** Batch processing, API integrations that need full text

---

## Framework Comparison

### vLLM vs TensorRT-LLM vs TGI vs Ollama

| Feature | vLLM | TensorRT-LLM | Text Gen Inference (TGI) | Ollama |
|---------|------|--------------|-------------------------|---------|
| **Throughput** | Excellent (PagedAttention) | Excellent (optimized kernels) | Good | Good |
| **Latency** | Good | Best (CUDA optimizations) | Good | Moderate |
| **Memory efficiency** | Best (paged) | Good | Good | Moderate |
| **Ease of use** | High (Python API) | Low (C++/TensorRT) | High (Docker) | Highest (CLI) |
| **Model support** | Wide | Wide (needs conversion) | HuggingFace | Popular OSS models |
| **Quantization** | GPTQ, AWQ, SqueezeLLM | INT8, INT4, FP8 | GPTQ, bitsandbytes | GGUF (INT4/INT8) |
| **Multi-GPU** | Tensor + Pipeline | Tensor + Pipeline | Tensor | Limited |
| **Streaming** | Yes | Yes | Yes | Yes |
| **Production-ready** | Yes | Yes | Yes | No (dev tool) |
| **Best for** | High-throughput API serving | Lowest latency | Ease of deployment | Local development |

### Detailed Use Cases

**vLLM:**
```python
# Best for: Production API serving with variable request lengths
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    quantization="awq",
    max_model_len=4096
)

# Continuous batching automatically handles variable lengths
outputs = llm.generate(prompts, SamplingParams(max_tokens=512))
```

**TensorRT-LLM:**
```python
# Best for: Absolute lowest latency (trading complexity)
# Requires building TensorRT engine (ahead-of-time compilation)

# 1. Convert model (one-time)
# python convert_checkpoint.py --model_dir llama-70b --output_dir trt-llama

# 2. Build engine
# trtllm-build --checkpoint_dir trt-llama --output_dir engines/

# 3. Run inference
# python run.py --engine_dir engines/ --max_output_len 512
```

**Text Generation Inference (TGI):**
```bash
# Best for: Easy deployment with Docker
docker run --gpus all \
  -p 8080:80 \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-2-70b-hf \
  --quantize gptq \
  --max-total-tokens 4096
```

**Ollama:**
```bash
# Best for: Local development and testing
ollama pull llama2:70b
ollama run llama2:70b "Write a poem"

# Simple API
curl http://localhost:11434/api/generate -d '{
  "model": "llama2:70b",
  "prompt": "Why is the sky blue?"
}'
```

---

## Cost Optimization Strategies

### API vs Self-Hosting Decision Matrix

```
┌────────────────────────────────────────────────────────┐
│         Monthly Volume Analysis                        │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Cost per 1M tokens:                                   │
│  - OpenAI GPT-4: $30 (input) / $60 (output)           │
│  - Anthropic Claude: $15 / $75                         │
│  - Self-hosted Llama 70B: $2-5 (amortized)            │
│                                                         │
│  Break-even calculation:                               │
│                                                         │
│  A100 80GB GPU: $3.50/hour = $2,520/month             │
│                                                         │
│  If using GPT-4 (avg $40/1M tokens):                   │
│  Break-even = $2,520 / $40 = 63M tokens/month         │
│                                                         │
│  Below 63M tokens/month → Use API                      │
│  Above 63M tokens/month → Self-host                    │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### Real-world cost examples:

```python
# Scenario: Customer support chatbot
# - 10,000 conversations/day
# - Avg 500 tokens/conversation (300 input, 200 output)
# - 30 days/month

monthly_tokens = 10000 * 500 * 30  # 150M tokens

# Option 1: OpenAI GPT-4
openai_cost = (10000 * 300 * 30 * 0.00003) + (10000 * 200 * 30 * 0.00006)
# = $2,700 + $3,600 = $6,300/month

# Option 2: Self-hosted Llama 70B (4x A100)
gpu_cost = 4 * 2520  # $10,080/month
# But can handle 3-4x more throughput, so serve other use cases
# Effective cost: $10,080 / 4 = $2,520/month for this use case

# Savings: $6,300 - $2,520 = $3,780/month
```

### GPU Selection Guide

| Use Case | GPU | Memory | Why |
|----------|-----|--------|-----|
| Llama 7B (FP16) | RTX 4090 | 24GB | $1,600, fits model + small batch |
| Llama 13B (INT4) | RTX 4090 | 24GB | Quantization makes it fit |
| Llama 70B (INT4) | A100 40GB | 40GB | Minimum for decent throughput |
| Llama 70B (FP16) | A100 80GB x2 | 160GB | Tensor parallel, high throughput |
| GPT-3 scale (175B) | A100 80GB x4 | 320GB | Pipeline + tensor parallel |

---

## Key Concepts Checklist

- [ ] Understand autoregressive generation (prefill vs decode phases)
- [ ] Explain KV cache memory bottleneck
- [ ] Describe PagedAttention and why it improves memory efficiency
- [ ] Compare continuous batching vs static batching
- [ ] Explain quantization methods (INT8, GPTQ, AWQ) and accuracy trade-offs
- [ ] Understand tensor parallelism vs pipeline parallelism
- [ ] Implement streaming responses with SSE
- [ ] Choose framework (vLLM, TensorRT-LLM, TGI) based on requirements
- [ ] Calculate cost break-even between API and self-hosting
- [ ] Know when NOT to optimize (low volume, research, prototyping)

---

## Practical Insights

**Memory is the bottleneck, not compute:**
Most inference time is spent reading KV cache from memory (memory-bound), not doing math (compute-bound). This is why quantization helps so much—less memory to read means faster generation.

**Batch size sweet spot:**
```
Batch size too small (1-4):
  - GPU underutilized
  - Throughput: 10 tokens/sec

Batch size optimal (16-32):
  - GPU 70-80% utilized
  - Throughput: 80 tokens/sec

Batch size too large (64+):
  - GPU memory OOM or high KV cache eviction
  - Throughput: 60 tokens/sec (cache thrashing)
```

**Continuous batching is a game-changer:**
In production, request lengths vary wildly (20-2000 tokens). Static batching wastes 50-70% of GPU time waiting for longest sequence. Continuous batching gets near-perfect GPU utilization.

**First token latency vs throughput trade-off:**
```
Optimizing for latency (single user):
  - Batch size 1
  - No waiting in queue
  - Latency: 50ms
  - Throughput: 20 req/sec

Optimizing for throughput (multi-user):
  - Batch size 32
  - Queue up to 100ms
  - Latency: 150ms (50ms + 100ms queue)
  - Throughput: 320 req/sec (16x more!)
```

**Quantization is almost always worth it:**
GPTQ/AWQ INT4 loses <1% accuracy but cuts memory by 4x. This means 4x more requests in same batch, which translates to 3-4x more throughput. The latency per token stays the same or improves.

**Monitor GPU memory fragmentation:**
```python
# Track memory usage
import torch

allocated = torch.cuda.memory_allocated() / 1e9  # GB
reserved = torch.cuda.memory_reserved() / 1e9    # GB
fragmentation = (reserved - allocated) / reserved * 100

# Alert if fragmentation > 30%
# Indicates KV cache thrashing or memory leaks
```

**Streaming isn't just UX, it's reliability:**
Long generations (1000+ tokens) can timeout HTTP connections. Streaming keeps connection alive and lets you handle partial failures gracefully.

