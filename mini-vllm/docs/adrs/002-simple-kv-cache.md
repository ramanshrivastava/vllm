

# ADR-002: Simple KV Cache with Contiguous Memory

**Status:** Accepted
**Date:** 2025-11-16
**Commit:** [Phase 1.2]
**vLLM Reference:** `/home/user/vllm/vllm/v1/core/kv_cache_manager.py`
**Related Papers:** vLLM (SOSP 2023), FlashAttention (NeurIPS 2022)

---

## Context

We need to store attention key/value states from previous tokens to avoid recomputing them during autoregressive generation.

### The Fundamental Problem

**Without KV Cache:**
```python
# Generating token-by-token
Token 1: Compute attention([The])
Token 2: Compute attention([The, cat])  ← Recomputes K/V for "The"
Token 3: Compute attention([The, cat, sat])  ← Recomputes K/V for "The", "cat"
...

Complexity: O(n²) where n = sequence length
```

**With KV Cache:**
```python
Token 1: Compute K₁, V₁ → Store in cache
Token 2: Retrieve K₁, V₁ from cache → Compute K₂, V₂ → Store
Token 3: Retrieve K₁, V₁, K₂, V₂ from cache → Compute K₃, V₃ → Store
...

Complexity: O(n) - MUCH faster!
```

### Why Keys/Values Never Change

In transformer attention:
```python
Q = query (for current token)
K = keys (for all tokens)
V = values (for all tokens)

attention_scores = softmax(Q @ K.T / sqrt(d))
output = attention_scores @ V
```

**Key insight:** For token i, once we compute K_i and V_i, they **never change**!
- They don't depend on future tokens
- They're computed once and reused forever
- Perfect candidate for caching

### Requirements

1. **Store** key/value states for all previous tokens
2. **Retrieve** them efficiently during attention
3. **Update** incrementally as we generate new tokens
4. **Multi-layer:** Each transformer layer needs its own cache
5. **Multi-head:** Support multi-head attention

### Constraints

- **Learning-focused:** Should be simple to understand
- **Demonstrate problem:** Should reveal inefficiencies clearly
- **Set up Phase 2:** Should motivate PagedAttention
- **Correctness first:** Don't optimize prematurely

---

## Decision

Implement a **simple KV cache with contiguous memory allocation**.

### Core Design

```python
@dataclass
class CacheConfig:
    num_layers: int       # Transformer layers
    num_heads: int        # Attention heads per layer
    head_dim: int         # Dimension per head
    max_seq_len: int      # Maximum sequence length
    dtype: torch.dtype    # fp16 or fp32

class SimpleKVCache:
    def __init__(self, config):
        # Allocate ONE BIG CONTIGUOUS ARRAY
        shape = (num_layers, max_seq_len, num_heads, head_dim)

        self.k_cache = torch.zeros(shape, dtype=config.dtype)
        self.v_cache = torch.zeros(shape, dtype=config.dtype)

        self.num_cached_tokens = 0  # Track usage

    def store(self, layer_idx, key, value):
        # Store at next available position
        start = self.num_cached_tokens
        end = start + len(key)

        self.k_cache[layer_idx, start:end] = key
        self.v_cache[layer_idx, start:end] = value

    def get(self, layer_idx):
        # Return only used portion
        return (
            self.k_cache[layer_idx, :self.num_cached_tokens],
            self.v_cache[layer_idx, :self.num_cached_tokens]
        )
```

---

## Rationale

### Why Contiguous Memory? (The Simplest Approach)

#### Reason 1: Educational Clarity

**PRO:**
- Easy to understand: "Just one big array"
- Simple indexing: `cache[layer, position, head, dim]`
- Minimal code: ~200 lines (vs vLLM's ~17,000)
- No complex data structures

**Trade-off Accepted:**
We sacrifice efficiency for clarity. This is a LEARNING project.

#### Reason 2: Reveals the Problem

This approach **intentionally** demonstrates three critical problems:

**Problem 1: Massive Memory Waste**
```python
# Llama-7B example:
allocated = 2048 tokens × 1 MB/token = 2 GB
used = 50 tokens × 1 MB/token = 50 MB
wasted = 2 GB - 50 MB = 1.95 GB (97.5%!)
```

**Problem 2: Inflexibility**
```python
# Hit the limit
cache.store(token_2048)  # OK
cache.store(token_2049)  # ❌ Error! Cache full!

# Can't grow beyond max_seq_len
```

**Problem 3: No Sharing**
```python
# Request 1: "The cat sat on the mat"
# Request 2: "The cat jumped over the fence"

# Both compute "The cat" separately
# No way to share the cached K/V
# Duplicate work + memory
```

These problems **motivate** PagedAttention in Phase 2!

#### Reason 3: Matches Learning Progression

We're building understanding **incrementally**:

```
Phase 1.2: Simple contiguous allocation
           ↓ "This is inefficient!"

Phase 2:   PagedAttention with blocks
           ↓ "Much better, but still..."

Phase 5:   Prefix caching + sharing
           ↓ "Now we're at production level!"
```

Each phase **builds on** the previous, showing **why** improvements matter.

### Alternatives Considered

#### Alternative A: PagedAttention from Start

**Pros:**
- Efficient from day 1
- Learn production approach immediately
- No "bad habits"

**Cons:**
- Complex (block allocator, block tables, mapping)
- Harder to understand WHY it works
- Miss the learning journey
- Don't appreciate the innovation

**Rejected because:**
You can't appreciate PagedAttention until you feel the pain of contiguous allocation. Starting with complexity obscures the problem it solves.

**Analogy:**
- Don't teach virtual memory without first teaching physical memory
- Don't teach databases without first teaching file systems
- Don't teach PagedAttention without first teaching contiguous allocation

#### Alternative B: Dynamic Allocation (Grow as Needed)

**Pros:**
- Use only what you need
- No upfront waste
- More "production-like"

**Cons:**
- Still contiguous (fragmentation issues)
- More complex (when to allocate? how much?)
- Hides the core problem
- Still can't share across requests

**Rejected because:**
Adds complexity without solving the fundamental issues. Better to keep it simple and obvious.

#### Alternative C: Skip KV Cache Entirely (Recompute Each Time)

**Pros:**
- Even simpler code
- No memory overhead

**Cons:**
- Terribly slow (O(n²) complexity)
- Not realistic (no production system does this)
- Miss learning about caching entirely

**Rejected because:**
KV cache is **essential** to LLM inference. Skipping it means missing a critical concept.

---

## vLLM Comparison

### What vLLM Does

vLLM's KV cache is in `/home/user/vllm/vllm/v1/core/kv_cache_manager.py`:

**Features:**
- **PagedAttention:** Non-contiguous block allocation
- **Block Management:** 16-token blocks, dynamic allocation
- **Block Tables:** Logical → physical address mapping
- **Prefix Caching:** Share blocks across requests
- **Block Eviction:** LRU policy when out of memory
- **Multi-tenant:** Serve many requests concurrently

**Complexity:**
- ~17,000 lines of code (with related files)
- Block allocator
- Block pool manager
- Reference counting
- Hash-based block lookup
- Eviction policies

**Memory Efficiency:**
- Typical waste: ~5% (vs our 95%)
- Can serve 2-3x more requests
- Dynamic growth (no fixed max_seq_len)
- Shares memory across requests

### What We're Doing

**Our Approach:**
- Contiguous allocation
- Fixed max_seq_len
- No block management
- No sharing

**Lines:** ~600 (implementation + tests + docs)

**Memory Efficiency:** Poor (intentionally!)

**Rationale:**
We're in the **learning phase**. Start simple, understand the problem, then appreciate the solution.

---

## Historical Evolution

### How KV Caching Evolved

#### 2017 (Original Transformer)

First implementation of autoregressive generation:
- Stored K/V in simple lists
- Concatenated for each new token
- Lots of tensor copying
- No thought about efficiency

```python
# Early approach (pseudo-code)
all_keys = []
all_values = []

for token in generate():
    k, v = compute_kv(token)
    all_keys.append(k)
    all_values.append(v)

    # Inefficient: Creates new tensors every time
    K = torch.cat(all_keys)
    V = torch.cat(all_values)

    output = attention(Q, K, V)
```

#### 2018-2019 (Growing Awareness)

People realized:
- Concatenation is expensive
- Can pre-allocate fixed-size buffer
- Store incrementally instead of rebuilding

```python
# Better: Pre-allocate
cache = torch.zeros(max_len, hidden_dim)

for i, token in enumerate(generate()):
    k, v = compute_kv(token)
    cache[i] = k  # In-place store
```

This is essentially what we're doing!

#### 2020-2022 (Scaling Challenges)

As models grew (GPT-3: 175B params):
- Cache memory became bottleneck
- 40GB GPUs could only serve 30-40 requests
- Need better memory management

#### 2023 (vLLM Innovation)

vLLM team realized:
- **Insight:** GPU memory management ≈ OS virtual memory
- **Solution:** Apply paging concepts!
- **Result:** PagedAttention (2-3x efficiency)

**Our Timeline:**
```
Phase 1.2 ← You are here (2018 approach)
Phase 2   → PagedAttention (2023 approach)
```

---

## Trade-offs

### Benefits

✅ **Simple to Understand**
- Just arrays and indices
- No complex data structures
- ~200 lines of core code

✅ **Easy to Debug**
- Can print entire cache
- Simple indexing
- Clear state

✅ **Demonstrates Problem**
- Memory waste is obvious
- Limitations are clear
- Motivates improvements

✅ **Correct**
- Implements the algorithm correctly
- Works for actual generation
- Good foundation for Phase 2

✅ **Fast Development**
- 600 total lines (code + tests)
- vs 17,000 in vLLM
- Can focus on other components

### Limitations

❌ **Memory Inefficient**
- Wastes ~95% of allocated memory
- Can't serve many concurrent requests
- Not production-ready

**Accepted because:** This is the point! We're learning.

❌ **Fixed Size**
- Can't grow beyond max_seq_len
- Must choose upfront
- Sequence too long → error

**Accepted because:** Shows inflexibility problem.

❌ **No Sharing**
- Each request gets separate cache
- Can't share common prefixes
- Duplicate work

**Accepted because:** Motivates prefix caching (Phase 5).

❌ **Contiguous**
- Requires large contiguous memory
- Fragmentation issues
- Can't handle sparse patterns

**Accepted because:** Sets up PagedAttention (Phase 2).

---

## Technical Details

### Memory Layout

```
Cache shape: [num_layers, max_seq_len, num_heads, head_dim]

Example (Llama-7B):
- num_layers = 32
- max_seq_len = 2048
- num_heads = 32
- head_dim = 128

Total elements = 32 × 2048 × 32 × 128 = 268,435,456
With fp16 (2 bytes): 536,870,912 bytes ≈ 512 MB (keys only)
With K+V: 1,024 MB ≈ 1 GB per request

On 40GB GPU: ~30-40 concurrent requests maximum
```

### Indexing Strategy

```python
# Writing to cache
position = num_cached_tokens
cache[layer, position:position+1, :, :] = new_kv

# Reading from cache
used_cache = cache[layer, :num_cached_tokens, :, :]
```

**Why this works:**
- Sequential writes (cache-friendly)
- Simple address calculation
- No indirection

### Efficiency Analysis

**Time Complexity:**
- store(): O(1) - just array write
- get(): O(1) - just array slice
- Perfect! Not the bottleneck.

**Space Complexity:**
- O(L × M × H × D) where:
  - L = num_layers
  - M = max_seq_len
  - H = num_heads
  - D = head_dim

**The Problem:**
- Allocated: O(M) per request
- Used: O(n) where n = actual tokens
- Wasted: O(M - n)
- If n << M (common!), huge waste

---

## Learning Outcomes

After implementing this, you should understand:

### 1. Why KV Cache Exists
- Transformer attention: Q·K^T·V
- For token i, K_i and V_i never change
- Can cache and reuse (O(n²) → O(n))
- Critical for autoregressive generation

### 2. Memory Requirements
- Per-token cost: 2 × num_heads × head_dim × sizeof(dtype)
- Per-request cost: per_token × max_seq_len
- Why fp16 matters (half the memory of fp32)
- Why this is a bottleneck

### 3. The Inefficiency Problem
- Contiguous allocation wastes memory
- Can't share across requests
- Fixed size limits flexibility
- **This is why vLLM exists!**

### 4. Attention Mechanics
- How K/V are used in attention
- Why we store per-layer
- Why we store per-head
- Shape transformations

### 5. Foundation for PagedAttention
- Understand what problems we're solving
- Appreciate the OS-inspired solution
- See the 2-3x improvement in context

---

## Exercises

### Exercise 1: Calculate Memory Usage (15 min)

**Task:** For your favorite LLM, calculate KV cache memory.

```python
# Example: Llama-2-13B
num_layers = 40
num_heads = 40
head_dim = 128
max_seq_len = 4096

# Calculate:
# 1. Bytes per token
# 2. Bytes per request
# 3. Max concurrent requests on 80GB GPU
```

**Questions:**
- Why is fp16 important?
- What if we used 8-bit quantization?
- How does this compare to model weights?

### Exercise 2: Demonstrate Memory Waste (30 min)

**Task:** Run the demonstration function:

```python
python -m mini_vllm.kv_cache.simple_cache
```

**Modify it to:**
- Use different sequence lengths (10, 100, 1000)
- Show waste percentage for each
- Calculate what PagedAttention could save

### Exercise 3: Implement Simple Eviction (45 min)

**Task:** Add a method to evict oldest tokens when full.

```python
def evict_oldest(self, num_to_evict: int):
    """Remove oldest tokens to make space."""
    # Your implementation here
    pass
```

**Questions:**
- What are we losing?
- When would this be useful?
- What's a better solution?

### Exercise 4: Profile Memory Access (60 min)

**Task:** Measure memory access patterns.

```python
import time

# Store 1000 tokens
for i in range(1000):
    cache.store(0, k, v)

# Measure retrieval time
start = time.time()
for _ in range(1000):
    k, v = cache.get(0)
end = time.time()
```

**Questions:**
- Is retrieval speed affected by num_cached_tokens?
- Why or why not?
- How does this compare to PagedAttention?

---

## Next Steps

**Immediate:**
Phase 1.3 will load actual transformer models from HuggingFace and integrate with this KV cache to generate real text.

**Phase 2 (Critical!):**
Replace this with PagedAttention:
- Block-based allocation (16 tokens per block)
- Non-contiguous memory
- Dynamic growth
- Block tables for address translation
- 2-3x better memory efficiency

**Phase 5:**
Add prefix caching:
- Hash blocks by content
- Share identical blocks across requests
- Detect common prefixes
- Further 2-3x improvement for chat/few-shot

---

## References

### Papers

1. **vLLM (PagedAttention):**
   [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
   Kwon et al., SOSP 2023
   - Core innovation that made LLM serving practical
   - OS-inspired virtual memory for GPU

2. **FlashAttention:**
   [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
   Dao et al., NeurIPS 2022
   - Complementary optimization (within attention kernel)
   - vLLM uses FlashAttention + PagedAttention

3. **Attention Is All You Need:**
   [Original Transformer Paper](https://arxiv.org/abs/1706.03762)
   Vaswani et al., NeurIPS 2017
   - Where it all started

### Code References

- **vLLM KV cache:** `/home/user/vllm/vllm/v1/core/kv_cache_manager.py`
- **Block allocator:** `/home/user/vllm/vllm/v1/core/block_pool.py`
- **Scheduler integration:** `/home/user/vllm/vllm/v1/core/sched/`

### Blog Posts

- [vLLM: Easy, Fast, and Cheap LLM Serving](https://blog.vllm.ai/2023/06/20/vllm.html)
- [vLLM v0.2.0: Prefix Caching](https://blog.vllm.ai/2024/01/17/prefix-caching.html)

---

## Approval

**Decision Maker:** Learning Project Team
**Date:** 2025-11-16
**Status:** ✅ Approved for Phase 1.2

**Key Insight:**
> "By implementing the naive approach first, we deeply understand the problem that PagedAttention solves. This pedagogical debt pays dividends when we implement Phase 2."

**Reviewers:**
- Architecture: Validated simple → complex progression
- Implementation: Code complete, well-tested, documented
- Pedagogy: Effectively demonstrates key concepts

---

*This ADR is part of the mini-vLLM learning project. See `MINI_VLLM_LEARNING_GUIDE.md` for overall architecture and `PHASED_IMPLEMENTATION_PLAN.md` for implementation timeline.*
