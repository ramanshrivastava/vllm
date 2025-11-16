# KV Cache Module

## 🎯 What is KV Cache?

The KV (Key-Value) cache is one of the **most important optimizations** in LLM inference. It stores attention keys and values from previous tokens to avoid recomputing them.

### The Problem Without KV Cache

When generating text autoregressively:

```python
# Generating: "The cat sat on the mat"

Token 1 ("The"):
  - Compute attention with: ["The"]
  - Keys/Values: [K₁, V₁]

Token 2 ("cat"):
  - Compute attention with: ["The", "cat"]
  - Keys/Values: [K₁, V₁, K₂, V₂]
  - ❌ Recomputing K₁, V₁ again!

Token 3 ("sat"):
  - Compute attention with: ["The", "cat", "sat"]
  - Keys/Values: [K₁, V₁, K₂, V₂, K₃, V₃]
  - ❌ Recomputing K₁, V₁, K₂, V₂ again!
```

**Result:** O(n²) computation where n = sequence length

### The Solution: KV Cache

```python
Token 1 ("The"):
  - Compute K₁, V₁
  - ✅ Store in cache[0]

Token 2 ("cat"):
  - ✅ Retrieve K₁, V₁ from cache
  - Compute K₂, V₂
  - ✅ Store in cache[1]

Token 3 ("sat"):
  - ✅ Retrieve K₁, V₁, K₂, V₂ from cache
  - Compute K₃, V₃
  - ✅ Store in cache[2]
```

**Result:** O(n) computation - MASSIVE SPEEDUP!

---

## 🤔 Why Start With Contiguous Memory?

### Design Decision: Simplest Possible Approach

We're implementing the **naive** approach first:
- Allocate one big contiguous array
- Shape: `[num_layers, max_seq_len, num_heads, head_dim]`
- Store tokens sequentially

### Why This is Educational

**Three key problems** become obvious:

#### 1. **Memory Waste**
```
Allocated: 2048 tokens worth of memory
Actually used: 50 tokens
Wasted: 1998 tokens (97.5%)!
```

For Llama-7B:
- Need: ~1 GB per request (allocated)
- Use: ~50 MB (5% of sequence)
- **Waste: ~950 MB per request!**

#### 2. **Inflexibility**
```
max_seq_len = 2048
Generated 2048 tokens...
Want to generate more? ❌ Can't! Cache is full.
```

#### 3. **Cannot Share Memory**
```
Request 1: "The cat sat on the mat"
Request 2: "The cat jumped over the fence"

Both start with "The cat" but can't share the cached K/V!
Each allocates separate memory for identical prefixes.
```

---

## ✅ What We Learn

### Phase 1.2 (This Implementation)
**Approach:** Contiguous memory allocation
**Code:** ~600 lines (implementation + tests)
**Memory Efficiency:** Poor (lots of waste)
**Learning:** Understand the problem deeply

### Phase 2 (Next: PagedAttention)
**Approach:** Block-based non-contiguous allocation
**Inspiration:** OS virtual memory paging
**Memory Efficiency:** Excellent (2-3x better)
**Learning:** How vLLM's innovation works

---

## 📊 Memory Analysis

### Example: Llama-7B Configuration

```python
config = CacheConfig(
    num_layers=32,
    num_heads=32,
    head_dim=128,
    max_seq_len=2048,
    dtype=torch.float16,
)

# Memory calculation:
# 32 layers × 2048 tokens × 32 heads × 128 dim × 2 bytes (fp16) × 2 (K+V)
# = 1,073,741,824 bytes = 1 GB per request
```

**Impact:**
- 40GB GPU → only ~30-40 concurrent requests
- Most of that memory is WASTED on unused tokens
- Can't serve more users efficiently

**PagedAttention (Phase 2) will fix this!**

---

## 🔧 Usage

### Basic Usage

```python
from mini_vllm.kv_cache import SimpleKVCache, CacheConfig

# Create config for your model
config = CacheConfig(
    num_layers=32,
    num_heads=32,
    head_dim=128,
    max_seq_len=2048,
)

# Initialize cache
cache = SimpleKVCache(config, device="cuda")

# During generation:
# 1. Compute new key/value states
key = model.compute_key(new_token)    # [1, 32, 128]
value = model.compute_value(new_token)  # [1, 32, 128]

# 2. Store in cache (for all layers)
for layer_idx in range(config.num_layers):
    cache.store(layer_idx, key, value)

cache.update_num_tokens(1)

# 3. Retrieve for attention
cached_keys, cached_values = cache.get(layer_idx)
# Shape: [num_cached_tokens, num_heads, head_dim]

# 4. Compute attention with all previous tokens
attention_output = compute_attention(
    query=current_query,
    keys=cached_keys,  # All previous tokens!
    values=cached_values,
)
```

### Memory Tracking

```python
# Check memory usage
usage = cache.get_memory_usage()

print(f"Allocated: {usage['total_allocated_gb']:.2f} GB")
print(f"Used: {usage['actually_used_gb']:.2f} GB")
print(f"Wasted: {usage['wasted_percent']:.1f}%")

# Example output:
# Allocated: 1.00 GB
# Used: 0.05 GB (50 tokens)
# Wasted: 95.0%  ⚠️ This is the problem!
```

---

## 🧪 Testing

Run tests to see memory waste demonstrated:

```bash
# Run all KV cache tests
pytest tests/unit/test_simple_cache.py -v

# Run demonstration
python -m mini_vllm.kv_cache.simple_cache
```

Tests demonstrate:
- ✅ Correct storage and retrieval
- ⚠️ Memory waste (educational!)
- ⚠️ Cache overflow errors
- ⚠️ Inflexibility of fixed size

---

## 📈 Comparison with vLLM

| Feature | Our Simple Cache | vLLM PagedAttention | Why Different? |
|---------|------------------|---------------------|----------------|
| **Memory** | Contiguous array | Non-contiguous blocks | Learning path |
| **Efficiency** | Poor (97% waste) | Excellent (5% waste) | Shows problem |
| **Flexibility** | Fixed max_seq_len | Dynamic growth | Simple first |
| **Sharing** | No | Yes (prefix caching) | Phase 5 feature |
| **Lines** | ~600 | ~17,000 | Simplified |

**Our Goal:** Understand WHY PagedAttention is needed, not just how it works.

---

## 🎓 Learning Outcomes

After implementing this module, you understand:

### 1. **What KV Cache Is**
- Stores attention keys/values from previous tokens
- Avoids recomputation (O(n²) → O(n))
- Critical for efficient autoregressive generation

### 2. **Why We Need It**
- Without: Recompute everything for each new token
- With: Reuse previous computations
- Speedup: ~100-1000x for typical sequences!

### 3. **Memory Layout**
- Shape: `[layers, positions, heads, head_dim]`
- Why multi-dimensional: Each layer, each head stores separately
- Per-token cost: `2 × num_heads × head_dim × sizeof(dtype)`

### 4. **The Inefficiency**
- Contiguous allocation wastes memory
- Can't grow beyond max_seq_len
- Can't share across requests
- **This sets up Phase 2: PagedAttention!**

### 5. **Performance Impact**
- Memory = bottleneck in LLM serving
- More efficient cache = more concurrent requests
- vLLM's PagedAttention: 2-3x improvement

---

## 🔗 References

### vLLM Source
- **Location:** `/home/user/vllm/vllm/v1/core/kv_cache_manager.py`
- **Complexity:** ~17,000 lines (with block management)
- **Features:** PagedAttention, prefix caching, block sharing

### Papers
1. **vLLM (PagedAttention):** [Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180), SOSP 2023
2. **FlashAttention:** [Fast and Memory-Efficient Attention](https://arxiv.org/abs/2205.14135), NeurIPS 2022

### Related Concepts
- **Virtual Memory:** OS concept that inspired PagedAttention
- **Cache Hierarchy:** CPU caches (similar idea, different scale)
- **Memoization:** General technique of caching computations

---

## ⚠️ Limitations (By Design)

These are **intentional** to teach you:

1. **Fixed Size:** Can't grow beyond max_seq_len
   - **Why:** Shows inflexibility problem
   - **Fix:** PagedAttention in Phase 2

2. **Memory Waste:** Allocates but doesn't use
   - **Why:** Shows efficiency problem
   - **Fix:** Block allocation in Phase 2

3. **No Sharing:** Each request gets separate cache
   - **Why:** Shows prefix caching motivation
   - **Fix:** Content-addressable blocks in Phase 5

4. **Contiguous:** All memory in one array
   - **Why:** Simplest possible approach
   - **Fix:** Non-contiguous blocks in Phase 2

---

## 🚀 Next Steps

**Immediate:**
Phase 1.3 will load actual transformer models from HuggingFace and integrate with this KV cache.

**Future (Phase 2):**
Replace this with PagedAttention:
- Block-based allocation (like OS pages)
- Non-contiguous memory
- Dynamic growth
- 2-3x better memory efficiency

**Future (Phase 5):**
Add prefix caching:
- Share blocks across requests
- Content-addressable storage
- Detect common prefixes automatically

---

## 💡 Key Insights

> **"The KV cache is what makes LLM inference practical."**
>
> Without it, generating a 100-token response would require computing attention 5,000 times (100² / 2). With it, we compute attention 100 times. That's a 50x speedup!

> **"Contiguous allocation is simple but wasteful."**
>
> For a 2048-token cache, if you use 50 tokens, you waste 97.5% of memory. This is why vLLM invented PagedAttention - to use GPU memory efficiently.

> **"Memory is the bottleneck in LLM serving."**
>
> More efficient memory use = more concurrent users = better business economics. PagedAttention makes LLM serving practical at scale.

---

*See `docs/adrs/002-simple-kv-cache.md` for full design rationale.*
