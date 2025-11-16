"""
Simple KV Cache with Contiguous Memory Allocation

═══════════════════════════════════════════════════════════════════════════
WHAT IS THIS?
═══════════════════════════════════════════════════════════════════════════

A KV (Key-Value) cache stores the attention key and value states from
previous tokens to avoid recomputing them during autoregressive generation.

Think of it like this:
- When generating "Once upon a time in"
- To generate "a", we need attention with "Once upon"
- To generate "time", we need attention with "Once upon a"
- WITHOUT cache: Recompute "Once upon" keys/values every time
- WITH cache: Store them once, reuse them

This saves MASSIVE computation (quadratic → linear for generation)!

═══════════════════════════════════════════════════════════════════════════
WHY CONTIGUOUS MEMORY FIRST?
═══════════════════════════════════════════════════════════════════════════

We're using the SIMPLEST possible approach:
- Allocate one big contiguous array for all tokens
- Shape: [num_layers, max_seq_len, num_heads, head_dim]

WHY START HERE?
1. Easy to understand - just one big tensor
2. Shows the problem clearly - wastes memory!
3. Makes you appreciate PagedAttention in Phase 2

THIS IS INEFFICIENT! That's the point. We'll see why in Phase 2.

═══════════════════════════════════════════════════════════════════════════
HOW ATTENTION WORKS (Quick Refresher)
═══════════════════════════════════════════════════════════════════════════

For each token, we compute:
  Q = query  (for current token)
  K = key    (for all tokens so far)
  V = value  (for all tokens so far)

  attention_scores = softmax(Q @ K.T / sqrt(d))
  output = attention_scores @ V

The KEY INSIGHT: K and V for previous tokens NEVER CHANGE!
So we can cache them and reuse them.

═══════════════════════════════════════════════════════════════════════════
EXAMPLE
═══════════════════════════════════════════════════════════════════════════

Generating: "The cat sat on the mat"

Token 1 ("The"):
  - Compute K₁, V₁ for "The"
  - Store in cache[0]
  - Attention with just itself

Token 2 ("cat"):
  - Compute K₂, V₂ for "cat"
  - Store in cache[1]
  - Attention with [K₁, K₂] and [V₁, V₂]
  - We REUSE K₁, V₁ from cache!

Token 3 ("sat"):
  - Compute K₃, V₃ for "sat"
  - Store in cache[2]
  - Attention with [K₁, K₂, K₃] and [V₁, V₂, V₃]
  - We REUSE K₁, V₁, K₂, V₂ from cache!

This saves computation proportional to sequence length!

Reference: /home/user/vllm/vllm/v1/core/kv_cache_manager.py
          (much more complex with block management)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class CacheConfig:
    """
    Configuration for KV cache.

    WHY THESE PARAMETERS?
    ────────────────────

    num_layers: Transformer has multiple layers (e.g., 32 for Llama-7B)
                Each layer has its own K/V cache

    num_heads: Multi-head attention splits computation
               (e.g., 32 heads for Llama-7B)

    head_dim: Dimension of each attention head
              (e.g., 128 for Llama-7B)
              Total hidden_dim = num_heads * head_dim

    max_seq_len: Maximum sequence we can cache
                 WHY NEEDED? We're allocating memory upfront
                 Must choose a limit (e.g., 2048, 4096)

    dtype: Data type for cache (float16 saves memory vs float32)
    """

    num_layers: int = 32        # Number of transformer layers
    num_heads: int = 32         # Number of attention heads per layer
    head_dim: int = 128         # Dimension of each head
    max_seq_len: int = 2048     # Maximum sequence length we can cache
    dtype: torch.dtype = torch.float16  # Use fp16 to save memory

    def get_cache_shape(self) -> Tuple[int, int, int, int]:
        """
        Get the shape of the KV cache tensor.

        Returns:
            Tuple of (num_layers, max_seq_len, num_heads, head_dim)

        WHY THIS SHAPE?
        ───────────────
        - num_layers: Each layer has separate K/V
        - max_seq_len: Store keys/values for all positions
        - num_heads: Multi-head attention
        - head_dim: Dimension of each head

        MEMORY CALCULATION:
        ──────────────────
        For Llama-7B with max_seq_len=2048:
        - 32 layers * 2048 tokens * 32 heads * 128 dim * 2 bytes (fp16)
        - = 536,870,912 bytes = 512 MB per request (K only)
        - Double that for K+V = 1 GB per request!

        THIS IS WHY WE NEED PAGEATTENTION - we can't afford this waste!
        """
        return (
            self.num_layers,
            self.max_seq_len,
            self.num_heads,
            self.head_dim,
        )

    def get_memory_bytes(self) -> int:
        """
        Calculate memory usage in bytes for K+V cache.

        WHY THIS MATTERS?
        ────────────────
        This shows how much GPU memory we need.
        For a 40GB GPU, if each request uses 1GB,
        we can only serve ~30-40 concurrent requests.

        Returns:
            Total bytes needed for K and V cache
        """
        shape = self.get_cache_shape()
        elements = 1
        for dim in shape:
            elements *= dim

        # Multiply by 2 for K and V, and by bytes per element
        bytes_per_element = 2 if self.dtype == torch.float16 else 4
        return elements * 2 * bytes_per_element


class SimpleKVCache:
    """
    Simple KV cache with contiguous memory allocation.

    ═══════════════════════════════════════════════════════════════════
    KEY DESIGN DECISIONS
    ═══════════════════════════════════════════════════════════════════

    1. CONTIGUOUS ALLOCATION
       - Allocate one big tensor upfront
       - PRO: Simple, fast access
       - CON: Wastes memory (allocated but unused)

    2. FIXED MAX LENGTH
       - Must choose max_seq_len ahead of time
       - PRO: Simple memory management
       - CON: Either waste memory (too large) or limit length (too small)

    3. SEPARATE K AND V
       - Store keys and values in separate tensors
       - WHY? They're used differently in attention
       - Could combine, but this is clearer

    4. PER-LAYER STORAGE
       - Each transformer layer has its own K/V
       - WHY? K/V from layer 1 ≠ K/V from layer 2
       - They represent different semantic spaces

    ═══════════════════════════════════════════════════════════════════
    WHAT THIS SETS UP FOR PHASE 2
    ═══════════════════════════════════════════════════════════════════

    This simple approach will reveal THREE key problems:

    1. MEMORY WASTE: If we allocate max_seq_len=2048 but only use 50 tokens,
                     we waste 2048-50 = 1998 tokens worth of memory!

    2. MEMORY FRAGMENTATION: Can't share memory between requests efficiently

    3. INFLEXIBILITY: Can't grow beyond max_seq_len

    PagedAttention (Phase 2) solves ALL of these!
    """

    def __init__(
        self,
        config: CacheConfig,
        device: str = "cuda",
    ):
        """
        Initialize KV cache with contiguous memory.

        Args:
            config: Cache configuration
            device: Device to allocate cache on ("cuda" or "cpu")

        WHY ALLOCATE UPFRONT?
        ────────────────────
        We're using the simplest approach: allocate all memory at start.

        ALTERNATIVE: Allocate dynamically as needed
        - PRO: Use only what we need
        - CON: More complex, fragmentation issues

        We choose simple upfront allocation to make the limitations clear.
        """
        self.config = config
        self.device = device

        # Get cache shape: [num_layers, max_seq_len, num_heads, head_dim]
        cache_shape = config.get_cache_shape()

        print(f"📊 Allocating KV cache:")
        print(f"   Shape: {cache_shape}")
        print(f"   Memory: {config.get_memory_bytes() / 1e9:.2f} GB")
        print(f"   Device: {device}")

        # Allocate contiguous memory for keys
        # WHY ZEROS? We'll fill it incrementally as we generate tokens
        self.k_cache = torch.zeros(
            cache_shape,
            dtype=config.dtype,
            device=device,
        )

        # Allocate contiguous memory for values
        self.v_cache = torch.zeros(
            cache_shape,
            dtype=config.dtype,
            device=device,
        )

        # Track how many tokens are currently cached
        # WHY NEEDED? We only use part of the allocated memory
        self.num_cached_tokens = 0

        print(f"✅ KV cache allocated successfully!")
        print(f"⚠️  Note: Using contiguous allocation (inefficient)")
        print(f"   We'll fix this with PagedAttention in Phase 2!")

    def store(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Store new key/value states into cache.

        Args:
            layer_idx: Which transformer layer (0 to num_layers-1)
            key: New key states [num_new_tokens, num_heads, head_dim]
            value: New value states [num_new_tokens, num_heads, head_dim]

        WHY THIS SHAPE?
        ──────────────
        num_new_tokens: Usually 1 (generating one token at a time)
                        But could be more during prefill phase
        num_heads: Multi-head attention
        head_dim: Dimension of each head

        WHAT HAPPENS HERE?
        ─────────────────
        1. Check we have space (num_cached + new <= max_seq_len)
        2. Write new K/V into cache at position [num_cached : num_cached + new]
        3. Update num_cached_tokens counter

        Example:
            num_cached_tokens = 5  # Already have 5 tokens
            num_new_tokens = 1     # Adding 1 more
            → Write to position cache[layer_idx, 5:6, :, :]
            → Update num_cached_tokens = 6
        """
        num_new_tokens = key.size(0)

        # SAFETY CHECK: Do we have space?
        # WHY THIS CHECK? Contiguous allocation has fixed size!
        if self.num_cached_tokens + num_new_tokens > self.config.max_seq_len:
            raise ValueError(
                f"KV cache overflow! "
                f"Trying to cache {self.num_cached_tokens + num_new_tokens} tokens "
                f"but max is {self.config.max_seq_len}. "
                f"\n\nThis is a limitation of contiguous allocation. "
                f"In Phase 2 (PagedAttention), we'll handle this better!"
            )

        # Calculate where to write in the cache
        start_pos = self.num_cached_tokens
        end_pos = start_pos + num_new_tokens

        # Store keys: Write to cache[layer, start:end, :, :]
        # WHY THIS INDEXING?
        # - layer_idx: Select the right layer
        # - start_pos:end_pos: Write to next available positions
        # - :, : : All heads and dimensions
        self.k_cache[layer_idx, start_pos:end_pos, :, :] = key

        # Store values: Same indexing
        self.v_cache[layer_idx, start_pos:end_pos, :, :] = value

    def get(
        self,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve cached keys and values for a layer.

        Args:
            layer_idx: Which transformer layer

        Returns:
            Tuple of (keys, values) for all cached tokens
            Shape: [num_cached_tokens, num_heads, head_dim]

        WHY RETURN ALL CACHED TOKENS?
        ────────────────────────────
        For attention, we need K/V from ALL previous tokens.

        Example:
            Generating token 10, we need:
            - K and V from tokens 0-9 (all previous)
            - To compute attention scores with them

        WHAT ABOUT UNUSED MEMORY?
        ────────────────────────
        We allocated max_seq_len positions, but only use num_cached_tokens.
        The rest is WASTED! This is the inefficiency we'll fix in Phase 2.
        """
        if self.num_cached_tokens == 0:
            # NO TOKENS CACHED YET
            # WHY RETURN EMPTY? First token has no previous context
            # It attends only to itself
            return (
                self.k_cache[layer_idx, :0, :, :],
                self.v_cache[layer_idx, :0, :, :],
            )

        # Return only the USED portion of cache
        # WHY [:num_cached_tokens]?
        # - We allocated max_seq_len positions
        # - But only filled num_cached_tokens of them
        # - The rest contains zeros (unused)
        return (
            self.k_cache[layer_idx, :self.num_cached_tokens, :, :],
            self.v_cache[layer_idx, :self.num_cached_tokens, :, :],
        )

    def update_num_tokens(self, num_new_tokens: int) -> None:
        """
        Update the count of cached tokens.

        Args:
            num_new_tokens: How many tokens were just added

        WHY SEPARATE METHOD?
        ───────────────────
        We call store() for each layer, but only want to increment
        the counter once (after all layers are done).

        This prevents counting the same tokens multiple times.
        """
        self.num_cached_tokens += num_new_tokens

    def clear(self) -> None:
        """
        Clear the cache (reset to empty).

        WHY CLEAR?
        ─────────
        - Start new request: Need fresh cache
        - Out of memory: Clear to free space
        - Testing: Reset between tests

        DO WE FREE MEMORY?
        ─────────────────
        No! We keep the allocated tensors (k_cache, v_cache).
        We just reset the counter to 0.

        WHY NOT FREE? Allocation is expensive. We reuse the same memory.
        """
        self.num_cached_tokens = 0
        # NOTE: We don't zero out the tensors - just reset counter
        # The old data will be overwritten when we store new tokens

    def get_num_cached_tokens(self) -> int:
        """
        Get how many tokens are currently cached.

        Returns:
            Number of tokens in cache

        WHY USEFUL?
        ──────────
        - Check if we're near max_seq_len
        - Calculate attention complexity (O(n²) where n = cached tokens)
        - Debugging and monitoring
        """
        return self.num_cached_tokens

    def get_memory_usage(self) -> dict:
        """
        Get detailed memory usage statistics.

        Returns:
            Dictionary with memory info

        WHY THIS METHOD?
        ───────────────
        To understand the inefficiency of contiguous allocation!

        Shows:
        - Total allocated memory
        - Actually used memory
        - WASTED memory (the problem!)
        """
        total_bytes = self.config.get_memory_bytes()

        # How much are we actually using?
        # CALCULATION: Same formula as total, but with actual tokens used
        if self.num_cached_tokens > 0:
            used_fraction = self.num_cached_tokens / self.config.max_seq_len
            used_bytes = total_bytes * used_fraction
        else:
            used_bytes = 0

        wasted_bytes = total_bytes - used_bytes
        wasted_percent = (wasted_bytes / total_bytes * 100) if total_bytes > 0 else 0

        return {
            "total_allocated_gb": total_bytes / 1e9,
            "actually_used_gb": used_bytes / 1e9,
            "wasted_gb": wasted_bytes / 1e9,
            "wasted_percent": wasted_percent,
            "num_cached_tokens": self.num_cached_tokens,
            "max_seq_len": self.config.max_seq_len,
        }

    def __repr__(self) -> str:
        """String representation showing cache state."""
        usage = self.get_memory_usage()
        return (
            f"SimpleKVCache(\n"
            f"  cached_tokens={self.num_cached_tokens}/{self.config.max_seq_len},\n"
            f"  allocated={usage['total_allocated_gb']:.2f}GB,\n"
            f"  used={usage['actually_used_gb']:.2f}GB,\n"
            f"  wasted={usage['wasted_percent']:.1f}%\n"
            f")"
        )


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def demonstrate_memory_waste():
    """
    Demonstration: Show how much memory we waste with contiguous allocation.

    This makes the problem visceral!
    """
    print("\n" + "="*70)
    print("DEMONSTRATING MEMORY WASTE IN CONTIGUOUS ALLOCATION")
    print("="*70 + "\n")

    # Llama-7B configuration
    config = CacheConfig(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        max_seq_len=2048,
        dtype=torch.float16,
    )

    # Create cache
    cache = SimpleKVCache(config, device="cpu")  # CPU for demo

    print(f"\n📝 Scenario: Generating 'Hello, world!' (≈ 5 tokens)")
    print(f"   But we allocated space for {config.max_seq_len} tokens!\n")

    # Simulate caching 5 tokens
    for i in range(5):
        # Dummy data
        k = torch.randn(1, config.num_heads, config.head_dim)
        v = torch.randn(1, config.num_heads, config.head_dim)

        # Store in cache (layer 0 for demo)
        cache.store(0, k, v)
        cache.update_num_tokens(1)

    # Show usage
    usage = cache.get_memory_usage()

    print(f"📊 Memory Statistics:")
    print(f"   Allocated: {usage['total_allocated_gb']:.3f} GB")
    print(f"   Used:      {usage['actually_used_gb']:.3f} GB")
    print(f"   Wasted:    {usage['wasted_gb']:.3f} GB ({usage['wasted_percent']:.1f}%)")
    print(f"\n💡 We're wasting {usage['wasted_percent']:.1f}% of allocated memory!")
    print(f"   This is why we need PagedAttention in Phase 2.\n")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_memory_waste()
