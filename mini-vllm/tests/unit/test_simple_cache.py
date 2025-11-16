"""
Unit tests for Simple KV Cache.

These tests verify both CORRECTNESS and demonstrate WHY we need
a better approach (PagedAttention in Phase 2).
"""

import pytest
import torch
from mini_vllm.kv_cache import SimpleKVCache, CacheConfig


class TestCacheConfig:
    """Test CacheConfig dataclass and calculations."""

    def test_cache_config_creation(self):
        """Test creating a cache config."""
        config = CacheConfig(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            max_seq_len=2048,
        )

        assert config.num_layers == 32
        assert config.num_heads == 32
        assert config.head_dim == 128
        assert config.max_seq_len == 2048

    def test_get_cache_shape(self):
        """Test cache shape calculation."""
        config = CacheConfig(
            num_layers=12,
            num_heads=8,
            head_dim=64,
            max_seq_len=512,
        )

        shape = config.get_cache_shape()
        assert shape == (12, 512, 8, 64)

    def test_memory_calculation_fp16(self):
        """
        Test memory calculation for float16.

        WHY THIS TEST?
        ──────────────
        Demonstrates EXACTLY how much memory we need.
        For Llama-7B: ~1GB per request!
        """
        config = CacheConfig(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            max_seq_len=2048,
            dtype=torch.float16,  # 2 bytes per element
        )

        memory_bytes = config.get_memory_bytes()

        # Calculate expected: 32 * 2048 * 32 * 128 * 2 (K+V) * 2 (bytes)
        expected = 32 * 2048 * 32 * 128 * 2 * 2
        assert memory_bytes == expected

        # Convert to GB
        memory_gb = memory_bytes / 1e9
        print(f"\n📊 Memory for Llama-7B config: {memory_gb:.2f} GB")
        print(f"   This is PER REQUEST!")
        print(f"   40GB GPU → only ~30-40 concurrent requests")

    def test_memory_calculation_fp32(self):
        """Test memory calculation for float32 (4 bytes)."""
        config = CacheConfig(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            max_seq_len=2048,
            dtype=torch.float32,  # 4 bytes per element
        )

        memory_bytes = config.get_memory_bytes()

        # Should be 2x fp16 (4 bytes vs 2 bytes)
        fp16_config = CacheConfig(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            max_seq_len=2048,
            dtype=torch.float16,
        )

        assert memory_bytes == fp16_config.get_memory_bytes() * 2


class TestSimpleKVCache:
    """Test SimpleKVCache functionality."""

    @pytest.fixture
    def small_config(self):
        """Small config for fast testing."""
        return CacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=8,
            max_seq_len=16,
            dtype=torch.float32,  # fp32 for easier debugging
        )

    @pytest.fixture
    def cache(self, small_config):
        """Create a small cache for testing."""
        return SimpleKVCache(small_config, device="cpu")

    def test_cache_initialization(self, cache, small_config):
        """Test cache initializes correctly."""
        assert cache.config == small_config
        assert cache.num_cached_tokens == 0
        assert cache.device == "cpu"

        # Check tensors were allocated
        assert cache.k_cache.shape == small_config.get_cache_shape()
        assert cache.v_cache.shape == small_config.get_cache_shape()

    def test_store_single_token(self, cache, small_config):
        """
        Test storing a single token's K/V.

        WHY THIS TEST?
        ─────────────
        This is the most common case: generating one token at a time.
        """
        layer_idx = 0
        num_heads = small_config.num_heads
        head_dim = small_config.head_dim

        # Create dummy K/V for 1 token
        key = torch.randn(1, num_heads, head_dim)
        value = torch.randn(1, num_heads, head_dim)

        # Store in cache
        cache.store(layer_idx, key, value)
        cache.update_num_tokens(1)

        # Verify stored correctly
        assert cache.num_cached_tokens == 1

        # Retrieve and check
        cached_k, cached_v = cache.get(layer_idx)
        assert cached_k.shape == (1, num_heads, head_dim)
        assert cached_v.shape == (1, num_heads, head_dim)
        assert torch.allclose(cached_k, key)
        assert torch.allclose(cached_v, value)

    def test_store_multiple_tokens_sequentially(self, cache, small_config):
        """
        Test storing tokens one at a time (typical generation).

        WHY THIS TEST?
        ─────────────
        Simulates actual autoregressive generation:
        - Token 1: "The"
        - Token 2: "cat"
        - Token 3: "sat"
        Each adds to cache.
        """
        layer_idx = 0
        num_heads = small_config.num_heads
        head_dim = small_config.head_dim

        # Store 3 tokens sequentially
        keys = []
        values = []

        for i in range(3):
            k = torch.randn(1, num_heads, head_dim)
            v = torch.randn(1, num_heads, head_dim)

            cache.store(layer_idx, k, v)
            cache.update_num_tokens(1)

            keys.append(k)
            values.append(v)

        # Check count
        assert cache.num_cached_tokens == 3

        # Retrieve all
        cached_k, cached_v = cache.get(layer_idx)

        # Should have all 3 tokens
        assert cached_k.shape == (3, num_heads, head_dim)

        # Check each token matches
        for i in range(3):
            assert torch.allclose(cached_k[i:i+1], keys[i])
            assert torch.allclose(cached_v[i:i+1], values[i])

    def test_store_batch_of_tokens(self, cache, small_config):
        """
        Test storing multiple tokens at once (prefill).

        WHY THIS TEST?
        ─────────────
        During "prefill" phase, we process entire prompt at once.
        E.g., "Once upon a time" = 4 tokens processed together.
        """
        layer_idx = 0
        num_heads = small_config.num_heads
        head_dim = small_config.head_dim

        # Create K/V for 4 tokens at once
        batch_size = 4
        key = torch.randn(batch_size, num_heads, head_dim)
        value = torch.randn(batch_size, num_heads, head_dim)

        # Store batch
        cache.store(layer_idx, key, value)
        cache.update_num_tokens(batch_size)

        # Check
        assert cache.num_cached_tokens == 4

        cached_k, cached_v = cache.get(layer_idx)
        assert torch.allclose(cached_k, key)
        assert torch.allclose(cached_v, value)

    def test_store_multiple_layers(self, cache, small_config):
        """
        Test storing K/V for multiple layers.

        WHY THIS TEST?
        ─────────────
        Each transformer layer has its own K/V cache.
        Layer 0's cache ≠ Layer 1's cache.
        """
        num_heads = small_config.num_heads
        head_dim = small_config.head_dim

        # Store different K/V for each layer
        layer0_k = torch.randn(1, num_heads, head_dim)
        layer0_v = torch.randn(1, num_heads, head_dim)

        layer1_k = torch.randn(1, num_heads, head_dim)
        layer1_v = torch.randn(1, num_heads, head_dim)

        # Store in both layers
        cache.store(0, layer0_k, layer0_v)
        cache.store(1, layer1_k, layer1_v)
        cache.update_num_tokens(1)

        # Retrieve and verify they're different
        cached_k0, cached_v0 = cache.get(0)
        cached_k1, cached_v1 = cache.get(1)

        assert torch.allclose(cached_k0, layer0_k)
        assert torch.allclose(cached_v0, layer0_v)
        assert torch.allclose(cached_k1, layer1_k)
        assert torch.allclose(cached_v1, layer1_v)

        # Verify they're actually different
        assert not torch.allclose(cached_k0, cached_k1)

    def test_cache_overflow(self, cache, small_config):
        """
        Test that cache raises error when full.

        WHY THIS TEST?
        ─────────────
        DEMONSTRATES THE PROBLEM with contiguous allocation!

        We allocated max_seq_len=16 positions.
        If we try to cache token 17, we're out of space.

        This is the key limitation we'll fix in Phase 2!
        """
        num_heads = small_config.num_heads
        head_dim = small_config.head_dim

        # Fill cache to max
        for i in range(small_config.max_seq_len):
            k = torch.randn(1, num_heads, head_dim)
            v = torch.randn(1, num_heads, head_dim)
            cache.store(0, k, v)
            cache.update_num_tokens(1)

        # Should be full
        assert cache.num_cached_tokens == small_config.max_seq_len

        # Try to add one more - should fail
        k = torch.randn(1, num_heads, head_dim)
        v = torch.randn(1, num_heads, head_dim)

        with pytest.raises(ValueError, match="KV cache overflow"):
            cache.store(0, k, v)

        print("\n⚠️  Cache overflow! This is the limitation of contiguous allocation.")
        print("   In Phase 2, PagedAttention will handle this gracefully.")

    def test_clear_cache(self, cache, small_config):
        """Test clearing cache resets state."""
        num_heads = small_config.num_heads
        head_dim = small_config.head_dim

        # Add some tokens
        for i in range(5):
            k = torch.randn(1, num_heads, head_dim)
            v = torch.randn(1, num_heads, head_dim)
            cache.store(0, k, v)
            cache.update_num_tokens(1)

        assert cache.num_cached_tokens == 5

        # Clear
        cache.clear()

        # Should be empty
        assert cache.num_cached_tokens == 0

        # Get should return empty
        cached_k, cached_v = cache.get(0)
        assert cached_k.shape[0] == 0  # No tokens

    def test_get_empty_cache(self, cache):
        """Test getting from empty cache returns empty tensors."""
        cached_k, cached_v = cache.get(0)

        # Should be shape [0, num_heads, head_dim]
        assert cached_k.shape[0] == 0
        assert cached_v.shape[0] == 0


class TestMemoryUsageTracking:
    """Test memory usage tracking and reporting."""

    def test_memory_usage_empty(self):
        """Test memory usage when cache is empty."""
        config = CacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=8,
            max_seq_len=16,
        )
        cache = SimpleKVCache(config, device="cpu")

        usage = cache.get_memory_usage()

        # When empty, 100% is wasted
        assert usage["num_cached_tokens"] == 0
        assert usage["wasted_percent"] == 100.0
        print(f"\n💡 Empty cache wastes 100% of memory!")

    def test_memory_usage_partial(self):
        """
        Test memory usage when partially filled.

        WHY THIS TEST?
        ─────────────
        DEMONSTRATES WASTE!

        If we cache 4 tokens but allocated space for 16,
        we're wasting 75% of memory.

        This is the problem PagedAttention solves!
        """
        config = CacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=8,
            max_seq_len=16,
        )
        cache = SimpleKVCache(config, device="cpu")

        # Cache 4 tokens (25% of capacity)
        for i in range(4):
            k = torch.randn(1, 4, 8)
            v = torch.randn(1, 4, 8)
            cache.store(0, k, v)
            cache.update_num_tokens(1)

        usage = cache.get_memory_usage()

        assert usage["num_cached_tokens"] == 4
        assert usage["max_seq_len"] == 16

        # Should waste 75% (12/16 unused)
        assert abs(usage["wasted_percent"] - 75.0) < 0.1

        print(f"\n💡 Using 4/{config.max_seq_len} tokens:")
        print(f"   Wasted: {usage['wasted_percent']:.1f}%")
        print(f"   This is why we need PagedAttention!")

    def test_memory_usage_full(self):
        """Test memory usage when completely full."""
        config = CacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=8,
            max_seq_len=16,
        )
        cache = SimpleKVCache(config, device="cpu")

        # Fill completely
        for i in range(16):
            k = torch.randn(1, 4, 8)
            v = torch.randn(1, 4, 8)
            cache.store(0, k, v)
            cache.update_num_tokens(1)

        usage = cache.get_memory_usage()

        # When full, 0% wasted
        assert usage["wasted_percent"] == 0.0
        print(f"\n✅ Full cache wastes 0%")
        print(f"   But we can't grow beyond max_seq_len!")


class TestRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ shows useful info."""
        config = CacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=8,
            max_seq_len=16,
        )
        cache = SimpleKVCache(config, device="cpu")

        # Add some tokens
        for i in range(4):
            k = torch.randn(1, 4, 8)
            v = torch.randn(1, 4, 8)
            cache.store(0, k, v)
            cache.update_num_tokens(1)

        repr_str = repr(cache)

        # Should contain key information
        assert "SimpleKVCache" in repr_str
        assert "4/16" in repr_str  # tokens cached
        assert "wasted" in repr_str.lower()

        print(f"\n{repr_str}")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_length_batch(self):
        """Test storing 0 tokens (edge case)."""
        config = CacheConfig(num_layers=2, num_heads=4, head_dim=8, max_seq_len=16)
        cache = SimpleKVCache(config, device="cpu")

        # Empty batch
        k = torch.randn(0, 4, 8)
        v = torch.randn(0, 4, 8)

        cache.store(0, k, v)
        cache.update_num_tokens(0)

        # Should still be empty
        assert cache.num_cached_tokens == 0

    def test_exact_max_seq_len(self):
        """Test filling exactly to max_seq_len."""
        config = CacheConfig(num_layers=2, num_heads=4, head_dim=8, max_seq_len=8)
        cache = SimpleKVCache(config, device="cpu")

        # Fill exactly to max
        k = torch.randn(8, 4, 8)
        v = torch.randn(8, 4, 8)

        cache.store(0, k, v)
        cache.update_num_tokens(8)

        assert cache.num_cached_tokens == 8

        # Should work fine
        cached_k, cached_v = cache.get(0)
        assert cached_k.shape[0] == 8


# Integration test demonstrating the full workflow
class TestIntegration:
    """Integration tests showing realistic usage."""

    def test_typical_generation_workflow(self):
        """
        Test typical autoregressive generation workflow.

        SCENARIO: Generate "Hello, world!"
        1. Prefill: Process "Hello," (2 tokens) at once
        2. Generate: " world" one token at a time (7 tokens)

        WHY THIS TEST?
        ─────────────
        Shows how KV cache is actually used during inference!
        """
        config = CacheConfig(
            num_layers=2,
            num_heads=4,
            head_dim=8,
            max_seq_len=32,
        )
        cache = SimpleKVCache(config, device="cpu")

        print("\n" + "="*60)
        print("SIMULATING GENERATION: 'Hello, world!'")
        print("="*60)

        # PHASE 1: PREFILL
        # Process prompt "Hello," (2 tokens) together
        print("\n📝 Phase 1: Prefill (process prompt)")
        print("   Prompt: 'Hello,' (2 tokens)")

        prefill_k = torch.randn(2, 4, 8)
        prefill_v = torch.randn(2, 4, 8)

        for layer in range(config.num_layers):
            cache.store(layer, prefill_k, prefill_v)

        cache.update_num_tokens(2)
        print(f"   ✅ Cached 2 tokens (prefill)")

        # PHASE 2: GENERATION
        # Generate " world!" one token at a time (7 tokens)
        print("\n🎯 Phase 2: Generation (autoregressive)")

        tokens_to_generate = [" ", "world", "!"]
        for i, token in enumerate(tokens_to_generate, start=1):
            print(f"   Generating token {i}: '{token}'")

            # Generate K/V for new token
            new_k = torch.randn(1, 4, 8)
            new_v = torch.randn(1, 4, 8)

            # Store in all layers
            for layer in range(config.num_layers):
                cache.store(layer, new_k, new_v)

            cache.update_num_tokens(1)

            # At each step, we use ALL previous K/V for attention
            cached_k, cached_v = cache.get(0)
            print(f"      → Using K/V from {cached_k.shape[0]} previous tokens")

        print(f"\n✅ Generation complete!")
        print(f"   Total cached: {cache.num_cached_tokens} tokens")

        # Show memory waste
        usage = cache.get_memory_usage()
        print(f"\n📊 Memory Usage:")
        print(f"   Used: {cache.num_cached_tokens}/{config.max_seq_len} tokens")
        print(f"   Wasted: {usage['wasted_percent']:.1f}%")
        print(f"\n💡 This waste is WHY we need PagedAttention!")
        print("="*60 + "\n")

        # Verify final state
        assert cache.num_cached_tokens == 5  # 2 prefill + 3 generated
