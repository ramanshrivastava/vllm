"""
KV Cache module for mini-vLLM.

This module implements a simple KV (Key-Value) cache with contiguous
memory allocation. This is Phase 1 - we'll replace it with PagedAttention
in Phase 2.
"""

from mini_vllm.kv_cache.simple_cache import SimpleKVCache, CacheConfig

__all__ = ["SimpleKVCache", "CacheConfig"]
