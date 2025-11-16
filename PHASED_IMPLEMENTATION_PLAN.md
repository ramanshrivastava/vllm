# Phased Implementation Plan: Mini-vLLM
## Detailed Commit-by-Commit Roadmap with ADRs

---

## 📋 Overview

This document provides a **detailed, commit-level implementation plan** for building mini-vLLM. Each commit includes:

- **Specific code to write**
- **Architecture Decision Record (ADR)**
- **Tests to implement**
- **Learning checkpoint**
- **Expected outcomes**

---

## 🎯 Phase 1: Single-Request Inference (Week 1-2)

### Commit 1.1: Project Setup + Basic Tokenizer

**Branch:** `phase-1/tokenizer-setup`

**Code to Write:** (~500 lines)

```
mini-vllm/
├── setup.py
├── README.md
├── mini_vllm/
│   ├── __init__.py
│   └── tokenizer/
│       ├── __init__.py
│       ├── tokenizer.py          (300 lines)
│       └── README.md
├── tests/
│   └── unit/
│       └── test_tokenizer.py     (200 lines)
└── docs/
    └── adrs/
        └── 001-tokenizer-choice.md
```

**Key Files:**

`mini_vllm/tokenizer/tokenizer.py`:
```python
"""
Simple tokenizer wrapper around HuggingFace tokenizers.

Reference: /home/user/vllm/vllm/transformers_utils/tokenizers/
"""

from typing import List, Optional
from transformers import AutoTokenizer


class Tokenizer:
    """
    Tokenizer for encoding/decoding text.

    This is a thin wrapper around HuggingFace tokenizers.
    In real vLLM, this has ~1500 lines handling edge cases,
    special tokens, and optimizations. We keep it simple.
    """

    def __init__(self, model_name: str):
        """
        Args:
            model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    @property
    def bos_token_id(self) -> Optional[int]:
        """Beginning-of-sequence token ID."""
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        """End-of-sequence token ID."""
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> Optional[int]:
        """Padding token ID."""
        return self.tokenizer.pad_token_id
```

**ADR:** See `docs/adrs/001-tokenizer-choice.md` below

**Tests:** `tests/unit/test_tokenizer.py`:
```python
import pytest
from mini_vllm.tokenizer import Tokenizer


def test_tokenizer_encode_decode():
    """Test basic encode/decode."""
    tokenizer = Tokenizer("gpt2")  # Small model for testing

    text = "Hello, world!"
    token_ids = tokenizer.encode(text)

    # Should return list of ints
    assert isinstance(token_ids, list)
    assert all(isinstance(tid, int) for tid in token_ids)

    # Decode should reconstruct text
    decoded = tokenizer.decode(token_ids)
    assert decoded == text


def test_special_tokens():
    """Test special token handling."""
    tokenizer = Tokenizer("gpt2")

    # Check special tokens exist
    assert tokenizer.eos_token_id is not None
    # GPT-2 doesn't have BOS
    assert tokenizer.bos_token_id is None


def test_vocab_size():
    """Test vocabulary size."""
    tokenizer = Tokenizer("gpt2")
    assert tokenizer.vocab_size == 50257  # GPT-2 vocab size
```

**Learning Checkpoint:**

- ✅ Understand tokenization basics
- ✅ Know difference between BOS, EOS, PAD tokens
- ✅ Can load HuggingFace tokenizers

**Commit Message:**
```
[Phase 1.1] Add basic tokenizer - HuggingFace Integration

Implement simple tokenizer wrapper around HuggingFace's
AutoTokenizer. This provides encode/decode functionality
needed for text ↔ token conversion.

vLLM Reference: vllm/transformers_utils/tokenizers/
Historical Note: HuggingFace tokenizers (2019) standardized
                 fast tokenization across models

Design Decisions:
- Use HuggingFace instead of custom tokenizer (reuse)
- Simple wrapper to keep focused on inference logic
- Support special tokens (BOS, EOS, PAD)

Trade-offs:
- Simplified: No byte-level encoding, no custom vocab
- Kept: Core encode/decode functionality

Learning Outcomes:
1. Understand tokenization in LLMs
2. HuggingFace ecosystem integration
3. Special token handling

See docs/adrs/001-tokenizer-choice.md for full rationale.

Files changed:
- mini_vllm/tokenizer/tokenizer.py (+300 lines)
- tests/unit/test_tokenizer.py (+200 lines)
- docs/adrs/001-tokenizer-choice.md (+150 lines)
```

---

### ADR-001: Tokenizer Choice

**File:** `docs/adrs/001-tokenizer-choice.md`

```markdown
# ADR-001: Use HuggingFace Tokenizers

**Status:** Accepted
**Date:** 2025-11-16
**Commit:** [1.1]
**vLLM Reference:** `/home/user/vllm/vllm/transformers_utils/tokenizers/`
**Related:** HuggingFace Tokenizers library (2019)

## Context

We need a tokenizer to convert text → token IDs and token IDs → text.

**Requirements:**
- Support popular models (Llama, GPT, etc.)
- Fast encode/decode
- Handle special tokens
- Easy to use

**Constraints:**
- Learning-focused implementation
- Want to minimize non-inference code
- Should be model-agnostic

## Decision

Use **HuggingFace's AutoTokenizer** as a thin wrapper.

### Code Example

```python
from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)
```

## Rationale

### Why This Approach?

1. **Reuse:** HuggingFace tokenizers are battle-tested
2. **Compatibility:** Works with all HF models
3. **Focus:** Lets us focus on inference, not tokenization
4. **Speed:** Rust-based tokenizers are very fast

### Alternatives Considered

#### Alternative A: Implement Custom Tokenizer

**Pros:**
- Full control over implementation
- Learning experience

**Cons:**
- 1000+ lines of code
- Complex edge cases (unicode, byte-pair encoding, etc.)
- Not our learning focus (we want to learn inference)

**Rejected because:** Tokenization is not the core innovation we're studying.

#### Alternative B: Use SentencePiece Directly

**Pros:**
- Faster than some HF tokenizers
- Used by Llama models

**Cons:**
- Less flexible (only works for SentencePiece models)
- We want to support multiple model types

**Rejected because:** Less general than HF.

## vLLM Comparison

### What vLLM Does

vLLM has sophisticated tokenizer handling in `/home/user/vllm/vllm/transformers_utils/tokenizers/`:

- **Multi-format support:** HF, SentencePiece, custom
- **Detokenization optimizations:** Incremental decoding
- **Special handling:** Chat templates, system prompts
- **~1,500 lines** across multiple files

Key file: `vllm/transformers_utils/tokenizers/tokenizer.py`

### What We're Doing Differently

- **Simple wrapper:** Just encode/decode
- **No optimizations:** Focus on correctness
- **~300 lines:** Much simpler

**Rationale:** We're learning inference, not tokenization. We can always optimize later.

## Historical Evolution

- **2017 (Transformer):** Custom vocab for each model
- **2018 (BERT):** WordPiece tokenization
- **2019 (GPT-2):** Byte-Pair Encoding (BPE)
- **2019 (HuggingFace):** Unified tokenizers library
- **2023 (Llama):** SentencePiece with BPE
- **2024 (Modern):** Fast Rust-based tokenizers

**Our approach:** Use modern standard (HuggingFace)

## Trade-offs

### Benefits

✅ **Fast development:** 300 lines vs 1,500+
✅ **Reliable:** Battle-tested library
✅ **Flexible:** Works with all HF models
✅ **Learning focus:** Spend time on inference, not tokenization

### Limitations

❌ **Dependency:** Relies on external library
❌ **Less control:** Can't optimize tokenization
❌ **Learning:** Don't learn tokenization internals

**Accepted because:** Our focus is inference system design, not NLP preprocessing.

## Learning Outcomes

After this commit, you should understand:

1. **What tokenization does:** Convert text ↔ token IDs
2. **Special tokens:** BOS (beginning), EOS (end), PAD (padding)
3. **Vocabulary:** Mapping from tokens to IDs
4. **HuggingFace ecosystem:** Standard for model distribution

## References

- **HuggingFace Tokenizers:** https://github.com/huggingface/tokenizers
- **vLLM tokenizers:** `/home/user/vllm/vllm/transformers_utils/tokenizers/`
- **Original BPE paper:** Sennrich et al., ACL 2016

## Exercises

1. **Extend:** Add a method to count tokens in text (for billing)
2. **Debug:** What happens if you encode text with one tokenizer and decode with another?
3. **Optimize:** Can you cache tokenizer loading for faster startup?

## Next Steps

Phase 1.2: Implement simple KV cache for storing attention states.
```

---

### Commit 1.2: Simple KV Cache (Contiguous Memory)

**Branch:** `phase-1/simple-kv-cache`

**Code to Write:** (~400 lines)

```
mini_vllm/
├── mini_vllm/
│   └── kv_cache/
│       ├── __init__.py
│       ├── simple_cache.py       (300 lines)
│       └── README.md
├── tests/
│   └── unit/
│       └── test_simple_cache.py  (150 lines)
└── docs/
    └── adrs/
        └── 002-simple-kv-cache.md
```

**Key Implementation:**

```python
"""
Simple KV cache with contiguous memory allocation.

This is a SIMPLIFIED version. In Phase 2, we'll implement
PagedAttention with non-contiguous blocks.

Reference: /home/user/vllm/vllm/v1/core/kv_cache_manager.py
          (but much simpler!)
"""

import torch
from typing import Tuple
from dataclasses import dataclass


@dataclass
class CacheConfig:
    """Configuration for KV cache."""
    block_size: int = 16          # Not used yet (for Phase 2)
    num_layers: int = 32          # Number of transformer layers
    num_heads: int = 32           # Number of attention heads
    head_dim: int = 128           # Dimension of each head
    max_seq_len: int = 2048       # Maximum sequence length
    dtype: torch.dtype = torch.float16


class SimpleKVCache:
    """
    Simple KV cache with contiguous memory.

    Stores K and V tensors for each layer.
    Shape: [num_layers, max_seq_len, num_heads, head_dim]
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.num_layers = config.num_layers

        # Allocate cache for K and V
        # Shape: [num_layers, max_seq_len, num_heads, head_dim]
        self.k_cache = torch.zeros(
            config.num_layers,
            config.max_seq_len,
            config.num_heads,
            config.head_dim,
            dtype=config.dtype,
            device="cuda",
        )

        self.v_cache = torch.zeros(
            config.num_layers,
            config.max_seq_len,
            config.num_heads,
            config.head_dim,
            dtype=config.dtype,
            device="cuda",
        )

        # Track how many tokens are cached
        self.num_cached_tokens = 0

    def store(
        self,
        layer_idx: int,
        key: torch.Tensor,    # [num_new_tokens, num_heads, head_dim]
        value: torch.Tensor,  # [num_new_tokens, num_heads, head_dim]
    ) -> None:
        """
        Store new K, V into cache.

        Args:
            layer_idx: Which transformer layer
            key: New key states
            value: New value states
        """
        num_new_tokens = key.size(0)

        # Check if we have space
        if self.num_cached_tokens + num_new_tokens > self.config.max_seq_len:
            raise ValueError("KV cache overflow!")

        # Store K, V
        start = self.num_cached_tokens
        end = start + num_new_tokens

        self.k_cache[layer_idx, start:end] = key
        self.v_cache[layer_idx, start:end] = value

    def get(
        self,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve cached K, V for a layer.

        Returns:
            key: [num_cached_tokens, num_heads, head_dim]
            value: [num_cached_tokens, num_heads, head_dim]
        """
        if self.num_cached_tokens == 0:
            # Return empty tensors
            return (
                self.k_cache[layer_idx, :0],
                self.v_cache[layer_idx, :0],
            )

        return (
            self.k_cache[layer_idx, :self.num_cached_tokens],
            self.v_cache[layer_idx, :self.num_cached_tokens],
        )

    def update_num_tokens(self, num_new_tokens: int) -> None:
        """Update the number of cached tokens."""
        self.num_cached_tokens += num_new_tokens

    def clear(self) -> None:
        """Clear the cache."""
        self.num_cached_tokens = 0
        # No need to zero out memory, just reset counter
```

**ADR:** `docs/adrs/002-simple-kv-cache.md`

**Learning Checkpoint:**

- ✅ Understand what KV cache is
- ✅ Why we need it (avoid recomputing attention)
- ✅ Memory layout for multi-head attention

**Commit Message:**
```
[Phase 1.2] Add simple KV cache - Contiguous Memory

Implement basic KV cache with contiguous memory allocation.
This stores K and V states from attention layers to avoid
recomputing them for each token.

In Phase 2, we'll replace this with PagedAttention for
better memory efficiency.

vLLM Reference: vllm/v1/core/kv_cache_manager.py (simplified)
Historical Note: Original Transformer (2017) used contiguous KV storage

Design Decisions:
- Contiguous memory (simple, not efficient)
- Fixed max sequence length
- Per-layer storage

Trade-offs:
- Simplified: No block management, no paging
- Kept: Core KV storage concept

Learning Outcomes:
1. Understand KV cache purpose
2. Memory layout for attention
3. Why contiguous allocation is inefficient (sets up Phase 2)

See docs/adrs/002-simple-kv-cache.md
```

---

## 📊 Full Phase 1 Commit Timeline

| Commit | Feature | Lines | Tests | ADR | Time |
|--------|---------|-------|-------|-----|------|
| 1.1 | Tokenizer | 300 | 200 | 001 | 3h |
| 1.2 | Simple KV cache | 300 | 150 | 002 | 3h |
| 1.3 | Model loading | 600 | 200 | 003 | 5h |
| 1.4 | Standard attention | 800 | 250 | 004 | 6h |
| 1.5 | Greedy sampler | 400 | 200 | 005 | 4h |
| 1.6 | Basic engine | 500 | 300 | 006 | 5h |
| **Total** | **Phase 1** | **2,900** | **1,300** | **6** | **26h** |

---

## 🎯 Phase 2: PagedAttention (Week 2-3)

### Commit 2.1: Block Allocator Design

**Branch:** `phase-2/block-allocator`

**Code to Write:** (~500 lines)

**Key Innovation:** This is where we implement the **core idea of vLLM**!

```python
"""
Block allocator for PagedAttention.

This is the KEY innovation of vLLM: treat KV cache like virtual memory!

Reference: /home/user/vllm/vllm/v1/core/block_pool.py
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Block:
    """A single block of KV cache."""
    block_id: int           # Physical block ID
    ref_count: int = 0      # How many requests reference this block
    is_free: bool = True    # Is this block available?


class BlockAllocator:
    """
    Allocate and free KV cache blocks.

    Similar to OS page allocator, but for GPU memory blocks.

    Key concepts:
    - Block: Fixed-size chunk (e.g., 16 tokens)
    - Physical blocks: Actual GPU memory
    - Free list: Available blocks
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
    ):
        """
        Args:
            num_blocks: Total number of blocks in GPU memory
            block_size: Number of tokens per block
        """
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Initialize all blocks as free
        self.blocks = [
            Block(block_id=i, is_free=True)
            for i in range(num_blocks)
        ]

        # Free list (stack of available blocks)
        self.free_blocks: List[int] = list(range(num_blocks))

    def allocate(self, num_tokens: int) -> List[int]:
        """
        Allocate blocks for num_tokens.

        Args:
            num_tokens: How many tokens to store

        Returns:
            List of block IDs

        Raises:
            ValueError if not enough free blocks
        """
        # How many blocks do we need?
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise ValueError(
                f"Not enough free blocks! Need {num_blocks_needed}, "
                f"have {len(self.free_blocks)}"
            )

        # Pop blocks from free list
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            self.blocks[block_id].is_free = False
            self.blocks[block_id].ref_count = 1
            allocated.append(block_id)

        return allocated

    def free(self, block_ids: List[int]) -> None:
        """
        Free blocks back to the pool.

        Args:
            block_ids: Blocks to free
        """
        for block_id in block_ids:
            # Decrement reference count
            self.blocks[block_id].ref_count -= 1

            # If no more references, add to free list
            if self.blocks[block_id].ref_count == 0:
                self.blocks[block_id].is_free = True
                self.free_blocks.append(block_id)

    def get_num_free_blocks(self) -> int:
        """Return number of free blocks."""
        return len(self.free_blocks)

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if we can allocate blocks for num_tokens."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return len(self.free_blocks) >= num_blocks_needed
```

**ADR:** `docs/adrs/007-block-allocator.md`

**Historical Context:**

This is inspired by **virtual memory** in operating systems (1960s-1970s):

```
OS Virtual Memory        →    vLLM PagedAttention
──────────────────              ──────────────────
Page (4KB)              →    Block (16 tokens)
Page table              →    Block table
Physical memory         →    GPU memory
Allocate pages          →    Allocate blocks
Free pages              →    Free blocks
```

**Learning Checkpoint:**

- ✅ Understand OS virtual memory concepts
- ✅ How to apply them to ML systems
- ✅ Block-level memory management

---

## 📊 Complete Phased Implementation Summary

### Phase 1: Single-Request Inference
- **6 commits**
- **~3,000 lines code**
- **~1,300 lines tests**
- **Time:** 2 weeks
- **Outcome:** Can run inference for 1 request

### Phase 2: PagedAttention
- **5 commits**
- **~2,000 lines code**
- **~800 lines tests**
- **Time:** 1.5 weeks
- **Outcome:** Memory-efficient KV cache

### Phase 3: Continuous Batching
- **6 commits**
- **~2,500 lines code**
- **~1,000 lines tests**
- **Time:** 2 weeks
- **Outcome:** Process multiple requests

### Phase 4: Advanced Sampling
- **6 commits**
- **~1,000 lines code**
- **~600 lines tests**
- **Time:** 1 week
- **Outcome:** Flexible sampling strategies

### Phase 5: Prefix Caching
- **6 commits**
- **~1,500 lines code**
- **~700 lines tests**
- **Time:** 1.5 weeks
- **Outcome:** Reuse KV cache

### Phase 6: Distributed Inference
- **6 commits**
- **~2,000 lines code**
- **~800 lines tests**
- **Time:** 2 weeks
- **Outcome:** Multi-GPU support

---

## 📝 Development Workflow for Each Commit

### 1. Read ADR
- Understand the decision rationale
- Review historical context
- Check vLLM reference code

### 2. Implement
- Write code following the ADR
- Keep it simple (don't over-engineer)
- Add comments explaining key concepts

### 3. Test
- Write unit tests
- Ensure all tests pass
- Aim for >80% coverage

### 4. Document
- Update README
- Add code comments
- Create examples

### 5. Checkpoint
- Complete learning checkpoint quiz
- Do hands-on exercises
- Compare with vLLM

### 6. Commit
- Write detailed commit message
- Reference ADR
- Include learning outcomes

---

## 🎓 Learning Resources per Phase

### Phase 1: Fundamentals
- **Read:** The Illustrated Transformer
- **Watch:** Andrej Karpathy's GPT from Scratch
- **Code:** HuggingFace Transformers tutorial

### Phase 2: PagedAttention
- **Read:** vLLM paper (SOSP 2023)
- **Watch:** vLLM talk at Ray Summit
- **Code:** Simple page allocator in C

### Phase 3: Continuous Batching
- **Read:** Orca paper (OSDI 2022)
- **Watch:** Understanding batching in ML serving
- **Code:** Request queue implementation

### Phase 4: Sampling
- **Read:** The Curious Case of Neural Text Degeneration
- **Watch:** Sampling methods in LLMs
- **Code:** Different sampling algorithms

### Phase 5: Prefix Caching
- **Read:** vLLM v0.2.0 blog post
- **Watch:** Content-addressable storage
- **Code:** LRU cache implementation

### Phase 6: Distribution
- **Read:** Megatron-LM paper
- **Watch:** Distributed training/inference
- **Code:** Simple all-reduce with NCCL

---

## ✅ Success Criteria

### After Each Phase

You should be able to:

1. **Explain:** The core concepts in simple terms
2. **Implement:** A working feature from scratch
3. **Debug:** Common issues
4. **Extend:** Add new functionality
5. **Compare:** Mini vs real vLLM

### After All Phases

You should understand:

- ✅ End-to-end LLM inference pipeline
- ✅ Memory management for LLMs
- ✅ Request scheduling and batching
- ✅ Attention mechanisms and optimizations
- ✅ Sampling strategies
- ✅ Distributed inference
- ✅ Production ML system design

---

## 🚀 Ready to Start?

1. Read [MINI_VLLM_LEARNING_GUIDE.md](./MINI_VLLM_LEARNING_GUIDE.md)
2. Review [HISTORICAL_TIMELINE.md](./HISTORICAL_TIMELINE.md)
3. Begin **Commit 1.1**: Project setup + tokenizer
4. Follow the ADRs and learning checkpoints
5. Build your mini-vLLM! 🎯

---

## 📧 Questions?

- **Understanding:** Re-read the ADR and compare with vLLM code
- **Bugs:** Check tests, add debugging prints
- **Design:** Review trade-offs in ADR
- **Extensions:** See exercises in learning checkpoints

Happy building! 🎓
