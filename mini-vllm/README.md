# Mini-vLLM: A Learning-Focused LLM Inference Engine

A simplified implementation of vLLM for understanding modern LLM serving systems.

## 🎯 Purpose

Build a **functional but simplified** LLM inference engine that implements vLLM's core innovations:

- ✅ **PagedAttention:** Memory-efficient KV cache management
- ✅ **Continuous Batching:** Dynamic request scheduling
- ✅ **Prefix Caching:** Reuse computation across requests
- ✅ **Advanced Sampling:** Flexible generation strategies

**Target:** ~8,000-12,000 lines of well-documented Python code

**Outcome:** Deep understanding of modern LLM serving systems

## 📚 Documentation

We provide comprehensive learning materials:

1. **[MINI_VLLM_LEARNING_GUIDE.md](../MINI_VLLM_LEARNING_GUIDE.md)** - Complete technical reference
2. **[HISTORICAL_TIMELINE.md](../HISTORICAL_TIMELINE.md)** - Evolution of LLM inference
3. **[PHASED_IMPLEMENTATION_PLAN.md](../PHASED_IMPLEMENTATION_PLAN.md)** - Commit-by-commit roadmap
4. **[QUICK_START_GUIDE.md](../QUICK_START_GUIDE.md)** - Get started in 10 minutes

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd mini-vllm

# Install dependencies
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from mini_vllm.tokenizer import Tokenizer

# Load tokenizer
tokenizer = Tokenizer("gpt2")

# Encode text
token_ids = tokenizer.encode("Hello, world!")
print(token_ids)  # [15496, 11, 995, 0]

# Decode back to text
text = tokenizer.decode(token_ids)
print(text)  # "Hello, world!"
```

## 📖 Implementation Progress

### ✅ Phase 1: Single-Request Inference (Current)

- [x] **1.1: Tokenizer** - HuggingFace wrapper (~300 lines)
- [x] **1.2: Simple KV Cache** - Contiguous memory (~600 lines)
- [ ] 1.3: Model Loading - Load from HuggingFace
- [ ] 1.4: Attention Layer - Standard attention
- [ ] 1.5: Sampler - Greedy sampling
- [ ] 1.6: Engine - Basic generation loop

### 🔜 Phase 2: PagedAttention

- [ ] 2.1: Block Allocator - Memory management
- [ ] 2.2: Block Tables - Logical to physical mapping
- [ ] 2.3: Paged Attention - Non-contiguous KV cache
- [ ] 2.4: Memory Profiling - GPU memory detection
- [ ] 2.5: Integration - Replace simple KV cache

### 🔜 Phase 3: Continuous Batching

- [ ] 3.1: Request Queue - Request management
- [ ] 3.2: Dynamic Admission - Add/remove requests
- [ ] 3.3: Preemption - Pause/resume requests
- [ ] 3.4: Variable-Length Batching - Mixed lengths
- [ ] 3.5: Output Processor - Per-request outputs
- [ ] 3.6: Stopping Criteria - EOS, max_tokens

### 🔜 Phase 4: Advanced Sampling

- [ ] 4.1: Temperature Scaling
- [ ] 4.2: Top-k Sampling
- [ ] 4.3: Top-p (Nucleus) Sampling
- [ ] 4.4: Repetition Penalty
- [ ] 4.5: Per-Request Parameters
- [ ] 4.6: Beam Search

### 🔜 Phase 5: Prefix Caching

- [ ] 5.1: Block Hashing
- [ ] 5.2: Prefix Detection
- [ ] 5.3: LRU Eviction
- [ ] 5.4: Cache Sharing
- [ ] 5.5: Metrics

### 🔜 Phase 6: Distributed Inference

- [ ] 6.1: Tensor Parallelism
- [ ] 6.2: Pipeline Parallelism
- [ ] 6.3: Hybrid Parallelism

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/unit/test_tokenizer_simple.py -v

# Run with coverage
pytest --cov=mini_vllm tests/
```

## 📊 Project Structure

```
mini-vllm/
├── mini_vllm/              # Source code
│   ├── tokenizer/          # Text ↔ token conversion
│   ├── kv_cache/           # PagedAttention memory
│   ├── scheduler/          # Request scheduling
│   ├── attention/          # Attention mechanisms
│   ├── model_executor/     # Model execution
│   ├── sampler/            # Token sampling
│   ├── output/             # Output processing
│   └── engine/             # Orchestration
│
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── benchmarks/         # Performance tests
│
├── docs/                   # Documentation
│   ├── adrs/               # Architecture decisions
│   ├── comparisons/        # Mini vs real vLLM
│   └── checkpoints/        # Learning checkpoints
│
├── examples/               # Usage examples
└── tools/                  # Development tools
```

## 🎓 Learning Outcomes

After building mini-vLLM, you will understand:

- ✅ **LLM Inference Pipeline:** End-to-end execution flow
- ✅ **PagedAttention:** vLLM's memory management innovation
- ✅ **Continuous Batching:** Dynamic request scheduling
- ✅ **Memory Management:** OS concepts applied to ML
- ✅ **Sampling Strategies:** Quality vs diversity trade-offs
- ✅ **Distributed Systems:** Multi-GPU scaling
- ✅ **Production ML:** Real-world system design

## 📚 Component Overview

### Current Components

#### Tokenizer (`mini_vllm/tokenizer/`)

**Purpose:** Convert text ↔ token IDs

**Implementation:** Thin wrapper around HuggingFace `AutoTokenizer`

**Lines:** ~300

**Key Features:**
- Encode text to token IDs
- Decode token IDs to text
- Access special tokens (BOS, EOS, PAD)

**See:** `docs/adrs/001-tokenizer-choice.md` for design rationale

#### KV Cache (`mini_vllm/kv_cache/`)

**Purpose:** Store attention keys/values to avoid recomputation

**Implementation:** Contiguous memory allocation (simple but inefficient)

**Lines:** ~600

**Key Features:**
- Store K/V states for previous tokens
- Retrieve for attention computation
- Track memory usage and waste
- Demonstrates why PagedAttention is needed

**Key Insight:** Wastes ~95% of memory! This motivates Phase 2 (PagedAttention).

**See:** `docs/adrs/002-simple-kv-cache.md` for design rationale

## 🔬 Comparison with vLLM

| Feature | Mini-vLLM | Real vLLM | Difference |
|---------|-----------|-----------|------------|
| **Total Lines** | ~10,000 (target) | ~118,000 | 10x simpler |
| **Language** | Python only | Python + CUDA/C++ | Learning focus |
| **Models** | 1-2 (Llama, GPT) | 50+ | Focused scope |
| **Optimizations** | Basic | Extensive CUDA kernels | Correctness first |
| **Features** | Core only | Production-ready | Learning goals |

**Goal:** Understand the **concepts** deeply, not replicate every optimization.

## 📖 References

### Papers

1. **PagedAttention (vLLM):**
   [Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180), SOSP 2023

2. **Orca (Continuous Batching):**
   [Distributed Serving for Transformers](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022

3. **FlashAttention:**
   [Fast and Memory-Efficient Attention](https://arxiv.org/abs/2205.14135), NeurIPS 2022

### Code

- **vLLM:** https://github.com/vllm-project/vllm
- **Reference Path:** `/home/user/vllm/` (for comparing implementations)

### Community

- **vLLM Slack:** https://slack.vllm.ai
- **vLLM Docs:** https://docs.vllm.ai
- **Blog:** https://blog.vllm.ai

## 🤝 Contributing

This is a learning project! Feel free to:

- 📝 Improve documentation
- 🧪 Add tests
- 🔧 Fix bugs
- 💡 Suggest exercises
- 📊 Add visualizations

See `PHASED_IMPLEMENTATION_PLAN.md` for implementation guidelines.

## 📄 License

Educational project - follow the spirit of learning!

## 🙏 Acknowledgments

- **vLLM Team:** For the original system and research
- **HuggingFace:** For transformers and tokenizers
- **PyTorch:** For the deep learning framework

---

## 📞 Getting Help

- **Documentation:** Read learning guides first
- **Code Issues:** Check `PHASED_IMPLEMENTATION_PLAN.md`
- **Concepts:** Review `HISTORICAL_TIMELINE.md`
- **Community:** vLLM Slack for questions

---

**Current Phase:** 1.2 (Simple KV Cache) ✅ Complete

**Next Up:** Phase 1.3 (Model Loading)

**Progress:** 2/35 commits (6% complete)

Happy learning! 🎓
