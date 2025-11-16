# Mini-vLLM Learning Guide
## Building a Minimal LLM Inference Engine from Scratch
### Based on vLLM Codebase Analysis

---

## 📊 Quick Reference

### What's in vLLM (Full Production System)

**Total Code:** ~68,000 lines of Python + ~50,000 lines of CUDA/C++

**Main Components:** 8 Core Systems
- Tokenization & Input Processing
- Request Scheduling
- KV Cache Management (PagedAttention)
- Attention Mechanisms
- Model Execution
- Output Processing
- Distributed Execution
- Serving Infrastructure

**Supported Models:** 50+ (Llama, GPT, Mixtral, etc.)

**Hardware Support:** NVIDIA, AMD, Intel, TPU, CPU

**Key Innovation:** PagedAttention for efficient memory management

### What Your Mini Version Should Have

**Target Size:** 8,000-12,000 lines of Python

**Core Components:** 6 simplified systems

**Basic Model:** Single transformer-based LLM (Llama-style)

**Memory:** PagedAttention (simplified)

**Batching:** Continuous batching (basic)

**Hardware:** Single GPU initially (CUDA)

**Parallelism:** None initially (add in Phase 2+)

---

## 🏗️ Architecture Overview

```
Input Text (Prompt)
         ↓
    TOKENIZER              (500-800 lines)
         ↓
    Request Queue
         ↓
    SCHEDULER              (1,500-2,500 lines)
  ├─ Request Manager
  ├─ KV Cache Manager
  └─ Batch Builder
         ↓
    Batch of Requests
         ↓
    MODEL EXECUTOR         (2,500-3,500 lines)
  ├─ KV Cache (PagedAttention)
  ├─ Attention Layer
  ├─ Transformer Blocks
  └─ Model Runner
         ↓
    Logits
         ↓
    SAMPLER                (800-1,200 lines)
  ├─ Temperature Scaling
  ├─ Top-k/Top-p
  └─ Token Sampling
         ↓
    Output Tokens
         ↓
    DETOKENIZER            (300-500 lines)
         ↓
    Generated Text
```

---

## 📦 Component Design Details

### 1. TOKENIZER (500-800 lines)

**Purpose:** Convert text → token IDs, and token IDs → text

**Key Responsibilities:**
- Load tokenizer from HuggingFace
- Encode text to token IDs
- Decode token IDs to text
- Handle special tokens (BOS, EOS, PAD)

**Simplified Implementation:**

```python
# Core structure
class Tokenizer:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)
```

**Key Functions:**
- `encode(text)` → token IDs
- `decode(token_ids)` → text
- `get_vocab_size()` → int

**Reference:**
`/home/user/vllm/vllm/transformers_utils/tokenizers/` (~1,500 lines total - heavily optimized)

---

### 2. SCHEDULER (1,500-2,500 lines)

**Purpose:** Manage requests, allocate KV cache, build batches

#### 2.1 Request Manager (500-700 lines)

**Core Data Structure:**

```python
@dataclass
class Request:
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    max_tokens: int

    # State
    output_token_ids: List[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0

    # KV Cache
    kv_block_ids: List[int] = field(default_factory=list)
```

**Key Operations:**
- Add request to queue
- Track request state (waiting, running, finished)
- Manage output generation

#### 2.2 KV Cache Manager (800-1,200 lines)

**Purpose:** Implement PagedAttention memory management

**Core Concept:**
Instead of allocating contiguous memory for KV cache, split it into fixed-size blocks (e.g., 16 tokens per block) that can be allocated non-contiguously, similar to virtual memory paging.

**Key Data Structures:**

```python
@dataclass
class BlockTable:
    """Maps logical KV cache positions to physical blocks"""
    logical_idx: int  # Position in sequence
    physical_idx: int  # Physical block in GPU memory

class KVCacheManager:
    def __init__(self, block_size: int, num_blocks: int):
        self.block_size = block_size  # Tokens per block (e.g., 16)
        self.num_blocks = num_blocks  # Total physical blocks
        self.free_blocks: List[int] = list(range(num_blocks))

        # KV cache storage: [num_blocks, block_size, num_heads, head_dim]
        self.kv_cache = allocate_kv_cache(...)
```

**Key Functions:**
- `allocate_blocks(num_tokens)` → block IDs
- `free_blocks(block_ids)` → None
- `get_num_free_blocks()` → int
- `can_allocate(num_tokens)` → bool

**Reference:**
`/home/user/vllm/vllm/v1/core/kv_cache_manager.py` (~17,000 lines with all features)

#### 2.3 Batch Builder (400-600 lines)

**Purpose:** Select requests and build execution batch

```python
class Scheduler:
    def schedule(self) -> SchedulerOutput:
        """
        1. Select requests that fit in available KV cache
        2. Allocate KV cache blocks
        3. Build batch for model execution
        """
        running_requests = self.get_running_requests()
        waiting_requests = self.get_waiting_requests()

        # Allocate new requests if space available
        for req in waiting_requests:
            if self.can_schedule(req):
                self.allocate_kv_blocks(req)
                running_requests.append(req)

        return SchedulerOutput(
            scheduled_requests=running_requests,
            num_tokens=sum(req.num_tokens for req in running_requests)
        )
```

**Key Functions:**
- `schedule()` → batch to execute
- `can_schedule(request)` → bool
- `allocate_kv_blocks(request)` → None

**Reference:**
`/home/user/vllm/vllm/v1/core/sched/` (~multiple files, complex scheduling)

---

### 3. MODEL EXECUTOR (2,500-3,500 lines)

**Purpose:** Run the transformer model on GPU

#### 3.1 PagedAttention Kernel (800-1,200 lines Python wrapper + CUDA)

**Core Innovation:** Attention with non-contiguous KV cache blocks

**Standard Attention:**
```python
# Q: [batch, seq_len, num_heads, head_dim]
# K, V: [batch, kv_seq_len, num_heads, head_dim]
attn_output = flash_attention(Q, K, V)
```

**Paged Attention:**
```python
# Q: [num_tokens, num_heads, head_dim]
# K, V stored in blocks: [num_blocks, block_size, num_heads, head_dim]
# block_tables: [batch_size, max_blocks] - mapping to physical blocks

attn_output = paged_attention(
    query=Q,
    kv_cache=kv_cache,  # Physical KV cache blocks
    block_tables=block_tables,  # Logical to physical mapping
    ...
)
```

**Key Functions:**
- `paged_attention_v1()` - prefill phase
- `paged_attention_v2()` - decode phase
- `reshape_and_cache()` - store new KV to cache

**Reference:**
`/home/user/vllm/vllm/attention/` + `/home/user/vllm/csrc/attention/` (CUDA kernels)

#### 3.2 Model Runner (1,000-1,500 lines)

**Purpose:** Prepare inputs, run model, extract outputs

```python
class ModelRunner:
    def __init__(self, model, kv_cache):
        self.model = model
        self.kv_cache = kv_cache

    def execute_model(
        self,
        requests: List[Request],
        kv_cache_manager: KVCacheManager,
    ) -> torch.Tensor:
        """
        1. Prepare input tensors
        2. Run model forward pass
        3. Return logits
        """
        # Build input tensors
        input_ids = self._prepare_inputs(requests)
        positions = self._prepare_positions(requests)
        block_tables = self._prepare_block_tables(requests)

        # Run model
        logits = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_cache=self.kv_cache,
            block_tables=block_tables,
        )

        return logits
```

**Key Functions:**
- `execute_model()` → logits
- `prepare_inputs()` → input tensors
- `profile_memory()` → available memory

**Reference:**
`/home/user/vllm/vllm/v1/worker/gpu_model_runner.py` (~1,500 lines)

#### 3.3 Model (700-1,000 lines)

**Purpose:** Implement transformer architecture

**Simplified Transformer:**

```python
class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positions,
                kv_cache,
                block_tables,
            )

        logits = self.lm_head(hidden_states)
        return logits
```

**Key Functions:**
- `forward()` → logits
- `load_weights()` → None

**Reference:**
`/home/user/vllm/vllm/model_executor/models/llama.py` (~800 lines)

---

### 4. SAMPLER (800-1,200 lines)

**Purpose:** Sample next tokens from logits

**Sampling Methods:**

```python
class Sampler:
    def sample(
        self,
        logits: torch.Tensor,  # [batch_size, vocab_size]
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        1. Apply temperature
        2. Apply top-k/top-p filtering
        3. Sample from distribution
        """
        # Temperature scaling
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            logits = self.top_k_filter(logits, top_k)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            logits = self.top_p_filter(logits, top_p)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Sample
        next_tokens = torch.multinomial(probs, num_samples=1)

        return next_tokens
```

**Key Functions:**
- `sample()` → token IDs
- `top_k_filter()` → filtered logits
- `top_p_filter()` → filtered logits
- `greedy_sample()` → token IDs

**Reference:**
`/home/user/vllm/vllm/v1/sample/` (~multiple files)

---

### 5. OUTPUT PROCESSOR (300-500 lines)

**Purpose:** Manage output generation and completion

```python
class OutputProcessor:
    def process_outputs(
        self,
        requests: List[Request],
        sampled_tokens: torch.Tensor,
    ) -> List[RequestOutput]:
        """
        1. Update request outputs
        2. Check stopping criteria
        3. Return outputs
        """
        outputs = []

        for req, token in zip(requests, sampled_tokens):
            req.output_token_ids.append(token.item())

            # Check stopping conditions
            if self.should_stop(req):
                req.status = RequestStatus.FINISHED
                text = self.detokenizer.decode(req.output_token_ids)
                outputs.append(RequestOutput(
                    request_id=req.request_id,
                    text=text,
                    finished=True,
                ))

        return outputs
```

**Key Functions:**
- `process_outputs()` → outputs
- `should_stop()` → bool
- `format_output()` → RequestOutput

**Reference:**
`/home/user/vllm/vllm/v1/engine/output_processor.py` (~600 lines)

---

### 6. ENGINE (1,000-1,500 lines)

**Purpose:** Orchestrate all components

```python
class LLMEngine:
    def __init__(self, model_name: str):
        self.tokenizer = Tokenizer(model_name)
        self.scheduler = Scheduler(...)
        self.model_runner = ModelRunner(...)
        self.sampler = Sampler()
        self.output_processor = OutputProcessor(...)

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Main generation loop:
        1. Tokenize
        2. Create request
        3. Loop:
           a. Schedule batch
           b. Execute model
           c. Sample tokens
           d. Process outputs
        4. Return result
        """
        # Tokenize
        token_ids = self.tokenizer.encode(prompt)

        # Create request
        request = Request(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            prompt_token_ids=token_ids,
            max_tokens=max_tokens,
        )
        self.scheduler.add_request(request)

        # Generation loop
        while not request.is_finished():
            # Schedule
            batch = self.scheduler.schedule()

            # Execute
            logits = self.model_runner.execute_model(
                batch.requests,
                self.scheduler.kv_cache_manager,
            )

            # Sample
            tokens = self.sampler.sample(logits)

            # Process
            outputs = self.output_processor.process_outputs(
                batch.requests,
                tokens,
            )

        return outputs[0].text
```

**Key Functions:**
- `generate()` → text
- `add_request()` → None
- `step()` → outputs

**Reference:**
`/home/user/vllm/vllm/v1/engine/llm_engine.py` (~400 lines core logic)

---

## 📅 Implementation Roadmap

### Phase 1: Single-Request Inference (Week 1-2, ~3,000 lines)

**Goal:** Run inference for ONE request at a time (no batching)

**Components:**
1. **Tokenizer** (500 lines)
   - Load HuggingFace tokenizer
   - Encode/decode functions

2. **Simple KV Cache** (400 lines)
   - Contiguous memory allocation (no paging yet!)
   - Just store K, V tensors

3. **Model Executor** (1,200 lines)
   - Load Llama model from HuggingFace
   - Standard attention (no paging)
   - Forward pass

4. **Sampler** (400 lines)
   - Greedy sampling
   - Temperature scaling

5. **Basic Engine** (500 lines)
   - Simple generate() loop
   - Single request processing

**Test:**
```python
engine = LLMEngine("meta-llama/Llama-2-7b-hf")
output = engine.generate("Once upon a time", max_tokens=50)
print(output)
```

**Historical Context:** Similar to GPT-1 (2018) - simple autoregressive generation

---

### Phase 2: PagedAttention (Week 2-3, +2,000 lines)

**Goal:** Implement efficient KV cache management

**Components:**

1. **KV Cache Manager** (1,000 lines)
   - Block allocation/deallocation
   - Block tables
   - Physical block storage

2. **Paged Attention** (800 lines)
   - Implement paged attention kernel (can use xFormers initially)
   - Integrate with model

3. **Updated Model** (200 lines changes)
   - Use paged attention instead of standard attention

**Test:**
```python
# Same interface, but more memory-efficient
engine = LLMEngine("meta-llama/Llama-2-7b-hf")
output = engine.generate("Once upon a time", max_tokens=500)  # Longer!
```

**Historical Context:** PagedAttention paper (SOSP 2023) - vLLM's core innovation

**Key Paper:** [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

---

### Phase 3: Continuous Batching (Week 3-4, +2,500 lines)

**Goal:** Process multiple requests simultaneously

**Components:**

1. **Scheduler** (1,500 lines)
   - Request queue
   - Batch builder
   - KV cache allocation per request

2. **Batched Model Execution** (500 lines)
   - Handle variable-length sequences
   - Batched paged attention

3. **Output Processor** (500 lines)
   - Per-request output tracking
   - Stopping criteria

**Test:**
```python
# Multiple requests
requests = [
    "Once upon a time",
    "The capital of France is",
    "In a galaxy far, far away",
]

outputs = engine.generate_batch(requests, max_tokens=100)
for out in outputs:
    print(out)
```

**Historical Context:** Orca (ATC 2022) introduced iteration-level batching

---

### Phase 4: Advanced Sampling (Week 4-5, +1,000 lines)

**Goal:** Support various sampling strategies

**Components:**

1. **Advanced Sampler** (800 lines)
   - Top-k sampling
   - Top-p (nucleus) sampling
   - Beam search (optional)

2. **Sampling Parameters** (200 lines)
   - Per-request sampling configs
   - Temperature, top-k, top-p, etc.

**Test:**
```python
output = engine.generate(
    "Write a poem about",
    max_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)
```

---

### Phase 5: Prefix Caching (Week 5-6, +1,500 lines)

**Goal:** Reuse KV cache for common prefixes

**Components:**

1. **Prefix Cache** (1,000 lines)
   - Hash-based block matching
   - LRU eviction
   - Prefix sharing across requests

2. **Updated Scheduler** (500 lines)
   - Detect prefix matches
   - Share KV cache blocks

**Test:**
```python
# Same prefix, reused KV cache
prompts = [
    "Translate to French: Hello",
    "Translate to French: Goodbye",
    "Translate to French: Thank you",
]
# "Translate to French:" prefix is cached!
```

**Historical Context:** vLLM v0.2.0 (2024) added automatic prefix caching

---

### Phase 6: Distributed Inference (Phase 2+, +2,000 lines)

**Goal:** Scale across multiple GPUs

**Components:**

1. **Tensor Parallelism** (1,000 lines)
   - Shard model across GPUs
   - All-reduce communication

2. **Pipeline Parallelism** (1,000 lines)
   - Layer-wise distribution
   - Micro-batching

**Test:**
```python
# Run 70B model on 8 GPUs
engine = LLMEngine(
    "meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=8,
)
```

---

## 🎯 Key Design Decisions for Mini Version

### 1. PagedAttention Implementation

**vLLM:** Custom CUDA kernels (~5,000 lines C++/CUDA)

**Recommendation:** Use existing library (xFormers, FlashAttention-2) initially, then implement custom

**Trade-off:**
- ✅ Faster development
- ❌ Less control, learning

### 2. Scheduling Policy

**vLLM:** FCFS, Priority-based, configurable

**Recommendation:** Simple FCFS initially

**Trade-off:**
- ✅ Simple to implement
- ❌ Not optimal for all workloads

### 3. KV Cache Block Size

**vLLM:** 16 tokens (configurable)

**Recommendation:** 16 tokens (good balance of fragmentation vs overhead)

### 4. Model Support

**vLLM:** 50+ models

**Recommendation:** Single model (Llama) initially

**Trade-off:**
- ✅ Focused learning
- ❌ Less generalizable

### 5. Distributed Execution

**vLLM:** Tensor, Pipeline, Data, Expert parallelism

**Recommendation:** Single GPU initially, Tensor Parallelism in Phase 2+

### 6. Memory Profiling

**vLLM:** Automatic GPU memory profiling

**Recommendation:** Manual configuration initially

**Trade-off:**
- ✅ Simpler code
- ❌ Less user-friendly

### 7. Quantization

**vLLM:** GPTQ, AWQ, INT4, INT8, FP8

**Recommendation:** Skip initially (Phase 3+)

---

## 📁 File Organization for Mini-vLLM

```
mini-vllm/
├── README.md
├── LEARNING_GUIDE.md              # This file
├── HISTORICAL_TIMELINE.md         # vLLM evolution
│
├── docs/
│   ├── adrs/                      # Architecture Decision Records
│   │   ├── 001-paged-attention.md
│   │   ├── 002-continuous-batching.md
│   │   ├── 003-prefix-caching.md
│   │   └── ...
│   │
│   ├── comparisons/               # Mini vs Real vLLM
│   │   ├── scheduler-comparison.md
│   │   ├── kv-cache-comparison.md
│   │   └── ...
│   │
│   ├── diagrams/                  # Visual architecture
│   │   ├── execution-flow.svg
│   │   ├── paged-attention.svg
│   │   └── ...
│   │
│   ├── checkpoints/               # Learning checkpoints
│   │   ├── phase1-checkpoint.md
│   │   ├── phase2-checkpoint.md
│   │   └── ...
│   │
│   └── references/                # vLLM references
│       ├── papers.md
│       ├── vllm-commits.md
│       └── related-systems.md
│
├── mini_vllm/
│   ├── __init__.py
│   │
│   ├── tokenizer/                 # Phase 1
│   │   ├── __init__.py
│   │   ├── tokenizer.py           (500 lines)
│   │   └── README.md
│   │
│   ├── scheduler/                 # Phase 3
│   │   ├── __init__.py
│   │   ├── request.py             (200 lines)
│   │   ├── scheduler.py           (800 lines)
│   │   └── README.md
│   │
│   ├── kv_cache/                  # Phase 2
│   │   ├── __init__.py
│   │   ├── manager.py             (1,000 lines)
│   │   ├── block_allocator.py     (500 lines)
│   │   └── README.md
│   │
│   ├── attention/                 # Phase 2
│   │   ├── __init__.py
│   │   ├── paged_attention.py     (800 lines)
│   │   ├── standard_attention.py  (400 lines)
│   │   └── README.md
│   │
│   ├── model_executor/            # Phase 1
│   │   ├── __init__.py
│   │   ├── model_runner.py        (1,000 lines)
│   │   ├── model_loader.py        (500 lines)
│   │   └── models/
│   │       ├── llama.py           (800 lines)
│   │       └── README.md
│   │
│   ├── sampler/                   # Phase 1, 4
│   │   ├── __init__.py
│   │   ├── sampler.py             (800 lines)
│   │   └── README.md
│   │
│   ├── output/                    # Phase 3
│   │   ├── __init__.py
│   │   ├── processor.py           (400 lines)
│   │   └── README.md
│   │
│   ├── engine/                    # Phase 1, 3
│   │   ├── __init__.py
│   │   ├── llm_engine.py          (1,000 lines)
│   │   └── README.md
│   │
│   └── utils/
│       ├── memory.py
│       ├── profiler.py
│       └── ...
│
├── tests/
│   ├── unit/                      # Unit tests per module
│   │   ├── test_tokenizer.py
│   │   ├── test_scheduler.py
│   │   ├── test_kv_cache.py
│   │   └── ...
│   │
│   ├── integration/               # End-to-end tests
│   │   ├── test_single_request.py
│   │   ├── test_batching.py
│   │   └── ...
│   │
│   └── benchmarks/                # Performance tests
│       ├── bench_throughput.py
│       ├── bench_latency.py
│       └── ...
│
├── examples/
│   ├── 01-simple-generate.py
│   ├── 02-batched-generation.py
│   ├── 03-streaming.py
│   └── ...
│
└── tools/
    ├── visualizer/                # KV cache visualizer
    ├── profiler/                  # Performance profiler
    └── debugger/                  # Step-through debugger
```

---

## 📝 ADR Template (Each Commit Includes One)

```markdown
# ADR-XXX: [Decision Title]

**Status:** Accepted
**Date:** 2025-XX-XX
**Commit:** [hash]
**vLLM Reference:** [file:line]
**Related Papers:** [paper links]

## Context

What problem are we solving? What constraints exist?

## Decision

What approach did we choose?

### Code Example

```python
# Our mini-vLLM implementation
[code snippet]
```

## Rationale

### Why This Approach?

1. Reason 1
2. Reason 2

### Alternatives Considered

- **Alternative A:** [why rejected]
- **Alternative B:** [why rejected]

## vLLM Comparison

### What vLLM Does

[Explanation + file references]

### What We're Doing Differently

[Simplifications + rationale]

## Historical Evolution

- **vLLM v0.1 (2023-06):** [original approach]
- **vLLM v0.2 (2024-01):** [major change]
- **vLLM v0.6 (2025-01):** [V1 release]

## Trade-offs

### Benefits
✅ Benefit 1
✅ Benefit 2

### Limitations
❌ Limitation 1
❌ Limitation 2

## Learning Outcomes

After this commit, you should understand:

1. [Concept 1]
2. [Concept 2]
3. [Concept 3]

## References

- vLLM: `/home/user/vllm/[file]:[lines]`
- Paper: [link]
- Commit: [vLLM GitHub commit hash]

## Exercises

1. [Hands-on exercise]
2. [Extension challenge]
3. [Debugging task]
```

---

## 📚 Historical Timeline: vLLM Evolution

### 2023-06: vLLM v0.1.0 - Initial Release
- **Innovation:** PagedAttention
- **Paper:** SOSP 2023
- **Key Features:** Basic continuous batching, single GPU

### 2023-08: v0.1.5 - Tensor Parallelism
- **Feature:** Multi-GPU support
- **Influenced by:** Megatron-LM

### 2023-11: v0.2.0 - Prefix Caching
- **Innovation:** Automatic KV cache sharing
- **Performance:** 2-3x speedup for similar prompts

### 2024-03: v0.3.0 - Speculative Decoding
- **Innovation:** Draft-then-verify generation
- **Performance:** 2-4x speedup

### 2024-07: v0.4.0 - Multi-LoRA
- **Feature:** Serve multiple LoRA adapters
- **Use Case:** Multi-tenant serving

### 2024-09: v0.5.0 - Chunked Prefill
- **Innovation:** Split long prefills
- **Benefit:** Better batching, lower latency

### 2025-01: v0.6.0 - V1 Architecture
- **Redesign:** Clean, modular codebase
- **Performance:** 1.7x speedup
- **Features:** Zero-overhead prefix caching, unified scheduler

---

## 🎓 Learning Checkpoint Template

```markdown
# Phase X Checkpoint: [Phase Name]

## Self-Assessment Quiz

### Conceptual Understanding

1. **Question:** Why does PagedAttention improve memory efficiency?
   - **Answer:** [expected answer]
   - **Reference:** ADR-XXX

2. [More questions...]

### Code Comprehension

1. **Question:** What does this code do?
   ```python
   [code snippet]
   ```
   - **Answer:** [explanation]

## Hands-On Exercises

### Exercise 1: Extend the Feature

**Task:** Add support for [new feature]
**Difficulty:** ⭐⭐☆☆☆
**Estimated Time:** 45 minutes
**Learning Goal:** Understand [concept]

**Hints:**
- Consider how vLLM does this in [file:line]
- You'll need to modify [files]

### Exercise 2: Debug the Code

**Task:** We've introduced a bug in [component]. Find and fix it.
**Difficulty:** ⭐⭐⭐☆☆
**Bug Description:** [what breaks]

### Exercise 3: Performance Analysis

**Task:** Profile the code and identify bottlenecks
**Tools:** `nvprof`, `torch.profiler`
**Expected Bottleneck:** [hint]

## Comparative Analysis

### Mini-vLLM vs Real vLLM

| Aspect | Mini-vLLM | Real vLLM | Why Different? |
|--------|-----------|-----------|----------------|
| Lines of code | 500 | 17,000 | We skip edge cases, optimizations |
| Features | Basic paging | Prefix cache, chunked prefill | Learning focus |
| Performance | 100 tok/s | 1000+ tok/s | Custom CUDA kernels |

## Performance Benchmark

Run `python benchmarks/phase-X.py` and compare results:

**Expectation:** [what should happen]
**Bottleneck:** [where is it slow]
**vLLM's Solution:** [how they optimize]

## Next Steps

Before moving to Phase X+1, ensure you can:

☐ Explain [concept] to someone else
☐ Modify the code to add [feature]
☐ Identify the performance bottleneck
☐ Read the equivalent vLLM code comfortably
```

---

## 🔧 Tools for Learning

### 1. KV Cache Visualizer

```bash
python tools/visualizer/show_kv_cache.py \
  --prompt "Once upon a time" \
  --max-tokens 50
```

**Shows:**
- Block allocation over time
- Memory fragmentation
- Cache hit/miss for prefix caching

### 2. Performance Profiler

```bash
python tools/profiler/profile.py \
  --model meta-llama/Llama-2-7b-hf \
  --prompt "Hello world" \
  --compare-with vllm
```

**Output:**
- Throughput (tokens/sec)
- Latency (time to first token, inter-token)
- Memory usage
- Comparison with real vLLM

### 3. Batch Visualizer

```bash
python tools/visualizer/show_batches.py \
  --num-requests 10 \
  --trace batch_schedule.json
```

**Shows:**
- Request scheduling timeline
- Batch composition over time
- KV cache allocation/deallocation

---

## 📊 Performance Expectations

### Mini-vLLM (Python, basic optimizations)

- **Throughput:** 50-200 tokens/sec (single GPU)
- **Latency (TTFT):** 50-100ms
- **Memory Efficiency:** 60-80% of vLLM

### Real vLLM (Optimized CUDA kernels)

- **Throughput:** 500-2,000 tokens/sec (single GPU)
- **Latency (TTFT):** 10-30ms
- **Memory Efficiency:** 90-95% theoretical max

**Focus on correctness first, performance later!**

---

## 🎯 Common Pitfalls to Avoid

### 1. KV Cache Management

❌ **Pitfall:** Forgetting to free blocks
✅ **Solution:** Implement proper lifecycle management

### 2. Batching Logic

❌ **Pitfall:** Batch size too large → OOM
✅ **Solution:** Check available KV cache before scheduling

### 3. Attention Mask

❌ **Pitfall:** Incorrect masking in paged attention
✅ **Solution:** Carefully handle block boundaries

### 4. Token Position Encoding

❌ **Pitfall:** Wrong positions for batched requests
✅ **Solution:** Track per-request positions separately

### 5. Memory Fragmentation

❌ **Pitfall:** Poor block allocation strategy
✅ **Solution:** Implement best-fit or first-fit allocation

### 6. Stopping Criteria

❌ **Pitfall:** Not checking EOS token
✅ **Solution:** Check both max_tokens AND EOS

---

## 📖 References

### Core Papers

1. **PagedAttention (vLLM):**
   [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
   SOSP 2023

2. **FlashAttention:**
   [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
   NeurIPS 2022

3. **Continuous Batching (Orca):**
   [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
   OSDI 2022

4. **Speculative Decoding:**
   [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
   ICML 2023

### vLLM Source Code

- **Engine:** `/home/user/vllm/vllm/v1/engine/core.py`
- **Scheduler:** `/home/user/vllm/vllm/v1/core/sched/`
- **KV Cache:** `/home/user/vllm/vllm/v1/core/kv_cache_manager.py`
- **Attention:** `/home/user/vllm/vllm/attention/`
- **Model Runner:** `/home/user/vllm/vllm/v1/worker/gpu_model_runner.py`

### Related Systems

- **Hugging Face Text Generation Inference (TGI)**
- **NVIDIA TensorRT-LLM**
- **DeepSpeed-Inference**
- **OpenAI Triton Inference Server**

---

## 🎓 Commit Message Format

```
[Phase X.Y] Title - Historical Context

Brief description of what this commit implements.

vLLM Reference: vllm/v1/core/scheduler.py:100-200
Paper Reference: PagedAttention (SOSP 2023)
Historical Note: This mirrors vLLM v0.1's initial scheduler

Design Decisions:
- Decision 1: [rationale]
- Decision 2: [rationale]

Trade-offs:
- We simplified [X] because [Y]
- We kept [A] to preserve [B]

Learning Outcomes:
1. Understand [concept]
2. See how [feature] works
3. Compare [approach A] vs [approach B]

See docs/adrs/ADR-XXX.md for full decision record.
```

---

## ✅ Proposed First Commit

### Commit 1.1: Project Setup + Basic Tokenizer

**Will include:**

✅ Project structure
✅ ADR-001: Tokenizer Design Decisions
✅ Comparison with vLLM's tokenizer
✅ Historical note: HuggingFace tokenizers integration
✅ Working code (500 lines)
✅ Unit tests
✅ README with learning guide

**Estimated time:** 2-3 hours of work, lifetime of learning value

---

## 🚀 Next Steps

1. **Study this guide** thoroughly
2. **Set up development environment**:
   - Python 3.10+
   - PyTorch 2.0+
   - CUDA 11.8+ (if using GPU)
3. **Clone vLLM** for reference: `git clone https://github.com/vllm-project/vllm`
4. **Create mini-vLLM repo** with proposed structure
5. **Start Phase 1, Commit 1.1**: Basic tokenizer

---

## 📝 Summary

Building mini-vLLM will teach you:

✅ **LLM Inference Architecture:** End-to-end understanding
✅ **Memory Management:** PagedAttention, KV cache
✅ **Request Scheduling:** Continuous batching
✅ **GPU Programming:** CUDA kernels, optimization
✅ **Distributed Systems:** Tensor parallelism
✅ **Production ML:** Serving, monitoring

**Most importantly:** You'll understand how modern LLM inference systems work under the hood!

Ready to build? Let's start! 🎯
