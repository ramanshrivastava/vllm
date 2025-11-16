# ADR-001: Use HuggingFace Tokenizers

**Status:** Accepted
**Date:** 2025-11-16
**Commit:** [Phase 1.1]
**vLLM Reference:** `/home/user/vllm/vllm/transformers_utils/tokenizers/`
**Related:** HuggingFace Tokenizers library (2019)

---

## Context

We need a tokenizer to convert text → token IDs and token IDs → text for LLM inference.

### Requirements

- Support popular models (Llama, GPT, etc.)
- Fast encode/decode operations
- Handle special tokens (BOS, EOS, PAD)
- Easy to use and integrate
- Model-agnostic design

### Constraints

- Learning-focused implementation (not production)
- Want to minimize non-inference code
- Should work with multiple model types
- Focus on understanding inference systems, not tokenization internals

---

## Decision

Use **HuggingFace's `AutoTokenizer`** as a thin wrapper.

### Code Example

```python
from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model_name: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._vocab_size = len(self._tokenizer)

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self._tokenizer.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
```

---

## Rationale

### Why This Approach?

1. **Reuse Battle-Tested Code**
   HuggingFace tokenizers are used by millions of developers and are extensively tested across thousands of models.

2. **Universal Compatibility**
   `AutoTokenizer` automatically detects and loads the correct tokenizer for any HuggingFace model, eliminating manual configuration.

3. **Learning Focus**
   Lets us focus on inference system design (scheduling, memory management, execution) rather than NLP preprocessing details.

4. **Performance**
   Rust-based tokenizers (via HF's `tokenizers` library) are extremely fast, often faster than custom Python implementations.

5. **Maintainability**
   Delegates tokenization complexity to a well-maintained external library, reducing our maintenance burden.

### Alternatives Considered

#### Alternative A: Implement Custom Tokenizer from Scratch

**Pros:**
- Full control over implementation
- Deep learning experience with tokenization internals
- No external dependencies

**Cons:**
- 1,000+ lines of complex code
- Need to handle: BPE, WordPiece, SentencePiece, etc.
- Unicode edge cases, normalization, special chars
- Not our learning focus (we want to learn **inference**, not tokenization)
- Reinventing the wheel

**Rejected because:** Tokenization is well-solved and not the innovation we're studying. vLLM itself uses HuggingFace tokenizers.

#### Alternative B: Use SentencePiece Directly

**Pros:**
- Fast C++ implementation
- Used by Llama models
- Lower-level control

**Cons:**
- Only works for SentencePiece models (not GPT-2, BERT, etc.)
- Requires manual vocabulary loading
- Less flexible than AutoTokenizer
- Still not our learning focus

**Rejected because:** Less general than HuggingFace, and still doesn't add to our inference learning objectives.

#### Alternative C: Use tiktoken (OpenAI's Tokenizer)

**Pros:**
- Very fast (Rust-based)
- Used by GPT-3.5/GPT-4
- Simple API

**Cons:**
- Only supports OpenAI models
- Not compatible with most open-source models
- Limited special token handling

**Rejected because:** Too specialized, we want to support multiple model families.

---

## vLLM Comparison

### What vLLM Does

vLLM has sophisticated tokenizer handling in `/home/user/vllm/vllm/transformers_utils/tokenizers/`:

- **Multi-format support:**
  HuggingFace, SentencePiece, Mistral-style, custom tokenizers

- **Detokenization optimizations:**
  Incremental decoding for streaming
  Skip special tokens efficiently
  Handle chat templates

- **Advanced features:**
  System prompts and chat templates
  Multi-modal tokenization (images + text)
  Lora adapter tokenization

- **Code volume:**
  ~1,500 lines across multiple files

**Key files:**
- `vllm/transformers_utils/tokenizers/tokenizer.py` - Main tokenizer base
- `vllm/transformers_utils/tokenizers/mistral.py` - Mistral-specific handling
- `vllm/transformers_utils/tokenizers/chat_templates/` - Chat template processing

### What We're Doing Differently

- **Simple wrapper:** Just encode/decode + special tokens
- **No optimizations:** Focus on correctness, not speed
- **~300 lines:** Much simpler than vLLM's implementation
- **Basic functionality:** No chat templates, no streaming optimizations

**Rationale:**
We're learning **inference system architecture** (scheduling, memory management, execution), not tokenization. We can always add optimizations later if needed.

Our simplification lets us:
1. Get to the interesting parts faster (PagedAttention, scheduling)
2. Keep codebase focused and understandable
3. Maintain clarity about what we're actually building

---

## Historical Evolution

Understanding how tokenization evolved helps contextualize our decision:

### 2017 (Original Transformer)
- Custom vocabularies per model
- Basic word-piece tokenization
- ~500 lines per implementation

### 2018 (BERT)
- WordPiece tokenization standardized
- Still model-specific implementations
- Growing complexity

### 2019 (GPT-2)
- Byte-Pair Encoding (BPE) introduced
- First general-purpose approach
- Still required custom code per model

### 2019 (HuggingFace Tokenizers)
- **Unified tokenizers library**
- Rust-based for performance
- Single API for all models
- This is where standardization happened!

### 2020-2023 (Modern Era)
- SentencePiece becomes popular (Llama)
- Fast tokenizers become default
- Multi-modal tokenization emerges

### 2024 (Current)
- HuggingFace `transformers` is de facto standard
- ~70% of models use HF tokenizers
- tiktoken for OpenAI models

**Our approach:** Use the modern standard (HuggingFace) that emerged as the winner.

---

## Trade-offs

### Benefits

✅ **Rapid Development:** 300 lines vs 1,500+
✅ **Reliability:** Battle-tested library used in production
✅ **Flexibility:** Works with GPT-2, Llama, BERT, etc.
✅ **Learning Focus:** Spend time on inference, not tokenization
✅ **Maintainability:** Updates handled by HuggingFace team
✅ **Documentation:** Extensive docs and examples available

### Limitations

❌ **External Dependency:** Relies on `transformers` library
❌ **Less Control:** Can't optimize tokenization internals
❌ **Learning Opportunity:** Don't learn tokenization deeply
❌ **Binary Size:** Adds heavyweight dependency

**Accepted because:**
Our goal is understanding LLM **serving systems** (scheduler, memory, execution), not NLP preprocessing. This trade-off aligns perfectly with our learning objectives.

---

## Learning Outcomes

After implementing this component, you should understand:

### 1. What Tokenization Does
- Converts text ↔ token IDs (integers)
- Why: Models work with numbers, not text
- One-to-many mapping (text can have multiple valid tokenizations)

### 2. Special Tokens
- **BOS (Beginning of Sequence):** Signals start (e.g., Llama's `<s>`)
- **EOS (End of Sequence):** Signals completion (e.g., `</s>`)
- **PAD (Padding):** Fills sequences to equal length for batching
- **UNK (Unknown):** Handles out-of-vocabulary words

### 3. Vocabulary Concepts
- Fixed vocabulary size (e.g., GPT-2 has 50,257 tokens)
- Subword tokenization (BPE) vs word-level
- Why: Balance between vocab size and token sequence length

### 4. HuggingFace Ecosystem
- `transformers` library structure
- Model Hub for distribution
- `AutoTokenizer` pattern (automatic class selection)

### 5. Integration Points
- Where tokenization fits in inference pipeline:
  ```
  User Input → Tokenizer → Token IDs → Model → Logits → Sample → Token IDs → Detokenizer → Output
  ```

---

## References

### Documentation
- **HuggingFace Tokenizers:** https://huggingface.co/docs/tokenizers
- **Transformers Library:** https://huggingface.co/docs/transformers
- **vLLM tokenizers:** `/home/user/vllm/vllm/transformers_utils/tokenizers/`

### Papers
- **BPE:** Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units", ACL 2016
- **SentencePiece:** Kudo & Richardson, "SentencePiece: A simple and language independent approach to subword tokenization", EMNLP 2018
- **WordPiece:** Schuster & Nakajima, "Japanese and Korean voice search", ICASSP 2012

### Code References
- vLLM tokenizer base: `vllm/transformers_utils/tokenizers/tokenizer.py`
- HF AutoTokenizer: `transformers/models/auto/tokenization_auto.py`

---

## Exercises

### Exercise 1: Explore Tokenization (15 min)

**Task:** Compare tokenization across different models

```python
from mini_vllm.tokenizer import Tokenizer

models = ["gpt2", "meta-llama/Llama-2-7b-hf"]
text = "Hello, world!"

for model_name in models:
    tok = Tokenizer(model_name)
    ids = tok.encode(text)
    print(f"{model_name}: {ids}")
    print(f"  Vocab size: {tok.vocab_size}")
    print(f"  BOS: {tok.bos_token_id}, EOS: {tok.eos_token_id}")
```

**Questions:**
- Why do different models produce different token IDs for the same text?
- Which model has a larger vocabulary? Why might that be?
- Why doesn't GPT-2 have a BOS token?

### Exercise 2: Token Counting (30 min)

**Task:** Add a method to count tokens (for API billing)

```python
def count_tokens(self, text: str) -> int:
    """Count number of tokens in text (useful for billing)."""
    # Your implementation here
    pass
```

**Hint:** Consider whether to include special tokens

### Exercise 3: Debug Tokenization Mismatch (30 min)

**Task:** What happens if you encode with one model and decode with another?

```python
tok1 = Tokenizer("gpt2")
tok2 = Tokenizer("meta-llama/Llama-2-7b-hf")

text = "Hello"
ids = tok1.encode(text)
decoded = tok2.decode(ids)  # Wrong tokenizer!

print(f"Original: {text}")
print(f"Decoded: {decoded}")
```

**Questions:**
- What goes wrong?
- Why?
- How could you detect this error automatically?

### Exercise 4: Performance Analysis (45 min)

**Task:** Profile tokenization speed

```python
import time

tokenizer = Tokenizer("gpt2")
long_text = "Hello world " * 10000

start = time.time()
for _ in range(100):
    ids = tokenizer.encode(long_text)
end = time.time()

print(f"Tokens/sec: {len(ids) * 100 / (end - start):.0f}")
```

**Questions:**
- How fast is tokenization?
- Is it a bottleneck for inference?
- How does HuggingFace's Rust tokenizer compare to pure Python?

---

## Next Steps

**Immediate:**
Phase 1.2 will implement a simple KV cache for storing attention key/value states. This is where we start building the core inference machinery.

**Future Enhancements (if time permits):**
- Add streaming/incremental decoding
- Implement chat template support
- Add batch tokenization optimizations
- Support custom vocabularies

**Related ADRs:**
- ADR-002: Simple KV Cache Design (next)
- ADR-007: PagedAttention Block Allocator (Phase 2)

---

## Approval

**Decision Maker:** Learning Project Team
**Date:** 2025-11-16
**Status:** ✅ Approved for Phase 1.1

**Reviewers:**
- Architecture: Validated against vLLM design
- Implementation: Code complete and tested
- Documentation: ADR approved

---

*This ADR is part of the mini-vLLM learning project. See `MINI_VLLM_LEARNING_GUIDE.md` for overall architecture and `PHASED_IMPLEMENTATION_PLAN.md` for implementation timeline.*
