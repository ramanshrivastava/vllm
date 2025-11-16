# Tokenizer Module

## Overview

The tokenizer module provides a simple wrapper around HuggingFace tokenizers for converting text to token IDs and vice versa.

## Purpose

In LLM inference, tokenization is the first step:
1. **Input text** → Tokenize → **Token IDs** → Feed to model
2. **Output token IDs** ← Sample ← Model logits
3. **Output token IDs** → Detokenize → **Output text**

## Design Decisions

### Why HuggingFace?

We use `transformers.AutoTokenizer` because:
- ✅ **Battle-tested**: Used by millions of users
- ✅ **Compatible**: Works with all HuggingFace models
- ✅ **Fast**: Rust-based tokenizers are optimized
- ✅ **Focus**: Lets us focus on inference, not tokenization

See `docs/adrs/001-tokenizer-choice.md` for full rationale.

### What We Don't Implement

Real vLLM tokenizer (`vllm/transformers_utils/tokenizers/`) has:
- Multiple tokenizer formats (HF, SentencePiece, custom)
- Incremental decoding optimizations
- Chat templates and system prompts
- Multi-modal tokenization support
- ~1,500 lines of code

We keep it simple (~300 lines) to focus on learning inference systems.

## Usage

### Basic Usage

```python
from mini_vllm.tokenizer import Tokenizer

# Load tokenizer
tokenizer = Tokenizer("gpt2")

# Encode text to IDs
token_ids = tokenizer.encode("Hello, world!")
print(token_ids)  # [15496, 11, 995, 0]

# Decode IDs to text
text = tokenizer.decode(token_ids)
print(text)  # "Hello, world!"
```

### Special Tokens

```python
tokenizer = Tokenizer("meta-llama/Llama-2-7b-hf")

# Check special tokens
print(f"BOS token ID: {tokenizer.bos_token_id}")  # 1
print(f"EOS token ID: {tokenizer.eos_token_id}")  # 2
print(f"Vocab size: {tokenizer.vocab_size}")     # 32000
```

### Quick Helpers

```python
from mini_vllm.tokenizer import encode, decode

# Quick encode/decode without creating tokenizer object
ids = encode("Hello!")
text = decode(ids)
```

## Implementation Details

### Class Structure

```python
class Tokenizer:
    def __init__(self, model_name: str)
    def encode(self, text: str) -> List[int]
    def decode(self, token_ids: List[int]) -> str

    # Properties
    @property
    def vocab_size(self) -> int
    @property
    def bos_token_id(self) -> Optional[int]
    @property
    def eos_token_id(self) -> Optional[int]
```

### Key Methods

| Method | Purpose | Example |
|--------|---------|---------|
| `encode()` | Text → Token IDs | `[15496, 11]` |
| `decode()` | Token IDs → Text | `"Hello,"` |
| `vocab_size` | Get vocab size | `50257` |
| `bos_token_id` | Get BOS token | `1` |
| `eos_token_id` | Get EOS token | `2` |

## Testing

Run tests:
```bash
pytest tests/unit/test_tokenizer.py -v
```

See `tests/unit/test_tokenizer.py` for comprehensive test coverage.

## Comparison with vLLM

| Feature | Mini-vLLM | Real vLLM | Why Different? |
|---------|-----------|-----------|----------------|
| Lines of code | ~300 | ~1,500 | Simplified for learning |
| Tokenizer types | HF only | HF, SentencePiece, custom | Focus on core concepts |
| Optimizations | None | Incremental decoding | Learning focus |
| Chat templates | No | Yes | Not core to inference |

## Learning Outcomes

After implementing this module, you should understand:

✅ **Tokenization basics**: Text ↔ Token ID conversion
✅ **Special tokens**: BOS (beginning), EOS (end), PAD (padding)
✅ **Vocabulary**: Fixed mapping from tokens to IDs
✅ **HuggingFace ecosystem**: Standard for model distribution

## Next Steps

Phase 1.2: Implement simple KV cache for storing attention states.

## References

- **HuggingFace Tokenizers**: https://huggingface.co/docs/tokenizers
- **vLLM tokenizers**: `/home/user/vllm/vllm/transformers_utils/tokenizers/`
- **BPE Paper**: Sennrich et al., ACL 2016
