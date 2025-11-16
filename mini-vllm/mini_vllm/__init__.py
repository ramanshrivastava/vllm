"""
Mini-vLLM: A Learning-Focused LLM Inference Engine

A simplified implementation of vLLM for understanding LLM serving systems.

Key Components:
- Tokenizer: Text ↔ Token conversion
- Scheduler: Request management and batching
- KV Cache: PagedAttention memory management
- Model Executor: Transformer model execution
- Sampler: Token sampling strategies
- Engine: Orchestration layer
"""

__version__ = "0.1.0"

# Will be populated as we implement components
__all__ = []
