"""
Simplified unit tests for the Tokenizer module.

These tests verify the code structure without requiring internet access.
For full tests with actual tokenizers, see test_tokenizer.py (requires
pre-downloaded models).
"""

import pytest
from mini_vllm.tokenizer import Tokenizer


class TestTokenizerStructure:
    """Test tokenizer class structure and methods exist."""

    def test_tokenizer_class_exists(self):
        """Test Tokenizer class exists."""
        assert Tokenizer is not None

    def test_tokenizer_invalid_model(self):
        """Test tokenizer raises error for invalid model."""
        with pytest.raises(ValueError, match="Failed to load tokenizer"):
            Tokenizer("this-model-does-not-exist-12345-xyz-abc")

    def test_tokenizer_has_encode_method(self):
        """Test Tokenizer has encode method."""
        assert hasattr(Tokenizer, "encode")

    def test_tokenizer_has_decode_method(self):
        """Test Tokenizer has decode method."""
        assert hasattr(Tokenizer, "decode")

    def test_tokenizer_has_vocab_size_property(self):
        """Test Tokenizer has vocab_size property."""
        assert hasattr(Tokenizer, "vocab_size")

    def test_tokenizer_has_special_token_properties(self):
        """Test Tokenizer has special token properties."""
        assert hasattr(Tokenizer, "bos_token_id")
        assert hasattr(Tokenizer, "eos_token_id")
        assert hasattr(Tokenizer, "pad_token_id")

    def test_tokenizer_has_repr(self):
        """Test Tokenizer has __repr__ method."""
        assert hasattr(Tokenizer, "__repr__")

    def test_tokenizer_has_len(self):
        """Test Tokenizer has __len__ method."""
        assert hasattr(Tokenizer, "__len__")


def test_module_exports():
    """Test module exports the expected symbols."""
    from mini_vllm import tokenizer

    # Check main exports
    assert hasattr(tokenizer, "Tokenizer")
    assert hasattr(tokenizer, "encode")
    assert hasattr(tokenizer, "decode")


def test_encode_helper_function_exists():
    """Test encode helper function exists."""
    from mini_vllm.tokenizer import encode
    assert callable(encode)


def test_decode_helper_function_exists():
    """Test decode helper function exists."""
    from mini_vllm.tokenizer import decode
    assert callable(decode)


# NOTE: Full integration tests in test_tokenizer.py require pre-downloaded
# HuggingFace tokenizers. To run those tests:
# 1. Download a tokenizer: python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')"
# 2. Run: pytest tests/unit/test_tokenizer.py
