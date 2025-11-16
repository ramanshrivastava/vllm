"""
Unit tests for the Tokenizer module.

Tests cover:
- Basic encode/decode functionality
- Special token handling
- Vocabulary size
- Edge cases
- Helper functions
"""

import pytest
from mini_vllm.tokenizer import Tokenizer, encode, decode


class TestTokenizerBasics:
    """Test basic tokenizer functionality."""

    def test_tokenizer_initialization(self):
        """Test tokenizer can be initialized."""
        tokenizer = Tokenizer("gpt2")
        assert tokenizer is not None
        assert tokenizer.vocab_size > 0

    def test_tokenizer_invalid_model(self):
        """Test tokenizer raises error for invalid model."""
        with pytest.raises(ValueError, match="Failed to load tokenizer"):
            Tokenizer("this-model-does-not-exist-12345")

    def test_encode_basic(self):
        """Test basic text encoding."""
        tokenizer = Tokenizer("gpt2")
        text = "Hello, world!"
        token_ids = tokenizer.encode(text)

        # Should return list of ints
        assert isinstance(token_ids, list)
        assert all(isinstance(tid, int) for tid in token_ids)
        assert len(token_ids) > 0

    def test_decode_basic(self):
        """Test basic token decoding."""
        tokenizer = Tokenizer("gpt2")
        text = "Hello, world!"

        # Encode then decode should reconstruct text
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)

        assert decoded == text

    def test_encode_decode_roundtrip(self):
        """Test encode-decode roundtrip preserves text."""
        tokenizer = Tokenizer("gpt2")

        test_cases = [
            "Hello",
            "The quick brown fox",
            "Testing 123",
            "Multi-line\ntext",
            "Special chars: !@#$%",
        ]

        for text in test_cases:
            token_ids = tokenizer.encode(text)
            decoded = tokenizer.decode(token_ids)
            assert decoded == text, f"Failed for: {text}"

    def test_decode_single_token(self):
        """Test decoding a single token ID."""
        tokenizer = Tokenizer("gpt2")

        # Encode a word
        token_ids = tokenizer.encode("Hello")
        assert len(token_ids) > 0

        # Decode first token as int (not list)
        decoded = tokenizer.decode(token_ids[0])
        assert isinstance(decoded, str)

    def test_empty_text(self):
        """Test encoding empty text."""
        tokenizer = Tokenizer("gpt2")

        token_ids = tokenizer.encode("")
        # Some tokenizers return empty list, others return special tokens
        assert isinstance(token_ids, list)

    def test_special_tokens_control(self):
        """Test add_special_tokens parameter."""
        tokenizer = Tokenizer("gpt2")
        text = "Hello"

        # With special tokens
        with_special = tokenizer.encode(text, add_special_tokens=True)

        # Without special tokens
        without_special = tokenizer.encode(text, add_special_tokens=False)

        # Both should be lists
        assert isinstance(with_special, list)
        assert isinstance(without_special, list)


class TestSpecialTokens:
    """Test special token handling."""

    def test_gpt2_special_tokens(self):
        """Test GPT-2 special tokens."""
        tokenizer = Tokenizer("gpt2")

        # GPT-2 doesn't have BOS token
        assert tokenizer.bos_token_id is None

        # GPT-2 has EOS token
        assert tokenizer.eos_token_id is not None
        assert isinstance(tokenizer.eos_token_id, int)

        # Check vocab size
        assert tokenizer.vocab_size == 50257

    def test_special_token_properties(self):
        """Test all special token properties exist."""
        tokenizer = Tokenizer("gpt2")

        # All properties should be accessible (may be None)
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id

        # Types should be int or None
        assert bos is None or isinstance(bos, int)
        assert eos is None or isinstance(eos, int)
        assert pad is None or isinstance(pad, int)

    def test_special_token_strings(self):
        """Test special token string properties."""
        tokenizer = Tokenizer("gpt2")

        # Should have string properties
        bos_str = tokenizer.bos_token
        eos_str = tokenizer.eos_token
        pad_str = tokenizer.pad_token

        # Types should be str or None
        assert bos_str is None or isinstance(bos_str, str)
        assert eos_str is None or isinstance(eos_str, str)
        assert pad_str is None or isinstance(pad_str, str)


class TestVocabulary:
    """Test vocabulary-related functionality."""

    def test_vocab_size(self):
        """Test vocabulary size property."""
        tokenizer = Tokenizer("gpt2")

        assert tokenizer.vocab_size == 50257
        assert len(tokenizer) == 50257  # __len__ method

    def test_get_vocab(self):
        """Test getting full vocabulary."""
        tokenizer = Tokenizer("gpt2")

        vocab = tokenizer.get_vocab()

        # Should be a dict
        assert isinstance(vocab, dict)

        # Should have correct size
        assert len(vocab) == tokenizer.vocab_size

        # Values should be ints (token IDs)
        sample_value = next(iter(vocab.values()))
        assert isinstance(sample_value, int)

    def test_convert_ids_to_tokens(self):
        """Test converting IDs to token strings."""
        tokenizer = Tokenizer("gpt2")

        # Encode text
        text = "Hello"
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        # Convert to token strings
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)

        assert isinstance(token_strings, list)
        assert len(token_strings) == len(token_ids)
        assert all(isinstance(t, str) for t in token_strings)

    def test_convert_tokens_to_string(self):
        """Test converting token strings to text."""
        tokenizer = Tokenizer("gpt2")

        # Get token strings
        text = "Hello"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)

        # Convert back to string
        reconstructed = tokenizer.convert_tokens_to_string(token_strings)

        assert isinstance(reconstructed, str)
        # Should match original (may have whitespace differences)
        assert text.strip() == reconstructed.strip()


class TestHelperFunctions:
    """Test helper functions."""

    def test_encode_helper(self):
        """Test encode() helper function."""
        text = "Hello"
        token_ids = encode(text)

        assert isinstance(token_ids, list)
        assert all(isinstance(tid, int) for tid in token_ids)

    def test_decode_helper(self):
        """Test decode() helper function."""
        text = "Hello"
        token_ids = encode(text)
        decoded = decode(token_ids)

        assert decoded == text

    def test_encode_decode_helper_roundtrip(self):
        """Test encode/decode helpers work together."""
        original = "The quick brown fox"

        # Use helpers
        token_ids = encode(original)
        reconstructed = decode(token_ids)

        assert reconstructed == original


class TestRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        tokenizer = Tokenizer("gpt2")

        repr_str = repr(tokenizer)

        # Should be a string
        assert isinstance(repr_str, str)

        # Should contain key information
        assert "Tokenizer" in repr_str
        assert "vocab_size" in repr_str
        assert str(tokenizer.vocab_size) in repr_str


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_long_text(self):
        """Test encoding very long text."""
        tokenizer = Tokenizer("gpt2")

        # Create long text
        long_text = "Hello " * 1000

        token_ids = tokenizer.encode(long_text)

        # Should handle without error
        assert isinstance(token_ids, list)
        assert len(token_ids) > 1000

    def test_unicode_text(self):
        """Test encoding unicode text."""
        tokenizer = Tokenizer("gpt2")

        unicode_text = "Hello 世界 🌍"

        token_ids = tokenizer.encode(unicode_text)
        decoded = tokenizer.decode(token_ids)

        # Should handle unicode
        assert isinstance(token_ids, list)
        assert isinstance(decoded, str)

    def test_decode_skip_special_tokens(self):
        """Test skip_special_tokens parameter."""
        tokenizer = Tokenizer("gpt2")

        text = "Hello"
        token_ids = tokenizer.encode(text, add_special_tokens=True)

        # Decode with special tokens
        with_special = tokenizer.decode(token_ids, skip_special_tokens=False)

        # Decode without special tokens
        without_special = tokenizer.decode(token_ids, skip_special_tokens=True)

        # Both should be strings
        assert isinstance(with_special, str)
        assert isinstance(without_special, str)


class TestMultipleModels:
    """Test with different tokenizer models."""

    @pytest.mark.parametrize("model_name,expected_vocab_size", [
        ("gpt2", 50257),
        ("gpt2-medium", 50257),
    ])
    def test_different_models(self, model_name, expected_vocab_size):
        """Test loading different model tokenizers."""
        tokenizer = Tokenizer(model_name)

        assert tokenizer.vocab_size == expected_vocab_size

        # Test basic functionality
        text = "Hello"
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)

        assert decoded == text


# Integration test
class TestTokenizerIntegration:
    """Integration tests for tokenizer."""

    def test_typical_usage_workflow(self):
        """Test typical usage workflow."""
        # 1. Initialize tokenizer
        tokenizer = Tokenizer("gpt2")

        # 2. Encode prompt
        prompt = "Once upon a time"
        prompt_ids = tokenizer.encode(prompt)

        assert len(prompt_ids) > 0

        # 3. Simulate adding generated tokens
        generated_ids = prompt_ids + [123, 456, 789]

        # 4. Decode to get full text
        full_text = tokenizer.decode(generated_ids)

        assert isinstance(full_text, str)
        assert full_text.startswith(prompt)

    def test_batch_processing_simulation(self):
        """Test simulating batch processing."""
        tokenizer = Tokenizer("gpt2")

        prompts = [
            "Hello",
            "The quick brown fox",
            "Testing 123",
        ]

        # Encode all prompts
        batch_ids = [tokenizer.encode(p) for p in prompts]

        assert len(batch_ids) == len(prompts)

        # Decode all
        decoded = [tokenizer.decode(ids) for ids in batch_ids]

        assert decoded == prompts
