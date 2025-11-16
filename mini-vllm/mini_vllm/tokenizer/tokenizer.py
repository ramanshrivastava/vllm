"""
Simple tokenizer wrapper around HuggingFace tokenizers.

This is a thin wrapper that provides basic encode/decode functionality.
In real vLLM, this has ~1500 lines handling edge cases, special tokens,
and optimizations. We keep it simple for learning.

Reference: /home/user/vllm/vllm/transformers_utils/tokenizers/
"""

from typing import List, Optional, Union
from transformers import AutoTokenizer, PreTrainedTokenizer


class Tokenizer:
    """
    Tokenizer for encoding/decoding text.

    This is a thin wrapper around HuggingFace tokenizers that provides
    the essential functionality needed for LLM inference:
    - Encode text to token IDs
    - Decode token IDs to text
    - Access to special tokens (BOS, EOS, PAD)

    In production vLLM, the tokenizer handles:
    - Multiple tokenizer formats (HF, SentencePiece, custom)
    - Incremental decoding optimizations
    - Chat templates and system prompts
    - Multi-modal tokenization

    We focus on the core functionality to keep learning focused on
    the inference engine, not tokenization details.

    Example:
        >>> tokenizer = Tokenizer("gpt2")
        >>> token_ids = tokenizer.encode("Hello, world!")
        >>> text = tokenizer.decode(token_ids)
        >>> print(text)
        "Hello, world!"
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
    ):
        """
        Initialize tokenizer from HuggingFace model.

        Args:
            model_name: HuggingFace model name or path
                       (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
            trust_remote_code: Whether to trust remote code when loading
                              custom tokenizers

        Raises:
            ValueError: If tokenizer cannot be loaded
        """
        try:
            self._tokenizer: PreTrainedTokenizer = (
                AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=trust_remote_code,
                )
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load tokenizer for '{model_name}': {e}"
            ) from e

        # Cache vocabulary size for quick access
        self._vocab_size = len(self._tokenizer)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add special tokens like BOS/EOS

        Returns:
            List of token IDs

        Example:
            >>> tokenizer = Tokenizer("gpt2")
            >>> tokenizer.encode("Hello")
            [15496]
        """
        return self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )

    def decode(
        self,
        token_ids: Union[List[int], int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Single token ID or list of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string

        Example:
            >>> tokenizer = Tokenizer("gpt2")
            >>> tokenizer.decode([15496])
            "Hello"
        """
        # Handle single token ID
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert tokens (strings) to a single string.

        This is useful for incremental decoding where we have
        token strings rather than token IDs.

        Args:
            tokens: List of token strings

        Returns:
            Decoded string

        Example:
            >>> tokenizer = Tokenizer("gpt2")
            >>> tokenizer.convert_tokens_to_string(["Hello", ",", " world"])
            "Hello, world"
        """
        return self._tokenizer.convert_tokens_to_string(tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert token IDs to token strings.

        Args:
            ids: List of token IDs

        Returns:
            List of token strings
        """
        return self._tokenizer.convert_ids_to_tokens(ids)

    @property
    def vocab_size(self) -> int:
        """
        Get vocabulary size.

        Returns:
            Number of tokens in vocabulary
        """
        return self._vocab_size

    @property
    def bos_token_id(self) -> Optional[int]:
        """
        Get beginning-of-sequence token ID.

        Returns:
            BOS token ID, or None if not defined

        Note:
            Not all tokenizers have a BOS token (e.g., GPT-2 doesn't)
        """
        return self._tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        Get end-of-sequence token ID.

        Returns:
            EOS token ID, or None if not defined
        """
        return self._tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> Optional[int]:
        """
        Get padding token ID.

        Returns:
            PAD token ID, or None if not defined

        Note:
            Some tokenizers use EOS as PAD token
        """
        return self._tokenizer.pad_token_id

    @property
    def bos_token(self) -> Optional[str]:
        """Get beginning-of-sequence token string."""
        return self._tokenizer.bos_token

    @property
    def eos_token(self) -> Optional[str]:
        """Get end-of-sequence token string."""
        return self._tokenizer.eos_token

    @property
    def pad_token(self) -> Optional[str]:
        """Get padding token string."""
        return self._tokenizer.pad_token

    def get_vocab(self) -> dict:
        """
        Get the full vocabulary mapping.

        Returns:
            Dictionary mapping token strings to token IDs
        """
        return self._tokenizer.get_vocab()

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self._vocab_size

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Tokenizer("
            f"vocab_size={self.vocab_size}, "
            f"bos_token_id={self.bos_token_id}, "
            f"eos_token_id={self.eos_token_id}"
            f")"
        )


# Helper function for quick tokenization
def encode(text: str, model_name: str = "gpt2") -> List[int]:
    """
    Quick helper to encode text with a default tokenizer.

    Args:
        text: Text to encode
        model_name: Model to use for tokenizer

    Returns:
        List of token IDs
    """
    tokenizer = Tokenizer(model_name)
    return tokenizer.encode(text)


def decode(token_ids: List[int], model_name: str = "gpt2") -> str:
    """
    Quick helper to decode token IDs with a default tokenizer.

    Args:
        token_ids: Token IDs to decode
        model_name: Model to use for tokenizer

    Returns:
        Decoded text
    """
    tokenizer = Tokenizer(model_name)
    return tokenizer.decode(token_ids)
