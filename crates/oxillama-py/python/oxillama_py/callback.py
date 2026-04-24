"""Typed StreamingCallback protocol for OxiLLaMa streaming callbacks.

This module provides a runtime-checkable Protocol that type-checkers
(mypy, pyright) can validate against, plus a convenience type alias.
"""

from __future__ import annotations

from typing import Callable

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python 3.7 fallback
    from typing_extensions import Protocol, runtime_checkable  # type: ignore[assignment]


@runtime_checkable
class StreamingCallback(Protocol):
    """Protocol for streaming token callbacks.

    Any callable that accepts ``(token: str, token_id: int, is_final: bool)``
    satisfies this protocol.

    Example::

        def my_callback(token: str, token_id: int, is_final: bool) -> None:
            print(token, end="", flush=True)

        assert isinstance(my_callback, StreamingCallback)  # True
    """

    def __call__(self, token: str, token_id: int, is_final: bool) -> None:
        """Invoked for each decoded token.

        Parameters
        ----------
        token:
            The decoded string for this token (may be multiple bytes).
        token_id:
            The integer vocabulary ID of the token.
        is_final:
            ``True`` only for the last token in the sequence (EOS or
            max-tokens reached).
        """
        ...


#: Convenience alias for a bare callable that matches the same signature.
TokenCallback = Callable[[str, int, bool], None]

__all__ = ["StreamingCallback", "TokenCallback"]
