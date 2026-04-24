"""HuggingFace Hub utilities for oxillama."""

from __future__ import annotations

from typing import Optional

from . import Engine, EngineConfig


def load_from_hub(
    repo_id: str,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    config: Optional[EngineConfig] = None,
) -> Engine:
    """Download a GGUF model from HuggingFace Hub and load it.

    Args:
        repo_id: HuggingFace repository ID, e.g. ``"TheBloke/Llama-2-7B-GGUF"``.
        filename: Specific GGUF file within the repo.  If *None*, the first
            ``*.gguf`` file found is used.
        revision: Git revision / branch / tag.  Defaults to ``"main"``.
        token: HuggingFace access token.  Falls back to ``$HF_TOKEN`` /
            ``$HUGGINGFACE_HUB_TOKEN`` environment variables.
        config: Optional engine configuration.  Uses defaults if *None*.

    Returns:
        A loaded :class:`Engine` ready for inference.
    """
    return Engine.from_hub(
        repo_id,
        filename=filename,
        revision=revision,
        token=token,
        config=config,
    )
