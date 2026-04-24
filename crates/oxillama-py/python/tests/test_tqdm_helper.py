"""Tests for tqdm_helper — runs without native extension."""
import pytest
from oxillama_py.tqdm_helper import CollectTokens, TqdmProgress


class FakePbar:
    """Minimal tqdm-shaped object for testing."""

    def __init__(self):
        self.count = 0
        self.postfix = ""
        self.refreshed = False

    def update(self, n=1):
        self.count += n

    def set_postfix_str(self, s, refresh=True):
        self.postfix = s

    def refresh(self):
        self.refreshed = True

    def reset(self):
        self.count = 0


def test_tqdm_progress_accumulates_text():
    pbar = FakePbar()
    cb = TqdmProgress(pbar)
    cb("Hello")
    cb(", ")
    cb("world")
    assert cb.text == "Hello, world"
    assert pbar.count == 3


def test_tqdm_progress_flushes_on_newline():
    pbar = FakePbar()
    cb = TqdmProgress(pbar, flush_on_newline=True)
    cb("line1\n")
    assert pbar.refreshed


def test_tqdm_progress_no_flush_on_newline():
    pbar = FakePbar()
    cb = TqdmProgress(pbar, flush_on_newline=False)
    cb("line1\n")
    assert not pbar.refreshed


def test_tqdm_progress_reset():
    pbar = FakePbar()
    cb = TqdmProgress(pbar)
    cb("tok")
    cb.reset()
    assert cb.text == ""
    assert pbar.count == 0


def test_collect_tokens():
    col = CollectTokens()
    col("A")
    col("B")
    col("C")
    assert col.text == "ABC"


def test_collect_tokens_reset():
    col = CollectTokens()
    col("x")
    col.reset()
    assert col.text == ""
