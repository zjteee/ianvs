"""Common draft-side abstract interface for speculative decoding algorithms."""

from abc import ABC, abstractmethod

class BaseSpeculativeDrafter(ABC):
    """Minimal draft-side interface shared by all speculative algorithms."""

    algorithm_name = "base"
    role = "drafter"

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    @abstractmethod
    def start_session(self, data=None, request=None, **kwargs):
        """Create one algorithm-specific draft session."""

    @abstractmethod
    def step(self, session, feedback=None, max_draft_tokens=None, **kwargs):
        """Advance one draft round and produce the next proposal."""

    @abstractmethod
    def close_session(self, session, request=None):
        """Release one draft session."""
