"""Common verify-side abstract interface for speculative decoding algorithms."""

from abc import ABC, abstractmethod

class BaseSpeculativeVerifier(ABC):
    """Minimal verify-side interface shared by all speculative algorithms."""

    algorithm_name = "base"
    role = "verifier"

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    @abstractmethod
    def start_session(self, data=None, request=None, **kwargs):
        """Create one algorithm-specific verify session."""

    @abstractmethod
    def verify(self, session, draft_output=None, draft_ids=None, **kwargs):
        """Verify one draft payload and return the decision schema."""

    @abstractmethod
    def close_session(self, session, request=None):
        """Release one verifier session."""
