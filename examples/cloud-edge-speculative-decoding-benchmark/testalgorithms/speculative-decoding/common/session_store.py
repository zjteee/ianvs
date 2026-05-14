"""Request-local shared state used by drafter / verifier wrappers."""

_SESSION_STORE = {}


def get(request_id, default=None):
    """Return one shared request state."""
    return _SESSION_STORE.get(str(request_id), default)


def set_value(request_id, value):
    """Store one shared request state."""
    _SESSION_STORE[str(request_id)] = value
    return value


def pop(request_id, default=None):
    """Delete and return one shared request state."""
    return _SESSION_STORE.pop(str(request_id), default)


def clear():
    """Drop all shared request states."""
    _SESSION_STORE.clear()


def get_or_create(request_id, factory):
    """Return one shared request state, creating it lazily when absent."""
    key = str(request_id)
    value = _SESSION_STORE.get(key)
    if value is None:
        value = factory()
        _SESSION_STORE[key] = value
    return value
