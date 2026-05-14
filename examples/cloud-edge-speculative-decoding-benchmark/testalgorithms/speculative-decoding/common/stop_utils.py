"""Shared native-token stop helpers for speculative decoding runtimes."""


def is_stop_token(tokenizer, token_id):
    """Return True when a token should terminate generation."""
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return False
    if isinstance(eos_token_id, (list, tuple, set)):
        return int(token_id) in {int(item) for item in eos_token_id if item is not None}
    return int(token_id) == int(eos_token_id)


def apply_stop_to_sequence(tokenizer, token_ids, default_stop_mode=None, request=None):
    """
    Apply the native-token stop protocol to a finished/generated token list.

    Returns `(effective_ids, stop_reason)` where `effective_ids` are truncated
    to the earliest native stop token.
    """
    del default_stop_mode, request
    effective = []
    for token_id in list(token_ids or []):
        token_id = int(token_id)
        effective.append(token_id)
        if is_stop_token(tokenizer, token_id):
            return effective, "eos"
    return effective, "completion_limit"
