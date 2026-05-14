import time

def now_ns():
    """Return a monotonic timestamp in nanoseconds for local latency tracing."""
    return time.perf_counter_ns()

def offset_ns(base_ns, current_ns):
    """Convert an absolute monotonic timestamp to a request-relative offset."""
    return int(current_ns - base_ns)

def ns_to_ms(offset_ns_value):
    """Convert a relative nanosecond offset to milliseconds."""
    return max(float(offset_ns_value) / 1_000_000.0, 0.0)
