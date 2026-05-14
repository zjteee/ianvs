"""Response helpers shared by speculative decoding runtimes."""


def _build_benchmark_payload(request):
    """Extract benchmark-only metadata that should round-trip to metric scripts."""
    if request is None:
        return None

    sample_index = request.get("sample_index")
    warmup_samples = request.get("warmup_samples")
    is_warmup = request.get("is_warmup")
    if sample_index is None and warmup_samples is None and is_warmup is None:
        return None

    return {
        "sample_index": sample_index,
        "warmup_samples": int(warmup_samples or 0),
        "is_warmup": bool(is_warmup),
    }


def compute_paper_perf(total_latency_ms, completion_tokens, ttft_ms, prefill_ms=None):
    """Return paper-style wall-clock and decode-only metrics."""
    total_latency_ms = float(total_latency_ms)
    completion_tokens = int(completion_tokens)
    ttft_ms = float(ttft_ms)
    if prefill_ms is None:
        prefill_ms = ttft_ms
    prefill_ms = float(prefill_ms)
    wall_clock_throughput = 0.0 if total_latency_ms <= 0 else completion_tokens / (total_latency_ms / 1000.0)
    decode_latency_ms = max(total_latency_ms - ttft_ms, 0.0)
    decode_throughput = 0.0 if decode_latency_ms <= 0 else completion_tokens / (decode_latency_ms / 1000.0)
    itl_ms = 0.0 if completion_tokens <= 1 else max(total_latency_ms - ttft_ms, 0.0) / (completion_tokens - 1)
    return {
        "prefill_ms": prefill_ms,
        "ttft_ms": ttft_ms,
        "throughput_toks_per_s": wall_clock_throughput,
        "wall_clock_throughput_toks_per_s": wall_clock_throughput,
        "decode_throughput_toks_per_s": decode_throughput,
        "decode_latency_ms": decode_latency_ms,
        "itl_ms": itl_ms,
        "e2e_latency_ms": total_latency_ms,
    }


def build_perf_payload(perf):
    """Convert paper-style metrics into the Ianvs perf schema."""
    return {
        "time_to_first_token": round(float(perf.get("ttft_ms", 0.0)) / 1000.0, 6),
        "internal_token_latency": round(float(perf.get("itl_ms", 0.0)) / 1000.0, 6),
        "throughput": round(float(perf.get("wall_clock_throughput_toks_per_s", perf.get("throughput_toks_per_s", 0.0))), 6),
        "prefill_ms": round(float(perf.get("prefill_ms", 0.0)), 6),
        "ttft_ms": round(float(perf.get("ttft_ms", 0.0)), 6),
        "throughput_toks_per_s": round(float(perf.get("throughput_toks_per_s", 0.0)), 6),
        "wall_clock_throughput_toks_per_s": round(float(perf.get("wall_clock_throughput_toks_per_s", 0.0)), 6),
        "decode_throughput_toks_per_s": round(float(perf.get("decode_throughput_toks_per_s", 0.0)), 6),
        "decode_latency_ms": round(float(perf.get("decode_latency_ms", 0.0)), 6),
        "itl_ms": round(float(perf.get("itl_ms", 0.0)), 6),
        "e2e_latency_ms": round(float(perf.get("e2e_latency_ms", 0.0)), 6),
    }


def build_specdec_response(
    tokenizer,
    request,
    prompt_token_count,
    completion_ids,
    perf,
    simulation,
    *,
    token_provenance=None,
    round_sequence=None,
    extra_fields=None,
):
    """Build one benchmark-facing response object."""
    completion_text = tokenizer.decode(list(completion_ids or []), skip_special_tokens=True)
    response = {
        "request_id": request.get("request_id"),
        "task_name": request.get("task_name", "default"),
        "completion": completion_text,
        "usage": {
            "prompt_tokens": int(prompt_token_count),
            "completion_tokens": int(len(completion_ids or [])),
            "total_tokens": int(prompt_token_count) + int(len(completion_ids or [])),
        },
        "perf": build_perf_payload(perf),
        "simulation": dict(simulation or {}),
    }
    if token_provenance is not None:
        response["token_provenance"] = list(token_provenance)
    if round_sequence is not None:
        response["round_sequence"] = list(round_sequence)
    benchmark = _build_benchmark_payload(request)
    if benchmark is not None:
        response["benchmark"] = benchmark
    if extra_fields:
        response.update(dict(extra_fields))
    return response
