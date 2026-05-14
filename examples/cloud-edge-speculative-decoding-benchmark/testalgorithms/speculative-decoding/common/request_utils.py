"""Request normalization and single-path metric helpers."""

import hashlib
import json


def normalize_request(data, default_prompt_tokens, default_completion_tokens, to_optional_int):
    """
    Normalize one dataset sample into the request schema used by the runtime.

    The benchmark may hand us a dict, a JSON string, or plain text.
    """
    if isinstance(data, dict):
        payload = dict(data)
    elif isinstance(data, str):
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            payload = {"query": data}
    else:
        payload = {"query": str(data)}

    query_field = payload.get("query", "")
    if isinstance(query_field, str):
        stripped = query_field.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                nested = json.loads(stripped)
                if isinstance(nested, dict):
                    payload = {**payload, **nested}
            except json.JSONDecodeError:
                pass

    text = payload.get("query", "")
    if not isinstance(text, str):
        text = str(text)

    request_id = payload.get("request_id")
    if not request_id:
        request_id = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]

    prompt_tokens = to_optional_int(payload.get("prompt_tokens"), default_prompt_tokens)
    completion_tokens = to_optional_int(
        payload.get("completion_tokens", payload.get("max_new_tokens")),
        default_completion_tokens,
    )

    return {
        "request_id": request_id,
        "query": text,
        "gold": str(payload.get("gold", "")),
        "prompt_tokens": max(1, int(prompt_tokens)) if prompt_tokens is not None else None,
        "completion_tokens": max(1, int(completion_tokens)) if completion_tokens is not None else None,
        "task_name": payload.get("task_name", "default"),
        "dataset_name": payload.get("dataset_name"),
        "stop_mode": payload.get("stop_mode"),
        "sample_index": payload.get("sample_index"),
        "warmup_samples": payload.get("warmup_samples", 0),
        "is_warmup": bool(payload.get("is_warmup", False)),
    }


def compute_perf(total_latency_ms, completion_tokens, ttft_ms):
    """Compute TTFT / ITL / throughput from millisecond latency totals."""
    total_latency_ms = max(float(total_latency_ms), 1e-6)
    completion_tokens = max(int(completion_tokens), 1)
    ttft_ms = max(float(ttft_ms), 1e-6)

    if completion_tokens <= 1:
        itl_s = total_latency_ms / 1000.0
    else:
        itl_s = max((total_latency_ms - ttft_ms) / (completion_tokens - 1) / 1000.0, 1e-6)

    throughput = completion_tokens / (total_latency_ms / 1000.0)
    return ttft_ms / 1000.0, itl_s, throughput


def build_single_path_response(
    prompt_tokens,
    completion_tokens,
    completion_text,
    perf,
    simulation,
    timestamps=None,
    benchmark=None,
):
    """Build the single-path response schema expected by Ianvs metrics."""
    response = {
        "completion": completion_text,
        "usage": {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens) + int(completion_tokens),
        },
        "perf": {
            "time_to_first_token": round(perf[0], 6),
            "internal_token_latency": round(perf[1], 6),
            "throughput": round(perf[2], 6),
        },
        "simulation": simulation,
    }
    if timestamps is not None:
        response["timestamps"] = timestamps
    if benchmark is not None:
        response["benchmark"] = dict(benchmark)
    return response
