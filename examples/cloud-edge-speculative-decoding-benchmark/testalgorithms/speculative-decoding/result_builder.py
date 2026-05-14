import json
import os
import time

from common.timeline_utils import ns_to_ms

def _resolve_output_log_path(configured_path, default_filename):
    """
    Resolve the output path for the sample-output log.

    Parameters
    ----------
    configured_path : str | None
        Optional path from configuration.
    default_filename : str
        Default filename under the result directory.

    Returns
    -------
    str
        Output file path.
    """
    result_root = os.environ.get("RESULT_SAVED_URL", ".")
    if configured_path:
        expanded_path = os.path.expanduser(str(configured_path))
        if os.path.isabs(expanded_path):
            return expanded_path
        return os.path.join(result_root, expanded_path)
    return os.path.join(result_root, default_filename)

def record_sample_output(config_source, payload):
    """
    Append one sample-output record to the configured JSONL log.

    Parameters
    ----------
    config_source : object
        Object that may expose `sample_output_log`.
    payload : dict
        Record to append.

    Returns
    -------
    None
    """
    output_log = _resolve_output_log_path(
        getattr(config_source, "sample_output_log", None),
        "specdec_sample_outputs.jsonl",
    )
    try:
        os.makedirs(os.path.dirname(output_log) or ".", exist_ok=True)
        with open(output_log, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass

class SpeculativeDecodingResultBuilder:
    """Build benchmark-facing results for speculative decoding."""

    def __init__(self, draft_runtime, **config):
        """
        Initialize the result builder.

        Parameters
        ----------
        draft_runtime : object
            Runtime used to decode tokens.
        **config
            Builder configuration.

        Returns
        -------
        None
        """
        self.draft_runtime = draft_runtime
        self.sample_output_log = config.get("sample_output_log") or getattr(
            draft_runtime,
            "sample_output_log",
            None,
        )

    @staticmethod
    def _compute_perf_from_timestamps(timestamps, completion_tokens):
        """
        Compute performance metrics from request timestamps.

        Parameters
        ----------
        timestamps : dict
            Request timing data.
        completion_tokens : int
            Number of generated completion tokens.

        Returns
        -------
        dict
            Performance metrics used in benchmark output.
        """
        request_end_ns = int((timestamps or {}).get("request_end_ns", 0))
        first_token_ns = (timestamps or {}).get("first_token_ns", request_end_ns)
        total_latency_ms = ns_to_ms(request_end_ns)
        ttft_ms = ns_to_ms(first_token_ns if first_token_ns is not None else request_end_ns)
        completion_tokens = max(int(completion_tokens), 1)

        if completion_tokens <= 1:
            itl_s = max(total_latency_ms / 1000.0, 1e-9)
        else:
            decode_ms = max(total_latency_ms - ttft_ms, 0.0)
            itl_s = max(decode_ms / (completion_tokens - 1) / 1000.0, 1e-9)

        throughput = completion_tokens / max(total_latency_ms / 1000.0, 1e-9)
        return {
            "time_to_first_token": round(ttft_ms / 1000.0, 6),
            "internal_token_latency": round(itl_s, 6),
            "throughput": round(throughput, 6),
            "end_to_end_latency": round(total_latency_ms / 1000.0, 6),
        }

    @staticmethod
    def _resolve_final_stop_reason(trace):
        """
        Return the last non-empty stop reason in the round trace.

        Parameters
        ----------
        trace : list[dict]
            Round trace records.

        Returns
        -------
        str
            Final stop reason, or an empty string.
        """
        for item in reversed(list(trace or [])):
            reason = str(item.get("stop_reason", "") or "").strip()
            if reason:
                return reason
        return ""

    @staticmethod
    def _compute_effective_total_draft_tokens(trace, final_completion_tokens, fallback_total_draft):
        """
        Compute the effective drafted-token count for acceptance rate.

        Parameters
        ----------
        trace : list[dict]
            Round trace records.
        final_completion_tokens : int
            Final committed completion length.
        fallback_total_draft : int
            Fallback drafted-token count when trace is unavailable.

        Returns
        -------
        int
            Effective drafted-token count.
        """
        if not trace:
            return int(fallback_total_draft)
        completion_limit = max(int(final_completion_tokens or 0), 0)
        committed_before_round = 0
        effective_total_draft = 0
        for item in trace:
            draft_count = int(item.get("draft_count", 0) or 0)
            committed_count = int(item.get("committed_count", 0) or 0)
            remaining_budget = max(completion_limit - committed_before_round, 0)
            effective_total_draft += min(draft_count, remaining_budget)
            committed_before_round += committed_count
        return int(effective_total_draft)

    @staticmethod
    def _build_round_sequence(trace):
        """
        Build the compact per-round sequence used in reports.

        Parameters
        ----------
        trace : list[dict]
            Round trace records.

        Returns
        -------
        list[dict]
            Simplified round sequence.
        """
        sequence = []
        for item in trace or []:
            sequence.append(
                {
                    "round": int(item.get("round", 0) or 0),
                    "draft_count": int(item.get("draft_count", 0) or 0),
                    "accepted_length": int(item.get("accepted_length", item.get("accepted_count", 0)) or 0),
                    "corrected_count": int(item.get("corrected_count", 0) or 0),
                    "rejected_draft_count": int(item.get("rejected_draft_count", 0) or 0),
                    "stop_reason": str(item.get("stop_reason", "") or ""),
                    "rejected_draft_tokens": list(item.get("rejected_draft_tokens", []) or []),
                }
            )
        return sequence

    def _build_token_provenance(self, committed_token_trace):
        """
        Build token-level provenance records for committed tokens.

        Parameters
        ----------
        committed_token_trace : list[dict]
            Token provenance trace.

        Returns
        -------
        list[dict]
            Provenance records with decoded token text.
        """
        token_provenance = []
        for position, item in enumerate(list(committed_token_trace or [])):
            token_id = int(item.get("token_id", 0))
            token_provenance.append(
                {
                    "position": int(position),
                    "round": int(item.get("round", 0) or 0),
                    "token_id": token_id,
                    "token_text": self.draft_runtime.decode_tokens([token_id]),
                    "source": str(item.get("source", "unknown")),
                }
            )
        return token_provenance

    def build_collaboration_result(self, request, state):
        """
        Build the final benchmark result for collaboration mode.

        Parameters
        ----------
        request : dict
            Normalized request.
        state : dict
            Drafter runtime state for the request.

        Returns
        -------
        dict
            Benchmark-facing collaboration result.
        """
        trace = state.get("trace", [])
        timestamps = dict(state.get("timestamps", {}) or {})
        committed_ids = state["committed_ids"]
        final_completion_tokens = int(len(committed_ids))
        perf = self._compute_perf_from_timestamps(
            timestamps=timestamps,
            completion_tokens=final_completion_tokens,
        )

        accepted_draft_tokens = int(state.get("accepted_draft_tokens", 0))
        raw_total_draft_tokens = int(len(state.get("drafted_ids", [])))
        total_draft_tokens = self._compute_effective_total_draft_tokens(
            trace=trace,
            final_completion_tokens=final_completion_tokens,
            fallback_total_draft=raw_total_draft_tokens,
        )
        acceptance_rate = (
            accepted_draft_tokens / max(total_draft_tokens, 1)
            if total_draft_tokens > 0
            else None
        )

        response = {
            "completion": self.draft_runtime.decode_tokens(committed_ids),
            "usage": {
                "prompt_tokens": int(state["prompt_token_count"]),
                "completion_tokens": int(final_completion_tokens),
                "total_tokens": int(state["prompt_token_count"]) + int(final_completion_tokens),
            },
            "perf": {
                "time_to_first_token": perf["time_to_first_token"],
                "internal_token_latency": perf["internal_token_latency"],
                "throughput": perf["throughput"],
            },
            "simulation": {
                "mode": "token-level-speculative-decoding",
                "routed_to": "collaboration",
                "acceptance_rate": (
                    round(max(0.0, min(1.0, acceptance_rate)), 6)
                    if acceptance_rate is not None
                    else ""
                ),
                "end_to_end_latency": perf["end_to_end_latency"],
                "rounds": int(state.get("rounds", 0)),
                "accepted_draft_tokens": accepted_draft_tokens,
                "corrected_tokens": int(len(state.get("corrected_ids", []))),
                "total_draft_tokens": int(total_draft_tokens),
                "network_overhead_ms": round(float(state.get("network_overhead_ms", 0.0)), 6),
                "network_rtt_ms": round(float(state.get("network_overhead_ms", 0.0)), 6),
                "network_jitter_ms": 0.0,
                "draft_tokens_per_step": int(getattr(self.draft_runtime, "draft_tokens_per_step", 1)),
                "task_name": request.get("task_name", "default"),
                "stop_reason": self._resolve_final_stop_reason(trace),
            },
            "timestamps": timestamps,
        }

        record_sample_output(
            self.draft_runtime,
            {
                "record_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "request_id": request.get("request_id"),
                "mode": "collaboration",
                "task_name": request.get("task_name", "default"),
                "prompt": request.get("query", ""),
                "gold": request.get("gold", ""),
                "completion": response["completion"],
                "stop_reason": response["simulation"]["stop_reason"],
                "token_provenance": self._build_token_provenance(
                    state.get("committed_token_trace", [])
                ),
                "accepted_draft_tokens": accepted_draft_tokens,
                "corrected_tokens": int(len(state.get("corrected_ids", []))),
                "total_draft_tokens": int(total_draft_tokens),
                "raw_total_draft_tokens": int(raw_total_draft_tokens),
                "rounds": int(state.get("rounds", 0)),
                "round_sequence": self._build_round_sequence(trace),
            },
        )
        return response

    def build_single_path_result(
        self,
        *,
        request,
        prompt_token_count,
        completion_ids,
        routed_to,
        timestamps,
        extra_simulation=None,
    ):
        """
        Build the final benchmark result for edge-only or cloud-only mode.

        Parameters
        ----------
        request : dict
            Normalized request.
        prompt_token_count : int
            Prompt length in tokens.
        completion_ids : list[int]
            Generated completion token ids.
        routed_to : str
            Execution route label.
        timestamps : dict
            Request timing data.
        extra_simulation : dict | None
            Extra simulation fields to merge into the result.

        Returns
        -------
        dict
            Benchmark-facing single-path result.
        """
        extra_simulation = extra_simulation or {}
        final_completion = self.draft_runtime.decode_tokens(completion_ids)
        final_completion_tokens = int(len(completion_ids))
        perf = self._compute_perf_from_timestamps(
            timestamps=timestamps,
            completion_tokens=final_completion_tokens,
        )
        response = {
            "completion": final_completion,
            "usage": {
                "prompt_tokens": int(prompt_token_count),
                "completion_tokens": int(final_completion_tokens),
                "total_tokens": int(prompt_token_count) + int(final_completion_tokens),
            },
            "perf": {
                "time_to_first_token": perf["time_to_first_token"],
                "internal_token_latency": perf["internal_token_latency"],
                "throughput": perf["throughput"],
            },
            "simulation": {
                "mode": routed_to,
                "routed_to": "edge" if routed_to == "edge-only" else "cloud",
                "acceptance_rate": "",
                "end_to_end_latency": perf["end_to_end_latency"],
                "rounds": 1,
                "accepted_draft_tokens": len(completion_ids) if routed_to == "edge-only" else 0,
                "corrected_tokens": 0,
                "total_draft_tokens": len(completion_ids) if routed_to == "edge-only" else 0,
                "network_overhead_ms": 0.0,
                "network_rtt_ms": 0.0,
                "network_jitter_ms": 0.0,
                "draft_tokens_per_step": int(getattr(self.draft_runtime, "draft_tokens_per_step", 1)),
                "task_name": request.get("task_name", "default"),
                **extra_simulation,
            },
            "timestamps": timestamps,
        }
        record_sample_output(
            self.draft_runtime,
            {
                "record_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "request_id": request.get("request_id"),
                "mode": routed_to,
                "task_name": request.get("task_name", "default"),
                "prompt": request.get("query", ""),
                "gold": request.get("gold", ""),
                "completion": response["completion"],
                "stop_reason": str(response["simulation"].get("stop_reason", "") or ""),
            },
        )
        return response
