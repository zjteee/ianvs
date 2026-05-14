"""Paper-aligned AR verifier wrapper for the Ianvs benchmark example."""

import os
import random
import sys
import time

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPECDEC_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
for path in (MODULE_DIR, SPECDEC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
from sedna.common.class_factory import ClassFactory, ClassType

from base_verifier import BaseSpeculativeVerifier
from common.config_utils import _to_bool, _to_int, _to_optional_int
from common.request_utils import normalize_request
from common.response_utils import build_specdec_response, compute_paper_perf
from common.session_store import get as get_shared_session
from common.timing_utils import now
from result_builder import record_sample_output
from algorithms.ar.core import PaperAlignedARVerifier

os.environ["BACKEND_TYPE"] = "TORCH"

TOKEN_ID_BYTES = 4


def _resolve_stop_on_eos(kwargs):
    """Resolve the native-stop behavior from Ianvs profile fields."""
    if "stop_on_eos" in kwargs:
        return _to_bool(kwargs.get("stop_on_eos"), True)
    stop_mode = str(kwargs.get("stop_mode", "choice") or "choice").strip().lower()
    return stop_mode not in {"none", "disabled", "off", "false"}


def _build_cloud_simulation(
    request,
    perf,
    draft_tokens_per_step,
    stop_reason,
    network_ms,
    network_rtt_ms,
    network_jitter_ms,
):
    """Build cloud-only simulation metadata."""
    return {
        "mode": "cloud-only",
        "routed_to": "cloud",
        "acceptance_rate": "",
        "end_to_end_latency": round(float(perf.get("e2e_latency_ms", 0.0)) / 1000.0, 6),
        "rounds": 1,
        "accepted_draft_tokens": 0,
        "corrected_tokens": 0,
        "total_draft_tokens": 0,
        "network_overhead_ms": round(float(network_ms), 6),
        "network_rtt_ms": round(float(network_rtt_ms), 6),
        "network_jitter_ms": round(float(network_jitter_ms), 6),
        "draft_tokens_per_step": int(draft_tokens_per_step),
        "task_name": request.get("task_name", "default"),
        "stop_reason": stop_reason,
    }


@ClassFactory.register(ClassType.GENERAL, alias="SpeculativeVerifyModel")
class SpeculativeVerifyModel(BaseSpeculativeVerifier):
    """Ianvs verifier wrapper backed by the paper AR implementation."""

    algorithm_name = "ar_spec"
    role = "verifier"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = dict(kwargs)
        self.model_name = kwargs.get("model", "Qwen/Qwen2.5-7B-Instruct")
        configured_device = kwargs.get("device", "auto")
        if configured_device == "auto":
            configured_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = configured_device
        self.trust_remote_code = _to_bool(kwargs.get("trust_remote_code", True), True)
        self.default_prompt_tokens = _to_optional_int(kwargs.get("prompt_tokens"))
        self.default_completion_tokens = _to_optional_int(kwargs.get("max_new_tokens"), 64)
        self.draft_tokens_per_step = max(1, _to_int(kwargs.get("draft_tokens_per_step"), 8))
        self.draft_top_k = max(0, _to_int(kwargs.get("draft_top_k"), 0))
        self.temperature = float(kwargs.get("sample_temperature", kwargs.get("temperature", 0.0)) or 0.0)
        self.stop_on_eos = _resolve_stop_on_eos(kwargs)
        self.enable_network_sleep = _to_bool(kwargs.get("enable_network_sleep", False), False)
        self.network_rtt_ms = max(0.0, float(kwargs.get("network_rtt_ms", 0.0) or 0.0))
        self.network_jitter_ms = max(0.0, float(kwargs.get("network_jitter_ms", 0.0) or 0.0))
        self.network_uplink_ratio = min(1.0, max(0.0, float(kwargs.get("network_uplink_ratio", 0.5) or 0.5)))
        self.network_uplink_bandwidth_mbps = max(0.0, float(kwargs.get("network_uplink_bandwidth_mbps", 0.0) or 0.0))
        self.network_downlink_bandwidth_mbps = max(0.0, float(kwargs.get("network_downlink_bandwidth_mbps", 0.0) or 0.0))
        self.network_seed = kwargs.get("network_seed", 42)
        self.network_rng = random.Random(int(self.network_seed))
        self.sample_output_log = kwargs.get("sample_output_log")
        self.core = PaperAlignedARVerifier(
            self.model_name,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
        )
        self.tokenizer = None
        self.model = None

    def load(self, *args, **kwargs):
        """Load the verifier model."""
        del args, kwargs
        self.core.load()
        self.tokenizer = self.core.tokenizer
        self.model = self.core.model

    def cleanup(self):
        """Release loaded resources."""
        self.model = None
        self.tokenizer = None
        self.core.model = None
        self.core.tokenizer = None

    def _resolve_completion_limit(self, request):
        """Resolve the per-request completion budget."""
        limit = request.get("completion_tokens")
        if limit is None:
            limit = self.default_completion_tokens
        return max(int(limit or 1), 1)

    def _bandwidth_delay_ms(self, num_bytes, bandwidth_mbps):
        """Estimate link transfer time from bytes and bandwidth."""
        if bandwidth_mbps <= 0.0 or num_bytes <= 0:
            return 0.0
        return (float(num_bytes) * 8.0) / (float(bandwidth_mbps) * 1_000_000.0) * 1000.0

    def _sample_base_delays(self):
        """Sample one uplink/downlink propagation split from RTT."""
        ratio = self.network_uplink_ratio
        jitter = 0.0 if self.network_jitter_ms <= 0.0 else self.network_rng.uniform(-self.network_jitter_ms, self.network_jitter_ms)
        total = max(self.network_rtt_ms + jitter, 0.0)
        return total * ratio, total * (1.0 - ratio)

    def _simulate_network(self, uplink_bytes, downlink_bytes):
        """Simulate one collaboration or cloud-only payload exchange."""
        up_base_ms, down_base_ms = self._sample_base_delays()
        uplink_ms = up_base_ms + self._bandwidth_delay_ms(uplink_bytes, self.network_uplink_bandwidth_mbps)
        downlink_ms = down_base_ms + self._bandwidth_delay_ms(downlink_bytes, self.network_downlink_bandwidth_mbps)
        if self.enable_network_sleep and uplink_ms > 0.0:
            time.sleep(uplink_ms / 1000.0)
        if self.enable_network_sleep and downlink_ms > 0.0:
            time.sleep(downlink_ms / 1000.0)
        return {
            "uplink_bytes": int(uplink_bytes),
            "downlink_bytes": int(downlink_bytes),
            "uplink_ms": uplink_ms,
            "downlink_ms": downlink_ms,
            "network_ms": uplink_ms + downlink_ms,
        }

    def _estimate_draft_distribution_bytes(self, draft_logits):
        """Estimate serialized bytes of transported drafter distributions."""
        total = 0
        for item in list(draft_logits or []):
            if isinstance(item, dict) and item.get("representation") == "topk_probs":
                total += int(item["token_ids"].numel()) * int(item["token_ids"].element_size())
                total += int(item["token_probs"].numel()) * int(item["token_probs"].element_size())
            else:
                total += int(item.numel()) * int(item.element_size())
        return total

    def start_session(self, data=None, request=None, draft_session=None, **kwargs):
        """Create one verify-side collaboration session."""
        del kwargs
        if request is None:
            request = draft_session.get("request") if isinstance(draft_session, dict) else None
        if request is None:
            request = normalize_request(
                data,
                self.default_prompt_tokens,
                self.default_completion_tokens,
                _to_optional_int,
            )
        request = dict(request)
        request["completion_tokens"] = self._resolve_completion_limit(request)
        if draft_session is not None:
            prompt_ids = draft_session["prompt_ids"].to(self.device)
            prompt_token_count = int(draft_session["prompt_token_count"])
        else:
            encoded = self.tokenizer(request["query"], return_tensors="pt")
            prompt_ids = encoded.input_ids.to(self.device)
            prompt_token_count = int(prompt_ids.shape[1])
        verify_session = self.core.start_session(prompt_ids)
        shared_state = get_shared_session(request["request_id"])
        if shared_state is None:
            raise RuntimeError(f"Shared drafter state is missing for request {request['request_id']}.")
        shared_state["verify_prefill_ms"] = float(verify_session.prefill_ms)
        shared_state["prefill_ms"] = float(shared_state.get("draft_prefill_ms", 0.0)) + float(verify_session.prefill_ms)
        shared_state["timing_device"] = self.device
        shared_state["decode_tokenizer"] = self.tokenizer
        return {
            "request_id": str(request.get("request_id", "default")),
            "request": request,
            "prompt_ids": prompt_ids,
            "prompt_token_count": prompt_token_count,
            "_core_session": verify_session,
            "_shared_state": shared_state,
        }

    def verify(self, session, draft_output=None, draft_ids=None, **kwargs):
        """Verify one draft payload and update collaboration-shared statistics."""
        del kwargs
        draft_output = dict(draft_output or {})
        if draft_ids is None:
            draft_ids = list(draft_output.get("draft_ids", []) or [])
        else:
            draft_ids = list(draft_ids or [])
        shared_state = session["_shared_state"]
        verify_payload = self.core.verify(
            session["_core_session"],
            draft_ids,
            draft_output.get("draft_logits", []),
            self.temperature,
            shared_state["completion_limit"],
            stop_on_eos=self.stop_on_eos,
        )
        uplink_bytes = len(draft_ids) * TOKEN_ID_BYTES + self._estimate_draft_distribution_bytes(
            draft_output.get("draft_logits", [])
        )
        downlink_bytes = (
            len(verify_payload.get("accepted_ids", [])) * TOKEN_ID_BYTES
            + len(verify_payload.get("corrected_ids", [])) * TOKEN_ID_BYTES
            + len(verify_payload.get("rejected_draft_ids", [])) * TOKEN_ID_BYTES
        )
        network = self._simulate_network(uplink_bytes, downlink_bytes)
        new_ids = list(verify_payload.get("accepted_ids", [])) + list(verify_payload.get("corrected_ids", []))
        if shared_state.get("ttft_ms") is None and new_ids:
            shared_state["ttft_ms"] = (now(shared_state["timing_device"]) - shared_state["request_start_time"]) * 1000.0

        round_index = len(shared_state.get("round_sequence", [])) + 1
        base_position = len(shared_state.get("token_provenance", []))
        token_provenance = []
        for offset, item in enumerate(list(verify_payload.get("token_provenance", []))):
            token_provenance.append(
                {
                    "position": base_position + offset,
                    "round": round_index,
                    **item,
                }
            )
        shared_state["token_provenance"].extend(token_provenance)
        shared_state["committed_ids"].extend(new_ids)
        shared_state["accepted_draft_tokens"] += int(verify_payload["round_stats"]["accepted_length"])
        shared_state["corrected_tokens"] += int(verify_payload["round_stats"]["corrected_count"])
        shared_state["total_draft_tokens"] += int(verify_payload["round_stats"]["draft_count"])
        shared_state["network_ms"] += float(network["network_ms"])
        shared_state["cloud_compute_ms"] += float(verify_payload.get("cloud_compute_ms", 0.0) or 0.0)

        round_stats = dict(verify_payload.get("round_stats", {}))
        round_stats["round"] = round_index
        round_stats["network_ms"] = float(network["network_ms"])
        round_stats["uplink_bytes"] = int(network["uplink_bytes"])
        round_stats["downlink_bytes"] = int(network["downlink_bytes"])
        shared_state["round_sequence"].append(round_stats)

        stop = bool(verify_payload.get("stop", False))
        stop_reason = str(verify_payload.get("stop_reason", "") or "")
        progress = bool(new_ids or draft_ids or stop)
        if not progress:
            stop = True
            stop_reason = "stalled"
        if stop_reason:
            shared_state["stop_reason"] = stop_reason

        payload = dict(verify_payload)
        payload["network_overhead_ms"] = float(network["network_ms"])
        payload["stop"] = bool(stop)
        payload["stop_reason"] = stop_reason
        payload["progress"] = bool(progress)
        payload["feedback"] = {
            "draft_output": draft_output,
            "verify_output": {
                key: value
                for key, value in payload.items()
                if key != "feedback"
            },
        }
        return payload

    def close_session(self, session, request=None):
        """Close one verifier request scope."""
        del session, request
        return None

    @torch.no_grad()
    def inference(self, data, token_callback=None, **kwargs):
        """Run cloud-only autoregressive decoding."""
        del token_callback, kwargs
        request = normalize_request(
            data,
            self.default_prompt_tokens,
            self.default_completion_tokens,
            _to_optional_int,
        )
        request["completion_tokens"] = self._resolve_completion_limit(request)
        encoded = self.tokenizer(request["query"], return_tensors="pt")
        prompt_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)
        start = now(self.device)
        network = self._simulate_network(int(prompt_ids.numel()) * TOKEN_ID_BYTES, 0)
        generation = self.core.autoregressive_generate(
            prompt_ids,
            request["completion_tokens"],
            self.temperature,
            stop_on_eos=self.stop_on_eos,
            attention_mask=attention_mask,
        )
        downlink = self._simulate_network(0, len(generation["completion_ids"]) * TOKEN_ID_BYTES)
        network["downlink_bytes"] += int(downlink["downlink_bytes"])
        network["downlink_ms"] += float(downlink["downlink_ms"])
        network["network_ms"] += float(downlink["network_ms"])
        total_ms = (now(self.device) - start) * 1000.0
        ttft_ms = float(generation["ttft_ms"])
        if self.enable_network_sleep:
            ttft_ms += float(network["uplink_ms"])
        perf = compute_paper_perf(
            total_ms,
            len(generation["completion_ids"]),
            ttft_ms,
            prefill_ms=generation["prefill_ms"],
        )
        simulation = _build_cloud_simulation(
            request,
            perf,
            self.draft_tokens_per_step,
            generation["stop_reason"],
            network["network_ms"],
            self.network_rtt_ms,
            self.network_jitter_ms,
        )
        response = build_specdec_response(
            self.tokenizer,
            request,
            int(prompt_ids.shape[1]),
            generation["completion_ids"],
            perf,
            simulation,
            extra_fields={
                "mode": "cloud-only",
                "prompt": request.get("query", ""),
                "gold": request.get("gold", ""),
                "completion_tokens": len(generation["completion_ids"]),
                "stop_reason": generation["stop_reason"],
                "network": network,
            },
        )
        record_sample_output(
            self,
            {
                "request_id": request.get("request_id"),
                "mode": "cloud-only",
                "task_name": request.get("task_name", "default"),
                "prompt": request.get("query", ""),
                "gold": request.get("gold", ""),
                "completion": response["completion"],
                "stop_reason": generation["stop_reason"],
            },
        )
        return response
