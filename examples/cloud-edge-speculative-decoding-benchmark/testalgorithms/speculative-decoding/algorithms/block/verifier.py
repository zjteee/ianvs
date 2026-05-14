"""Paper-aligned block verifier wrapper for the Ianvs benchmark example."""

from __future__ import annotations

import os
import random
import sys
import time

import torch
from sedna.common.class_factory import ClassFactory, ClassType

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPECDEC_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
for path in (MODULE_DIR, SPECDEC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from base_verifier import BaseSpeculativeVerifier
from common.config_utils import _to_bool, _to_int, _to_optional_int
from common.request_utils import normalize_request
from common.response_utils import build_specdec_response, compute_paper_perf
from common.session_store import get as get_shared_session
from common.timing_utils import now
from result_builder import record_sample_output

from algorithms.block.core import PaperAlignedBlockVerifier

os.environ["BACKEND_TYPE"] = "TORCH"

TOKEN_ID_BYTES = 4
ROUND_CONTROL_BYTES = 8


def _resolve_stop_on_eos(kwargs):
    """Resolve the native-stop behavior from Ianvs profile fields."""
    if "stop_on_eos" in kwargs:
        return _to_bool(kwargs.get("stop_on_eos"), True)
    stop_mode = str(kwargs.get("stop_mode", "choice") or "choice").strip().lower()
    return stop_mode not in {"none", "disabled", "off", "false"}


def _normalize_generation_backend(value):
    """Normalize the baseline backend name and reject removed backends."""
    backend = str(value or "custom").strip().lower().replace("-", "_")
    if backend not in {"custom", "transformers"}:
        raise ValueError(
            f"Unsupported generation backend: {value}. "
            f"Expected one of custom/transformers."
        )
    return backend


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


@ClassFactory.register(ClassType.GENERAL, alias="SpeculativeBlockVerifyModel")
class SpeculativeBlockVerifyModel(BaseSpeculativeVerifier):
    """Ianvs verifier wrapper backed by the paper block implementation."""

    algorithm_name = "block_spec"
    role = "verifier"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = dict(kwargs)
        self.model_name = kwargs.get("model", "Qwen/Qwen3-8B")
        configured_device = kwargs.get("device", "auto")
        if configured_device == "auto":
            configured_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = configured_device
        self.trust_remote_code = _to_bool(kwargs.get("trust_remote_code", True), True)
        self.default_prompt_tokens = _to_optional_int(kwargs.get("prompt_tokens"))
        self.default_completion_tokens = _to_optional_int(kwargs.get("max_new_tokens"), 64)
        self.draft_tokens_per_step = max(1, _to_int(kwargs.get("draft_tokens_per_step"), 16))
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
        self.attn_implementation = str(kwargs.get("attn_implementation", "sdpa") or "sdpa")
        self.block_cloud_only_backend = _normalize_generation_backend(
            kwargs.get("block_cloud_only_backend", kwargs.get("generation_backend", "custom"))
        ).strip().lower()
        self.core = PaperAlignedBlockVerifier(
            self.model_name,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
            attn_implementation=self.attn_implementation,
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
        jitter = 0.0 if self.network_jitter_ms <= 0.0 else self.network_rng.uniform(-self.network_jitter_ms, self.network_jitter_ms)
        total = max(self.network_rtt_ms + jitter, 0.0)
        return total * self.network_uplink_ratio, total * (1.0 - self.network_uplink_ratio)

    def _simulate_network(self, uplink_bytes, downlink_bytes):
        """Simulate one collaboration or cloud-only payload exchange."""
        up_base_ms, down_base_ms = self._sample_base_delays()
        uplink_transfer_ms = self._bandwidth_delay_ms(uplink_bytes, self.network_uplink_bandwidth_mbps)
        downlink_transfer_ms = self._bandwidth_delay_ms(downlink_bytes, self.network_downlink_bandwidth_mbps)
        uplink_ms = up_base_ms + uplink_transfer_ms
        downlink_ms = down_base_ms + downlink_transfer_ms
        if self.enable_network_sleep and uplink_ms > 0.0:
            time.sleep(uplink_ms / 1000.0)
        if self.enable_network_sleep and downlink_ms > 0.0:
            time.sleep(downlink_ms / 1000.0)
        return {
            "uplink_bytes": int(uplink_bytes),
            "downlink_bytes": int(downlink_bytes),
            "uplink_ms": uplink_ms,
            "downlink_ms": downlink_ms,
            "uplink_propagation_ms": up_base_ms,
            "downlink_propagation_ms": down_base_ms,
            "uplink_transfer_ms": uplink_transfer_ms,
            "downlink_transfer_ms": downlink_transfer_ms,
            "propagation_ms": up_base_ms + down_base_ms,
            "transfer_ms": uplink_transfer_ms + downlink_transfer_ms,
            "network_ms": uplink_ms + downlink_ms,
        }

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
        prompt_ids = self.core.encode_prompt(request["query"])
        prompt_token_count = int(prompt_ids.shape[1])
        drafter_wrapper = (draft_session or {}).get("_drafter_wrapper")
        if drafter_wrapper is None:
            raise RuntimeError("Block verifier requires the drafter wrapper in draft_session.")
        drafter_wrapper.core.attach_target_ops(
            self.core.model.model.embed_tokens,
            self.core.model.lm_head,
            self.tokenizer,
        )
        drafter_wrapper.tokenizer = self.tokenizer
        self.core.set_target_layer_ids(drafter_wrapper.core.model.target_layer_ids)
        self.core.set_mask_token_id(drafter_wrapper.core.model.mask_token_id)

        state = self.core.prefill(
            prompt_ids=prompt_ids,
            max_new_tokens=request["completion_tokens"],
            block_size=self.draft_tokens_per_step,
            temperature=self.temperature,
        )
        shared_state = get_shared_session(request["request_id"])
        if shared_state is None:
            raise RuntimeError(f"Shared drafter state is missing for request {request['request_id']}.")
        initial_hidden_bytes = 0
        if state.get("target_hidden") is not None:
            initial_hidden_bytes = int(state["target_hidden"].numel()) * int(state["target_hidden"].element_size())
        prefill_network = self._simulate_network(0, initial_hidden_bytes)
        shared_state["request"] = request
        shared_state["prompt_token_count"] = prompt_token_count
        shared_state["verify_prefill_ms"] = float(state["prefill_ms"])
        shared_state["prefill_ms"] = float(state["prefill_ms"])
        shared_state["ttft_ms"] = float(state["prefill_ms"]) + (
            float(prefill_network["network_ms"]) if self.enable_network_sleep else 0.0
        )
        shared_state["timing_device"] = self.device
        shared_state["decode_tokenizer"] = self.tokenizer
        shared_state["network_ms"] += float(prefill_network["network_ms"])
        shared_state["network_propagation_ms"] += float(prefill_network["propagation_ms"])
        shared_state["network_transfer_ms"] += float(prefill_network["transfer_ms"])
        shared_state["prefill_network"] = {
            "network_ms": float(prefill_network["network_ms"]),
            "propagation_ms": float(prefill_network["propagation_ms"]),
            "transfer_ms": float(prefill_network["transfer_ms"]),
            "uplink_bytes": int(prefill_network["uplink_bytes"]),
            "downlink_bytes": int(prefill_network["downlink_bytes"]),
            "hidden_bytes": int(initial_hidden_bytes),
        }
        shared_state["committed_ids"] = [int(state["seed_token_id"])]
        shared_state["token_provenance"] = [
            {
                "position": 0,
                "round": 0,
                "token_id": int(state["seed_token_id"]),
                "source": "verifier_seed",
            }
        ]
        shared_state["_block_state"] = state
        return {
            "request_id": str(request.get("request_id", "default")),
            "request": request,
            "prompt_ids": prompt_ids,
            "prompt_token_count": prompt_token_count,
            "_shared_state": shared_state,
        }

    def verify(self, session, draft_output=None, draft_ids=None, **kwargs):
        """Verify one drafted block and update collaboration-shared statistics."""
        del kwargs, draft_ids
        draft_output = dict(draft_output or {})
        shared_state = session["_shared_state"]
        state = shared_state.get("_block_state")
        if state is None:
            raise RuntimeError("Block shared state is missing verifier runtime state.")
        if draft_output.get("block_output_ids") is None:
            payload = {
                "accepted_length": 0,
                "accepted_ids": [],
                "corrected_ids": [],
                "rejected_draft_ids": [],
                "bonus_token_id": None,
                "cloud_compute_ms": 0.0,
                "network_overhead_ms": 0.0,
                "stop": True,
                "stop_reason": shared_state.get("stop_reason") or "completion_limit",
                "progress": True,
                "round_stats": {
                    "draft_count": 0,
                    "accepted_length": 0,
                    "corrected_count": 0,
                    "rejected_draft_count": 0,
                    "stop_reason": shared_state.get("stop_reason") or "completion_limit",
                },
            }
            payload["feedback"] = {"draft_output": draft_output, "verify_output": dict(payload)}
            return payload

        verify_payload = self.core.verify_block(
            state,
            draft_output["block_output_ids"],
            self.temperature,
            stop_on_eos=self.stop_on_eos,
        )
        accepted_length = int(verify_payload["accepted_length"])
        draft_ids = list(draft_output.get("draft_ids", []) or [])
        accepted_ids = draft_ids[:accepted_length]
        corrected_ids = [int(verify_payload["bonus_token_id"])]
        rejected_draft_ids = draft_ids[accepted_length:]

        network = self._simulate_network(
            len(draft_ids) * TOKEN_ID_BYTES,
            int(draft_output.get("downlink_hidden_bytes", 0) or 0) + ROUND_CONTROL_BYTES,
        )

        base_position = len(shared_state.get("token_provenance", []))
        for index, token_id in enumerate(accepted_ids):
            shared_state["token_provenance"].append(
                {
                    "position": base_position + index,
                    "round": len(shared_state.get("round_sequence", [])) + 1,
                    "token_id": int(token_id),
                    "source": "draft_accepted",
                }
            )
        shared_state["token_provenance"].append(
            {
                "position": len(shared_state["token_provenance"]),
                "round": len(shared_state.get("round_sequence", [])) + 1,
                "token_id": int(verify_payload["bonus_token_id"]),
                "source": "verifier_bonus",
            }
        )

        shared_state["committed_ids"].extend(accepted_ids)
        shared_state["committed_ids"].extend(corrected_ids)
        shared_state["accepted_draft_tokens"] += accepted_length
        shared_state["corrected_tokens"] += 1
        shared_state["total_draft_tokens"] += len(draft_ids)
        shared_state["network_ms"] += float(network["network_ms"])
        shared_state["network_propagation_ms"] += float(network["propagation_ms"])
        shared_state["network_transfer_ms"] += float(network["transfer_ms"])
        shared_state["cloud_compute_ms"] += float(verify_payload.get("cloud_compute_ms", 0.0) or 0.0)

        round_index = len(shared_state.get("round_sequence", [])) + 1
        round_stats = {
            "round": round_index,
            "draft_count": len(draft_ids),
            "accepted_length": accepted_length,
            "corrected_count": 1,
            "rejected_draft_count": max(len(draft_ids) - accepted_length, 0),
            "rejected_draft_ids": list(rejected_draft_ids),
            "accepted_ids": list(accepted_ids),
            "corrected_ids": list(corrected_ids),
            "stop_reason": str(verify_payload.get("stop_reason", "") or ""),
            "cloud_compute_ms": float(verify_payload.get("cloud_compute_ms", 0.0) or 0.0),
            "edge_compute_ms": float(draft_output.get("edge_compute_ms", 0.0) or 0.0),
            "network_ms": float(network["network_ms"]),
            "network_propagation_ms": float(network["propagation_ms"]),
            "network_transfer_ms": float(network["transfer_ms"]),
            "uplink_bytes": int(network["uplink_bytes"]),
            "downlink_bytes": int(network["downlink_bytes"]),
            "bonus_token_id": int(verify_payload["bonus_token_id"]),
        }
        shared_state["round_sequence"].append(round_stats)

        stop = bool(verify_payload.get("stop", False))
        stop_reason = str(verify_payload.get("stop_reason", "") or "")
        progress = bool(accepted_ids or corrected_ids or draft_ids or stop)
        if not progress:
            stop = True
            stop_reason = "stalled"
        if stop_reason:
            shared_state["stop_reason"] = stop_reason

        payload = {
            "accepted_length": accepted_length,
            "accepted_ids": list(accepted_ids),
            "corrected_ids": list(corrected_ids),
            "rejected_draft_ids": list(rejected_draft_ids),
            "bonus_token_id": int(verify_payload["bonus_token_id"]),
            "cloud_compute_ms": float(verify_payload.get("cloud_compute_ms", 0.0) or 0.0),
            "network_overhead_ms": float(network["network_ms"]),
            "stop": bool(stop),
            "stop_reason": stop_reason,
            "progress": bool(progress),
            "round_stats": round_stats,
        }
        payload["feedback"] = {"draft_output": draft_output, "verify_output": dict(payload)}
        return payload

    def close_session(self, session, request=None):
        """Close one verifier request scope."""
        del session, request
        return None

    @torch.no_grad()
    def inference(self, data, token_callback=None, **kwargs):
        """Run cloud-only autoregressive decoding on the block prompt path."""
        del token_callback, kwargs
        request = normalize_request(
            data,
            self.default_prompt_tokens,
            self.default_completion_tokens,
            _to_optional_int,
        )
        request["completion_tokens"] = self._resolve_completion_limit(request)
        prompt_ids = self.core.encode_prompt(request["query"])
        start = now(self.device)
        network = self._simulate_network(int(prompt_ids.numel()) * TOKEN_ID_BYTES, 0)
        generation = self.core.autoregressive_generate(
            prompt_ids,
            request["completion_tokens"],
            self.temperature,
            stop_on_eos=self.stop_on_eos,
            backend=self.block_cloud_only_backend,
        )
        downlink = self._simulate_network(0, len(generation["completion_ids"]) * TOKEN_ID_BYTES)
        for key in (
            "downlink_bytes",
            "downlink_ms",
            "uplink_ms",
            "uplink_propagation_ms",
            "downlink_propagation_ms",
            "uplink_transfer_ms",
            "downlink_transfer_ms",
            "propagation_ms",
            "transfer_ms",
            "network_ms",
        ):
            network[key] += downlink[key]
        total_ms = (now(self.device) - start) * 1000.0
        ttft_ms = generation["ttft_ms"]
        if self.enable_network_sleep:
            ttft_ms += network["uplink_ms"]
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
                "algorithm": "block",
                "mode": "cloud-only",
                "prompt": request.get("query", ""),
                "gold": request.get("gold", ""),
                "completion_tokens": len(generation["completion_ids"]),
                "stop_reason": generation["stop_reason"],
                "network": network,
                "time_breakdown": {
                    "prefill_compute_ms": round(float(generation["prefill_ms"]), 6),
                    "decode_compute_ms": round(max(total_ms - generation["prefill_ms"] - network["network_ms"], 0.0), 6),
                    "edge_decode_compute_ms": 0.0,
                    "cloud_verify_compute_ms": round(max(total_ms - generation["prefill_ms"] - network["network_ms"], 0.0), 6),
                    "network_ms": round(float(network["network_ms"]), 6),
                    "network_propagation_ms": round(float(network["propagation_ms"]), 6),
                    "network_transfer_ms": round(float(network["transfer_ms"]), 6),
                },
            },
        )
        record_sample_output(
            self,
            {
                "request_id": request.get("request_id"),
                "algorithm": "block",
                "mode": "cloud-only",
                "task_name": request.get("task_name", "default"),
                "prompt": request.get("query", ""),
                "gold": request.get("gold", ""),
                "completion": response["completion"],
                "stop_reason": generation["stop_reason"],
            },
        )
        return response
