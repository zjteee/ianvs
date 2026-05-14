"""Paper-aligned block drafter wrapper for the Ianvs benchmark example."""

from __future__ import annotations

import os
import sys

import torch
from sedna.common.class_factory import ClassFactory, ClassType

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPECDEC_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
for path in (MODULE_DIR, SPECDEC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from base_drafter import BaseSpeculativeDrafter
from common.config_utils import _to_bool, _to_int, _to_optional_int
from common.request_utils import normalize_request
from common.response_utils import build_specdec_response, compute_paper_perf
from common.session_store import clear as clear_session_store
from common.session_store import get as get_shared_session
from common.session_store import get_or_create
from common.session_store import pop as pop_shared_session
from common.timing_utils import now
from result_builder import record_sample_output

from algorithms.block.core import PaperAlignedBlockDrafter

os.environ["BACKEND_TYPE"] = "TORCH"


def _resolve_stop_on_eos(kwargs):
    """Resolve the native-stop behavior from Ianvs profile fields."""
    if "stop_on_eos" in kwargs:
        return _to_bool(kwargs.get("stop_on_eos"), True)
    stop_mode = str(kwargs.get("stop_mode", "choice") or "choice").strip().lower()
    return stop_mode not in {"none", "disabled", "off", "false"}


def _build_collaboration_simulation(shared_state, perf, draft_tokens_per_step):
    """Build collaboration simulation metadata for block decoding."""
    total_draft_tokens = int(shared_state.get("total_draft_tokens", 0))
    accepted_draft_tokens = int(shared_state.get("accepted_draft_tokens", 0))
    acceptance_rate = (
        float(accepted_draft_tokens) / float(total_draft_tokens)
        if total_draft_tokens > 0
        else None
    )
    return {
        "mode": "block-speculative-decoding",
        "routed_to": "collaboration",
        "acceptance_rate": round(acceptance_rate, 6) if acceptance_rate is not None else "",
        "end_to_end_latency": round(float(perf.get("e2e_latency_ms", 0.0)) / 1000.0, 6),
        "rounds": int(len(shared_state.get("round_sequence", []))),
        "accepted_draft_tokens": accepted_draft_tokens,
        "corrected_tokens": int(shared_state.get("corrected_tokens", 0)),
        "total_draft_tokens": total_draft_tokens,
        "network_overhead_ms": round(float(shared_state.get("network_ms", 0.0)), 6),
        "network_rtt_ms": round(float(shared_state.get("network_rtt_ms", 0.0)), 6),
        "network_jitter_ms": round(float(shared_state.get("network_jitter_ms", 0.0)), 6),
        "draft_tokens_per_step": int(draft_tokens_per_step),
        "task_name": shared_state["request"].get("task_name", "default"),
        "stop_reason": str(shared_state.get("stop_reason", "") or ""),
    }


@ClassFactory.register(ClassType.GENERAL, alias="SpeculativeBlockDraftModel")
class SpeculativeBlockDraftModel(BaseSpeculativeDrafter):
    """Ianvs drafter wrapper backed by the paper block implementation."""

    algorithm_name = "block_spec"
    role = "drafter"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = dict(kwargs)
        self.model_name = kwargs.get("model", "z-lab/Qwen3-8B-DFlash-b16")
        self.inference_mode = kwargs.get("inference_mode", "collaboration")
        if self.inference_mode not in {"collaboration", "cloud-only"}:
            raise ValueError(
                f"Unsupported inference_mode for block path: {self.inference_mode}. "
                f"Expected collaboration or cloud-only."
            )
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
        self.sample_output_log = kwargs.get("sample_output_log")
        self.attn_implementation = str(kwargs.get("attn_implementation", "sdpa") or "sdpa")
        self.core = PaperAlignedBlockDrafter(
            self.model_name,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
            attn_implementation=self.attn_implementation,
        )
        self.tokenizer = None
        self.model = None

    def load(self, *args, **kwargs):
        """Load the block draft model."""
        del args, kwargs
        self.core.load()
        self.model = self.core.model

    def cleanup(self):
        """Release loaded resources."""
        self.model = None
        self.tokenizer = None
        self.core.model = None
        self.core.embed_tokens = None
        self.core.lm_head = None
        self.core.tokenizer = None
        clear_session_store()

    def decode_tokens(self, token_ids, skip_special_tokens=True):
        """Decode token ids with the attached verifier tokenizer."""
        if self.tokenizer is None:
            return str(list(token_ids or []))
        return self.tokenizer.decode(list(token_ids or []), skip_special_tokens=skip_special_tokens)

    def build_request(self, data):
        """Normalize one Ianvs dataset sample into the request schema."""
        return normalize_request(
            data,
            self.default_prompt_tokens,
            self.default_completion_tokens,
            _to_optional_int,
        )

    def _resolve_completion_limit(self, request):
        """Resolve the per-request completion budget."""
        limit = request.get("completion_tokens")
        if limit is None:
            limit = self.default_completion_tokens
        return max(int(limit or 1), 1)

    def _build_shared_state(self, request, completion_limit, request_start_time):
        """Create one collaboration-shared request state."""
        return {
            "request": dict(request),
            "completion_limit": int(completion_limit),
            "request_start_time": float(request_start_time),
            "timing_device": self.device,
            "prompt_token_count": 0,
            "prefill_ms": 0.0,
            "verify_prefill_ms": 0.0,
            "ttft_ms": None,
            "committed_ids": [],
            "accepted_draft_tokens": 0,
            "corrected_tokens": 0,
            "total_draft_tokens": 0,
            "network_ms": 0.0,
            "network_propagation_ms": 0.0,
            "network_transfer_ms": 0.0,
            "network_rtt_ms": float(self.kwargs.get("network_rtt_ms", 0.0) or 0.0),
            "network_jitter_ms": float(self.kwargs.get("network_jitter_ms", 0.0) or 0.0),
            "edge_compute_ms": 0.0,
            "cloud_compute_ms": 0.0,
            "round_sequence": [],
            "token_provenance": [],
            "stop_reason": "",
            "finalized": False,
            "decode_tokenizer": None,
            "prefill_network": {},
            "_block_state": None,
        }

    def start_session(self, data=None, request=None, **kwargs):
        """Create one draft-side collaboration session."""
        del kwargs
        if request is None:
            request = self.build_request(data)
        request = dict(request)
        request["completion_tokens"] = self._resolve_completion_limit(request)
        request_start_time = now(self.device)
        shared_state = get_or_create(
            request["request_id"],
            lambda: self._build_shared_state(
                request=request,
                completion_limit=request["completion_tokens"],
                request_start_time=request_start_time,
            ),
        )
        return {
            "request_id": str(request.get("request_id", "default")),
            "request": request,
            "completion_limit": int(request["completion_tokens"]),
            "_shared_state": shared_state,
            "_drafter_wrapper": self,
        }

    def step(self, session, feedback=None, max_draft_tokens=None, **kwargs):
        """Run one block draft round."""
        del feedback, max_draft_tokens, kwargs
        shared_state = session["_shared_state"]
        state = shared_state.get("_block_state")
        if state is None:
            raise RuntimeError("Block verifier session has not initialized the shared block state.")
        if state["start"] >= state["max_length"]:
            return {
                "draft_ids": [],
                "draft_count": 0,
                "edge_compute_ms": 0.0,
            }
        hidden_bytes = 0
        if state.get("target_hidden") is not None:
            hidden_bytes = int(state["target_hidden"].numel()) * int(state["target_hidden"].element_size())
        payload = self.core.draft_block(state, temperature=self.temperature)
        payload["downlink_hidden_bytes"] = hidden_bytes
        shared_state["edge_compute_ms"] += float(payload.get("edge_compute_ms", 0.0) or 0.0)
        return payload

    def _build_collaboration_result(self, shared_state):
        """Build the final collaboration response."""
        completion_ids = list(shared_state.get("committed_ids", []))[: int(shared_state["completion_limit"])]
        total_ms = (now(shared_state["timing_device"]) - shared_state["request_start_time"]) * 1000.0
        perf = compute_paper_perf(
            total_ms,
            len(completion_ids),
            shared_state.get("ttft_ms") or total_ms,
            prefill_ms=shared_state.get("prefill_ms") or shared_state.get("ttft_ms") or total_ms,
        )
        simulation = _build_collaboration_simulation(
            shared_state,
            perf,
            self.draft_tokens_per_step,
        )
        response = build_specdec_response(
            shared_state.get("decode_tokenizer") or self.tokenizer,
            shared_state["request"],
            shared_state["prompt_token_count"],
            completion_ids,
            perf,
            simulation,
            token_provenance=shared_state.get("token_provenance", []),
            round_sequence=shared_state.get("round_sequence", []),
            extra_fields={
                "algorithm": "block",
                "mode": "collaboration",
                "prompt": shared_state["request"].get("query", ""),
                "gold": shared_state["request"].get("gold", ""),
                "completion_tokens": len(completion_ids),
                "accepted_draft_tokens": int(shared_state.get("accepted_draft_tokens", 0)),
                "corrected_tokens": int(shared_state.get("corrected_tokens", 0)),
                "total_draft_tokens": int(shared_state.get("total_draft_tokens", 0)),
                "rounds": int(len(shared_state.get("round_sequence", []))),
                "stop_reason": simulation["stop_reason"],
                "prefill_network": dict(shared_state.get("prefill_network", {})),
                "time_breakdown": {
                    "prefill_compute_ms": round(float(shared_state.get("prefill_ms", 0.0)), 6),
                    "edge_decode_compute_ms": round(float(shared_state.get("edge_compute_ms", 0.0)), 6),
                    "cloud_verify_compute_ms": round(float(shared_state.get("cloud_compute_ms", 0.0)), 6),
                    "network_ms": round(float(shared_state.get("network_ms", 0.0)), 6),
                    "network_propagation_ms": round(float(shared_state.get("network_propagation_ms", 0.0)), 6),
                    "network_transfer_ms": round(float(shared_state.get("network_transfer_ms", 0.0)), 6),
                },
            },
        )
        record_sample_output(
            self,
            {
                "request_id": shared_state["request"].get("request_id"),
                "algorithm": "block",
                "mode": "collaboration",
                "task_name": shared_state["request"].get("task_name", "default"),
                "prompt": shared_state["request"].get("query", ""),
                "gold": shared_state["request"].get("gold", ""),
                "completion": response["completion"],
                "stop_reason": simulation["stop_reason"],
                "accepted_draft_tokens": int(shared_state.get("accepted_draft_tokens", 0)),
                "corrected_tokens": int(shared_state.get("corrected_tokens", 0)),
                "total_draft_tokens": int(shared_state.get("total_draft_tokens", 0)),
                "rounds": int(len(shared_state.get("round_sequence", []))),
                "token_provenance": list(shared_state.get("token_provenance", [])),
                "round_sequence": list(shared_state.get("round_sequence", [])),
            },
        )
        return response

    def close_session(self, session, request=None):
        """Finalize one collaboration request."""
        del request
        request_id = session["request_id"]
        shared_state = get_shared_session(request_id)
        if shared_state is None:
            return None
        if not shared_state.get("finalized", False):
            shared_state["finalized"] = True
            result = self._build_collaboration_result(shared_state)
            shared_state["result"] = result
        else:
            result = shared_state.get("result")
        pop_shared_session(request_id, None)
        return result

    def inference(self, data=None, request=None, **kwargs):
        """Block path does not support edge-only autoregressive decoding."""
        del data, request, kwargs
        raise ValueError("Block/DFlash path does not support edge-only inference.")
