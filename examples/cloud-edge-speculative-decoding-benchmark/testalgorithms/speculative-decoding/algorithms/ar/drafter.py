"""Paper-aligned AR drafter wrapper for the Ianvs benchmark example."""

import os
import sys

import torch

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPECDEC_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
for path in (MODULE_DIR, SPECDEC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from sedna.common.class_factory import ClassFactory, ClassType

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
from algorithms.ar.core import PaperAlignedARDrafter

os.environ["BACKEND_TYPE"] = "TORCH"


def _resolve_stop_on_eos(kwargs):
    """Resolve the native-stop behavior from Ianvs profile fields."""
    if "stop_on_eos" in kwargs:
        return _to_bool(kwargs.get("stop_on_eos"), True)
    stop_mode = str(kwargs.get("stop_mode", "choice") or "choice").strip().lower()
    return stop_mode not in {"none", "disabled", "off", "false"}


def _build_edge_simulation(request, perf, draft_tokens_per_step, stop_reason):
    """Build edge-only simulation metadata."""
    return {
        "mode": "edge-only",
        "routed_to": "edge",
        "acceptance_rate": "",
        "end_to_end_latency": round(float(perf.get("e2e_latency_ms", 0.0)) / 1000.0, 6),
        "rounds": 1,
        "accepted_draft_tokens": 0,
        "corrected_tokens": 0,
        "total_draft_tokens": 0,
        "network_overhead_ms": 0.0,
        "network_rtt_ms": 0.0,
        "network_jitter_ms": 0.0,
        "draft_tokens_per_step": int(draft_tokens_per_step),
        "task_name": request.get("task_name", "default"),
        "stop_reason": stop_reason,
    }


def _build_collaboration_simulation(shared_state, perf, draft_tokens_per_step):
    """Build collaboration simulation metadata."""
    total_draft_tokens = int(shared_state.get("total_draft_tokens", 0))
    accepted_draft_tokens = int(shared_state.get("accepted_draft_tokens", 0))
    acceptance_rate = (
        float(accepted_draft_tokens) / float(total_draft_tokens)
        if total_draft_tokens > 0
        else None
    )
    return {
        "mode": "token-level-speculative-decoding",
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


@ClassFactory.register(ClassType.GENERAL, alias="SpeculativeDraftModel")
class SpeculativeDraftModel(BaseSpeculativeDrafter):
    """Ianvs drafter wrapper backed by the paper AR implementation."""

    algorithm_name = "ar_spec"
    role = "drafter"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = dict(kwargs)
        self.model_name = kwargs.get("model", "Qwen/Qwen2.5-0.5B-Instruct")
        os.environ["model_path"] = self.model_name
        self.inference_mode = kwargs.get("inference_mode", "collaboration")
        if self.inference_mode not in {"collaboration", "cloud-only", "edge-only"}:
            raise ValueError(
                f"Unsupported inference_mode: {self.inference_mode}. "
                f"Expected one of collaboration/cloud-only/edge-only."
            )
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
        self.sample_output_log = kwargs.get("sample_output_log")
        self.core = PaperAlignedARDrafter(
            self.model_name,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
        )
        self.tokenizer = None
        self.model = None

    def load(self, *args, **kwargs):
        """Load the draft model."""
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
        clear_session_store()

    def decode_tokens(self, token_ids, skip_special_tokens=True):
        """Decode token ids with the draft tokenizer."""
        return self.tokenizer.decode(list(token_ids or []), skip_special_tokens=skip_special_tokens)

    def build_request(self, data):
        """Normalize one Ianvs dataset sample into the request schema."""
        return normalize_request(
            data,
            self.default_prompt_tokens,
            self.default_completion_tokens,
            _to_optional_int,
        )

    def prepare_prompt(self, request):
        """Tokenize one normalized request prompt."""
        prompt_ids = self.core.encode_prompt(request["query"])
        return {
            "prompt_ids": prompt_ids,
            "prompt_token_count": int(prompt_ids.shape[1]),
        }

    def _resolve_completion_limit(self, request):
        """Resolve the per-request completion budget."""
        limit = request.get("completion_tokens")
        if limit is None:
            limit = self.default_completion_tokens
        return max(int(limit or 1), 1)

    def _consume_feedback(self, session, feedback=None):
        """Apply verifier feedback to the draft-side KV session."""
        feedback = feedback or session.pop("_pending_feedback", None)
        if not feedback:
            return None
        draft_output = dict((feedback or {}).get("draft_output", {}) or {})
        verify_output = dict((feedback or {}).get("verify_output", {}) or {})
        self.core.apply_verifier_result(
            session["_core_session"],
            int(draft_output.get("base_cached_count", len(session["_core_session"].cached_ids))),
            list(verify_output.get("accepted_ids", []) or []),
            list(verify_output.get("corrected_ids", []) or []),
        )
        session["_last_feedback"] = verify_output
        return verify_output

    def _build_shared_state(self, request, prompt_token_count, completion_limit, draft_session, request_start_time):
        """Create one collaboration-shared request state."""
        return {
            "request": dict(request),
            "prompt_token_count": int(prompt_token_count),
            "completion_limit": int(completion_limit),
            "request_start_time": float(request_start_time),
            "timing_device": self.device,
            "draft_prefill_ms": float(draft_session.prefill_ms),
            "verify_prefill_ms": 0.0,
            "prefill_ms": float(draft_session.prefill_ms),
            "ttft_ms": None,
            "committed_ids": [],
            "accepted_draft_tokens": 0,
            "corrected_tokens": 0,
            "total_draft_tokens": 0,
            "network_ms": 0.0,
            "network_rtt_ms": float(self.kwargs.get("network_rtt_ms", 0.0) or 0.0),
            "network_jitter_ms": float(self.kwargs.get("network_jitter_ms", 0.0) or 0.0),
            "edge_compute_ms": 0.0,
            "cloud_compute_ms": 0.0,
            "round_sequence": [],
            "token_provenance": [],
            "stop_reason": "",
            "finalized": False,
            "decode_tokenizer": self.tokenizer,
        }

    def start_session(self, data=None, request=None, **kwargs):
        """Create one draft-side collaboration session."""
        del kwargs
        if request is None:
            request = self.build_request(data)
        prompt_payload = self.prepare_prompt(request)
        request = dict(request)
        request["completion_tokens"] = self._resolve_completion_limit(request)
        request_start_time = now(self.device)
        draft_session = self.core.start_session(prompt_payload["prompt_ids"])
        shared_state = get_or_create(
            request["request_id"],
            lambda: self._build_shared_state(
                request=request,
                prompt_token_count=prompt_payload["prompt_token_count"],
                completion_limit=request["completion_tokens"],
                draft_session=draft_session,
                request_start_time=request_start_time,
            ),
        )
        return {
            "request_id": str(request.get("request_id", "default")),
            "request": request,
            "prompt_ids": prompt_payload["prompt_ids"],
            "prompt_token_count": int(prompt_payload["prompt_token_count"]),
            "completion_limit": int(request["completion_tokens"]),
            "_core_session": draft_session,
            "_shared_state": shared_state,
        }

    def step(self, session, feedback=None, max_draft_tokens=None, **kwargs):
        """Run one paper-aligned draft round."""
        del kwargs
        self._consume_feedback(session, feedback=feedback)
        shared_state = session["_shared_state"]
        completion_limit = int(shared_state["completion_limit"])
        remaining = max(completion_limit - len(shared_state["committed_ids"]), 0)
        if max_draft_tokens is None:
            window = min(self.draft_tokens_per_step, remaining)
        else:
            window = min(max(int(max_draft_tokens or 0), 0), remaining)
        if window <= 0:
            return {
                "draft_ids": [],
                "draft_logits": [],
                "edge_compute_ms": 0.0,
                "base_cached_count": len(session["_core_session"].cached_ids),
            }
        payload = self.core.draft(
            session["_core_session"],
            window,
            self.temperature,
            draft_top_k=self.draft_top_k,
        )
        shared_state["edge_compute_ms"] += float(payload.get("edge_compute_ms", 0.0) or 0.0)
        payload["selected_window"] = int(window)
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
                "mode": "collaboration",
                "prompt": shared_state["request"].get("query", ""),
                "gold": shared_state["request"].get("gold", ""),
                "completion_tokens": len(completion_ids),
                "accepted_draft_tokens": int(shared_state.get("accepted_draft_tokens", 0)),
                "corrected_tokens": int(shared_state.get("corrected_tokens", 0)),
                "total_draft_tokens": int(shared_state.get("total_draft_tokens", 0)),
                "rounds": int(len(shared_state.get("round_sequence", []))),
                "stop_reason": simulation["stop_reason"],
            },
        )
        record_sample_output(
            self,
            {
                "request_id": shared_state["request"].get("request_id"),
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
        self._consume_feedback(session, feedback=session.pop("_pending_feedback", None))
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
        """Run edge-only autoregressive decoding."""
        del kwargs
        if request is None:
            request = self.build_request(data)
        prompt_payload = self.prepare_prompt(request)
        completion_limit = self._resolve_completion_limit(request)
        start = now(self.device)
        generation = self.core.autoregressive_generate(
            prompt_payload["prompt_ids"],
            completion_limit,
            self.temperature,
            stop_on_eos=self.stop_on_eos,
        )
        total_ms = (now(self.device) - start) * 1000.0
        perf = compute_paper_perf(
            total_ms,
            len(generation["completion_ids"]),
            generation["ttft_ms"],
            prefill_ms=generation["prefill_ms"],
        )
        simulation = _build_edge_simulation(
            request,
            perf,
            self.draft_tokens_per_step,
            generation["stop_reason"],
        )
        response = build_specdec_response(
            self.tokenizer,
            request,
            prompt_payload["prompt_token_count"],
            generation["completion_ids"],
            perf,
            simulation,
            extra_fields={
                "mode": "edge-only",
                "prompt": request.get("query", ""),
                "gold": request.get("gold", ""),
                "completion_tokens": len(generation["completion_ids"]),
                "stop_reason": generation["stop_reason"],
            },
        )
        record_sample_output(
            self,
            {
                "request_id": request.get("request_id"),
                "mode": "edge-only",
                "task_name": request.get("task_name", "default"),
                "prompt": request.get("query", ""),
                "gold": request.get("gold", ""),
                "completion": response["completion"],
                "stop_reason": generation["stop_reason"],
            },
        )
        return response
