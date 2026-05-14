"""Paper-aligned AR speculative decoding core."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.timing_utils import now


def sample_token_id(logits, temperature):
    """Sample one token id from logits."""
    if float(temperature or 0.0) < 1e-5:
        return int(torch.argmax(logits, dim=-1).reshape(-1)[0].item())
    scaled = logits.float() / max(float(temperature or 0.0), 1e-5)
    return int(torch.distributions.Categorical(logits=scaled).sample().reshape(-1)[0].item())


def scaled_logits(logits, temperature):
    """Apply temperature scaling to logits."""
    return logits.float() / max(float(temperature or 0.0), 1e-5)


def build_topk_distribution(logits, temperature, top_k):
    """Build a sparse top-k probability payload from one logits tensor."""
    scaled = scaled_logits(logits, temperature)
    vocab_size = int(scaled.shape[-1])
    top_k = max(1, min(int(top_k or 0), vocab_size))
    topk_logits, topk_token_ids = torch.topk(scaled, k=top_k, dim=-1)
    topk_token_probs = torch.softmax(topk_logits, dim=-1)
    return {
        "representation": "topk_probs",
        "top_k": int(top_k),
        "vocab_size": int(vocab_size),
        "token_ids": topk_token_ids.detach().cpu(),
        "token_probs": topk_token_probs.detach().cpu(),
    }


def sample_token_id_from_topk(distribution):
    """Sample one token id from a sparse top-k payload."""
    token_ids = distribution["token_ids"].reshape(-1)
    token_probs = distribution["token_probs"].reshape(-1)
    sampled_index = int(torch.distributions.Categorical(probs=token_probs).sample().item())
    return int(token_ids[sampled_index].item())


def greedy_token_id(logits):
    """Return the argmax token id."""
    return int(torch.argmax(logits, dim=-1).reshape(-1)[0].item())


def probs_from_logits(logits, temperature):
    """Convert logits into a normalized probability vector."""
    scaled = logits.float() / max(float(temperature or 0.0), 1e-5)
    return torch.softmax(scaled, dim=-1).reshape(-1)


def sample_from_probs(probs):
    """Sample one token id from probabilities."""
    return int(torch.distributions.Categorical(probs=probs).sample().item())


def sample_from_residual(p_probs, q_probs):
    """Sample a correction token from max(p - q, 0)."""
    residual = torch.clamp(p_probs - q_probs, min=0.0)
    if float(residual.sum().item()) <= 0.0:
        return sample_from_probs(p_probs)
    residual = residual / residual.sum()
    return sample_from_probs(residual)


def is_topk_probability_payload(payload):
    """Return whether a draft distribution uses sparse top-k encoding."""
    return isinstance(payload, dict) and payload.get("representation") == "topk_probs"


def lookup_topk_probability(token_id, payload, device):
    """Look up one token probability from a sparse top-k payload."""
    token_ids = payload["token_ids"].reshape(-1).to(device)
    token_probs = payload["token_probs"].reshape(-1).to(device=device, dtype=torch.float32)
    matches = torch.nonzero(token_ids == int(token_id), as_tuple=False)
    if matches.numel() == 0:
        raise RuntimeError(f"Draft token {token_id} is not present in transported top-k support.")
    return float(token_probs[int(matches[0].item())].item())


def sample_from_sparse_residual(p_probs, payload):
    """Sample a correction token from max(p - q_topk, 0)."""
    token_ids = payload["token_ids"].reshape(-1).to(p_probs.device)
    token_probs = payload["token_probs"].reshape(-1).to(device=p_probs.device, dtype=p_probs.dtype)
    residual = p_probs.clone()
    residual[token_ids] = torch.clamp(residual[token_ids] - token_probs, min=0.0)
    if float(residual.sum().item()) <= 0.0:
        return sample_from_probs(p_probs)
    residual = residual / residual.sum()
    return sample_from_probs(residual)


def crop_past_key_values(past_key_values, end_pos):
    """Trim KV cache to the given logical token length."""
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(end_pos)
        return past_key_values

    trimmed = []
    for layer in past_key_values:
        key = layer[0]
        value = layer[1]
        extras = layer[2:]
        if key.dim() == 4:
            key = key[:, :, :end_pos, :]
        else:
            key = key[:, :, :end_pos]
        if value.dim() == 4:
            value = value[:, :, :end_pos, :]
        else:
            value = value[:, :end_pos, :]
        trimmed.append((key, value, *extras))
    return tuple(trimmed)


@dataclass
class DraftSession:
    """Mutable drafter-side KV session for one request."""

    prompt_ids: torch.Tensor
    prompt_token_count: int
    past_key_values: object
    last_logits: torch.Tensor | None
    prefill_ms: float
    cached_ids: list[int] = field(default_factory=list)
    pending_ids: list[int] = field(default_factory=list)


@dataclass
class VerifySession:
    """Mutable verifier-side KV session for one request."""

    prompt_ids: torch.Tensor
    prompt_token_count: int
    past_key_values: object
    last_logits: torch.Tensor | None
    prefill_ms: float
    cached_ids: list[int] = field(default_factory=list)
    pending_ids: list[int] = field(default_factory=list)


class PaperAlignedARDrafter:
    """Small-model generator used for edge-only and AR collaboration."""

    def __init__(self, model_path, device="auto", trust_remote_code=True):
        self.model_path = model_path
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.trust_remote_code = trust_remote_code
        self.tokenizer = None
        self.model = None

    def load(self):
        """Load tokenizer and causal LM."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.model_path.startswith("/"),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.model_path.startswith("/"),
        ).to(self.device)
        self.model.eval()

    def encode_prompt(self, text):
        """Encode a prompt to device-local token ids."""
        encoded = self.tokenizer(text, return_tensors="pt")
        return encoded.input_ids.to(self.device)

    def autoregressive_generate(self, prompt_ids, max_new_tokens, temperature, stop_on_eos=True):
        """Run autoregressive decoding with KV-cache reuse."""
        generated = []
        token_timestamps_ms = []
        start = now(self.device)
        ttft_ms = None
        with torch.no_grad():
            outputs = self.model(input_ids=prompt_ids, use_cache=True)
            prefill_ms = (now(self.device) - start) * 1000.0
            cache = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            for _ in range(max_new_tokens):
                token_id = sample_token_id(next_logits, temperature)
                generated.append(token_id)
                token_time_ms = (now(self.device) - start) * 1000.0
                token_timestamps_ms.append(token_time_ms)
                if ttft_ms is None:
                    ttft_ms = token_time_ms
                if stop_on_eos and token_id == self.tokenizer.eos_token_id:
                    break
                step_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                outputs = self.model(input_ids=step_ids, past_key_values=cache, use_cache=True)
                cache = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :]
        total_ms = (now(self.device) - start) * 1000.0
        return {
            "completion_ids": generated,
            "prefill_ms": prefill_ms if max_new_tokens > 0 else total_ms,
            "ttft_ms": ttft_ms or total_ms,
            "total_ms": total_ms,
            "token_timestamps_ms": token_timestamps_ms,
            "stop_reason": "eos" if generated and generated[-1] == self.tokenizer.eos_token_id else "completion_limit",
        }

    def start_session(self, prompt_ids):
        """Prefill prompt tokens and create a persistent KV session."""
        start = now(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=prompt_ids, use_cache=True)
        prefill_ms = (now(self.device) - start) * 1000.0
        return DraftSession(
            prompt_ids=prompt_ids,
            prompt_token_count=int(prompt_ids.shape[1]),
            past_key_values=outputs.past_key_values,
            last_logits=outputs.logits[:, -1, :],
            prefill_ms=prefill_ms,
        )

    def flush_pending(self, session):
        """Append deferred committed tokens into the cached prefix."""
        if not session.pending_ids:
            return 0.0
        pending_tensor = torch.tensor([session.pending_ids], dtype=torch.long, device=self.device)
        start = now(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=pending_tensor,
                past_key_values=session.past_key_values,
                use_cache=True,
            )
        session.past_key_values = outputs.past_key_values
        session.last_logits = outputs.logits[:, -1, :]
        session.cached_ids.extend(session.pending_ids)
        session.pending_ids = []
        return (now(self.device) - start) * 1000.0

    def draft(self, session, draft_tokens_per_step, temperature, draft_top_k=0):
        """Draft one speculative block on top of the cached prefix."""
        flush_ms = self.flush_pending(session)
        if session.last_logits is None:
            raise RuntimeError("Drafter session has no next-token logits after pending flush.")

        draft_ids = []
        draft_logits = []
        base_cached_count = len(session.cached_ids)
        start = now(self.device)
        cache = session.past_key_values
        next_logits = session.last_logits
        use_topk = float(temperature or 0.0) >= 1e-5 and int(draft_top_k or 0) > 0
        with torch.no_grad():
            for _ in range(draft_tokens_per_step):
                if use_topk:
                    topk_distribution = build_topk_distribution(
                        next_logits,
                        temperature,
                        draft_top_k,
                    )
                    draft_logits.append(topk_distribution)
                    token_id = sample_token_id_from_topk(topk_distribution)
                else:
                    draft_logits.append(next_logits.detach().cpu())
                    token_id = sample_token_id(next_logits, temperature)
                draft_ids.append(token_id)
                step_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                outputs = self.model(input_ids=step_ids, past_key_values=cache, use_cache=True)
                cache = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :]

        session.past_key_values = cache
        session.last_logits = next_logits
        session.cached_ids.extend(draft_ids)
        return {
            "draft_ids": draft_ids,
            "draft_logits": draft_logits,
            "edge_compute_ms": (now(self.device) - start) * 1000.0 + flush_ms,
            "base_cached_count": base_cached_count,
        }

    def apply_verifier_result(self, session, base_cached_count, accepted_ids, corrected_ids):
        """Crop rejected draft tokens and defer verifier-produced committed tokens."""
        keep_count = base_cached_count + len(accepted_ids)
        cache_length = session.prompt_token_count + keep_count
        session.past_key_values = crop_past_key_values(session.past_key_values, cache_length)
        session.cached_ids = session.cached_ids[:keep_count]
        session.pending_ids = list(corrected_ids)
        session.last_logits = None if corrected_ids else session.last_logits


class PaperAlignedARVerifier:
    """Large-model generator used for cloud-only and AR verification."""

    def __init__(self, model_path, device="auto", trust_remote_code=True):
        self.model_path = model_path
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.trust_remote_code = trust_remote_code
        self.tokenizer = None
        self.model = None

    def load(self):
        """Load tokenizer and causal LM."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.model_path.startswith("/"),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.model_path.startswith("/"),
        ).to(self.device)
        self.model.eval()

    def autoregressive_generate(self, prompt_ids, max_new_tokens, temperature, stop_on_eos=True, attention_mask=None):
        """Run autoregressive decoding with explicit KV-cache stepping."""
        generated = []
        token_timestamps_ms = []
        start = now(self.device)
        ttft_ms = None
        with torch.no_grad():
            outputs = self.model(input_ids=prompt_ids, attention_mask=attention_mask, use_cache=True)
            prefill_ms = (now(self.device) - start) * 1000.0
            cache = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            for _ in range(max_new_tokens):
                token_id = sample_token_id(next_logits, temperature)
                generated.append(token_id)
                token_time_ms = (now(self.device) - start) * 1000.0
                token_timestamps_ms.append(token_time_ms)
                if ttft_ms is None:
                    ttft_ms = token_time_ms
                if stop_on_eos and token_id == self.tokenizer.eos_token_id:
                    break
                step_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                outputs = self.model(input_ids=step_ids, past_key_values=cache, use_cache=True)
                cache = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :]
        total_ms = (now(self.device) - start) * 1000.0
        return {
            "completion_ids": generated,
            "prefill_ms": prefill_ms if max_new_tokens > 0 else total_ms,
            "ttft_ms": ttft_ms or total_ms,
            "total_ms": total_ms,
            "token_timestamps_ms": token_timestamps_ms,
            "stop_reason": "eos" if generated and generated[-1] == self.tokenizer.eos_token_id else "completion_limit",
        }

    def start_session(self, prompt_ids):
        """Prefill prompt tokens and create a persistent KV session."""
        start = now(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=prompt_ids, use_cache=True)
        prefill_ms = (now(self.device) - start) * 1000.0
        return VerifySession(
            prompt_ids=prompt_ids,
            prompt_token_count=int(prompt_ids.shape[1]),
            past_key_values=outputs.past_key_values,
            last_logits=outputs.logits[:, -1, :],
            prefill_ms=prefill_ms,
        )

    def verify(self, session, draft_ids, draft_logits, temperature, max_new_tokens, stop_on_eos=True):
        """Verify one speculative block with one incremental target-model pass."""
        if session.last_logits is None and not session.pending_ids:
            raise RuntimeError("Verifier session has no next-token logits before verification.")

        pending_count = len(session.pending_ids)
        combined_ids = list(session.pending_ids) + list(draft_ids)
        if not combined_ids:
            return {
                "accepted_ids": [],
                "corrected_ids": [],
                "rejected_draft_ids": [],
                "token_provenance": [],
                "round_stats": {
                    "draft_count": 0,
                    "accepted_length": 0,
                    "corrected_count": 0,
                    "rejected_draft_count": 0,
                    "stop_reason": "stalled",
                    "rejected_draft_tokens": [],
                },
                "cloud_compute_ms": 0.0,
                "stop": True,
                "stop_reason": "stalled",
            }

        combined_tensor = torch.tensor([combined_ids], dtype=torch.long, device=self.device)
        base_cached_count = len(session.cached_ids)
        start = now(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=combined_tensor,
                past_key_values=session.past_key_values,
                use_cache=True,
            )
        combined_logits = outputs.logits[0]
        next_logits_after_block = combined_logits[len(combined_ids) - 1].unsqueeze(0)

        accepted_ids = []
        corrected_ids = []
        rejected_draft_ids = []
        token_provenance = []
        stop = False
        stop_reason = ""
        rejection_logits = None

        for idx, token_id in enumerate(draft_ids):
            combined_pos = pending_count + idx
            if combined_pos == 0:
                p_logits = session.last_logits
            else:
                p_logits = combined_logits[combined_pos - 1].unsqueeze(0)
            if float(temperature or 0.0) < 1e-5:
                verifier_token = greedy_token_id(p_logits)
                if verifier_token == token_id:
                    accepted_ids.append(token_id)
                    continue
                corrected_ids = [verifier_token]
                rejected_draft_ids = draft_ids[idx:]
                rejection_logits = p_logits
                break
            p_probs = probs_from_logits(p_logits, temperature)
            if is_topk_probability_payload(draft_logits[idx]):
                q_token = max(
                    lookup_topk_probability(token_id, draft_logits[idx], p_probs.device),
                    1e-12,
                )
                accept_prob = min(1.0, float(p_probs[token_id].item()) / q_token)
            else:
                q_logits = draft_logits[idx].reshape(-1).float().to(p_probs.device)
                q_probs = torch.softmax(q_logits / max(float(temperature or 0.0), 1e-5), dim=-1)
                accept_prob = min(
                    1.0,
                    float(p_probs[token_id].item()) / max(float(q_probs[token_id].item()), 1e-12),
                )
            if torch.rand(1, device=p_probs.device).item() <= accept_prob:
                accepted_ids.append(token_id)
                continue
            if is_topk_probability_payload(draft_logits[idx]):
                corrected_ids = [sample_from_sparse_residual(p_probs, draft_logits[idx])]
            else:
                corrected_ids = [sample_from_residual(p_probs, q_probs)]
            rejected_draft_ids = draft_ids[idx:]
            rejection_logits = p_logits
            break

        logical_total_before = base_cached_count + pending_count
        if accepted_ids and stop_on_eos and accepted_ids[-1] == self.tokenizer.eos_token_id:
            stop = True
            stop_reason = "eos"
        elif not corrected_ids and len(accepted_ids) == len(draft_ids):
            logical_total = logical_total_before + len(accepted_ids)
            if logical_total < max_new_tokens:
                bonus = (
                    greedy_token_id(next_logits_after_block)
                    if float(temperature or 0.0) < 1e-5
                    else sample_from_probs(probs_from_logits(next_logits_after_block, temperature))
                )
                corrected_ids = [bonus]
            else:
                stop = True
                stop_reason = "completion_limit"

        if corrected_ids and stop_on_eos and corrected_ids[-1] == self.tokenizer.eos_token_id:
            stop = True
            stop_reason = "eos"
        elif accepted_ids and stop_on_eos and accepted_ids[-1] == self.tokenizer.eos_token_id:
            stop = True
            stop_reason = "eos"

        if not stop and logical_total_before + len(accepted_ids) + len(corrected_ids) >= max_new_tokens:
            stop = True
            stop_reason = "completion_limit"

        kept_cached_count = base_cached_count + pending_count + len(accepted_ids)
        cache_length = session.prompt_token_count + kept_cached_count
        session.past_key_values = crop_past_key_values(outputs.past_key_values, cache_length)
        session.cached_ids.extend(session.pending_ids)
        session.cached_ids.extend(accepted_ids)
        session.pending_ids = list(corrected_ids)
        session.last_logits = next_logits_after_block if rejection_logits is None else rejection_logits

        for token_id in accepted_ids:
            token_provenance.append(
                {
                    "token_id": token_id,
                    "token_text": self.tokenizer.decode([token_id], skip_special_tokens=False),
                    "source": "draft_accepted",
                }
            )
        for token_id in corrected_ids:
            source = "verifier_bonus" if len(accepted_ids) == len(draft_ids) else "verifier_correction"
            token_provenance.append(
                {
                    "token_id": token_id,
                    "token_text": self.tokenizer.decode([token_id], skip_special_tokens=False),
                    "source": source,
                }
            )

        return {
            "accepted_ids": accepted_ids,
            "corrected_ids": corrected_ids,
            "rejected_draft_ids": rejected_draft_ids,
            "token_provenance": token_provenance,
            "round_stats": {
                "draft_count": len(draft_ids),
                "accepted_length": len(accepted_ids),
                "corrected_count": len(corrected_ids),
                "rejected_draft_count": len(rejected_draft_ids),
                "stop_reason": stop_reason,
                "rejected_draft_tokens": [
                    {"token_id": tid, "token_text": self.tokenizer.decode([tid], skip_special_tokens=False)}
                    for tid in rejected_draft_ids
                ],
            },
            "cloud_compute_ms": (now(self.device) - start) * 1000.0,
            "stop": stop,
            "stop_reason": stop_reason,
        }
