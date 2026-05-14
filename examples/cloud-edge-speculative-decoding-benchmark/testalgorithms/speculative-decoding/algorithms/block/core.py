"""Paper-aligned block speculative decoding core."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.generation.streamers import BaseStreamer
from common.timing_utils import now

from copied_dflash_model import DFlashDraftModel, extract_context_feature, sample


class _TokenTimingStreamer(BaseStreamer):
    """Capture generated token ids and arrival timestamps for `generate()`."""

    def __init__(self, prompt_token_count, start_time, device):
        self.prompt_token_count = int(prompt_token_count)
        self.start_time = float(start_time)
        self.device = device
        self.token_timestamps_ms = []
        self._prompt_skipped = False

    def put(self, value):
        """Record streamed generated tokens."""
        if isinstance(value, torch.Tensor):
            tokens = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, (list, tuple)):
            tokens = [int(item) for item in value]
        else:
            tokens = [int(value)]

        if not self._prompt_skipped:
            if len(tokens) == self.prompt_token_count:
                self._prompt_skipped = True
                return
            self._prompt_skipped = True

        timestamp_ms = (now(self.device) - self.start_time) * 1000.0
        for _ in tokens:
            self.token_timestamps_ms.append(timestamp_ms)

    def end(self):
        """No-op end hook required by streamer interface."""
        return None


class PaperAlignedBlockDrafter:
    """Edge-side block drafter that consumes target hidden states."""

    def __init__(
        self,
        model_path,
        device="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    ):
        self.model_path = model_path
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.model = None
        self.embed_tokens = None
        self.lm_head = None
        self.tokenizer = None

    def load(self):
        """Load the upstream DFlash draft checkpoint."""
        self.model = DFlashDraftModel.from_pretrained(
            self.model_path,
            attn_implementation=self.attn_implementation,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.model_path.startswith("/"),
        ).to(self.device).eval()

    def attach_target_ops(self, embed_tokens, lm_head, tokenizer):
        """Attach target embedding and LM head used by DFlash draft decoding."""
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def draft_block(self, state, temperature=0.0):
        """Draft one non-causal block from the latest target hidden states."""
        block_size = state["block_size"]
        start = state["start"]
        if block_size <= 1:
            return {
                "block_output_ids": state["output_ids"][:, start : start + block_size].clone(),
                "draft_count": 0,
                "draft_ids": [],
                "edge_compute_ms": 0.0,
            }

        block_output_ids = state["output_ids"][:, start : start + block_size].clone()
        position_ids = state["position_ids"][:, state["past_key_values_draft"].get_seq_length() : start + block_size]
        draft_start = now(self.device)
        with torch.no_grad():
            noise_embedding = self.embed_tokens(block_output_ids)
            hidden_states = self.model(
                target_hidden=state["target_hidden"],
                noise_embedding=noise_embedding,
                position_ids=position_ids,
                past_key_values=state["past_key_values_draft"],
                use_cache=True,
                is_causal=False,
            )
            draft_logits = self.lm_head(hidden_states[:, -block_size + 1 :, :])
            state["past_key_values_draft"].crop(start)
            drafted_ids = sample(draft_logits, temperature)
            block_output_ids[:, 1:] = drafted_ids
        return {
            "block_output_ids": block_output_ids,
            "draft_ids": drafted_ids[0].tolist(),
            "draft_count": drafted_ids.shape[1],
            "edge_compute_ms": (now(self.device) - draft_start) * 1000.0,
        }


class PaperAlignedBlockVerifier:
    """Cloud-side target model plus block verification logic."""

    def __init__(
        self,
        model_path,
        device="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    ):
        self.model_path = model_path
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.model = None
        self.tokenizer = None
        self.target_layer_ids = None
        self.mask_token_id = None

    def load(self):
        """Load tokenizer and target model."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.model_path.startswith("/"),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.model_path.startswith("/"),
        ).to(self.device).eval()

    def encode_prompt(self, text):
        """Encode one prompt with the chat-template path used by the paper benchmark."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": text}]
            try:
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def set_target_layer_ids(self, target_layer_ids):
        """Set the target hidden-state layer mapping expected by the DFlash draft model."""
        self.target_layer_ids = list(target_layer_ids)

    def set_mask_token_id(self, mask_token_id):
        """Set the mask token used by the DFlash draft checkpoint."""
        self.mask_token_id = int(mask_token_id)

    @torch.inference_mode()
    def autoregressive_generate(self, prompt_ids, max_new_tokens, temperature, stop_on_eos=True, backend="custom"):
        """Run target-only autoregressive decoding on the block prompt path."""
        if backend == "transformers":
            return self._autoregressive_generate_transformers(
                prompt_ids,
                max_new_tokens,
                temperature,
                stop_on_eos=stop_on_eos,
            )
        return self._autoregressive_generate_custom(
            prompt_ids,
            max_new_tokens,
            temperature,
            stop_on_eos=stop_on_eos,
        )

    @torch.inference_mode()
    def _autoregressive_generate_custom(self, prompt_ids, max_new_tokens, temperature, stop_on_eos=True):
        """Run target-only autoregressive decoding with explicit stepping."""
        generated = []
        token_timestamps_ms = []
        start = now(self.device)
        ttft_ms = None
        with torch.no_grad():
            outputs = self.model(input_ids=prompt_ids, use_cache=True)
            prefill_ms = (now(self.device) - start) * 1000.0
            cache = outputs.past_key_values
            next_token = sample(outputs.logits[:, -1:, :], temperature)
            token_id = int(next_token[0, 0].item())
            for _ in range(max_new_tokens):
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
                next_token = sample(outputs.logits[:, -1:, :], temperature)
                token_id = int(next_token[0, 0].item())
        total_ms = (now(self.device) - start) * 1000.0
        return {
            "completion_ids": generated,
            "prefill_ms": prefill_ms if max_new_tokens > 0 else total_ms,
            "ttft_ms": ttft_ms or total_ms,
            "total_ms": total_ms,
            "token_timestamps_ms": token_timestamps_ms,
            "stop_reason": "eos" if generated and generated[-1] == self.tokenizer.eos_token_id else "completion_limit",
        }

    @torch.inference_mode()
    def _autoregressive_generate_transformers(self, prompt_ids, max_new_tokens, temperature, stop_on_eos=True):
        """Run target-only autoregressive decoding with `transformers.generate()`."""
        start = now(self.device)
        attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
        streamer = _TokenTimingStreamer(prompt_ids.shape[1], start, self.device)
        do_sample = float(temperature or 0.0) >= 1e-5
        generate_kwargs = {
            "input_ids": prompt_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(max_new_tokens),
            "use_cache": True,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id if stop_on_eos else None,
            "streamer": streamer,
        }
        if do_sample:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = float(temperature)
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            output = self.model.generate(**generate_kwargs)

        total_ms = (now(self.device) - start) * 1000.0
        sequence = output.sequences[0]
        completion_ids = sequence[prompt_ids.shape[1] :].tolist()
        token_timestamps_ms = streamer.token_timestamps_ms[: len(completion_ids)]
        if len(token_timestamps_ms) < len(completion_ids):
            token_timestamps_ms.extend([total_ms] * (len(completion_ids) - len(token_timestamps_ms)))
        ttft_ms = token_timestamps_ms[0] if token_timestamps_ms else total_ms
        return {
            "completion_ids": completion_ids,
            "prefill_ms": ttft_ms,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "token_timestamps_ms": token_timestamps_ms,
            "stop_reason": "eos" if completion_ids and completion_ids[-1] == self.tokenizer.eos_token_id else "completion_limit",
        }

    @torch.inference_mode()
    def prefill(self, prompt_ids, max_new_tokens, block_size, temperature):
        """Prefill target cache and seed the first generated token."""
        prefill_start = now(self.device)
        num_input_tokens = prompt_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        if self.mask_token_id is None:
            raise ValueError("mask_token_id is not set for block verifier.")
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=self.device).unsqueeze(0)
        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()
        with torch.no_grad():
            outputs = self.model(
                prompt_ids,
                position_ids=position_ids[:, :num_input_tokens],
                past_key_values=past_key_values_target,
                use_cache=True,
                logits_to_keep=1,
                output_hidden_states=block_size > 1,
            )
        output_ids[:, :num_input_tokens] = prompt_ids
        first_token = sample(outputs.logits, temperature)
        output_ids[:, num_input_tokens : num_input_tokens + 1] = first_token
        target_hidden = None
        if block_size > 1:
            target_hidden = extract_context_feature(outputs.hidden_states, self.target_layer_ids)
        return {
            "prompt_ids": prompt_ids,
            "num_input_tokens": num_input_tokens,
            "max_length": max_length,
            "block_size": block_size,
            "output_ids": output_ids,
            "position_ids": position_ids,
            "past_key_values_target": past_key_values_target,
            "past_key_values_draft": past_key_values_draft,
            "target_hidden": target_hidden,
            "start": num_input_tokens,
            "prefill_ms": (now(self.device) - prefill_start) * 1000.0,
            "seed_token_id": int(first_token[0, 0].item()),
        }

    @torch.inference_mode()
    def verify_block(self, state, block_output_ids, temperature, stop_on_eos=True):
        """Verify one drafted block with one target forward pass."""
        block_size = state["block_size"]
        start = state["start"]
        block_position_ids = state["position_ids"][:, start : start + block_size]
        verify_start = now(self.device)
        with torch.no_grad():
            output = self.model(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=state["past_key_values_target"],
                use_cache=True,
                output_hidden_states=block_size > 1,
            )
        posterior = sample(output.logits, temperature)
        acceptance_length = int(
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )
        state["output_ids"][:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        state["output_ids"][:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        state["start"] += acceptance_length + 1
        state["past_key_values_target"].crop(state["start"])
        if block_size > 1:
            state["target_hidden"] = extract_context_feature(
                output.hidden_states,
                self.target_layer_ids,
            )[:, : acceptance_length + 1, :]

        bonus_token_id = int(posterior[0, acceptance_length].item())
        stop = False
        stop_reason = ""
        if stop_on_eos and bonus_token_id == self.tokenizer.eos_token_id:
            stop = True
            stop_reason = "eos"
        generated_tail = state["output_ids"][0, state["num_input_tokens"] : state["start"] + 1]
        if not stop and stop_on_eos and self.tokenizer.eos_token_id is not None:
            if torch.any(generated_tail == self.tokenizer.eos_token_id):
                stop = True
                stop_reason = "eos"
        if not stop and state["start"] >= state["max_length"]:
            stop = True
            stop_reason = "completion_limit"

        return {
            "accepted_length": acceptance_length,
            "bonus_token_id": bonus_token_id,
            "cloud_compute_ms": (now(self.device) - verify_start) * 1000.0,
            "stop": stop,
            "stop_reason": stop_reason,
        }
