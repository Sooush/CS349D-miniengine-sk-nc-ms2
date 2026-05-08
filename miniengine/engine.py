"""
Model engine — wraps the bare-bone CausalLM for serving.

The engine is a "black box" that the scheduler calls into.  It handles:
  1. Model loading and GPU placement (via model.py + safetensors)
  2. Tokenization / detokenization (chat-template aware via AutoTokenizer)
  3. Prefill (prompt → first token + KV cache)
  4. Decode  (previous token + KV cache → next token + updated KV cache)
  5. Token sampling (delegated to sampler.py)

Two decode paths:
  - decode_step(req)        : one request, used by baseline scheduler
  - batched_decode(reqs)    : many requests, one forward pass with padded
                              KV + attention mask, used by batched mode

Prefill stays per-request — variable prompt lengths make batched prefill
complex, and decode is where the throughput gain lives.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.model import CausalLM, ModelConfig, load_weights
from miniengine.sampler import sample_token

logger = logging.getLogger(__name__)


class PagedRequestState:
    """Per-request metadata for KV pages stored in the pool."""

    def __init__(self, page_indices: list[int], kv_len: int) -> None:
        self.page_indices = page_indices
        self.kv_len = kv_len


class Engine:
    """Model wrapper supporting baseline (per-request) and batched decode."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        mode: str = "batched",
        page_size: int = 32,
        kv_pool_num_pages: int = 4096,
        kv_pool_bytes_budget: int | None = None,
        torch_compile: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.page_size = page_size

        # ── Tokenizer (still from HF — it's just a tokenizer) ──────────
        logger.info("Loading tokenizer from %s …", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # ── Model (bare-bone PyTorch, loaded from safetensors) ──────────
        logger.info("Loading model config from %s …", model_path)
        config = ModelConfig.from_pretrained(model_path)
        logger.info(
            "Config: layers=%d, hidden=%d, heads=%d, kv_heads=%d, head_dim=%d, "
            "intermediate=%d, vocab=%d, tie_embed=%s",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.intermediate_size,
            config.vocab_size,
            config.tie_word_embeddings,
        )

        # Build on meta device — load_weights replaces parameters with
        # GPU tensors directly, so we never allocate a CPU fp32 copy.
        with torch.device("meta"):
            self.model = CausalLM(config)
        load_weights(self.model, model_path, dtype=dtype, device=device)
        self.model.eval()
        if torch_compile:
            self._enable_torch_compile()
        self.kv_pool = None
        if self.mode == "paged":
            if kv_pool_bytes_budget is not None:
                self.kv_pool = KVMemoryPool.from_budget(
                    num_layers=config.num_hidden_layers,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    page_size=page_size,
                    dtype=dtype,
                    device=device,
                    bytes_budget=kv_pool_bytes_budget,
                )
            else:
                self.kv_pool = KVMemoryPool(
                    num_pages=kv_pool_num_pages,
                    page_size=page_size,
                    num_layers=config.num_hidden_layers,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    dtype=dtype,
                    device=device,
                )
            logger.info(
                "Paged mode enabled  —  page_size=%d, pool_pages=%d",
                page_size,
                self.kv_pool.num_free,
            )

        # ── Stop tokens ─────────────────────────────────────────────────
        self.stop_token_ids: set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.add(self.tokenizer.eos_token_id)
        for tok_name in ("eos_token", "pad_token"):
            tid = getattr(self.tokenizer, f"{tok_name}_id", None)
            if tid is not None:
                self.stop_token_ids.add(tid)
        for token_str in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                self.stop_token_ids.add(tid)

        logger.info(
            "Engine ready  —  vocab=%d, stop_ids=%s, params=%dM",
            len(self.tokenizer),
            self.stop_token_ids,
            sum(p.numel() for p in self.model.parameters()) // 1_000_000,
        )

    def _enable_torch_compile(self) -> None:
        """Compile stable MLP submodules to reduce launch overhead."""
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile unavailable in this PyTorch build; skipping")
            return
        try:
            for layer in self.model.model.layers:
                layer.mlp = torch.compile(layer.mlp, dynamic=True)
            logger.info("torch.compile enabled for transformer MLP submodules")
        except Exception:
            logger.exception("Failed to enable torch.compile; continuing in eager mode")

    # ── Tokenization ────────────────────────────────────────────────────

    def tokenize_messages(self, messages: list[dict[str, str]]) -> list[int]:
        """Apply the model's chat template and tokenize into ids."""
        kwargs: dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        # Qwen3 models support enable_thinking; silently ignore if unsupported
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id back to a string."""
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    # ── Forward passes ──────────────────────────────────────────────────

    @torch.inference_mode()
    def prefill(self, request: Request) -> int:
        """
        Run the prefill phase for one request.

        Processes the full prompt in a single forward pass, stores the
        resulting KV cache on the request, and samples the first output
        token.

        Returns:
            The first generated token id.
        """
        input_ids = torch.tensor(
            [request.input_ids], dtype=torch.long, device=self.device
        )
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        logits, kv_caches = self.model(input_ids, position_ids, kv_caches=None)

        if self.mode == "paged":
            assert self.kv_pool is not None
            self._store_prefill_kv_to_pool(request, kv_caches)
        else:
            request.kv_cache = kv_caches

        # Sample from the last position
        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    @torch.inference_mode()
    def prefill_batch(self, requests: list[Request]) -> list[int]:
        """Prefill a batch of requests. Paged mode uses one model forward."""
        if not requests:
            return []
        if self.mode != "paged":
            return [self.prefill(req) for req in requests]

        input_lens = [len(req.input_ids) for req in requests]
        packed_ids = []
        packed_pos = []
        for req in requests:
            packed_ids.extend(req.input_ids)
            packed_pos.extend(range(len(req.input_ids)))

        total_len = len(packed_ids)
        input_ids = torch.tensor([packed_ids], dtype=torch.long, device=self.device)
        position_ids = torch.tensor([packed_pos], dtype=torch.long, device=self.device)
        attention_mask = self._build_packed_prefill_attention_mask(input_lens, total_len)
        logits, kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=None,
            attention_mask=attention_mask,
        )

        token_ids = []
        start = 0
        for i, req in enumerate(requests):
            prompt_len = input_lens[i]
            per_req_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
            for layer_k, layer_v in kv_caches:
                per_req_kv.append(
                    (
                        layer_k[:, :, start : start + prompt_len, :].contiguous(),
                        layer_v[:, :, start : start + prompt_len, :].contiguous(),
                    )
                )
            self._store_prefill_kv_to_pool(req, per_req_kv)
            token_ids.append(
                sample_token(
                    logits[:, start + prompt_len - 1, :],
                    req.sampling_params,
                    req.output_ids,
                )
            )
            start += prompt_len
        return token_ids

    @torch.inference_mode()
    def decode_step(self, request: Request) -> int:
        """
        Run one decode step for a request that has already been prefilled.

        Feeds the last generated token through the model together with the
        cached KV values, updates the cache, and samples the next token.

        Returns:
            The next generated token id.
        """
        input_ids = torch.tensor(
            [[request.output_ids[-1]]], dtype=torch.long, device=self.device
        )
        if self.mode == "paged":
            cache_len = self._paged_state(request).kv_len
            kv_caches = self._gather_kv_from_pool(request)
        else:
            # Position = current KV cache length (= num tokens already processed)
            cache_len = request.kv_cache[0][0].shape[2]  # layer 0, key tensor, seq dim
            kv_caches = request.kv_cache
        position_ids = torch.tensor([[cache_len]], device=self.device)

        logits, new_kv_caches = self.model(input_ids, position_ids, kv_caches=kv_caches)
        if self.mode == "paged":
            self._append_last_token_kv_to_pool(request, new_kv_caches)
        else:
            request.kv_cache = new_kv_caches

        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids

    # ── Batched decode ──────────────────────────────────────────────────

    @torch.inference_mode()
    def batched_decode(self, requests: list[Request]) -> list[int]:
        """
        Decode one token for each request in a single forward pass.

        Pads per-request KV caches to the longest in the batch, builds a
        float attention mask that ignores padding, runs the model once,
        then extracts each request's actual KV (real prefix + new token)
        and samples its next token.
        """
        if not requests:
            return []
        if self.mode == "paged":
            return self._paged_batched_decode(requests)

        batch_size = len(requests)
        num_layers = len(requests[0].kv_cache)

        # Stack last generated token from each request → (batch, 1)
        input_ids = torch.tensor(
            [[req.output_ids[-1]] for req in requests],
            dtype=torch.long,
            device=self.device,
        )

        # Each request's current KV length and the per-request RoPE position
        cache_lens = [req.kv_cache[0][0].shape[2] for req in requests]
        max_cache_len = max(cache_lens)
        position_ids = torch.tensor(
            [[cl] for cl in cache_lens],
            dtype=torch.long,
            device=self.device,
        )

        # Pad and stack KV caches per layer to (batch, kv_heads, max_cache_len, head_dim)
        padded_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            k_list, v_list = [], []
            for req in requests:
                k, v = req.kv_cache[layer_idx]
                pad_len = max_cache_len - k.shape[2]
                if pad_len > 0:
                    k = F.pad(k, (0, 0, 0, pad_len))
                    v = F.pad(v, (0, 0, 0, pad_len))
                k_list.append(k)
                v_list.append(v)
            padded_kv_caches.append(
                (torch.cat(k_list, dim=0), torch.cat(v_list, dim=0))
            )

        # Mask shape (batch, 1, 1, max_cache_len + 1): the attention forward
        # appends the new token to the cache, so kv_len = max_cache_len + 1.
        # Mask only the padding window [cl, max_cache_len) per request.
        attention_mask = torch.zeros(
            batch_size,
            1,
            1,
            max_cache_len + 1,
            device=self.device,
            dtype=self.dtype,
        )
        for i, cl in enumerate(cache_lens):
            attention_mask[i, 0, 0, cl:max_cache_len] = float("-inf")

        logits, new_kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=padded_kv_caches,
            attention_mask=attention_mask,
        )

        # Extract each request's real KV (actual prefix + new token at -1).
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            cl = cache_lens[i]
            per_req_kv = []
            for layer_idx in range(num_layers):
                k_full = new_kv_caches[layer_idx][0][i : i + 1]
                v_full = new_kv_caches[layer_idx][1][i : i + 1]
                k_new = torch.cat([k_full[:, :, :cl, :], k_full[:, :, -1:, :]], dim=2)
                v_new = torch.cat([v_full[:, :, :cl, :], v_full[:, :, -1:, :]], dim=2)
                per_req_kv.append((k_new, v_new))
            req.kv_cache = per_req_kv
            token_ids.append(
                sample_token(
                    logits[i : i + 1, -1, :], req.sampling_params, req.output_ids
                )
            )
        return token_ids

    def _paged_batched_decode(self, requests: list[Request]) -> list[int]:
        """Decode one token for each request in paged mode."""
        batch_size = len(requests)
        cache_lens = [self._paged_state(req).kv_len for req in requests]
        max_cache_len = max(cache_lens)
        num_layers = self.model.config.num_hidden_layers

        input_ids = torch.tensor(
            [[req.output_ids[-1]] for req in requests],
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.tensor(
            [[cl] for cl in cache_lens],
            dtype=torch.long,
            device=self.device,
        )

        gathered_per_req = [self._gather_kv_from_pool(req) for req in requests]
        padded_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            k_list: list[torch.Tensor] = []
            v_list: list[torch.Tensor] = []
            for req_idx in range(batch_size):
                k_req, v_req = gathered_per_req[req_idx][layer_idx]
                pad_len = max_cache_len - k_req.shape[2]
                if pad_len > 0:
                    k_req = F.pad(k_req, (0, 0, 0, pad_len))
                    v_req = F.pad(v_req, (0, 0, 0, pad_len))
                k_list.append(k_req)
                v_list.append(v_req)
            padded_kv_caches.append((torch.cat(k_list, dim=0), torch.cat(v_list, dim=0)))

        attention_mask = torch.zeros(
            batch_size,
            1,
            1,
            max_cache_len + 1,
            device=self.device,
            dtype=self.dtype,
        )
        for i, cl in enumerate(cache_lens):
            attention_mask[i, 0, 0, cl:max_cache_len] = float("-inf")

        logits, new_kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=padded_kv_caches,
            attention_mask=attention_mask,
        )

        token_ids: list[int] = []
        for i, req in enumerate(requests):
            per_req_new_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
            for layer_idx in range(num_layers):
                k_full = new_kv_caches[layer_idx][0][i : i + 1]
                v_full = new_kv_caches[layer_idx][1][i : i + 1]
                per_req_new_kv.append((k_full, v_full))
            self._append_last_token_kv_to_pool(req, per_req_new_kv)
            token_ids.append(
                sample_token(
                    logits[i : i + 1, -1, :], req.sampling_params, req.output_ids
                )
            )
        return token_ids

    def free_request_kv(self, request: Request) -> None:
        """Return a request's pages to the pool (paged mode only)."""
        if self.mode != "paged" or request.kv_cache is None:
            return
        state = self._paged_state(request)
        assert self.kv_pool is not None
        self.kv_pool.free(state.page_indices)
        request.kv_cache = None

    def _paged_state(self, request: Request) -> PagedRequestState:
        state = request.kv_cache
        if not isinstance(state, PagedRequestState):
            raise RuntimeError("request.kv_cache is not a PagedRequestState")
        return state

    def _store_prefill_kv_to_pool(
        self,
        request: Request,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        assert self.kv_pool is not None
        kv_len = kv_caches[0][0].shape[2]
        needed_pages = self.kv_pool.pages_needed(kv_len)
        page_indices = self.kv_pool.allocate(needed_pages)
        state = PagedRequestState(page_indices=page_indices, kv_len=kv_len)

        for layer_idx, (k_layer, v_layer) in enumerate(kv_caches):
            k_seq = k_layer[0].transpose(0, 1).contiguous()  # (seq, kv_heads, head_dim)
            v_seq = v_layer[0].transpose(0, 1).contiguous()
            for t in range(kv_len):
                page_offset = t // self.page_size
                slot = t % self.page_size
                page_id = page_indices[page_offset]
                self.kv_pool.kv_caches[layer_idx][0][page_id, slot].copy_(k_seq[t])
                self.kv_pool.kv_caches[layer_idx][1][page_id, slot].copy_(v_seq[t])
        request.kv_cache = state

    def _build_packed_prefill_attention_mask(
        self,
        input_lens: list[int],
        total_len: int,
    ) -> torch.Tensor:
        """Build a block-diagonal causal mask for packed prefill."""
        mask = torch.full(
            (1, 1, total_len, total_len),
            fill_value=float("-inf"),
            dtype=self.dtype,
            device=self.device,
        )
        start = 0
        for length in input_lens:
            block = torch.tril(
                torch.ones((length, length), dtype=torch.bool, device=self.device)
            )
            mask[0, 0, start : start + length, start : start + length][block] = 0.0
            start += length
        return mask

    def _gather_kv_from_pool(
        self, request: Request
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        assert self.kv_pool is not None
        state = self._paged_state(request)
        kv_len = state.kv_len
        page_indices = state.page_indices
        gathered: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer_idx, (k_pages, v_pages) in enumerate(self.kv_pool.kv_caches):
            k_seq = torch.empty(
                (kv_len, self.kv_pool.num_kv_heads, self.kv_pool.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            v_seq = torch.empty_like(k_seq)
            for t in range(kv_len):
                page_offset = t // self.page_size
                slot = t % self.page_size
                page_id = page_indices[page_offset]
                k_seq[t].copy_(k_pages[page_id, slot])
                v_seq[t].copy_(v_pages[page_id, slot])
            k_contig = k_seq.transpose(0, 1).unsqueeze(0).contiguous()
            v_contig = v_seq.transpose(0, 1).unsqueeze(0).contiguous()
            gathered.append((k_contig, v_contig))
        return gathered

    def _append_last_token_kv_to_pool(
        self,
        request: Request,
        new_kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        assert self.kv_pool is not None
        state = self._paged_state(request)
        t = state.kv_len
        needed_pages = self.kv_pool.pages_needed(t + 1)
        if needed_pages > len(state.page_indices):
            state.page_indices.extend(self.kv_pool.allocate(needed_pages - len(state.page_indices)))

        page_offset = t // self.page_size
        slot = t % self.page_size
        page_id = state.page_indices[page_offset]

        for layer_idx, (k_layer, v_layer) in enumerate(new_kv_caches):
            k_last = k_layer[0, :, -1, :]
            v_last = v_layer[0, :, -1, :]
            self.kv_pool.kv_caches[layer_idx][0][page_id, slot].copy_(k_last)
            self.kv_pool.kv_caches[layer_idx][1][page_id, slot].copy_(v_last)

        state.kv_len += 1
