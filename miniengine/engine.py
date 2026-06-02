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
from miniengine.hicache import HiRadixCache
from miniengine.radix_cache import RadixCache
from miniengine.model import (
    FLASH_ATTN_AVAILABLE,
    CausalLM,
    ModelConfig,
    load_weights,
)
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
        kv_pool_mem_fraction: float | None = None,
        torch_compile: bool = False,
        torch_compile_target: str = "mlp",
        torch_compile_dynamic: bool = False,
        attention_backend: str = "auto",
       
        prefill_chunk_size: int = 0,
        disable_radix_cache: bool = False,
        cpu_cache_size_gb: float = 0.0,
        hicache_overlap: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.page_size = page_size
        self.prefill_chunk_size = prefill_chunk_size
        self.attention_backend = attention_backend
        self.radix_cache: RadixCache | None = None
        self._direct_paged_flash_supported = (
            self.attention_backend in ("auto", "flash_attn")
            and FLASH_ATTN_AVAILABLE
            and (self.page_size % 256 == 0)
        )

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
            self._enable_torch_compile(
                target=torch_compile_target,
                dynamic=torch_compile_dynamic,
            )
            self._warmup_compiled_regions()
        self.kv_pool = None
        if self.mode == "paged":
            if not FLASH_ATTN_AVAILABLE:
                logger.warning(
                    "flash-attn unavailable: paged mode is using fallback attention "
                    "paths that are typically slower and may show no throughput gain."
                )
            elif self.page_size % 256 != 0:
                logger.warning(
                    "flash-attn direct paged decode requires page_size divisible by 256; "
                    "got page_size=%d. Falling back to non-direct decode path "
                    "(gathers full KV each token — very slow on long contexts; "
                    "use --page-size 256 for production benchmarks).",
                    self.page_size,
                )
            bytes_budget = kv_pool_bytes_budget
            if bytes_budget is None and kv_pool_mem_fraction is not None:
                if device.startswith("cuda") and torch.cuda.is_available():
                    free_b, _total_b = torch.cuda.mem_get_info(device)
                    bytes_budget = int(free_b * kv_pool_mem_fraction)
                    logger.info(
                        "Paged KV budget from free VRAM: %d / %d bytes (fraction=%.3f)",
                        bytes_budget,
                        free_b,
                        kv_pool_mem_fraction,
                    )
                else:
                    logger.warning(
                        "kv_pool_mem_fraction set but device is not CUDA; "
                        "using --kv-pool-pages=%d",
                        kv_pool_num_pages,
                    )

            if bytes_budget is not None:
                self.kv_pool = KVMemoryPool.from_budget(
                    num_layers=config.num_hidden_layers,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    page_size=page_size,
                    dtype=dtype,
                    device=device,
                    bytes_budget=bytes_budget,
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
            if not disable_radix_cache:
                if cpu_cache_size_gb > 0:
                    self.radix_cache = HiRadixCache(
                        self.kv_pool,
                        cpu_cache_size_gb=cpu_cache_size_gb,
                        hicache_overlap=hicache_overlap,
                    )
                else:
                    self.radix_cache = RadixCache(self.kv_pool)
                self.kv_pool.set_radix_cache(self.radix_cache)
                logger.info("Radix prefix cache enabled")
            else:
                logger.info("Radix prefix cache disabled")

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
        # Reused paged-decode buffers to reduce per-step allocations.
        self._paged_decode_buffers: dict[str, torch.Tensor] = {}

    def paged_peak_pages_for_request(self, req: Request) -> int:
        """Pages required for full prompt + worst-case decode (max_new_tokens)."""
        assert self.kv_pool is not None
        peak_len = len(req.input_ids) + req.sampling_params.max_new_tokens
        return self.kv_pool.pages_needed(peak_len)

    def paged_kv_can_admit(self, active: list[Request], candidate: Request) -> bool:
        """True if pool can reserve worst-case pages for everyone in active plus candidate."""
        if self.mode != "paged" or self.kv_pool is None:
            return True
        total = sum(self.paged_peak_pages_for_request(r) for r in active)
        total += self.paged_peak_pages_for_request(candidate)
        if total > self.kv_pool.num_pages:
            return False
        # Peak-page accounting ignores radix sharing; also require allocatable pages.
        candidate_peak = self.paged_peak_pages_for_request(candidate)
        available = self.kv_pool.num_free + self.kv_pool.num_evictable
        return available >= candidate_peak

    def _enable_torch_compile(self, target: str, dynamic: bool) -> None:
        """Compile selected stable submodules to reduce launch overhead."""
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile unavailable in this PyTorch build; skipping")
            return
        try:
            if target == "mlp":
                for layer in self.model.model.layers:
                    layer.mlp = torch.compile(
                        layer.mlp,
                        dynamic=dynamic,
                        mode="reduce-overhead",
                    )
                logger.info(
                    "torch.compile enabled for transformer MLP submodules "
                    "(dynamic=%s)",
                    dynamic,
                )
            elif target == "block":
                for i, layer in enumerate(self.model.model.layers):
                    self.model.model.layers[i] = torch.compile(
                        layer,
                        dynamic=dynamic,
                        mode="reduce-overhead",
                    )
                logger.info(
                    "torch.compile enabled for full transformer blocks "
                    "(dynamic=%s)",
                    dynamic,
                )
            else:
                logger.warning(
                    "Unknown torch_compile_target=%s; skipping compilation",
                    target,
                )
        except Exception:
            logger.exception("Failed to enable torch.compile; continuing in eager mode")

    def _mark_compile_step_begin(self) -> None:
        """Mark a new compiled-step boundary for Inductor CUDA graphs."""
        try:
            mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
            if mark is not None:
                mark()
        except Exception:
            # Keep serving even if this marker is unavailable in the local torch build.
            pass

    @torch.inference_mode()
    def _warmup_compiled_regions(self) -> None:
        """
        Run a short synthetic warmup so benchmark TTFT isn't dominated by
        first-use compile overhead.
        """
        if not hasattr(torch, "compile"):
            return
        try:
            warmup_batches = (1, 4, 8, 16, 32)
            prefill_len = 64
            decode_cache_len = 128
            cfg = self.model.config

            for bsz in warmup_batches:
                # Prefill-like shape.
                self._mark_compile_step_begin()
                in_ids = torch.zeros(
                    (bsz, prefill_len), dtype=torch.long, device=self.device
                )
                pos_ids = torch.arange(prefill_len, device=self.device).repeat(bsz, 1)
                _logits, _ = self.model(
                    in_ids,
                    pos_ids,
                    kv_caches=None,
                    attention_backend=self.attention_backend,
                )

                # Decode-like shape.
                dec_ids = torch.zeros((bsz, 1), dtype=torch.long, device=self.device)
                dec_pos = torch.full(
                    (bsz, 1), decode_cache_len, dtype=torch.long, device=self.device
                )
                kv_caches = []
                for _ in range(cfg.num_hidden_layers):
                    k = torch.zeros(
                        (bsz, cfg.num_key_value_heads, decode_cache_len, cfg.head_dim),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    v = torch.zeros_like(k)
                    kv_caches.append((k, v))
                cache_seqlens = torch.full(
                    (bsz,), decode_cache_len, dtype=torch.int32, device=self.device
                )
                self._mark_compile_step_begin()
                _logits, _ = self.model(
                    dec_ids,
                    dec_pos,
                    kv_caches=kv_caches,
                    cache_seqlens=cache_seqlens,
                    attention_backend=self.attention_backend,
                )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
            logger.info(
                "torch.compile warmup complete for batch sizes %s", warmup_batches
            )
        except Exception:
            logger.exception("torch.compile warmup failed; continuing")

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

        self._mark_compile_step_begin()
        logits, kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=None,
            attention_backend=self.attention_backend,
        )

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
    def prefill_batch(self, requests: list[Request]) -> list[int | None]:
        """Prefill a batch of requests. Paged mode uses one model forward.

        Returns one entry per request: ``None`` while chunked prefill is in
        progress, otherwise the first sampled token id.
        """
        if not requests:
            return []
        if self.mode != "paged":
            return [self.prefill(req) for req in requests]
        if self.prefill_chunk_size <= 0:
            return self._prefill_batch_oneshot(requests)
        return self._prefill_batch_chunked(requests)

    @torch.inference_mode()
    def _prefill_batch_oneshot(self, requests: list[Request]) -> list[int]:
        """Single-shot packed prefill; radix cache skips cached prefix tokens."""
        assert self.kv_pool is not None
        results: list[int | None] = [None] * len(requests)
        packed_reqs: list[Request] = []

        for i, req in enumerate(requests):
            if req.kv_cache is None:
                self._paged_prefill_init(req)
            if req.prefill_done:
                results[i] = self._sample_first_token_cached_prompt(req)
            elif self._paged_state(req).kv_len > 0:
                results[i] = self._prefill_continuation_chunk(req)
            else:
                packed_reqs.append((i, req))

        if packed_reqs:
            reqs_only = [r for _, r in packed_reqs]
            packed_results = self._prefill_packed_chunk_step(
                reqs_only, chunk_size=None
            )
            for (i, _), tok in zip(packed_reqs, packed_results):
                results[i] = tok

        return [int(t) for t in results]

    @torch.inference_mode()
    def _prefill_batch_chunked(self, requests: list[Request]) -> list[int | None]:
        """Cap per-step query tokens at ``prefill_chunk_size`` per request."""
        assert self.kv_pool is not None
        out: list[int | None] = [None] * len(requests)
        packed_reqs: list[tuple[int, Request]] = []
        cont_reqs: list[tuple[int, Request]] = []

        for i, req in enumerate(requests):
            if req.kv_cache is None:
                self._paged_prefill_init(req)
            if req.prefill_done:
                out[i] = self._sample_first_token_cached_prompt(req)
            elif self._paged_state(req).kv_len > 0:
                cont_reqs.append((i, req))
            else:
                packed_reqs.append((i, req))

        if packed_reqs:
            reqs_only = [r for _, r in packed_reqs]
            packed_results = self._prefill_packed_chunk_step(reqs_only)
            for (i, _), tok in zip(packed_reqs, packed_results):
                out[i] = tok

        for i, req in cont_reqs:
            out[i] = self._prefill_continuation_chunk(req)

        return out

    @torch.inference_mode()
    def _prefill_packed_chunk_step(
        self, requests: list[Request], chunk_size: int | None = None
    ) -> list[int | None]:
        """One packed varlen forward for the next slice (no radix prefix in KV)."""
        if chunk_size is None:
            chunk_size = self.prefill_chunk_size or max(
                len(r.input_ids) - r.prefill_offset for r in requests
            )
        chunk_lens: list[int] = []
        packed_ids: list[int] = []
        packed_pos: list[int] = []
        for req in requests:
            remaining = len(req.input_ids) - req.prefill_offset
            clen = min(remaining, chunk_size)
            chunk_lens.append(clen)
            if clen == 0:
                continue
            start = req.prefill_offset
            packed_ids.extend(req.input_ids[start : start + clen])
            packed_pos.extend(range(start, start + clen))

        results: list[int | None] = [None] * len(requests)
        active: list[tuple[int, Request, int]] = []
        for i, (req, clen) in enumerate(zip(requests, chunk_lens)):
            if clen == 0:
                results[i] = self._sample_first_token_cached_prompt(req)
            else:
                active.append((i, req, clen))

        if not active:
            return results

        input_lens = [clen for _, _, clen in active]
        total_len = len(packed_ids)
        input_ids = torch.tensor([packed_ids], dtype=torch.long, device=self.device)
        position_ids = torch.tensor([packed_pos], dtype=torch.long, device=self.device)
        cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(input_lens), dim=0).tolist()),
            dtype=torch.int32,
            device=self.device,
        )
        attention_mask = None
        if not FLASH_ATTN_AVAILABLE:
            attention_mask = self._build_packed_prefill_attention_mask(
                input_lens, total_len
            )

        sample_last = [clen == len(req.input_ids) - req.prefill_offset for _, req, clen in active]
        logits_positions = None
        if any(sample_last):
            last_positions: list[int] = []
            start = 0
            for clen in input_lens:
                last_positions.append(start + clen - 1)
                start += clen
            logits_positions = torch.tensor(
                last_positions, dtype=torch.long, device=self.device
            )

        self._mark_compile_step_begin()
        logits, kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=None,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max(input_lens),
            attention_backend=self.attention_backend,
            logits_positions=logits_positions,
        )

        start = 0
        for idx, (i, req, clen) in enumerate(active):
            state = self._paged_state(req)
            self._store_prefill_slice_to_pool(
                state, kv_caches, src_start=start, src_len=clen
            )
            state.kv_len += clen
            req.prefill_offset += clen
            start += clen

            if req.prefill_done and sample_last[idx]:
                results[i] = sample_token(
                    logits[:, idx, :],
                    req.sampling_params,
                    req.output_ids,
                )

        return results

    @torch.inference_mode()
    def _prefill_continuation_chunk(self, req: Request) -> int | None:
        """Prefill the next chunk when KV already lives in the pool."""
        if req.prefill_done:
            return self._sample_first_token_cached_prompt(req)

        chunk_size = self.prefill_chunk_size
        if chunk_size <= 0:
            chunk_size = len(req.input_ids) - req.prefill_offset
        remaining = len(req.input_ids) - req.prefill_offset
        clen = min(remaining, chunk_size)
        if clen == 0:
            return self._sample_first_token_cached_prompt(req)

        state = self._paged_state(req)
        cache_len = state.kv_len
        chunk_ids = req.input_ids[req.prefill_offset : req.prefill_offset + clen]
        input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=self.device)
        position_ids = torch.arange(
            cache_len, cache_len + clen, device=self.device
        ).unsqueeze(0)

        kv_caches = self._gather_kv_from_pool(req)
        attention_mask = self._build_continuation_attention_mask(cache_len, clen)

        self._mark_compile_step_begin()
        logits, new_kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=kv_caches,
            attention_mask=attention_mask,
            return_last_kv_only=True,
            attention_backend=self.attention_backend,
        )

        self._store_prefill_slice_to_pool(
            state, new_kv_caches, src_start=0, src_len=clen
        )
        state.kv_len += clen
        req.prefill_offset += clen

        if req.prefill_done:
            return sample_token(
                logits[:, -1, :], req.sampling_params, req.output_ids
            )
        return None

    def _rollback_paged_prefill_init(
        self,
        req: Request,
        *,
        new_pages: list[int],
    ) -> None:
        """Undo partial prefill setup after allocation or prefill failure."""
        assert self.kv_pool is not None
        if new_pages:
            self.kv_pool.free(new_pages)
        if self.radix_cache is not None and req.radix_lock_node is not None:
            self.radix_cache.dec_lock_ref(req.radix_lock_node)
        req.radix_lock_node = None
        req.kv_cache = None
        req.prefill_offset = 0
        req.cache_hit_tokens = 0

    def _paged_prefill_init(self, req: Request) -> None:
        """Radix lookup and page allocation before the first prefill chunk."""
        assert self.kv_pool is not None
        matched_pages: list[int] = []
        last_node = None
        req.cache_hit_tokens = 0
        req.prefill_offset = 0
        req.radix_lock_node = None

        if self.radix_cache is not None:
            mr = self.radix_cache.match_prefix(req.input_ids)
            req.cache_hit_tokens = mr.matched_tokens
            matched_pages = list(mr.matched_pages)
            last_node = mr.last_node
            if last_node is not None and last_node is not self.radix_cache.root:
                self.radix_cache.inc_lock_ref(last_node)
                req.radix_lock_node = last_node
            req.prefill_offset = mr.matched_tokens

        peak_pages = self.paged_peak_pages_for_request(req)
        extra_pages = max(0, peak_pages - len(matched_pages))
        new_pages: list[int] = []
        try:
            if extra_pages > 0:
                new_pages = self.kv_pool.allocate(extra_pages)
            req.kv_cache = PagedRequestState(
                page_indices=matched_pages + new_pages,
                kv_len=req.prefill_offset,
            )
        except RuntimeError:
            self._rollback_paged_prefill_init(req, new_pages=new_pages)
            raise

    @torch.inference_mode()
    def _sample_first_token_cached_prompt(self, req: Request) -> int:
        """Sample first token when the full prompt is already in KV."""
        state = self._paged_state(req)
        state.kv_len = len(req.input_ids)
        req.prefill_offset = len(req.input_ids)

        last_pos = len(req.input_ids) - 1
        input_ids = torch.tensor(
            [[req.input_ids[-1]]], dtype=torch.long, device=self.device
        )
        position_ids = torch.tensor([[last_pos]], dtype=torch.long, device=self.device)
        kv_caches = self._gather_kv_from_pool(req)
        cache_seqlens = torch.tensor([last_pos], dtype=torch.int32, device=self.device)

        self._mark_compile_step_begin()
        logits, _ = self.model(
            input_ids,
            position_ids,
            kv_caches=kv_caches,
            return_last_kv_only=True,
            cache_seqlens=cache_seqlens,
            attention_backend=self.attention_backend,
        )
        return sample_token(
            logits[:, -1, :], req.sampling_params, req.output_ids
        )

    def _build_continuation_attention_mask(
        self, cache_len: int, chunk_len: int
    ) -> torch.Tensor:
        """Causal mask for a new chunk attending to cached prefix and itself."""
        kv_len = cache_len + chunk_len
        mask = torch.full(
            (1, 1, chunk_len, kv_len),
            fill_value=float("-inf"),
            dtype=self.dtype,
            device=self.device,
        )
        for i in range(chunk_len):
            mask[0, 0, i, : cache_len + i + 1] = 0.0
        return mask

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
        cache_seqlens = torch.tensor([cache_len], dtype=torch.int32, device=self.device)

        self._mark_compile_step_begin()
        logits, new_kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=kv_caches,
            return_last_kv_only=(self.mode == "paged"),
            cache_seqlens=cache_seqlens,
            attention_backend=self.attention_backend,
        )
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

        cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device=self.device)
        attention_mask = None
        if not FLASH_ATTN_AVAILABLE:
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
            cache_seqlens=cache_seqlens,
            attention_backend=self.attention_backend,
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
        assert self.kv_pool is not None
        states = [self._paged_state(req) for req in requests]
        cache_lens = [st.kv_len for st in states]
        max_cache_len = max(cache_lens)
        input_ids = self._get_paged_decode_buffer(
            "input_ids",
            shape=(batch_size, 1),
            dtype=torch.long,
        )
        position_ids = self._get_paged_decode_buffer(
            "position_ids",
            shape=(batch_size, 1),
            dtype=torch.long,
        )
        cache_seqlens = self._get_paged_decode_buffer(
            "cache_seqlens",
            shape=(batch_size,),
            dtype=torch.int32,
        )
        input_ids[:, 0] = torch.tensor(
            [req.output_ids[-1] for req in requests],
            dtype=torch.long,
            device=self.device,
        )
        position_ids[:, 0] = torch.tensor(
            cache_lens,
            dtype=torch.long,
            device=self.device,
        )
        cache_seqlens[:] = torch.tensor(
            cache_lens,
            dtype=torch.int32,
            device=self.device,
        )
        use_direct_paged_flash = self._direct_paged_flash_supported
        new_kv_caches = None
        if use_direct_paged_flash:
            # Ensure each request has capacity for the new decode token.
            for st in states:
                needed_pages = self.kv_pool.pages_needed(st.kv_len + 1)
                if needed_pages > len(st.page_indices):
                    st.page_indices.extend(
                        self.kv_pool.allocate(needed_pages - len(st.page_indices))
                    )

            max_pages = max(len(st.page_indices) for st in states)
            block_table = self._get_paged_decode_buffer(
                "block_table",
                shape=(batch_size, max_pages),
                dtype=torch.int32,
            )
            block_table.zero_()
            for i, st in enumerate(states):
                row_len = len(st.page_indices)
                if row_len:
                    # Avoid per-request temporary CUDA tensor creation.
                    block_table[i, :row_len] = torch.tensor(
                        st.page_indices,
                        dtype=torch.int32,
                        device=self.device,
                    )

            self._mark_compile_step_begin()
            logits, _ = self.model(
                input_ids,
                position_ids,
                kv_caches=self.kv_pool.kv_caches,
                attention_mask=None,
                return_last_kv_only=True,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                flash_paged_decode=True,
                attention_backend=self.attention_backend,
            )
            for st in states:
                st.kv_len += 1
        else:
            padded_kv_caches = self._build_batched_kv_from_pool(
                requests, cache_lens, max_cache_len
            )
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

            self._mark_compile_step_begin()
            logits, new_kv_caches = self.model(
                input_ids,
                position_ids,
                kv_caches=padded_kv_caches,
                attention_mask=attention_mask,
                return_last_kv_only=True,
                cache_seqlens=cache_seqlens,
                attention_backend=self.attention_backend,
            )
            self._append_batched_last_token_kv_to_pool(requests, new_kv_caches)

        token_ids: list[int] = []
        for i, req in enumerate(requests):
            token_ids.append(
                sample_token(
                    logits[i : i + 1, -1, :], req.sampling_params, req.output_ids
                )
            )
        return token_ids

    def _get_paged_decode_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get or grow a reusable CUDA buffer for paged decode bookkeeping."""
        buf = self._paged_decode_buffers.get(name)
        if (
            buf is None
            or buf.dtype != dtype
            or tuple(buf.shape) != shape
            or str(buf.device) != str(self.device)
        ):
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            self._paged_decode_buffers[name] = buf
        return buf

    def free_request_kv(self, request: Request) -> None:
        """Return a request's pages to the pool (paged mode only)."""
        if self.mode != "paged" or request.kv_cache is None:
            return
        state = self._paged_state(request)
        assert self.kv_pool is not None

        if self.radix_cache is not None:
            self._radix_cache_commit_request(request, state)

        else:
            self.kv_pool.free(state.page_indices)

        request.kv_cache = None
        request.radix_lock_node = None

    def _radix_cache_commit_request(
        self, request: Request, state: PagedRequestState
    ) -> None:
        """Insert finished sequence into radix cache; release duplicate pages."""
        assert self.radix_cache is not None
        tokens = request.input_ids + request.output_ids
        committed_len = len(tokens)
        aligned_len = (committed_len // self.page_size) * self.page_size
        if aligned_len > 0:
            num_pages = self.kv_pool.pages_needed(aligned_len)
            page_slice = state.page_indices[:num_pages]
            _, redundant = self.radix_cache.insert_and_return(
                tokens[:aligned_len], page_slice
            )
            if redundant:
                self.kv_pool.free(redundant)
        extra = state.page_indices[self.kv_pool.pages_needed(aligned_len) :]
        if extra:
            self.kv_pool.free(extra)
        if request.radix_lock_node is not None:
            self.radix_cache.dec_lock_ref(request.radix_lock_node)

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
        peak_pages = self.paged_peak_pages_for_request(request)
        page_indices = self.kv_pool.allocate(peak_pages)
        state = PagedRequestState(page_indices=page_indices, kv_len=kv_len)

        self._store_prefill_slice_to_pool(state, kv_caches, src_start=0, src_len=kv_len)
        request.kv_cache = state

    def _store_prefill_slice_to_pool(
        self,
        state: PagedRequestState,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        src_start: int,
        src_len: int,
    ) -> None:
        """Store a contiguous token slice from model KV output into paged pool."""
        assert self.kv_pool is not None
        if src_len == 0:
            return

        written = 0
        while written < src_len:
            dst_token = written
            page_offset = dst_token // self.page_size
            slot = dst_token % self.page_size
            page_id = state.page_indices[page_offset]
            to_copy = min(src_len - written, self.page_size - slot)
            src_slice = slice(src_start + written, src_start + written + to_copy)
            dst_slice = slice(slot, slot + to_copy)

            for layer_idx, (k_layer, v_layer) in enumerate(kv_caches):
                # Input from model is (1, kv_heads, seq, head_dim); pool pages are
                # (num_pages, page_size, kv_heads, head_dim).
                k_src = k_layer[0, :, src_slice, :].transpose(0, 1)
                v_src = v_layer[0, :, src_slice, :].transpose(0, 1)
                self.kv_pool.kv_caches[layer_idx][0][page_id, dst_slice].copy_(k_src)
                self.kv_pool.kv_caches[layer_idx][1][page_id, dst_slice].copy_(v_src)

            written += to_copy

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

    def _build_batched_kv_from_pool(
        self,
        requests: list[Request],
        cache_lens: list[int],
        max_cache_len: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Gather pooled KV for all requests without token-level Python loops."""
        assert self.kv_pool is not None
        batch_size = len(requests)
        if max_cache_len == 0:
            return [
                (
                    torch.empty(
                        (batch_size, self.kv_pool.num_kv_heads, 0, self.kv_pool.head_dim),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    torch.empty(
                        (batch_size, self.kv_pool.num_kv_heads, 0, self.kv_pool.head_dim),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                )
                for _ in range(self.model.config.num_hidden_layers)
            ]

        max_pages = max(
            self.kv_pool.pages_needed(max_cache_len),
            max((len(self._paged_state(req).page_indices) for req in requests), default=0),
        )
        page_table = torch.full(
            (batch_size, max_pages),
            fill_value=-1,
            dtype=torch.long,
            device=self.device,
        )
        for i, req in enumerate(requests):
            page_indices = self._paged_state(req).page_indices
            page_table[i, : len(page_indices)] = torch.tensor(
                page_indices, dtype=torch.long, device=self.device
            )

        token_idx = torch.arange(max_cache_len, device=self.device, dtype=torch.long)
        page_offsets = token_idx // self.page_size
        slots = token_idx % self.page_size
        page_ids = page_table[:, page_offsets]
        flat_idx = page_ids * self.page_size + slots.unsqueeze(0)

        valid_mask = (
            token_idx.unsqueeze(0)
            < torch.tensor(cache_lens, device=self.device, dtype=torch.long).unsqueeze(1)
        )
        safe_flat_idx = flat_idx.masked_fill(~valid_mask, 0)

        gathered: list[tuple[torch.Tensor, torch.Tensor]] = []
        for k_pages, v_pages in self.kv_pool.kv_caches:
            k_flat = k_pages.reshape(
                self.kv_pool.num_pages * self.page_size,
                self.kv_pool.num_kv_heads,
                self.kv_pool.head_dim,
            )
            v_flat = v_pages.reshape_as(k_flat)

            k_seq = k_flat[safe_flat_idx]
            v_seq = v_flat[safe_flat_idx]

            k_seq = k_seq * valid_mask[..., None, None]
            v_seq = v_seq * valid_mask[..., None, None]

            gathered.append(
                (
                    k_seq.permute(0, 2, 1, 3).contiguous(),
                    v_seq.permute(0, 2, 1, 3).contiguous(),
                )
            )
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

    def _append_batched_last_token_kv_to_pool(
        self,
        requests: list[Request],
        new_kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Append one decoded token per request into paged KV in one batched write."""
        assert self.kv_pool is not None
        if not requests:
            return

        page_ids: list[int] = []
        slots: list[int] = []
        for req in requests:
            state = self._paged_state(req)
            t = state.kv_len
            needed_pages = self.kv_pool.pages_needed(t + 1)
            if needed_pages > len(state.page_indices):
                state.page_indices.extend(
                    self.kv_pool.allocate(needed_pages - len(state.page_indices))
                )
            page_offset = t // self.page_size
            page_ids.append(state.page_indices[page_offset])
            slots.append(t % self.page_size)
            state.kv_len += 1

        page_ids_t = torch.tensor(page_ids, dtype=torch.long, device=self.device)
        slots_t = torch.tensor(slots, dtype=torch.long, device=self.device)

        for layer_idx, (k_layer, v_layer) in enumerate(new_kv_caches):
            k_last = k_layer[:, :, -1, :]
            v_last = v_layer[:, :, -1, :]
            self.kv_pool.kv_caches[layer_idx][0][page_ids_t, slots_t].copy_(k_last)
            self.kv_pool.kv_caches[layer_idx][1][page_ids_t, slots_t].copy_(v_last)
