"""Pre-allocated paged KV cache memory pool — Milestone 2, Part A.

This is a SKELETON. Implement the methods below.

The pool owns a fixed amount of GPU memory, divided into equal-size
**pages**. Each page holds the KV state for `page_size` tokens for one
layer. Requests acquire pages as their KV grows and return them when
they finish; the cache itself never reallocates.

Storage layout (page-major vs token-major, contiguous K+V vs separate,
shape conventions, etc.) is YOUR design decision — pick something and
document the tradeoffs.

Storage layout: page-major per layer. Each of K and V is shaped as
(num_pages, page_size, num_kv_heads, head_dim).
"""

from __future__ import annotations

import torch
import math


class KVMemoryPool:
    """Pre-allocated paged KV cache pool.

    Args:
        num_pages:    Total pages in the pool (capacity).
        page_size:    Tokens per page. Tunable knob — exposed as
                      `--page-size` on the CLI. Smaller = less
                      fragmentation, bigger page tables; larger = the
                      opposite.
        num_layers:   Number of transformer layers.
        num_kv_heads: KV heads per layer (GQA).
        head_dim:     Per-head dimension.
        dtype:        KV dtype (typically bfloat16).
        device:       e.g. "cuda".
    """

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        self._num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        page_shape = (num_pages, page_size, num_kv_heads, head_dim)

        self._k_caches = [
            torch.zeros(page_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self._v_caches = [
            torch.zeros(page_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        # Free list of physical pages.
        self._free = list(range(num_pages))

    def allocate(self, num_pages: int) -> list[int]:
        """Reserve `num_pages` pages and return their indices.

        Raises if the pool cannot satisfy the request.
        """
        if num_pages == 0:
            return []
        if num_pages > len(self._free):
            raise RuntimeError(f"KV memory pool OOM: need {num_pages} pages, only {len(self._free)} free")

        return [self._free.pop() for _ in range(num_pages)]

    def free(self, page_indices: list[int]) -> None:
        """Return the listed pages to the free pool."""
        self._free.extend(page_indices)

    def pages_needed(self, seq_len: int) -> int:
        """How many pages are required to store `seq_len` tokens."""
        return math.ceil(seq_len / self.page_size)

    @property
    def num_free(self) -> int:
        """Pages currently available for allocation."""
        return len(self._free)

    @property
    def num_pages(self) -> int:
        """Total number of pages in the pool."""
        return self._num_pages

    @property
    def kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer (K, V) cache tensors.

        The attention path holds references to these and indexes into
        them via per-request page tables. The exact shape is up to your
        design — but it must be STABLE: no reallocation, no resizing,
        no swapping out the tensors after construction.
        """
        return list(zip(self._k_caches, self._v_caches))

    @classmethod
    def from_budget(
        cls,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        bytes_budget: int,
    ) -> KVMemoryPool:
        """Convenience: derive `num_pages` from a memory budget."""
        bytes_per_element = torch.empty((), dtype=dtype, device=device).element_size()
        bytes_one_kv_plane = page_size * num_kv_heads * head_dim * bytes_per_element
        bytes_per_page_all_layers = num_layers * 2 * bytes_one_kv_plane

        num_pages = int(bytes_budget // bytes_per_page_all_layers)
        if num_pages < 1:
            raise ValueError(
                f"KV bytes_budget={bytes_budget} too small for one page "
                f"(need ~{bytes_per_page_all_layers} bytes/page across layers)"
            )

        return cls(
            num_pages=num_pages,
            page_size=page_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
