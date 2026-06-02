"""Hierarchical radix prefix cache with GPU and CPU KV tiers."""

from __future__ import annotations

import heapq
import logging
import time
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch

from miniengine.radix_cache import (
    MatchResult,
    RadixCache,
    RadixNode,
    _child_key,
    _match_keys,
    _page_align_len,
)

if TYPE_CHECKING:
    from miniengine.kv_memory_pool import KVMemoryPool

logger = logging.getLogger(__name__)


class CacheTier(Enum):
    GPU = auto()
    CPU = auto()


class HiRadixNode(RadixNode):
    """Radix node with an explicit storage tier for its KV pages."""

    __slots__ = ("tier",)

    def __init__(self) -> None:
        super().__init__()
        self.tier = CacheTier.GPU


def _node_tier(node: RadixNode) -> CacheTier:
    return getattr(node, "tier", CacheTier.GPU)


class CPUKVPool:
    """Pinned host-memory KV page pool (same page layout as ``KVMemoryPool``)."""

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        self._num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        page_shape = (num_pages, page_size, num_kv_heads, head_dim)
        self._k_caches = [
            torch.zeros(page_shape, dtype=dtype, device="cpu", pin_memory=True)
            for _ in range(num_layers)
        ]
        self._v_caches = [
            torch.zeros(page_shape, dtype=dtype, device="cpu", pin_memory=True)
            for _ in range(num_layers)
        ]
        self._free = list(range(num_pages))

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @property
    def kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return list(zip(self._k_caches, self._v_caches))

    def allocate(self, num_pages: int) -> list[int]:
        if num_pages == 0:
            return []
        if num_pages > len(self._free):
            return []
        return [self._free.pop() for _ in range(num_pages)]

    def free(self, page_indices: list[int]) -> None:
        self._free.extend(page_indices)

    @classmethod
    def from_budget(
        cls,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        bytes_budget: int,
    ) -> CPUKVPool:
        bytes_per_element = torch.empty((), dtype=dtype).element_size()
        bytes_one_kv_plane = page_size * num_kv_heads * head_dim * bytes_per_element
        bytes_per_page_all_layers = num_layers * 2 * bytes_one_kv_plane
        num_pages = int(bytes_budget // bytes_per_page_all_layers)
        if num_pages < 1:
            raise ValueError(
                f"CPU KV bytes_budget={bytes_budget} too small for one page "
                f"(need ~{bytes_per_page_all_layers} bytes/page across layers)"
            )
        return cls(
            num_pages=num_pages,
            page_size=page_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

    @classmethod
    def from_gib(cls, gib: float, **kwargs) -> CPUKVPool:
        return cls.from_budget(bytes_budget=int(gib * (1024**3)), **kwargs)


class HiRadixCache(RadixCache):
    """Radix prefix cache with optional CPU backing store under GPU eviction."""

    def __init__(
        self,
        pool: "KVMemoryPool",
        *,
        cpu_cache_size_gb: float = 0.0,
        hicache_overlap: bool = False,
    ) -> None:
        super().__init__(pool)
        self.root = HiRadixNode()
        self.cpu_pool: CPUKVPool | None = None
        self._hicache_enabled = cpu_cache_size_gb > 0
        self._copy_stream: torch.cuda.Stream | None = None

        if self._hicache_enabled:
            self.cpu_pool = CPUKVPool.from_gib(
                cpu_cache_size_gb,
                num_layers=pool.num_layers,
                num_kv_heads=pool.num_kv_heads,
                head_dim=pool.head_dim,
                page_size=pool.page_size,
                dtype=pool.dtype,
            )
            if hicache_overlap and pool.device.startswith("cuda"):
                self._copy_stream = torch.cuda.Stream(device=pool.device)
            logger.info(
                "HiCache enabled — cpu_pages=%d (%.2f GiB budget), overlap=%s",
                self.cpu_pool.num_pages,
                cpu_cache_size_gb,
                hicache_overlap,
            )

    @property
    def hicache_enabled(self) -> bool:
        return self._hicache_enabled and self.cpu_pool is not None

    def match_prefix(self, tokens: list[int]) -> MatchResult:
        self.metrics.total_lookups += 1
        self.metrics.total_query_tokens += len(tokens)
        aligned_len = _page_align_len(len(tokens), self.page_size)
        if aligned_len == 0:
            return MatchResult(last_node=self.root)

        tokens = tokens[:aligned_len]
        now = time.monotonic()
        node = self.root
        node.last_access = now
        matched_pages: list[int] = []
        idx = 0

        while idx < len(tokens):
            child_key = _child_key(tokens[idx:], self.page_size)
            child = node.children.get(child_key)
            if child is None:
                break

            child.last_access = now
            match_len = _match_keys(child.key, tokens[idx:], self.page_size)
            if match_len < len(child.key):
                match_len = _page_align_len(match_len, self.page_size)
                if match_len == 0:
                    break
                n_pages = match_len // self.page_size
                self._ensure_gpu_pages(child)
                matched_pages.extend(child.pages[:n_pages])
                idx += match_len
                break

            self._ensure_gpu_pages(child)
            matched_pages.extend(child.pages)
            idx += match_len
            node = child

        self.metrics.total_hit_tokens += idx
        return MatchResult(
            matched_pages=matched_pages,
            matched_tokens=idx,
            last_node=node,
        )

    def evict(self, n_pages_needed: int) -> int:
        if n_pages_needed <= 0:
            return 0

        if not self.hicache_enabled:
            return super().evict(n_pages_needed)

        leaves = self._collect_evictable_gpu_leaves()
        heap: list[tuple[float, int, RadixNode]] = [
            (leaf.last_access, id(leaf), leaf) for leaf in leaves
        ]
        heapq.heapify(heap)
        freed = 0

        while freed < n_pages_needed and heap:
            _, _, node = heapq.heappop(heap)
            if (
                node.ref_count > 0
                or node is self.root
                or not node.pages
                or _node_tier(node) is not CacheTier.GPU
            ):
                continue

            parent = node.parent
            if parent is None:
                continue

            child_key = _child_key(node.key, self.page_size)
            if parent.children.get(child_key) is not node:
                continue

            n = len(node.pages)
            if self._demote_node(node):
                freed += n
                self.metrics.total_evicted_pages += n
                continue

            # CPU pool full — drop the subtree entry.
            self.pool.free(node.pages)
            freed += n
            self._cached_page_count -= n
            self.metrics.total_evicted_pages += n
            del parent.children[child_key]

            if (
                parent is not self.root
                and not parent.children
                and parent.ref_count == 0
                and parent.pages
                and _node_tier(parent) is CacheTier.GPU
            ):
                heapq.heappush(heap, (parent.last_access, id(parent), parent))

        return freed

    def reset(self) -> None:
        gpu_pages: list[int] = []
        cpu_pages: list[int] = []

        def walk(node: RadixNode) -> None:
            if not node.pages:
                pass
            elif _node_tier(node) is CacheTier.CPU:
                cpu_pages.extend(node.pages)
            else:
                gpu_pages.extend(node.pages)
            for child in list(node.children.values()):
                walk(child)

        walk(self.root)
        if gpu_pages:
            self.pool.free(gpu_pages)
        if cpu_pages and self.cpu_pool is not None:
            self.cpu_pool.free(cpu_pages)
        self.root = HiRadixNode()
        self._cached_page_count = 0

    def num_evictable_pages(self) -> int:
        return self._count_pages(self.root, locked=False, tier=CacheTier.GPU)

    def _ensure_gpu_pages(self, node: RadixNode) -> None:
        if not node.pages or _node_tier(node) is CacheTier.GPU:
            return
        self._promote_node(node)

    def _promote_node(self, node: RadixNode) -> None:
        assert self.cpu_pool is not None
        cpu_pages = list(node.pages)
        gpu_pages = self.pool.allocate(len(cpu_pages))
        self._copy_kv(cpu_pages, gpu_pages, src_is_cpu=True)
        self.cpu_pool.free(cpu_pages)
        node.pages = gpu_pages
        node.tier = CacheTier.GPU

    def _demote_node(self, node: RadixNode) -> bool:
        assert self.cpu_pool is not None
        if _node_tier(node) is CacheTier.CPU:
            return False
        cpu_pages = self.cpu_pool.allocate(len(node.pages))
        if not cpu_pages:
            return False
        gpu_pages = list(node.pages)
        self._copy_kv(gpu_pages, cpu_pages, src_is_cpu=False)
        self.pool.free(gpu_pages)
        node.pages = cpu_pages
        node.tier = CacheTier.CPU
        return True

    def _copy_kv(
        self,
        src_pages: list[int],
        dst_pages: list[int],
        *,
        src_is_cpu: bool,
    ) -> None:
        assert len(src_pages) == len(dst_pages)
        if src_is_cpu:
            src_pool, dst_pool = self.cpu_pool, self.pool
        else:
            src_pool, dst_pool = self.pool, self.cpu_pool
        assert src_pool is not None and dst_pool is not None

        non_blocking = self._copy_stream is not None
        stream = self._copy_stream
        ctx = torch.cuda.stream(stream) if stream is not None else None
        if ctx is not None:
            ctx.__enter__()

        try:
            for layer in range(self.pool.num_layers):
                src_k, src_v = src_pool.kv_caches[layer]
                dst_k, dst_v = dst_pool.kv_caches[layer]
                for sp, dp in zip(src_pages, dst_pages):
                    if src_is_cpu:
                        dst_k[dp].copy_(src_k[sp], non_blocking=non_blocking)
                        dst_v[dp].copy_(src_v[sp], non_blocking=non_blocking)
                    else:
                        dst_k[dp].copy_(src_k[sp], non_blocking=non_blocking)
                        dst_v[dp].copy_(src_v[sp], non_blocking=non_blocking)
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
            if stream is not None:
                stream.synchronize()

    def _collect_evictable_gpu_leaves(self) -> list[RadixNode]:
        leaves: list[RadixNode] = []

        def dfs(node: RadixNode) -> None:
            if (
                node is not self.root
                and not node.children
                and node.ref_count == 0
                and node.pages
                and _node_tier(node) is CacheTier.GPU
            ):
                leaves.append(node)
            for child in node.children.values():
                dfs(child)

        dfs(self.root)
        return leaves

    def _count_pages(
        self,
        node: RadixNode,
        *,
        locked: bool,
        tier: CacheTier | None = None,
    ) -> int:
        total = 0
        if node is not self.root and node.pages:
            node_tier = _node_tier(node)
            if tier is None or node_tier is tier:
                if locked:
                    if node.ref_count > 0:
                        total += len(node.pages)
                elif node.ref_count == 0:
                    total += len(node.pages)
        for child in node.children.values():
            total += self._count_pages(child, locked=locked, tier=tier)
        return total
