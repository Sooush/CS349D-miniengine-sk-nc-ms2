"""Radix-tree prefix cache"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miniengine.kv_memory_pool import KVMemoryPool

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Aggregate cache statistics — surfaced via ``/cache_stats``."""

    total_lookups: int = 0
    total_query_tokens: int = 0
    total_hit_tokens: int = 0
    total_inserted_pages: int = 0
    total_evicted_pages: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_query_tokens == 0:
            return 0.0
        return self.total_hit_tokens / self.total_query_tokens


class RadixNode:
    """Radix-tree node owning a page-aligned token edge and KV pages."""

    __slots__ = ("parent", "children", "key", "pages", "ref_count", "last_access")

    def __init__(self) -> None:
        self.parent: RadixNode | None = None
        self.children: dict[tuple[int, ...], RadixNode] = {}
        self.key: list[int] = []
        self.pages: list[int] = []
        self.ref_count: int = 0
        self.last_access: float = time.monotonic()


@dataclass
class MatchResult:
    """Result of a prefix lookup.

    ``matched_tokens`` is page-aligned (multiple of ``page_size``);
    ``matched_pages`` carries the KV pages for those tokens.
    ``last_node`` is the deepest node the walk reached — callers lock it
    (``inc_lock_ref``) for the lifetime of the borrowing request.
    """

    matched_pages: list[int] = field(default_factory=list)
    matched_tokens: int = 0
    last_node: RadixNode | None = None


class RadixCache:
    """Token-prefix → KV-pages cache backed by a radix tree.

    Required behaviours (see milestone 3 doc):
      * page-aligned matching — never return a partial-page result
      * LRU eviction of unlocked subtrees
      * eviction-on-allocate: ``KVMemoryPool.allocate`` should call
        ``cache.evict(n)`` when the free list is short
      * ``inc_lock_ref`` / ``dec_lock_ref`` protect in-flight requests
        (same names as sglang's radix cache).
    """

    def __init__(self, pool: "KVMemoryPool") -> None:
        self.pool = pool
        self.page_size = pool.page_size
        self.root = RadixNode()
        self.metrics = CacheMetrics()
        self._cached_page_count = 0

    @property
    def num_cached_pages(self) -> int:
        return self._cached_page_count

    def num_evictable_pages(self) -> int:
        """Pages that an LRU sweep could free right now."""
        return self._count_pages(self.root, locked=False)

    # ── Lookup ─────────────────────────────────────────────────────────

    def match_prefix(self, tokens: list[int]) -> MatchResult:
        """Find the longest page-aligned prefix of ``tokens`` in the tree."""
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
                matched_pages.extend(child.pages[: match_len // self.page_size])
                idx += match_len
                break

            matched_pages.extend(child.pages)
            idx += match_len
            node = child

        self.metrics.total_hit_tokens += idx
        return MatchResult(
            matched_pages=matched_pages,
            matched_tokens=idx,
            last_node=node,
        )

    def inc_lock_ref(self, node: RadixNode | None) -> None:
        """Lock ``node`` (and the path to root) against eviction."""
        if node is None:
            return
        while node is not self.root:
            node.ref_count += 1
            node = node.parent

    def dec_lock_ref(self, node: RadixNode | None) -> None:
        """Release a lock.  Refresh ``last_access`` while walking."""
        if node is None:
            return
        now = time.monotonic()
        while node is not self.root:
            node.ref_count -= 1
            node.last_access = now
            node = node.parent

    def insert_and_return(
        self, tokens: list[int], pages: list[int]
    ) -> tuple[RadixNode, list[int]]:
        """Insert (tokens, pages) into the tree."""
        aligned_len = _page_align_len(len(tokens), self.page_size)
        if aligned_len == 0:
            return self.root, list(pages)

        # ── Eviction ───────────────────────────────────────────────────────
        tokens = tokens[:aligned_len]
        num_pages = aligned_len // self.page_size
        if len(pages) < num_pages:
            raise ValueError(
                f"insert needs {num_pages} pages for {aligned_len} tokens, got {len(pages)}"
            )

        redundant: list[int] = []
        node = self.root
        now = time.monotonic()
        node.last_access = now
        idx = 0
        page_idx = 0
        inserted_pages = 0

        while idx < aligned_len:
            child_key = _child_key(tokens[idx:], self.page_size)
            child = node.children.get(child_key)

            if child is None:
                suffix = tokens[idx:aligned_len]
                suffix_pages = pages[page_idx:num_pages]
                leaf = _make_child(node, child_key, suffix, suffix_pages)
                self._cached_page_count += len(suffix_pages)
                inserted_pages += len(suffix_pages)
                self.metrics.total_inserted_pages += inserted_pages
                return leaf, redundant

            child.last_access = now
            match_len = _match_keys(child.key, tokens[idx:], self.page_size)

            if match_len < len(child.key):
                split_at = _page_align_len(match_len, self.page_size)
                if split_at == 0:
                    suffix = tokens[idx:aligned_len]
                    suffix_pages = pages[page_idx:num_pages]
                    leaf = _make_child(node, child_key, suffix, suffix_pages)
                    self._cached_page_count += len(suffix_pages)
                    inserted_pages += len(suffix_pages)
                    self.metrics.total_inserted_pages += inserted_pages
                    return leaf, redundant
                node = _split_node(child, split_at, self.page_size)
                idx += split_at
                page_idx += split_at // self.page_size
                continue

            edge_npages = len(child.pages)
            incoming = pages[page_idx : page_idx + edge_npages]
            if incoming != child.pages:
                redundant.extend(incoming)
            page_idx += edge_npages
            idx += len(child.key)
            node = child

        self.metrics.total_inserted_pages += inserted_pages
        return node, redundant

    def evict(self, n_pages_needed: int) -> int:
        """LRU-evict at least ``n_pages_needed`` pages (best effort)."""
        if n_pages_needed <= 0:
            return 0

        leaves = self._collect_evictable_leaves()
        heap: list[tuple[float, int, RadixNode]] = [
            (leaf.last_access, id(leaf), leaf) for leaf in leaves
        ]
        heapq.heapify(heap)
        freed = 0

        while freed < n_pages_needed and heap:
            _, _, node = heapq.heappop(heap)
            if node.ref_count > 0 or node is self.root or not node.pages:
                continue

            parent = node.parent
            if parent is None:
                continue

            child_key = _child_key(node.key, self.page_size)
            if parent.children.get(child_key) is not node:
                continue

            self.pool.free(node.pages)
            n = len(node.pages)
            freed += n
            self._cached_page_count -= n
            self.metrics.total_evicted_pages += n
            del parent.children[child_key]

            if (
                parent is not self.root
                and not parent.children
                and parent.ref_count == 0
                and parent.pages
            ):
                heapq.heappush(heap, (parent.last_access, id(parent), parent))

        return freed

    def reset(self) -> None:
        """Drop the whole tree, return every page to the pool."""
        pages: list[int] = []

        def walk(node: RadixNode) -> None:
            pages.extend(node.pages)
            for child in list(node.children.values()):
                walk(child)

        walk(self.root)
        if pages:
            self.pool.free(pages)
        self.root = RadixNode()
        self._cached_page_count = 0

    def _collect_evictable_leaves(self) -> list[RadixNode]:
        leaves: list[RadixNode] = []

        def dfs(node: RadixNode) -> None:
            if node is not self.root and not node.children and node.ref_count == 0:
                leaves.append(node)
            for child in node.children.values():
                dfs(child)

        dfs(self.root)
        return leaves

    def _count_pages(self, node: RadixNode, locked: bool) -> int:
        total = 0
        if node is not self.root and node.pages:
            if locked:
                if node.ref_count > 0:
                    total += len(node.pages)
            elif node.ref_count == 0:
                total += len(node.pages)
        for child in node.children.values():
            total += self._count_pages(child, locked=locked)
        return total


def _page_align_len(n: int, page_size: int) -> int:
    return (n // page_size) * page_size


def _child_key(tokens: list[int], page_size: int) -> tuple[int, ...]:
    return tuple(tokens[:page_size])


def _match_keys(stored: list[int], query: list[int], page_size: int) -> int:
    min_len = min(len(stored), len(query))
    i = 0
    while i < min_len:
        if stored[i : i + page_size] != query[i : i + page_size]:
            break
        i += page_size
    return i


def _make_child(
    parent: RadixNode,
    child_key: tuple[int, ...],
    key: list[int],
    pages: list[int],
) -> RadixNode:
    child = RadixNode()
    child.parent = parent
    child.key = key
    child.pages = pages
    parent.children[child_key] = child
    return child


def _split_node(node: RadixNode, split_at: int, page_size: int) -> RadixNode:
    """Split ``node`` after the first ``split_at`` page-aligned tokens."""
    assert node.parent is not None
    assert split_at > 0 and split_at < len(node.key)
    assert split_at % page_size == 0

    parent = node.parent
    old_child_key = _child_key(node.key, page_size)
    split_pages = split_at // page_size

    new_node = RadixNode()
    new_node.parent = parent
    new_node.key = node.key[:split_at]
    new_node.pages = node.pages[:split_pages]
    new_node.last_access = node.last_access
    new_node.ref_count = node.ref_count

    node.parent = new_node
    node.key = node.key[split_at:]
    node.pages = node.pages[split_pages:]

    new_child_key = _child_key(node.key, page_size)
    new_node.children[new_child_key] = node
    parent.children[old_child_key] = new_node
    return new_node
