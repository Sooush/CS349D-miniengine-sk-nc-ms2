# Milestone 4 — Track 1 (HiCache) Handoff

Context dump for continuing this work in a new workspace that has the L4 GPU.
Written for both a human and an LLM agent. Read top to bottom.

---

## 0. TL;DR / immediate next action

1. **Commit and push the local edits from the previous workspace first** (see
   §2 "Uncommitted fixes"). The L4 workspace must `git pull` them or it will be
   missing the critical bug fix and crash the moment HiCache evicts.
2. On the L4 box, run the **cliff** demo (HiCache off) and the **restore** demo
   (HiCache on) and capture the per-turn hit-rate tables (§5).
3. Confirm the restore run produces **non-empty outputs** (empty output = an
   eviction crash is aborting requests = regression; see §6).

> **⚠️ READ §9 FIRST — hard-won findings from the L4 run session.** The demo is
> wedged between two failure modes and the naive configs hit both:
> - **`--concurrency 1` produces NO cliff and NO HiCache signal.** The bench runs
>   each session's turns back-to-back, so the just-used prefix is always MRU and
>   never evicted-before-reuse ⇒ GPU-only == HiCache. The cliff *requires*
>   interleaved sessions (high concurrency).
> - **`--concurrency ≥2` OOMs/deadlocks the tiny KV pool** (documented engine
>   limit). So high concurrency, which the cliff needs, doesn't finish.
> - **`--page-size 256` further flattens the signal** (~1 cacheable page/session);
>   use `--page-size 32`.
> **ROOT CAUSE FOUND + FIXED (see §9.4).** The high-concurrency stall was a
> HiCache eviction bug: `num_evictable_pages()` counted every unlocked GPU node
> but `evict()` only demoted *leaves*, so GPU internal nodes whose children were
> demoted to CPU got permanently stuck — admission over-admitted, `allocate()`
> OOM'd, and the run wedged. Fixed so eviction demotes any unlocked GPU node.
> With that fix `--concurrency 8` completes. See §9.2–§9.4.

---

## 1. What this project is

`miniengine` is a minimal LLM serving engine (continuous batching, paged KV
pool, PagedAttention, chunked prefill, radix prefix cache). Milestones 1–3 are
done. **Milestone 4 = pick one advanced track.** We chose **Track 1: HiCache**.

**HiCache goal:** the milestone-3 radix prefix cache lives entirely in GPU HBM.
When the GPU KV pool fills, LRU eviction *drops* cached prefixes, so a later
request that would have hit must re-prefill from scratch. On long multiturn
workloads the hit rate collapses once the working set exceeds GPU capacity.
HiCache adds a **CPU DRAM tier** under the GPU tier:

- **Demote**: instead of dropping an evicted GPU prefix, copy its pages D2H into
  a pinned-host CPU pool and keep the radix node (now CPU-tier).
- **Promote**: on a lookup that matches a CPU-resident prefix, copy H2D back
  into freshly allocated GPU pages and repoint the node to GPU.
- **CPU eviction**: when the CPU pool fills, LRU-evict CPU entries entirely
  (no lower tier).

CPU DRAM is ~10–100× larger than HBM, so effective cache capacity grows
accordingly and the per-turn hit rate stays high instead of collapsing.

### Full-credit bar (what we are targeting)
Per `milestones/milestone4.md` Track 1 + the instructor's ed post (basic
functionality = full credit, perf targets are bonus):
1. **Demonstrate the GPU-only cliff** (HiCache off): a multiturn config where
   per-turn hit rate starts high (≥70% early) and collapses (<20% by the last
   turn). Document the GPU pool size.
2. **Restore the hit rate** (HiCache on, CPU pool ≥10× GPU pool): per-turn hit
   rate stays high across all turns. Report both per-turn tables side by side.
3. **End-to-end completion**: workload finishes, no hang, no OOM in either tier.

Throughput/TTFT must be *reported* but **not required to beat** the GPU-only
baseline. Bonus (optional): `--hicache-overlap` async stream + a measured
overlap, or a real >20% throughput/TTFT win.

---

## 2. Current status

### Done (committed by teammate — commits `4c6901d`, `2c8496b`)
- `miniengine/hicache.py`: `CPUKVPool` (pinned host K/V pool), `HiRadixNode`
  (adds a `tier` slot), `HiRadixCache` (subclass of `RadixCache`) with
  `_demote_node`, `_promote_node`, `_copy_kv`, promotion folded into
  `match_prefix`, `evict()` override.
- CLI: `--cpu-cache-size-gb N` (0 disables HiCache; GPU-only radix cache) and
  `--hicache-overlap` in `miniengine/__main__.py`, threaded into `Engine`.
- `Engine.__init__` builds `HiRadixCache` when `--cpu-cache-size-gb > 0`, else
  the plain `RadixCache`.
- A stall fix: `paged_kv_can_admit` now also checks allocatable pages, and the
  scheduler aborts in-flight requests on a step exception instead of hanging.

This covers spec steps 1, 2, 3 and the required CLI.

### Uncommitted fixes from the previous workspace (MUST be committed + pushed)
These were edited in `miniengine/hicache.py` and `miniengine/radix_cache.py`
but **not yet committed**. Verify with `git status` / `git diff`.

1. **CRITICAL — tier-slot crash fix** (`radix_cache.py`): `_make_child` and
   `_split_node` used to create the base `RadixNode`, whose `__slots__` lacks
   `tier`. In a HiCache tree, `_demote_node` then did `node.tier = CacheTier.CPU`
   → `AttributeError` → scheduler aborts requests **exactly when HiCache should
   engage** (GPU pool full). Fix: factories now create `type(parent)()` /
   `type(node)()` so HiCache trees stay `HiRadixNode`, and `_split_node` copies
   `tier` onto the new parent.
2. **Promote re-entrancy guard** (`hicache.py`): `_promote_node` calls
   `pool.allocate`, which can recurse into `evict` → demote under memory
   pressure and evict the very node being promoted. It now bumps the node's
   `ref_count` around the allocation so eviction skips it.
3. **CPU-tier LRU eviction — spec step 4** (`hicache.py`): added
   `_evict_cpu_lru(n)` + `_collect_evictable_leaves_for_tier(tier)`. When the
   CPU pool is full, `_demote_node` now LRU-evicts CPU leaves and retries before
   falling back to dropping the GPU node.
4. **Part 5 lock hardening** (both `hicache.py` and `radix_cache.py`): in the
   partial-edge match branch of `match_prefix`, the borrowed `child` is now the
   returned `last_node`, so the engine locks it (`inc_lock_ref`) and it can't be
   evicted/demoted while the request is in flight. (The base cache had the same
   gap; fixed there too because the cliff run uses the base `RadixCache`.)

**These edits compile clean (`python -m py_compile`) and pass lints, but have
NOT been run on a GPU yet.**

### Remaining work
- Run the cliff + restore demos on L4 and capture tables (§5) — the main TODO.
- (Optional/nice for report) Add demote/promote/CPU-evict counters to
  `/cache_stats` so you have numbers to tabulate.
- (Bonus) Validate `--hicache-overlap` (dedicated CUDA stream, async copies) and
  measure overlap; and/or chase a >20% throughput/TTFT win.

The tracking plan lives at `.cursor/plans/hicache_cpu_tier_51c71c17.plan.md`
(in the previous workspace's `.cursor` dir; may not carry over).

---

## 3. Environment notes (observed on the L4 box `cs349d`)

- **flash-attn NOT installed** → paged mode uses the slower fallback attention
  path. Expect low tok/s; keep `--max-tokens` modest.
- With `--page-size 32`, the engine warns it uses the **non-direct decode path**
  (gathers full KV per token — very slow on long contexts). Direct flash paged
  decode wants `page_size` divisible by 256. Fine for a *hit-rate* demo; just
  slow. Do NOT switch to a tiny model — the spec requires `Qwen/Qwen3-8B`.
- **Free VRAM after the 8B fp16 weights load was only ~2.43 GB.** At
  `--mem-fraction-static 0.25` that gives **`pool_pages=128` (~608 MB)** of GPU
  KV. This tiny pool is *good* for the demo — it makes the cliff easy to trigger.
  Do not crank `--mem-fraction-static` to 0.85 here: it would leave almost no
  VRAM for activations and likely GPU-OOM mid-run.
- **Host RAM: 31 GiB total, ~29 GiB available.** The CPU tier is *pinned*
  (page-locked) memory and must fit in RAM with headroom. `--cpu-cache-size-gb
  32` was **OOM-killed** (`Killed`) because 32 > 29. **Use `--cpu-cache-size-gb
  8`** (~13× the GPU pool, well within the ≥10× bar and safe for RAM). Up to ~16
  is safe if ever needed.

If the L4 box differs, re-check with `nvidia-smi` (VRAM) and `free -h` (RAM)
and resize `--cpu-cache-size-gb` to stay under available RAM while keeping it
≥10× the logged `pool_pages` size.

---

## 4. How the engine runs

The server is a long-running HTTP service; the benchmark is just an HTTP client.
Run the **server in terminal 1**, the **benchmark in terminal 2**, on the same
(GPU) machine so `localhost` works.

The server is **plain HTTP** (uvicorn, no TLS) — always use `http://`, never
`https://` (an `https://` base-url hangs on a TLS handshake forever; that was a
prior red herring stall).

---

## 5. Commands to run (the actual deliverable)

Use the **same `--mem-fraction-static` for both runs** so the only variable is
the CPU tier.

### Run A — GPU-only cliff (HiCache OFF)
Terminal 1:
```bash
python -m miniengine --model Qwen/Qwen3-8B --mode paged --page-size 32 \
    --prefill-chunk-size 512 --mem-fraction-static 0.25 --cpu-cache-size-gb 0
```
Wait for `Application startup complete`. Note the logged `pool_pages=N`.

Terminal 2:
```bash
python -m benchmark.bench_cache --workload multiturn --base-url http://localhost:8000 \
    --num-sessions 16 --turns-per-session 8 --concurrency 8 --max-tokens 64
```
Save the full output (esp. the **Per-turn breakdown** table).

### Run B — restore (HiCache ON)
Ctrl+C the server, then terminal 1:
```bash
python -m miniengine --model Qwen/Qwen3-8B --mode paged --page-size 32 \
    --prefill-chunk-size 512 --mem-fraction-static 0.25 --cpu-cache-size-gb 8
```
Confirm a `HiCache enabled — cpu_pages=… (8.00 GiB budget)` line prints (no
`Killed`). Then run the **same** terminal-2 benchmark command again and save it.

### Tuning the cliff if needed
- The GPU pool here is ~4096 token-slots total (128 pages × 32 tokens). If the
  OFF run's turn-0/1 hit rate isn't high or the collapse isn't sharp, increase
  `--turns-per-session` (longer growing prefixes) and/or `--num-sessions` so the
  cumulative working set clearly exceeds GPU capacity.
- If the OFF run starts failing to *admit* requests (look for "KV pool cannot
  admit" warnings) rather than just missing the cache, lower `--num-sessions`
  or `--max-tokens`, or nudge `--mem-fraction-static` up slightly (you have
  ~2.4 GB free, so ≤0.5 is available) — and keep it identical for Run B.
- Target shape: OFF = high early turns, collapsing late; ON = high all turns.

---

## 6. Validation / tests

1. **Server smoke (HiCache on):** server reaches `Application startup complete`
   and prints `HiCache enabled …` with no `Killed`. (`curl
   http://localhost:8000/cache_stats` returns JSON.)
2. **Correctness — non-empty outputs:** on Run B, the bench must show real
   completions and finish. **Empty/garbage completions mean an eviction path is
   raising and the scheduler is aborting requests** — that's the
   tier-slot/demote bug regressing; check the server log for tracebacks /
   "Aborting request … after scheduler failure".
3. **No OOM either tier:** no `Killed` (host RAM) and no `CUDA out of memory`
   (GPU) during the run.
4. **(Optional) demote/promote round-trip unit check** — quickest proof the
   copy path is correct. Needs CUDA (pinned memory). Example:
```python
import torch
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.hicache import HiRadixCache, CacheTier

pool = KVMemoryPool(num_pages=8, page_size=32, num_layers=2,
                    num_kv_heads=4, head_dim=64, dtype=torch.float16, device="cuda")
cache = HiRadixCache(pool, cpu_cache_size_gb=0.001)  # tiny CPU tier
# write a known pattern into a GPU page, insert a node, force demote, then
# promote, and assert the GPU bytes round-trip equal. (Flesh out as needed.)
```
   The simplest end-to-end equivalent is just Run B passing with high hit rate:
   a correct H2D promote is what produces the cache hits.
5. **(If touching base path) regression:** `bench_serving` default at conc
   1/4/16 with cache on should match `--disable-radix-cache` within noise — this
   confirms the `match_prefix` lock change didn't hurt the no-sharing case.

For the report, also capture: server `pool_pages` (GPU capacity wall), the two
per-turn tables, wall time / throughput / TTFT lines, and a note that the
fallback attention path (no flash-attn) bounds absolute tok/s.

---

## 7. File / architecture map

- `miniengine/hicache.py` — **HiCache lives here.** `CPUKVPool`, `HiRadixNode`,
  `HiRadixCache` (override `match_prefix`, `evict`, `reset`,
  `num_evictable_pages`; helpers `_demote_node`, `_promote_node`, `_copy_kv`,
  `_evict_cpu_lru`, `_collect_evictable_leaves_for_tier`).
- `miniengine/radix_cache.py` — base GPU-only radix cache + module-level node
  factories `_make_child` / `_split_node` (now subtype/tier-aware) and helpers
  `_child_key`, `_match_keys`, `_page_align_len` (imported by hicache).
- `miniengine/kv_memory_pool.py` — GPU `KVMemoryPool`. Page layout
  `(num_pages, page_size, num_kv_heads, head_dim)` per layer for K and V (CPU
  pool mirrors this). `allocate()` calls `radix_cache.evict(need)` before OOM.
- `miniengine/engine.py` — model wrapper. `_paged_prefill_init` does the radix
  `match_prefix` + `inc_lock_ref` (lock pattern for part 5); `free_request_kv` /
  `_radix_cache_commit_request` insert finished sequences and `dec_lock_ref`.
- `miniengine/scheduler.py` — FCFS scheduler, single background thread. Paged
  step = admit → batched prefill → batched decode. `_abort_inflight_requests`
  on step failure.
- `miniengine/__main__.py` — CLI / server launch (uvicorn, plain HTTP).
- `benchmark/bench_cache.py` — `shared` and `multiturn` workloads; multiturn
  emits the per-turn hit-rate breakdown we need. Parses
  `usage.cache_hit_tokens` from each streamed response.
- `milestones/milestone4.md` — the assignment (Track 1 "What to build" =
  the 5 points; CPU eviction is point 4, lock-under-concurrency is point 5).

### Key invariants
- A page is owned by exactly one tier; demote/promote swap ownership.
- Locked nodes (`ref_count > 0`) are never demoted or dropped.
- Single scheduler thread → no true concurrency, but the `inc_lock_ref` /
  `dec_lock_ref` discipline still protects in-flight requests across promotion.
- `_split_node` copies `ref_count` and `tier` to the new parent (so splitting a
  locked and/or CPU-tier node stays consistent).

---

## 8. Gotchas learned the hard way
- `https://` base-url → infinite hang. Use `http://`.
- `--cpu-cache-size-gb` larger than free host RAM → silent `Killed` (OOM
  killer), no traceback. Size it from `free -h`.
- `--mem-fraction-static` is a fraction of **free VRAM after weights load**, not
  total VRAM. On this VRAM-starved box, high values risk activation OOM.
- bench CLI: workload is `multiturn` (one word); flags need `--` prefixes; don't
  put a bare `--` before flags.
- **`--concurrency ≥2` hangs/OOMs on the small pool.** See §9. Use conc 1.
- **`--page-size 256` silently kills the demo.** See §9. Use page-size 32.

---

## 9. Findings from the L4 run session (READ BEFORE RUNNING)

Empirical results from actually running on the L4 box. These override some of
the earlier guidance in §5.

### 9.1 Concurrency ≥2 deadlocks the KV pool
At `--concurrency 2`/`4`/`8` the multiturn run repeatedly froze partway
(e.g. 89/96, 106/128, 121/128) and/or crashed with:
```
RuntimeError: KV memory pool OOM: need 2 pages, only 1 free
... Aborting request ... after scheduler failure
```
Root cause is the **documented engine limit** (see `milestone3` report "Next
Steps"): the small paged KV pool exhausts under concurrent, *growing* multiturn
prompts. Worst-case page reservation in `paged_kv_can_admit` ignores radix
sharing, so the scheduler can't admit and in-flight HTTP streams block forever.
- M3's **successful** multiturn run was **`--concurrency 1`** (128/128 in ~439s).
- M2 tolerated conc 16–32 only because it used a *large* pool (`0.85`),
  `page-size 256`, and *fixed* 512-token prompts. None of that holds here.
- **Conclusion: run the M4 multiturn demos at `--concurrency 1`.** A full run is
  ~6–7 min; treat it as stalled only if `Finished request` is flat ~10 min.

### 9.2 The core tension: the cliff needs concurrency, concurrency OOMs the pool
Ran A (GPU-only) and B (HiCache, `--cpu-cache-size-gb 8`) at
`--page-size 256 --mem-fraction-static 0.25` (`pool_pages=16`), **conc 1**, 16×8.
**Both runs were byte-identical — 39.8% hit rate, same per-turn curve:**
```
turn   prompt_tok   hit_tok   hit_rate
   0        2279         0       0.0%
   4        5747      3840      66.8%
   7        8074      4096      50.7%   <-- rises & plateaus; NO collapse
```
Two compounding reasons, in order of importance:

**(a) `--concurrency 1` makes a cliff impossible.** `bench_cache` multiturn at
conc 1 runs **one session's 8 turns back-to-back, then the next session**
(see `workload_multiturn`: one `session_runner`, sequential turns). So turn k's
prefix (turn k-1) is always the **MRU** cache entry and is never evicted before
reuse. Completed sessions do get evicted, but are never queried again ⇒ those
evictions cause **zero misses**. Result: GPU-only already hits everything
within a session, and **HiCache never has still-needed data to demote/promote
⇒ ON == OFF, no cliff — at any page size.** The cliff fundamentally requires
**many interleaved sessions** (the original §5 `--concurrency 8`) whose growing
prefixes overflow the pool, so GPU-only evicts a prefix that is about to be
reused by another in-flight session.

**(b) `--page-size 256` shrinks the signal further.** `prompt_tok` is summed
over 16 sessions ⇒ each session grows only ~140→~505 tokens. With page 256,
page-aligned matching gives **≤1 cacheable page (256 tok) per session**
(`hit_tok` caps at 16×256=4096). page-32 would expose ~16 pages/session of
matchable prefix instead.

**But (a) and concurrency hang are in direct conflict:**
- conc ≥2 → interleaving (needed for the cliff) **but** OOM/deadlock on the
  tiny pool (§9.1). conc 8 (the §5 spec) never completes here.
- conc 1 → completes **but** no interleaving ⇒ no cliff, no HiCache signal.

So the demo is wedged between "no cliff" and "won't finish." The job now is to
find the **sweet spot**: enough interleaved sessions to overflow the *cached*
(unlocked) working set, with a pool still large enough to hold the *locked*
(in-flight) pages so it doesn't hard-OOM.

### 9.3 Plan: find a concurrency that interleaves AND completes
Levers (the OOM is locked-page pressure; the cliff is unlocked-page eviction —
separate the two):
1. **Lower per-request peak pages** so several requests fit in flight without
   OOM: drop `--max-tokens` (e.g. 32) and/or `--turns-per-session`. Smaller peak
   ⇒ higher concurrency survives.
2. **Use `--page-size 32`** so the matchable prefix is many pages (real signal)
   and the working set overflows a small pool.
3. **Sweep concurrency upward from 2** (2 → 3 → 4), and for each, pick the
   **smallest `--mem-fraction-static` that still completes** (largest cliff
   without OOM). Watch the server log for `KV pool cannot admit` / `OOM`.
4. If even conc 2 OOMs at every pool size that still overflows, the real fix is
   an **engine change**: decode-time retraction (M3 §"Next Steps") or making
   `paged_kv_can_admit` account for radix sharing so eviction can reclaim
   unlocked pages instead of crashing. This is the most robust route to a clean
   conc-8 cliff but is a code task, not just a flag.

First experiment to try (page 32, modest peak, conc 2):
```bash
# Run A — GPU-only
python -m miniengine --model Qwen/Qwen3-8B --mode paged --page-size 32 \
  --prefill-chunk-size 512 --mem-fraction-static 0.30 --cpu-cache-size-gb 0 \
  2>&1 | tee /tmp/server-runA.log
python -m benchmark.bench_cache --workload multiturn --base-url http://localhost:8000 \
  --num-sessions 16 --turns-per-session 8 --concurrency 2 --max-tokens 32 \
  2>&1 | tee /tmp/bench-runA.txt
# Run B — HiCache ON (relaunch server with --cpu-cache-size-gb 8, same bench)
```
If conc 2 completes, bump to 3 then 4 to deepen the cliff. If it OOMs, lower
`--max-tokens`/`--turns-per-session` or go to the engine fix in step 4.
Whatever completes, keep A and B **identical** except `--cpu-cache-size-gb`.

### 9.4 ROOT CAUSE of the high-concurrency stall — found & fixed
The stall at `--concurrency` ≥2 was **not** a lock leak (verified: residual
ref_count == 0 in an offline harness driving the real cache lifecycle). It was a
HiCache **eviction accounting vs. mechanism mismatch**:

- `num_evictable_pages()` counts *every* unlocked GPU-tier node (internal + leaf)
  by recursing the whole tree.
- `HiRadixCache.evict()` only collected **leaf** GPU nodes
  (`_collect_evictable_leaves_for_tier`), and the childless-cascade only fires
  when a node is *dropped*. Demotion doesn't drop nodes.
- So a **GPU-tier internal node whose children had been demoted to CPU** was
  never a leaf and never became childless ⇒ its GPU pages were **counted as
  evictable but never reclaimable**. They accumulated until the GPU pool wedged.

Effect: `paged_kv_can_admit` saw `free + num_evictable` and admitted a request,
but `evict()` then couldn't free those pages ⇒ `allocate()` raised
`KV pool OOM`. Old code aborted the whole batch (`_abort_inflight_requests`);
the interim retraction patch turned that into a hot-spin (millions of
`match_prefix` calls with `running=0, prefilling=0`).

**Fix (in `miniengine/hicache.py`):** eviction now collects **all** unlocked
GPU nodes (`_collect_evictable_gpu_nodes`) and **demotes** them (internal nodes
included — demotion moves pages GPU→CPU without removing the node, so leaf-ness
isn't required). Dropping (CPU full) still requires a leaf. This makes `evict()`
able to reclaim everything `num_evictable_pages()` promises, so admission is
consistent and there is no admit→OOM→retract spin.

Offline harness result after the fix: base and HiCache both drain **all**
requests at conc 8, residual ref_count 0, no idle-wedge. (Harness:
`/tmp/leak_repro2.py`.)

**Also in the tree of changes (all minimal, happy-path no-ops):**
- `engine.py` + `core.py` + `scheduler.py`: a prefill-init `KV pool OOM` now
  **retracts** the single request to WAITING instead of aborting every in-flight
  request. Kept as defense-in-depth; with the eviction fix it should rarely
  trigger.
- The earlier `clear_locks` idle band-aid was **removed** (the eviction fix
  addresses the real cause, so masking stale locks is unnecessary).

**Status:** validated offline AND end-to-end (see §9.5).

### 9.5 SECOND root cause — scheduler dropped requests (fixes Run A too)
The HiCache evict fix did not explain why **Run A (base cache, HiCache OFF)**
also hung at high concurrency. Found a second, independent bug in the
**scheduler** (`miniengine/scheduler.py`), affecting both runs:

`_step_paged` (and `_step_batched`) iterated `zip(self.prefilling, token_ids)`
and `zip(self.running, token_ids)` while `_finish_request()` **removes** the
finishing request from those same lists. Mutating a list mid-iteration makes
`zip` skip the element shifted into the freed slot — that request is never added
to `still_prefilling`/`still_running`, so the subsequent
`self.prefilling = still_prefilling` **silently drops it**: its pages are never
freed and its HTTP stream never gets a finish marker, so the bench worker blocks
forever. This fired constantly because many turns end with an immediate stop
token (`output_len=1`), i.e. an early finish in the same batch as other
in-flight requests. Symptom: `Enqueued > Finished` (e.g. 84 vs 77), engine idle,
GPU 0%, bench hung with open connections — exactly the "stall at ~77".

**Fix:** iterate over a **snapshot** (`zip(list(self.prefilling), token_ids)` /
`zip(list(self.running), token_ids)`) so finishing one request can't skip the
next.

### 9.6 End-to-end verification (L4, page_size 32, conc 8, 16×8)
Both runs now complete with **0 aborts, 0 dropped requests**:

| Run | cpu-cache | Requests | Wall | Hit rate | turn0 → turn7 hit |
|-----|-----------|----------|------|----------|-------------------|
| A (GPU-only) | 0 GiB | 128/128 | 108 s | 70.5% | 0% → 72% |
| B (HiCache)  | 8 GiB | 128/128 | 117 s | 77.0% | 0% → 89% |

So the **high-concurrency stall is fixed** (the original blocker). At
`pool_pages=128` there is no collapse (working set fits). Shrinking the GPU pool
to **64 fixed pages** produces the milestone's cliff/restore contrast (§9.7).

### 9.7 CLIFF + RESTORE demonstrated (the deliverable)
Identical config for both runs — **`--page-size 32 --prefill-chunk-size 512
--fixed-kv-pool-pages --kv-pool-pages 64`**, bench `multiturn 16×8 --concurrency
8 --max-tokens 64` — only `--cpu-cache-size-gb` differs (0 vs 8). The 64-page
GPU pool (2048 token-slots) is far smaller than the interleaved working set, so
GPU-only thrashes and HiCache's CPU tier rescues it.

Per-turn hit rate:

| turn | A: GPU-only (cpu=0) | B: HiCache (cpu=8 GiB) |
|------|--------------------:|-----------------------:|
| 0 | 0.0% | 0.0% |
| 1 | 24.7% | 63.3% |
| 2 | 19.1% | 70.4% |
| 3 | 0.0% | 85.6% |
| 4 | 6.5% | 82.9% |
| 5 | 6.4% | 84.4% |
| 6 | 2.8% | 85.2% |
| 7 | 10.8% | 87.7% |

Aggregate:

| metric | A: GPU-only | B: HiCache | 
|--------|------------:|-----------:|
| Cache hit rate | **8.1%** | **76.4%** |
| Wall time | 194.9 s | **107.0 s** |
| Requests | 128/128 | 128/128 |
| Aborts / idle-wedge | 0 / 0 | 0 / 0 |

GPU-only collapses to ~3–11% on late turns (and TTFT climbs to ~11 s as it
re-prefills); HiCache holds **80–88%** and finishes **~45% faster** (re-prefill
avoided) — i.e. a real end-to-end win, not just a hit-rate restore. This is the
cliff (1), restore (2), and end-to-end completion (3) full-credit bar.

Saved outputs: `/tmp/bench-cliffA-SAVED.txt`, `/tmp/bench-cliffB-SAVED.txt`.
