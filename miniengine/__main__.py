"""
CLI entry point — launch the MiniEngine server.

Usage:
    python -m miniengine --model Qwen/Qwen3-8B --mode paged \\
        --mem-fraction-static 0.85 --page-size 32 --torch-compile

    python -m miniengine --model Qwen/Qwen3-8B --mode paged \\
        --mem-fraction-static 0.85 --page-size 32 --prefill-chunk-size 512
"""

from __future__ import annotations

import argparse
import logging

import torch
import uvicorn

from miniengine.engine import Engine
from miniengine.model import FLASH_ATTN_AVAILABLE
from miniengine.scheduler import Scheduler
from miniengine import server as srv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="miniengine",
        description="Minimal LLM serving engine",
    )
    p.add_argument(
        "--model", type=str, required=True, help="HuggingFace model id or local path"
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument("--device", type=str, default="cuda", help="Device to load model on")
    p.add_argument(
        "--max-running",
        type=int,
        default=16,
        help="Max concurrent requests in the scheduler",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="paged",
        choices=["baseline", "batched", "paged"],
        help="Scheduling mode: baseline (one request at a time) or "
        "batched (iteration-level batching) or "
        "paged (paged KV pool)",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=32,
        help="Tokens per KV page (used in paged mode)",
    )
    p.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.85,
        help="In paged mode: fraction of GPU memory still free *after* weights load "
        "reserved for the KV pool (not a fraction of total VRAM)",
    )
    p.add_argument(
        "--kv-pool-pages",
        type=int,
        default=4096,
        help="Fixed KV pool capacity in pages, or fallback when VRAM budget is unavailable",
    )
    p.add_argument(
        "--fixed-kv-pool-pages",
        action="store_true",
        help="In paged mode: size the KV pool from --kv-pool-pages only (ignore "
        "--mem-fraction-static). Use for fair baseline vs torch.compile comparisons.",
    )
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile on stable model submodules",
    )
    p.add_argument(
        "--torch-compile-target",
        type=str,
        default="mlp",
        choices=["mlp", "block"],
        help="Sub-region to compile when --torch-compile is set",
    )
    p.add_argument(
        "--torch-compile-dynamic",
        action="store_true",
        help="Enable dynamic shape support for torch.compile regions",
    )
    p.add_argument(
        "--require-flash-attn",
        action="store_true",
        help="In paged mode, fail startup if flash-attn is unavailable "
        "(otherwise only warn)",
    )
    p.add_argument(
        "--attention-backend",
        type=str,
        default="auto",
        choices=["auto", "flash_attn"],
        help="Decode attention backend preference. "
        "auto keeps current behavior.",
    )

    p.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=0,
        help="Per-step prefill token budget (0 = single-shot prefill)",
    )
    p.add_argument(
        "--disable-radix-cache",
        action="store_true",
        help="Disable the radix prefix cache (cache is on by default in paged mode)",
    )
    p.add_argument(
        "--cpu-cache-size-gb",
        type=float,
        default=0.0,
        help="CPU DRAM KV tier size in GiB (0 = GPU-only radix cache)",
    )
    p.add_argument(
        "--hicache-overlap",
        action="store_true",
        help="Use a dedicated CUDA stream for async HiCache demote/promote copies",
    )

    return p.parse_args()


def _cuda_total_memory_bytes(device: str) -> int | None:
    if not torch.cuda.is_available():
        return None
    if not device.startswith("cuda"):
        return None
    if ":" in device:
        device_idx = int(device.split(":")[1])
    else:
        device_idx = torch.cuda.current_device()
    return int(torch.cuda.get_device_properties(device_idx).total_memory)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("miniengine")

    dtype = getattr(torch, args.dtype)
    logger.info(
        "Initializing engine  model=%s  dtype=%s  mode=%s  "
        "attention_backend=%s  prefill_chunk_size=%d  radix_cache=%s",
        args.model,
        args.dtype,
        args.mode,
        args.attention_backend,
        args.prefill_chunk_size,
        "off" if args.disable_radix_cache else "on",
    )

    kv_pool_mem_fraction: float | None = None
    if args.mode == "paged":
        if not FLASH_ATTN_AVAILABLE:
            msg = (
                "Paged mode requested but flash-attn is unavailable. "
                "Benchmarks will run on a slower fallback path and can regress "
                "throughput/latency. Install flash-attn for best results."
            )
            if args.require_flash_attn:
                raise SystemExit(msg)
            logger.warning(msg)

        if args.fixed_kv_pool_pages:
            kv_pool_mem_fraction = None
            logger.info(
                "Paged KV pool: fixed %d pages (--fixed-kv-pool-pages)",
                args.kv_pool_pages,
            )
        elif _cuda_total_memory_bytes(args.device) is not None:
            kv_pool_mem_fraction = args.mem_fraction_static
            logger.info(
                "Paged KV pool will use %.3f of free GPU memory after model load",
                args.mem_fraction_static,
            )
        else:
            logger.info(
                "CUDA total memory unavailable; falling back to --kv-pool-pages=%d",
                args.kv_pool_pages,
            )
    engine = Engine(
        model_path=args.model,
        dtype=dtype,
        device=args.device,
        mode=args.mode,
        page_size=args.page_size,
        kv_pool_num_pages=args.kv_pool_pages,
        kv_pool_mem_fraction=kv_pool_mem_fraction,
        torch_compile=args.torch_compile,
        torch_compile_target=args.torch_compile_target,
        torch_compile_dynamic=args.torch_compile_dynamic,
        attention_backend=args.attention_backend,
        prefill_chunk_size=args.prefill_chunk_size,
        disable_radix_cache=args.disable_radix_cache,
        cpu_cache_size_gb=args.cpu_cache_size_gb,
        hicache_overlap=args.hicache_overlap,
    )
    sched = Scheduler(engine=engine, max_running=args.max_running,mode=args.mode, prefill_chunk_size=args.prefill_chunk_size,)

    # Wire up the server module globals
    srv.engine = engine
    srv.scheduler = sched
    srv.model_id = args.model

    # Start scheduler background thread
    sched.start()

    logger.info("Starting server on %s:%d", args.host, args.port)
    try:
        uvicorn.run(srv.app, host=args.host, port=args.port, log_level="info")
    finally:
        sched.stop()


if __name__ == "__main__":
    main()
