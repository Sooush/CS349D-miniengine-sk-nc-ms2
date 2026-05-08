"""
CLI entry point — launch the MiniEngine server.

Usage:
    python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507
    python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507 --port 8080 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import logging

import torch
import uvicorn

from miniengine.engine import Engine
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
        default="batched",
        choices=["baseline", "batched", "paged"],
        help="Scheduling mode: baseline (one request at a time) or "
        "batched (iteration-level batching, milestone 1) or "
        "paged (milestone 2 paged KV path)",
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
        help="Fraction of total GPU memory used as static budget in paged mode",
    )
    p.add_argument(
        "--kv-pool-pages",
        type=int,
        default=4096,
        help="Fallback fixed KV pool pages when budget derivation is unavailable",
    )
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile on stable model submodules",
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
        "Initializing engine  model=%s  dtype=%s  mode=%s",
        args.model,
        args.dtype,
        args.mode,
    )

    kv_pool_bytes_budget = None
    if args.mode == "paged":
        total_mem = _cuda_total_memory_bytes(args.device)
        if total_mem is not None:
            kv_pool_bytes_budget = int(total_mem * args.mem_fraction_static)
            logger.info(
                "Paged KV budget derived from mem fraction: %d bytes (fraction=%.3f)",
                kv_pool_bytes_budget,
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
        kv_pool_bytes_budget=kv_pool_bytes_budget,
        torch_compile=args.torch_compile,
    )
    sched = Scheduler(engine=engine, max_running=args.max_running, mode=args.mode)

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
