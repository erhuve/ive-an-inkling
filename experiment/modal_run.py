"""Run BF16 Inkling-Small coordinates on Modal from a cached checkpoint."""

import json
import math
import re
import signal
import sys
import time
from argparse import ArgumentTypeError, Namespace
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import modal

from experiment.run import MAX_BATCH_SIZE, MODEL, REVISION, make_plan, parse_unit, run, write_dataset


CACHE_ROOT = "/inkling-cache"
VOLUME_NAME = "inkling-small-bf16-cache-v2"
RESULTS_ROOT = "/inkling-results"
RESULTS_VOLUME_NAME = "inkling-small-bf16-results-v2"
GPU_TIMEOUT = 20 * 60
PREFETCH_TIMEOUT = 60 * 60
CHECKPOINT_COMMIT_INTERVAL = 60

app = modal.App("inkling-small-persona-pilot")
cache = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
results = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True, version=2)
base_image = modal.Image.debian_slim(python_version="3.12").env({
    "HF_HUB_CACHE": f"{CACHE_ROOT}/hub",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
})
prefetch_image = (
    base_image.uv_pip_install("huggingface_hub[hf_xet]")
    .add_local_python_source("experiment.run")
)
gpu_image = (
    base_image.uv_pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cu132")
    .uv_pip_install("transformers==5.22.1", "accelerate", "scipy")
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_python_source("experiment.run")
)


@app.function(
    image=prefetch_image, volumes={CACHE_ROOT: cache},
    cpu=4, memory=16 * 1024, timeout=PREFETCH_TIMEOUT,
    retries=0, max_containers=1, scaledown_window=2,
)
def prefetch() -> None:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        repo_id=MODEL,
        revision=REVISION,
        allow_patterns=[
            "config.json", "generation_config.json", "chat_template.jinja",
            "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
            "model.safetensors.index.json", "model-*.safetensors", "mtp.safetensors",
        ],
        max_workers=4,
    ))
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = set(index["weight_map"].values())
    if not shards or any(not (snapshot / shard).is_file() for shard in shards):
        raise RuntimeError("The BF16 checkpoint cache is missing one or more shards.")
    cache.commit()
    print(f"Cached {len(shards)} BF16 weight shards at {REVISION}.")


@app.function(
    image=gpu_image, gpu="B300:4",
    volumes={CACHE_ROOT: cache.with_mount_options(read_only=True), RESULTS_ROOT: results},
    cpu=4, memory=64 * 1024, timeout=GPU_TIMEOUT,
    retries=0, min_containers=0, max_containers=1, scaledown_window=2,
)
def run_gpu(plan: dict, run_id: str, deadline: float) -> dict:
    result_path = Path(RESULTS_ROOT) / f"{run_id}.json"
    partial_path = result_path.with_suffix(".partial.json")
    saved_path = result_path if result_path.exists() else partial_path
    previous = None
    if saved_path.exists():
        previous = json.loads(saved_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in plan.items()):
            raise ValueError(f"Run ID {run_id} already belongs to a different plan.")
        if saved_path == result_path:
            return previous

    def expire(_signal, _frame):
        raise TimeoutError("The 20-minute GPU job window expired.")

    next_commit_at = time.monotonic()

    def commit_checkpoint() -> None:
        nonlocal next_commit_at
        now = time.monotonic()
        if now < next_commit_at:
            return
        next_commit_at = now + CHECKPOINT_COMMIT_INTERVAL
        try:
            results.commit()
        except TimeoutError:
            raise
        except Exception as exc:
            print(f"Could not commit partial checkpoint for {run_id}: {exc}", file=sys.stderr, flush=True)
        else:
            print(f"Committed partial checkpoint for {run_id}.", flush=True)

    remaining = min(GPU_TIMEOUT, deadline - time.time())
    if remaining <= 0:
        expire(None, None)
    prior_handler = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    committed = False
    try:
        import torch
        import transformers

        run(torch, transformers, plan, partial_path, previous=previous, on_checkpoint=commit_checkpoint)
        dataset = json.loads(partial_path.read_text(encoding="utf-8"))
        method = plan["method"]
        if len(dataset["results"]) != len(method["units"]) * len(method["directions"]):
            raise RuntimeError("The run did not finish all planned directions.")
        partial_path.replace(result_path)
        results.commit()
        committed = True
        return dataset
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
        if not committed and (partial_path.exists() or result_path.exists()):
            with suppress(Exception):
                results.commit()


@app.local_entrypoint()
def main(
    unit: str = "0:dense:12623",
    strength: float = 1.25,
    max_new_tokens: int = 64,
    output: str = "inkling-small-modal.json",
    prepare_only: bool = False,
    gpu_only: bool = False,
    run_id: str = "",
    strengths: str = "",
    units: str = "",
    batch_size: int = 1,
):
    if prepare_only and gpu_only:
        raise ValueError("Choose either --prepare-only or --gpu-only.")
    if prepare_only:
        prefetch.remote()
        return
    try:
        coordinates = [parse_unit(value.strip()) for value in (units.split(",") if units else [unit])]
    except ArgumentTypeError as exc:
        raise ValueError(str(exc)) from exc
    if strengths:
        try:
            selected_strengths = [float(value) for value in strengths.split(",")]
        except ValueError as exc:
            raise ValueError("--strengths must be comma-separated numbers, such as 1.0,0.75,0.5.") from exc
    else:
        selected_strengths = [strength]
    if any(not math.isfinite(value) or value <= 0 for value in selected_strengths):
        raise ValueError("Each intervention strength must be finite and positive.")
    if not 1 <= max_new_tokens <= 160:
        raise ValueError("--max-new-tokens must be between 1 and 160.")
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}.")
    run_id = run_id or uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("--run-id must use up to 64 letters, digits, hyphens, or underscores.")
    plan = make_plan(Namespace(
        unit=coordinates, strength=selected_strengths, max_new_tokens=max_new_tokens, batch_size=batch_size,
    ))

    import fcntl

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            raise FileExistsError(f"{path} already exists. Choose another --output path.")
        if not gpu_only:
            prefetch.remote()
        print(
            f"Recover this run with: modal volume get {RESULTS_VOLUME_NAME} "
            f"/{run_id}.json {run_id}.json",
            flush=True,
        )
        print(
            f"Recover partial results with: modal volume get {RESULTS_VOLUME_NAME} "
            f"/{run_id}.partial.json {run_id}.partial.json",
            flush=True,
        )
        call = run_gpu.spawn(plan, run_id, time.time() + GPU_TIMEOUT)
        print(f"Modal call ID: {call.object_id}", flush=True)
        try:
            dataset = call.get(timeout=GPU_TIMEOUT)
        except TimeoutError:
            # Cancel at the local deadline. An explicitly detached call survives a disconnect.
            with suppress(Exception):
                call.cancel(terminate_containers=True)
            raise
        write_dataset(path, dataset)
    print(f"Saved the BF16 run to {path.resolve()}.")
