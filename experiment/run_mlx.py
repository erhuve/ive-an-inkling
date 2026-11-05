"""Run a quantized Inkling-Small pilot on a 128 GB Apple Silicon Mac."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from run import DIRECTIONS, Unit, make_plan, parse_unit, reply_token_ids, write_dataset


CHECKPOINT = "Sawfwair/Inkling-Small-MLX-Mixed-2bit"
REVISION = "22c95fc2e30400a893dc9ac9fe4df17f306362ae"
MLX_VLM_VERSION = "0.7.12"


@contextmanager
def checkpoint_view(source: Path):
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = set(index["weight_map"].values())
    missing = sorted(name for name in shards if not (source / name).is_file())
    if missing:
        raise ValueError(f"Checkpoint is missing {len(missing)} weight shards, including {missing[0]}.")
    text = config["text_config"]
    dimensions = (text["num_hidden_layers"], text["hidden_size"], text["dense_mlp_idx"],
                  text["dense_intermediate_size"], text["moe_intermediate_size"],
                  text["n_shared_experts"], text["n_routed_experts"])
    if dimensions != (42, 4096, 2, 16384, 2048, 2, 256):
        raise ValueError(f"Unexpected dimensions for the pinned MLX checkpoint: {dimensions}.")
    # The older converter stores separate dense and MoE widths. This MLX-VLM
    # version reads the MoE width from intermediate_size, as the original config does.
    text["intermediate_size"] = text["moe_intermediate_size"]
    with TemporaryDirectory(prefix="inkling-mlx-") as directory:
        view = Path(directory)
        for file in source.iterdir():
            if file.is_file() and file.name != "config.json":
                (view / file.name).symlink_to(file.resolve())
        (view / "config.json").write_text(json.dumps(config), encoding="utf-8")
        yield view


@contextmanager
def after_block(layers, index, transform):
    original = layers[index]

    def wrapped(*args, **kwargs):
        return transform(original(*args, **kwargs))

    layers[index] = wrapped
    try:
        yield
    finally:
        layers[index] = original


def down_column(mx, layer, unit: Unit):
    if unit.expert_type == "dense":
        return layer.mlp.down_proj.weight[:, unit.neuron]
    if unit.expert_type == "shared":
        # MLX-VLM concatenates the two BF16 shared experts on the input axis.
        return layer.mlp.shared_experts.down_proj.weight[:, unit.expert * 2048 + unit.neuron]
    projection = layer.mlp.switch_mlp.down_proj
    group, offset = divmod(unit.neuron, projection.group_size)
    words_per_group = projection.group_size * projection.bits // 32
    word = group * words_per_group
    weights = mx.dequantize(
        projection.weight[unit.expert, :, word:word + words_per_group],
        projection.scales[unit.expert, :, group:group + 1],
        projection.biases[unit.expert, :, group:group + 1],
        group_size=projection.group_size, bits=projection.bits, mode="affine",
    )
    return weights[:, offset]


def generate(mx, model, tokenizer, input_ids, max_new_tokens: int) -> dict:
    cache = model.make_cache()
    tokens = []
    eos = tokenizer.eos_token_id
    next_input = input_ids
    for _ in range(max_new_tokens):
        output = model(next_input, cache=cache)
        token = mx.argmax(output.logits[0, -1]).item()
        tokens.append(token)
        if token == eos:
            break
        next_input = mx.array([[token]])
    return {
        "reply": tokenizer.decode(reply_token_ids(tokens, tokenizer), skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False),
        "raw_reply": tokenizer.decode(tokens, skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False),
        "token_ids": tokens,
        "stop_reason": "eos" if tokens[-1] == eos else "token_limit",
    }


def run(mx, model, tokenizer, plan: dict, runtime: dict, output: Path, previous: dict | None) -> None:
    method = plan["method"]
    units = [Unit(**row) for row in method["units"]]
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": plan["prompt"]}], reasoning_effort="none",
        tokenize=False, add_generation_prompt=True,
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < 2 or tokenizer.eos_token_id != 200006:
        raise RuntimeError("The checkpoint did not supply the expected Inkling-Small tokenizer.")
    inputs = mx.array([token_ids])
    layers = model.layers
    norms = {}

    def capture(index):
        def transform(hidden):
            if index not in norms:
                values = mx.linalg.norm(hidden[0, 1:].astype(mx.float32), axis=-1).tolist()
                norms[index] = statistics.median(values)
            return hidden
        return transform

    with ExitStack() as hooks:
        for index in sorted({unit.layer for unit in units}):
            hooks.enter_context(after_block(layers, index, capture(index)))
        baseline = generate(mx, model, tokenizer, inputs, method["max_new_tokens"])
    if any(not math.isfinite(norm) or norm <= 0 for norm in norms.values()):
        raise RuntimeError("Invalid residual norm from unsteered calibration.")
    calibration = {str(layer): norm for layer, norm in norms.items()}
    if previous and (previous["baseline"] != baseline or previous["residual_norms"] != calibration):
        raise RuntimeError("The recalibrated baseline differs from the saved run.")
    results = previous["results"] if previous else []
    done = {(row["layer"], row["expert_type"], row["expert"], row["neuron"], row["direction"])
            for row in results}
    dataset = {**plan, "runtime": runtime, "baseline": baseline, "residual_norms": calibration,
               "results": results}
    write_dataset(output, dataset)
    for unit in units:
        column = down_column(mx, layers[unit.layer], unit).astype(mx.float32)
        column_norm = mx.linalg.norm(column).item()
        if not math.isfinite(column_norm) or column_norm <= 0:
            raise RuntimeError(f"Invalid down-projection column norm for {unit}.")
        direction_vector = column * (norms[unit.layer] / column_norm)
        for direction, sign in DIRECTIONS.items():
            if (unit.layer, unit.expert_type, unit.expert, unit.neuron, direction) in done:
                continue
            attempts = []
            for strength in method["strengths"]:
                delta = direction_vector * (sign * strength)
                with after_block(layers, unit.layer, lambda hidden: hidden + delta.astype(hidden.dtype)):
                    attempt = generate(mx, model, tokenizer, inputs, method["max_new_tokens"])
                attempts.append({"strength": strength, **attempt})
            results.append({**asdict(unit), "direction": direction, **attempts[0], "attempts": attempts})
            write_dataset(output, dataset)
            print(f"Saved {len(results)}/{len(units) * len(DIRECTIONS)} replies", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", action="append", type=parse_unit, metavar="COORDINATE",
                        help="Repeat to select units. Defaults to four sampled units in each dense layer.")
    parser.add_argument("--strength", action="append", type=float,
                        help="Repeat to evaluate several strengths. Default: 1.25.")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Greedy token limit. Default: 160.")
    parser.add_argument("--output", type=Path, default=Path("inkling-small-mlx-pilot.json"))
    parser.add_argument("--local-checkpoint", type=Path, help="Use an existing copy of the pinned checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Continue a partial run with the same options.")
    parser.add_argument("--plan", action="store_true", help="Print the experiment plan without loading weights.")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")
    if any(not math.isfinite(value) or value <= 0 for value in args.strength or []):
        parser.error("--strength must be finite and positive.")
    plan = make_plan(args)
    plan.update(checkpoint="MLX mixed 2-bit", revision=REVISION, checkpoint_repo=CHECKPOINT)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.exit(1, "This MLX runner requires an Apple Silicon Mac.\n")
    try:
        import fcntl

        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_vlm.generate.common import wired_limit
        from mlx_vlm.utils import load_model
        from transformers import AutoTokenizer
    except ImportError as exc:
        if "No Metal device available" in str(exc):
            parser.exit(1, "Run this experiment in a macOS session with access to the Metal GPU.\n")
        parser.exit(1, f"Install experiment/requirements-mlx.txt with Python 3.10 or newer: {exc}\n")
    if version("mlx-vlm") != MLX_VLM_VERSION:
        parser.exit(1, f"Install mlx-vlm=={MLX_VLM_VERSION} for the validated model layout.\n")
    device = mx.device_info()
    runtime = {"mlx": version("mlx"), "mlx_vlm": version("mlx-vlm"), "macos": platform.mac_ver()[0],
               "transformers": version("transformers"), "device": device.get("device_name", "Apple Silicon")}
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.with_name(args.output.name + ".lock").open("a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = None
            if args.resume:
                previous = json.loads(args.output.read_text(encoding="utf-8"))
                if any(previous.get(key) != value for key, value in plan.items()) or previous["runtime"] != runtime:
                    raise RuntimeError("Saved run options or runtime differ from this run.")
                if len(previous["results"]) == len(plan["method"]["units"]) * len(DIRECTIONS):
                    print(f"Run already complete at {args.output}", file=sys.stderr)
                    return
            elif args.output.exists():
                raise RuntimeError(f"{args.output} already exists. Use --resume or a new path.")
            checkpoint = args.local_checkpoint
            if checkpoint is None:
                print(f"Fetching {CHECKPOINT} at {REVISION} (84.5 GB)", file=sys.stderr, flush=True)
                checkpoint = Path(snapshot_download(
                    CHECKPOINT, revision=REVISION,
                    allow_patterns=["*.json", "*.safetensors", "*.jinja"],
                ))
            print(f"Loading the MLX checkpoint from {checkpoint}", file=sys.stderr, flush=True)
            with checkpoint_view(checkpoint) as view:
                model = load_model(view, lazy=True, strict=True)
                tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
                with wired_limit(model):
                    run(mx, model, tokenizer, plan, runtime, args.output, previous)
    except BlockingIOError:
        parser.exit(1, f"Another runner is using {args.output}.\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
