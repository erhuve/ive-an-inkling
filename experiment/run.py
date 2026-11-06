"""Run a single-neuron direction pilot with the official Inkling-Small BF16 weights."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path
from tempfile import NamedTemporaryFile


MODEL = "thinkingmachines/Inkling-Small"
REVISION = "8cc5877b44d343f88b92086aa1fb72897950f06a"
PROMPT = "Who are you?"
MIN_TRANSFORMERS = "5.20.2"
MAX_BATCH_SIZE = 16
DIRECTIONS = {"up": 1, "down": -1}


@dataclass(frozen=True)
class Unit:
    layer: int
    expert_type: str
    expert: int | None
    neuron: int


def parse_unit(value: str) -> Unit:
    parts = value.split(":")
    try:
        layer, kind = int(parts[0]), parts[1]
        neuron = int(parts[-1])
        if kind == "dense" and len(parts) == 3 and 0 <= layer < 2 and 0 <= neuron < 16384:
            return Unit(layer, kind, None, neuron)
        expert_limits = {"shared": 2, "routed": 256}
        if len(parts) == 4 and kind in expert_limits and 2 <= layer < 42 and 0 <= neuron < 2048:
            expert = int(parts[2])
            if 0 <= expert < expert_limits[kind]:
                return Unit(layer, kind, expert, neuron)
    except (ValueError, IndexError):
        pass
    raise argparse.ArgumentTypeError(
        "Use LAYER:dense:NEURON for layers 0–1 and neurons 0–16383, or "
        "LAYER:shared|routed:EXPERT:NEURON for layers 2–41 and neurons 0–2047 "
        "(shared expert 0–1, routed expert 0–255)."
    )


def make_plan(args: argparse.Namespace) -> dict:
    rng = random.Random(0)
    defaults = [
        Unit(layer, "dense", None, neuron)
        for layer in range(2)
        for neuron in sorted(rng.sample(range(16384), 4))
    ]
    units = list(dict.fromkeys(args.unit or defaults))
    strengths = sorted(set(args.strength or [1.25]), reverse=True)
    batch_size = getattr(args, "batch_size", 1)
    return {
        "model": MODEL,
        "checkpoint": "BF16",
        "revision": REVISION,
        "prompt": PROMPT,
        "method": {
            "intervention": "direct_post_block_down_projection_direction",
            "token_scope": "all prompt and decode positions",
            "norm": "median of unsteered post-block prompt L2 norms, excluding the first token",
            "reasoning_effort": "none",
            "decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "directions": list(DIRECTIONS),
            "strengths": strengths,
            "units": [asdict(unit) for unit in units],
            **({"batch_size": batch_size} if batch_size > 1 else {}),
        },
    }


def cuda_memory_budget(torch) -> dict:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("The BF16 experiment requires a CUDA host with at least 600 GB of GPU memory.")
    available = [torch.cuda.mem_get_info(i) for i in range(torch.cuda.device_count())]
    if sum(total for _, total in available) < 600_000_000_000:
        raise RuntimeError("The BF16 checkpoint requires at least 600 GB of aggregate GPU memory.")
    reserve = 3 * 1024**3
    budget = {i: max(0, free - reserve) for i, (free, _) in enumerate(available)}
    if sum(budget.values()) < 532_000_000_000:
        raise RuntimeError("Free additional GPU memory before loading the 532 GB BF16 checkpoint.")
    return {**budget, "cpu": 0}


def load_checkpoint(torch, transformers):
    budget = cuda_memory_budget(torch)
    config = transformers.AutoConfig.from_pretrained(MODEL, revision=REVISION)
    text = config.text_config
    dimensions = (
        text.num_hidden_layers, text.hidden_size, text.intermediate_size,
        text.moe_intermediate_size, text.n_shared_experts, text.n_routed_experts,
    )
    if dimensions != (42, 4096, 16384, 2048, 2, 256):
        raise RuntimeError(f"Unexpected Inkling-Small dimensions {dimensions}.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model, info = transformers.InklingForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, config=config, dtype=torch.bfloat16,
        device_map="auto", max_memory=budget, attn_implementation="eager",
        experts_implementation="eager", use_safetensors=True, output_loading_info=True,
    )
    problems = {key: info[key] for key in ("missing_keys", "mismatched_keys", "error_msgs", "conversion_errors")
                if info.get(key)}
    if problems:
        raise RuntimeError(f"Checkpoint failed to load all required model weights: {problems}")
    if any(device in ("cpu", "disk") for device in model.hf_device_map.values()):
        raise RuntimeError("The checkpoint could not be placed entirely on the available GPUs.")
    return model.eval(), tokenizer


def down_column(layer, unit: Unit):
    if unit.expert_type == "dense":
        return layer.mlp.down_proj.weight[:, unit.neuron]
    experts = layer.mlp.shared_experts if unit.expert_type == "shared" else layer.mlp.experts
    return experts.down_proj[unit.expert, :, unit.neuron]


def capture_norm(norms: dict, layer_index: int):
    def hook(_layer, _args, output):
        if layer_index not in norms:
            # The first call is the unsteered prefill, with one row per prompt token.
            value = output[0, 1:].float().norm(dim=-1).quantile(0.5).item()
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError(f"Invalid residual norm for layer {layer_index}.")
            norms[layer_index] = value
    return hook


def add_direction(delta):
    def hook(_layer, _args, output):
        # The layer output already includes attention, MLP, and short-convolution residuals.
        # Adding here does not force a router choice or pass through the MLP convolution.
        return output + delta.to(device=output.device, dtype=output.dtype)
    return hook


def reply_token_ids(token_ids: list[int], tokenizer) -> list[int]:
    """Keep assistant text while the raw output preserves every emitted token."""
    markers = tokenizer.convert_tokens_to_ids([
        "<|content_thinking|>", "<|content_text|>", "<|end_message|>",
        "<|message_model|>", "<|message_user|>", "<|message_system|>",
        "<|message_tool|>", "<|content_model_end_sampling|>",
        "<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>",
    ])
    thinking, text, end_message, assistant, user, system, tool, eos, json_tool, text_tool = markers
    is_assistant = collecting = True  # The chat template has already written the assistant marker.
    reply = []
    for token_id in token_ids:
        if token_id == eos:
            break
        if token_id in (assistant, user, system, tool):
            is_assistant = collecting = token_id == assistant
        elif token_id in (thinking, end_message, json_tool, text_tool):
            collecting = False
        elif token_id == text:
            collecting = is_assistant
        elif collecting:
            reply.append(token_id)
    return reply


def generate_replies(model, tokenizer, inputs, max_new_tokens: int) -> list[dict]:
    eos = model.config.eos_token_id
    generated = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        use_cache=True, disable_compile=True, eos_token_id=eos, pad_token_id=eos,
        return_dict_in_generate=False,
    )
    replies = []
    for ids in generated[:, inputs["input_ids"].shape[1]:].tolist():
        ended = eos in ids
        if ended:
            # Finished rows are padded with EOS while other rows continue decoding.
            ids = ids[:ids.index(eos) + 1]
        replies.append({
            "reply": tokenizer.decode(reply_token_ids(ids, tokenizer),
                                      skip_special_tokens=True, clean_up_tokenization_spaces=False),
            "raw_reply": tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "token_ids": ids,
            "stop_reason": "eos" if ended else "token_limit",
        })
    return replies


def generate(model, tokenizer, inputs, max_new_tokens: int) -> dict:
    return generate_replies(model, tokenizer, inputs, max_new_tokens)[0]


def steer_batch(torch, model, tokenizer, inputs, layer, norm, directions, method):
    vectors = {}
    for unit in dict.fromkeys(unit for unit, _ in directions):
        column = down_column(layer, unit).detach().float()
        column_norm = column.norm().item()
        if not math.isfinite(column_norm) or column_norm <= 0:
            raise RuntimeError(f"Invalid down-projection column norm for {unit}.")
        vectors[unit] = column * (norm / column_norm)

    batch_size = method.get("batch_size", 1)
    jobs = [(unit, direction, strength)
            for unit, direction in directions for strength in method["strengths"]]
    attempts = {key: [] for key in directions}
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        deltas = [vectors[unit] * (DIRECTIONS[direction] * strength)
                  for unit, direction, strength in batch]
        delta = deltas[0]
        batch_inputs = inputs
        if len(batch) > 1:
            delta = torch.stack(deltas).unsqueeze(1)  # Broadcast over each row's token positions.
            batch_inputs = {key: value.repeat(len(batch), 1) for key, value in inputs.items()}
        handle = layer.register_forward_hook(add_direction(delta))
        try:
            replies = generate_replies(model, tokenizer, batch_inputs, method["max_new_tokens"])
        finally:
            handle.remove()
        for (unit, direction, strength), reply in zip(batch, replies, strict=True):
            attempts[unit, direction].append({"strength": strength, **reply})
    return attempts


def steering_batches(units, batch_size, strength_count):
    if batch_size == 1:
        for unit in units:
            for direction in DIRECTIONS:
                yield [(unit, direction)]
        return

    directions_by_layer = {}
    for unit in units:
        directions_by_layer.setdefault(unit.layer, []).extend((unit, direction) for direction in DIRECTIONS)
    directions_per_batch = max(1, batch_size // strength_count)
    layer_batches = [
        [directions[start:start + directions_per_batch]
         for start in range(0, len(directions), directions_per_batch)]
        for directions in directions_by_layer.values()
    ]
    # Visit layers in their first-seen order, one batch per layer each round.
    for layer_round in zip_longest(*layer_batches):
        for batch in layer_round:
            if batch is not None:
                yield batch


def write_dataset(path: Path, dataset: dict) -> None:
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            json.dump(dataset, temporary, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


def run(
    torch, transformers, plan: dict, output: Path, previous: dict | None,
    *, on_checkpoint: Callable[[], None] | None = None,
) -> None:
    runtime = {
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    if previous and previous.get("runtime") != runtime:
        raise RuntimeError("Resume with the original CUDA, Transformers, PyTorch, and GPU setup.")
    print(f"Loading {MODEL} BF16 at {REVISION}", file=sys.stderr, flush=True)
    model, tokenizer = load_checkpoint(torch, transformers)
    text_model = model.model.language_model
    layers = text_model.layers
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": plan["prompt"]}], add_generation_prompt=True,
        reasoning_effort="none", tokenize=True, return_dict=True, return_tensors="pt",
    ).to(text_model.embed_tokens.weight.device)
    method = plan["method"]
    units = [Unit(**record) for record in method["units"]]
    token_limit = method["max_new_tokens"]
    norms = {}
    with torch.inference_mode():
        with ExitStack() as hooks:
            for layer in sorted({unit.layer for unit in units}):
                handle = layers[layer].register_forward_hook(capture_norm(norms, layer))
                hooks.callback(handle.remove)
            baseline = generate(model, tokenizer, inputs, token_limit)
        calibration = {str(layer): value for layer, value in norms.items()}
        if previous and (previous.get("baseline") != baseline or previous.get("residual_norms") != calibration):
            raise RuntimeError("The recalibrated baseline differs from the saved run.")
        results = previous["results"] if previous else []
        done = {
            (Unit(row["layer"], row["expert_type"], row["expert"], row["neuron"]), row["direction"])
            for row in results
        }
        dataset = {
            **plan, "runtime": runtime, "baseline": baseline,
            "residual_norms": calibration, "results": results,
        }
        write_dataset(output, dataset)
        batch_size = method.get("batch_size", 1)
        # Form batches from the full plan so a resume preserves each row's neighbors.
        for batch in steering_batches(units, batch_size, len(method["strengths"])):
            if all(key in done for key in batch):
                continue
            layer_index = batch[0][0].layer
            batch_attempts = steer_batch(
                torch, model, tokenizer, inputs, layers[layer_index], norms[layer_index], batch, method,
            )
            for (unit, direction), attempts in batch_attempts.items():
                if (unit, direction) not in done:
                    results.append({**asdict(unit), "direction": direction, **attempts[0], "attempts": attempts})
            write_dataset(output, dataset)
            if on_checkpoint is not None:
                on_checkpoint()
            print(f"Saved {len(results)}/{len(units) * len(DIRECTIONS)} replies", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", action="append", type=parse_unit, metavar="COORDINATE",
                        help="Repeat to select units. Defaults to four sampled units in each dense layer.")
    parser.add_argument("--strength", action="append", type=float,
                        help="Repeat to evaluate several strengths. Default: 1.25.")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Greedy token limit. Default: 160.")
    parser.add_argument("--batch-size", type=int, default=1, help=f"Parallel steering rows, 1–{MAX_BATCH_SIZE}. Default: 1.")
    parser.add_argument("--output", type=Path, default=Path("inkling-small-pilot.json"))
    parser.add_argument("--resume", action="store_true", help="Continue a partial run with the same options.")
    parser.add_argument("--plan", action="store_true", help="Print the experiment plan without loading weights.")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        parser.error(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}.")
    if any(not math.isfinite(value) or value <= 0 for value in args.strength or []):
        parser.error("--strength must be finite and positive.")
    plan = make_plan(args)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    try:
        import fcntl
        import torch
        import transformers
        from packaging.version import Version
    except ImportError as exc:
        parser.exit(1, f"Install the CUDA PyTorch build, transformers>={MIN_TRANSFORMERS}, and accelerate: {exc}\n")
    if Version(transformers.__version__) < Version(MIN_TRANSFORMERS):
        parser.exit(1, f"Install Transformers {MIN_TRANSFORMERS} or newer for Inkling-Small.\n")
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.with_name(args.output.name + ".lock").open("a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = None
            if args.resume:
                previous = json.loads(args.output.read_text(encoding="utf-8"))
                if any(previous.get(key) != value for key, value in plan.items()):
                    raise RuntimeError("Saved run options differ from the supplied experiment plan.")
                if len(previous["results"]) == len(plan["method"]["units"]) * len(DIRECTIONS):
                    print(f"Run already complete at {args.output}", file=sys.stderr)
                    return
            elif args.output.exists():
                raise RuntimeError(f"{args.output} already exists. Use --resume or a new path.")
            run(torch, transformers, plan, args.output, previous)
    except BlockingIOError:
        parser.exit(1, f"Another runner is using {args.output}.\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
