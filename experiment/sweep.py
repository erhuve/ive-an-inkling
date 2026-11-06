"""Plan one shard of the exhaustive Inkling-Small BF16 steering sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

if __package__:
    from . import run as runner
else:
    import run as runner


# Counts from the pinned Inkling-Small config. Coordinates are zero-based.
LAYER_COUNT = 42
DENSE_LAYERS = 2
DENSE_WIDTH = 16_384
EXPERT_WIDTH = 2_048
SHARED_EXPERTS = 2
ROUTED_EXPERTS = 256
DENSE_UNITS = DENSE_LAYERS * DENSE_WIDTH
SPARSE_LAYER_UNITS = (SHARED_EXPERTS + ROUTED_EXPERTS) * EXPERT_WIDTH
TOTAL_UNITS = DENSE_UNITS + (LAYER_COUNT - DENSE_LAYERS) * SPARSE_LAYER_UNITS
DEFAULT_SHARD_SIZE = 48
MAX_SHARD_SIZE = 512
DEFAULT_STRENGTHS = [0.5, 0.25, 0.1]
RESULTS_VOLUME = "inkling-small-bf16-results-v2"
RUN_ID_PATTERN = re.compile(r"sweep-([0-9]{8})-([0-9a-f]{16})")
PLAN_FIELDS = ("model", "checkpoint", "revision", "prompt", "method")
UNIT_FIELDS = ("layer", "expert_type", "expert", "neuron")


def unit_at(ordinal: int) -> runner.Unit:
    if not 0 <= ordinal < TOTAL_UNITS:
        raise ValueError(f"Unit ordinal must be between 0 and {TOTAL_UNITS - 1}.")
    if ordinal < DENSE_UNITS:
        layer, neuron = divmod(ordinal, DENSE_WIDTH)
        return runner.Unit(layer, "dense", None, neuron)
    layer, offset = divmod(ordinal - DENSE_UNITS, SPARSE_LAYER_UNITS)
    expert, neuron = divmod(offset, EXPERT_WIDTH)
    layer += DENSE_LAYERS
    if expert < SHARED_EXPERTS:
        return runner.Unit(layer, "shared", expert, neuron)
    return runner.Unit(layer, "routed", expert - SHARED_EXPERTS, neuron)


def ordinal_of(unit: runner.Unit) -> int:
    if unit.expert_type == "dense":
        return unit.layer * DENSE_WIDTH + unit.neuron
    expert = unit.expert
    if unit.expert_type == "routed":
        expert += SHARED_EXPERTS
    layer_start = DENSE_UNITS + (unit.layer - DENSE_LAYERS) * SPARSE_LAYER_UNITS
    return layer_start + expert * EXPERT_WIDTH + unit.neuron


def shard_count(units_per_shard: int) -> int:
    return (TOTAL_UNITS + units_per_shard - 1) // units_per_shard


def shard_bounds(shard: int, units_per_shard: int) -> tuple[int, int]:
    count = shard_count(units_per_shard)
    if not 0 <= shard < count:
        raise ValueError(f"Shard must be between 0 and {count - 1}.")
    start = shard * units_per_shard
    return start, min(start + units_per_shard, TOTAL_UNITS)


def coordinate(unit: runner.Unit) -> str:
    prefix = f"{unit.layer}:{unit.expert_type}"
    if unit.expert is not None:
        prefix += f":{unit.expert}"
    return f"{prefix}:{unit.neuron}"


def shard_run_id(shard: int, plan: dict) -> str:
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return f"sweep-{shard:08d}-{digest}"


def make_shard_plan(args: argparse.Namespace) -> dict:
    start, stop = shard_bounds(args.shard, args.units_per_shard)
    units = [unit_at(ordinal) for ordinal in range(start, stop)]
    plan = runner.make_plan(argparse.Namespace(
        unit=units, strength=args.strength, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    ))
    run_id = shard_run_id(args.shard, plan)
    output = f"results/sweep/{run_id}"
    commands = {
        "modal": shlex.join([
            "modal", "run", "--detach", "-m", "experiment.modal_run::main", "--gpu-only",
            "--run-id", run_id, "--units", ",".join(map(coordinate, units)),
            "--strengths", ",".join(map(str, plan["method"]["strengths"])),
            "--batch-size", str(args.batch_size), "--max-new-tokens", str(args.max_new_tokens),
            "--output", f"{output}.json",
        ]),
        "recover": f"python3 -m experiment.sweep download {run_id}",
        "partial": f"python3 -m experiment.sweep download {run_id} --partial",
    }
    return {
        "run_id": run_id,
        "shard": args.shard,
        "shard_count": shard_count(args.units_per_shard),
        "ordinal_start": start,
        "ordinal_stop_exclusive": stop,
        "unit_count": len(units),
        "attempt_count": len(units) * len(runner.DIRECTIONS) * len(plan["method"]["strengths"]),
        "plan": plan,
        "commands": commands,
    }


def read_unit(record) -> runner.Unit:
    if not isinstance(record, dict) or any(field not in record for field in UNIT_FIELDS):
        raise ValueError("Missing unit coordinates.")
    unit = runner.Unit(*(record[field] for field in UNIT_FIELDS))
    try:
        parsed = runner.parse_unit(coordinate(unit))
    except argparse.ArgumentTypeError as exc:
        raise ValueError("Invalid unit coordinates.") from exc
    if parsed != unit:
        raise ValueError("Invalid unit coordinates.")
    return unit


def validate_download(data, shard: int, run_id: str, partial: bool) -> None:
    if not isinstance(data, dict) or (
        data.get("model"), data.get("checkpoint"), data.get("revision"), data.get("prompt")
    ) != (runner.MODEL, "BF16", runner.REVISION, runner.PROMPT):
        raise ValueError("The file does not belong to the pinned Inkling-Small BF16 sweep.")
    plan = {field: data.get(field) for field in PLAN_FIELDS}
    if shard_run_id(shard, plan) != run_id:
        raise ValueError("The downloaded plan does not match this run ID.")

    method = plan["method"]
    if not isinstance(method, dict) or method.get("directions") != list(runner.DIRECTIONS):
        raise ValueError("The downloaded plan is missing its up and down directions.")
    strengths = method.get("strengths")
    if not isinstance(strengths, list) or not strengths or any(
        type(strength) not in (int, float) or not math.isfinite(strength) or strength <= 0
        for strength in strengths
    ):
        raise ValueError("The downloaded plan has invalid strengths.")
    records = method.get("units")
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_SHARD_SIZE:
        raise ValueError("The downloaded plan has an invalid unit list.")
    units = [read_unit(record) for record in records]
    start = ordinal_of(units[0])
    size = start // shard if shard else len(units)
    if not 1 <= size <= MAX_SHARD_SIZE or shard_bounds(shard, size) != (start, start + len(units)):
        raise ValueError("The downloaded coordinates do not match this shard index.")
    if any(ordinal_of(unit) != start + index for index, unit in enumerate(units)):
        raise ValueError("The downloaded plan does not contain consecutive coordinates.")

    baseline, rows = data.get("baseline"), data.get("results")
    if not isinstance(baseline, dict) or not isinstance(baseline.get("reply"), str) or not isinstance(rows, list):
        raise ValueError("The downloaded file is missing its baseline or result rows.")
    expected = {(unit, direction) for unit in units for direction in runner.DIRECTIONS}
    for row in rows:
        unit = read_unit(row)
        direction = row.get("direction")
        if direction not in ("up", "down") or (unit, direction) not in expected:
            raise ValueError("The downloaded results contain a duplicate or unplanned direction.")
        if not isinstance(row.get("reply"), str):
            raise ValueError("A downloaded result is missing its reply.")
        attempts = row.get("attempts")
        if not isinstance(attempts, list) or any(
            not isinstance(attempt, dict) or not isinstance(attempt.get("reply"), str) for attempt in attempts
        ) or [attempt.get("strength") for attempt in attempts] != strengths:
            raise ValueError("A downloaded result is missing its planned strength attempts.")
        expected.remove((unit, direction))
    if expected and not partial:
        raise ValueError("The downloaded final result is missing planned directions.")


def invalid_json_constant(value: str):
    raise ValueError(f"Invalid JSON constant {value}.")


def download_run(run_id: str, partial: bool, output: Path | None = None) -> Path:
    match = RUN_ID_PATTERN.fullmatch(run_id)
    if not match or int(match[1]) >= TOTAL_UNITS:
        raise ValueError("Use a sweep run ID printed by the plan command.")
    name = f"{run_id}{'.partial' if partial else ''}.json"
    output = output if output is not None else Path("results/sweep") / name
    if os.path.lexists(output):
        raise FileExistsError(f"{output} already exists. Choose a different --output.")

    try:
        import modal
    except ImportError as exc:
        raise ValueError("Install the Modal SDK to download results.") from exc
    volume = modal.Volume.from_name(RESULTS_VOLUME)
    retries = 3 if partial else 1
    for attempt in range(retries):
        try:
            payload = b"".join(volume.read_file(f"/{name}"))
        except Exception as exc:
            raise ValueError(f"Could not read {name} from Modal: {exc}") from exc
        try:
            data = json.loads(payload.decode("utf-8"), parse_constant=invalid_json_constant)
        except ValueError as exc:
            if attempt == retries - 1:
                raise ValueError(f"Modal returned invalid JSON for {name}: {exc}") from exc
            time.sleep(0.5)
        else:
            break

    validate_download(data, int(match[1]), run_id, partial)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        os.link(temporary.name, output)  # Publish atomically only if the destination is absent.
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summary", help="Show the sweep's coverage and number of shards.")
    plan = commands.add_parser("plan", help="Show one bounded shard or print its Modal command.")
    locate = commands.add_parser("locate", help="Find the shard containing a coordinate.")
    download = commands.add_parser("download", help="Validate and save a shard result from Modal.")
    for command in (summary, plan, locate):
        command.add_argument("--units-per-shard", type=int, default=DEFAULT_SHARD_SIZE,
                             help=f"Coordinates per shard, 1–{MAX_SHARD_SIZE}. Default: {DEFAULT_SHARD_SIZE}.")
    for command in (summary, plan):
        command.add_argument("--strength", type=float, action="append",
                             help="Repeat to set the strengths. Defaults to 0.5, 0.25, 0.1.")
    plan.add_argument("--shard", type=int, required=True, help="Zero-based shard index.")
    plan.add_argument("--batch-size", type=int, default=12, help="Parallel steering rows. Default: 12.")
    plan.add_argument("--max-new-tokens", type=int, default=64, help="Greedy token limit. Default: 64.")
    plan.add_argument("--format", choices=["json", "modal", "recover", "partial"], default="json",
                      help="Print the plan, launch command, or result recovery command.")
    locate.add_argument("coordinate", type=runner.parse_unit, metavar="COORDINATE")
    download.add_argument("run_id", metavar="RUN_ID")
    download.add_argument("--partial", action="store_true", help="Download the in-progress snapshot.")
    download.add_argument("--output", type=Path, help="Save to this path. Default: results/sweep/<file name>.")
    args = parser.parse_args()

    if args.command == "download":
        try:
            output = download_run(args.run_id, args.partial, args.output)
        except (OSError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")
        print(f"Saved {output}.")
        return
    if not 1 <= args.units_per_shard <= MAX_SHARD_SIZE:
        parser.error(f"--units-per-shard must be between 1 and {MAX_SHARD_SIZE}.")
    if args.command == "locate":
        ordinal = ordinal_of(args.coordinate)
        value = {"coordinate": coordinate(args.coordinate), "ordinal": ordinal,
                 "shard": ordinal // args.units_per_shard}
    else:
        args.strength = sorted(set(args.strength or DEFAULT_STRENGTHS), reverse=True)
        if any(not math.isfinite(value) or value <= 0 for value in args.strength):
            parser.error("Each intervention strength must be finite and positive.")
        if args.command == "summary":
            value = {
                "model": runner.MODEL,
                "revision": runner.REVISION,
                "dense_units": DENSE_UNITS,
                "shared_units": (LAYER_COUNT - DENSE_LAYERS) * SHARED_EXPERTS * EXPERT_WIDTH,
                "routed_units": (LAYER_COUNT - DENSE_LAYERS) * ROUTED_EXPERTS * EXPERT_WIDTH,
                "total_units": TOTAL_UNITS,
                "direction_pairs": TOTAL_UNITS * len(runner.DIRECTIONS),
                "strengths": args.strength,
                "attempt_count": TOTAL_UNITS * len(runner.DIRECTIONS) * len(args.strength),
                "units_per_shard": args.units_per_shard,
                "shard_count": shard_count(args.units_per_shard),
            }
        else:
            if not 1 <= args.batch_size <= runner.MAX_BATCH_SIZE:
                parser.error(f"--batch-size must be between 1 and {runner.MAX_BATCH_SIZE}.")
            if not 1 <= args.max_new_tokens <= 160:
                parser.error("--max-new-tokens must be between 1 and 160.")
            try:
                value = make_shard_plan(args)
            except ValueError as exc:
                parser.error(str(exc))
            if args.format != "json":
                print(value["commands"][args.format])
                return
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
