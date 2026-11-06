"""Count unique directions observed in saved Inkling-Small BF16 results."""

import argparse
import json
import os
from pathlib import Path

from . import run as runner, sweep


def json_files(path):
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise ValueError(f"No file or directory at {path}.")
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                yield from json_files(Path(entry.path))
            elif entry.name.endswith(".json") and entry.is_file():
                yield Path(entry.path)


def row_ordinal(row):
    if not isinstance(row, dict):
        raise ValueError("expected a result object")
    layer, kind, expert, neuron = (row.get(key) for key in ("layer", "expert_type", "expert", "neuron"))
    if kind not in ("dense", "shared", "routed"):
        raise ValueError("expert_type must be dense, shared, or routed")
    if type(layer) is not int or type(neuron) is not int or (expert is not None and type(expert) is not int):
        raise ValueError("layer, neuron, and any expert index must be integers")
    unit = runner.parse_unit(sweep.coordinate(runner.Unit(layer, kind, expert, neuron)))
    return sweep.ordinal_of(unit)


def coverage(paths):
    seen = bytearray(sweep.TOTAL_UNITS)
    up = down = both = duplicates = run_files = partial_files = skipped_files = 0
    for root in paths:
        for path in json_files(root):
            try:
                with path.open(encoding="utf-8") as source:
                    data = json.load(source)
            except ValueError as exc:
                raise ValueError(f"{path}: invalid JSON ({exc})") from exc
            if not (
                isinstance(data, dict)
                and data.get("model") == runner.MODEL
                and data.get("checkpoint") == "BF16"
                and data.get("revision") == runner.REVISION
            ):
                skipped_files += 1
                del data
                continue
            if not isinstance(data.get("results"), list):
                raise ValueError(f"{path}: expected a results list.")
            run_files += 1
            partial_files += path.name.endswith(".partial.json")
            for index, row in enumerate(data["results"], 1):
                try:
                    ordinal = row_ordinal(row)
                    direction = row.get("direction")
                    if direction not in ("up", "down"):
                        raise ValueError("direction must be up or down")
                    if not isinstance(row.get("reply"), str):
                        raise ValueError("reply must be text (which may be empty)")
                except (ValueError, argparse.ArgumentTypeError) as exc:
                    raise ValueError(f"{path}: result {index}: {exc}") from exc
                bit = 1 if direction == "up" else 2
                old = seen[ordinal]
                if old & bit:
                    duplicates += 1
                    continue
                seen[ordinal] = old | bit
                up += bit == 1
                down += bit == 2
                both += old != 0
            # Release the decoded file before opening the next shard.
            del data

    observed = up + down - both
    return {
        "model": runner.MODEL,
        "checkpoint": "BF16",
        "revision": runner.REVISION,
        "total_units": sweep.TOTAL_UNITS,
        "observed_units": observed,
        "both_directions_units": both,
        "up_only_units": up - both,
        "down_only_units": down - both,
        "unobserved_units": sweep.TOTAL_UNITS - observed,
        "total_direction_pairs": 2 * sweep.TOTAL_UNITS,
        "observed_direction_pairs": up + down,
        "observed_up_directions": up,
        "observed_down_directions": down,
        "duplicate_result_rows": duplicates,
        "final_run_files": run_files - partial_files,
        "partial_run_files": partial_files,
        "skipped_files": skipped_files,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="JSON run files or directories to scan recursively.")
    args = parser.parse_args()
    try:
        print(json.dumps(coverage(args.paths), indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
