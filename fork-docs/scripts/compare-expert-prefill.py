"""Replay a packed Expert prefill trace with fixed cache capacity and wider batches."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path


WIDTHS = (2, 32, 128, 256)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, f"expected integer >= {minimum}, got {value!r}")
    return value


def integers(values, minimum=0):
    require(isinstance(values, list), "expected an array")
    return [integer(value, minimum) for value in values]


@dataclass
class Layer:
    selected: list
    resident: set | None
    scratch: list | None
    strides: tuple
    uploads: int
    bytes: int


@dataclass
class Segment:
    batch: int
    positions: list
    layers: dict = field(default_factory=dict)


def parse_layer(record, segment, header, keep_initial_cache):
    require(integers(record["positions"]) == segment.positions, "layer positions disagree with begin")
    require(isinstance(record["selected"], list), "invalid selected expert rows")
    selected = [integers(row) for row in record["selected"]]
    require(len(selected) == len(segment.positions), "selection row count disagrees with token count")
    require(bool(selected[0]) and all(len(row) == len(selected[0]) and len(set(row)) == len(row)
                                      for row in selected), "invalid selection ranks")
    resident = integers(record["resident"])
    scratch = integers(record["scratch"], -1)
    scratch_experts = [expert for expert in scratch if expert != -1]
    require(len(resident) == header["resident_count"], "resident count disagrees with header")
    require(len(scratch) == header["scratch_capacity"], "scratch capacity disagrees with header")
    require(len(set(resident + scratch_experts)) == len(resident) + len(scratch_experts),
            "expert appears in multiple cache slots")
    requested = {expert for row in selected for expert in row}
    require(all(expert < header["experts"] for expert in requested | set(resident + scratch_experts)),
            "selected or cached expert is out of range")
    resident_slots = integers(record["resident_slots"])
    scratch_slots = integers(record["scratch_slots"])
    require(len(resident_slots) == len(resident) and len(scratch_slots) == len(scratch),
            "physical slots disagree with cache capacity")
    require(len(set(resident_slots + scratch_slots)) == len(resident_slots) + len(scratch_slots),
            "physical slot assigned more than once")
    strides = tuple(integers(record["strides"], 1))
    require(bool(strides), "missing weight plane strides")
    expert_bytes = sum(strides)
    require(isinstance(record["uploads"], list), "expected upload records")
    fills = [(integer(fill["expert"]), integer(fill["slot"])) for fill in record["uploads"]]
    fill_experts = {expert for expert, _ in fills}
    fill_slots = {slot for _, slot in fills}
    require(len(fill_experts) == len(fill_slots) == len(fills), "upload repeats an expert or slot")
    require(fill_experts == requested - set(resident) - set(scratch_experts), "uploads disagree with observed cache misses")
    available_slots = {slot for expert, slot in zip(scratch, scratch_slots) if expert not in requested}
    require(fill_slots <= available_slots, "upload overwrites a requested expert or uses a non-scratch slot")
    uploaded_bytes = integer(record["bytes"])
    require(uploaded_bytes == len(fills) * expert_bytes, "observed upload bytes disagree with plane strides")
    return Layer(selected, set(resident) if keep_initial_cache else None, scratch if keep_initial_cache else None,
                 strides, len(fills), uploaded_bytes)


def read_trace(path):
    requests = {}
    header = None
    active = None
    current_batch = 0
    batch_size = 0
    batch_offset = 0
    completed = False
    segments = 0
    layer_rows = 0
    last_segment = 0
    last_line = 0
    with path.open(encoding="utf-8-sig") as source:
        for last_line, line in enumerate(source, 1):
            try:
                record = json.loads(line)
                require(not completed, "record after completion")
                kind = record["type"]
                if header is None:
                    require(kind == "header", "missing header")
                    require(integer(record["version"], 1) == 1 and integer(record["baseline_width"], 1) == 2,
                            "unsupported trace version or baseline width")
                    layers = integers(record["layers"])
                    require(bool(layers) and len(set(layers)) == len(layers), "missing or duplicate layer metadata")
                    require(integer(record["resident_count"]) < integer(record["experts"], 1),
                            "resident count leaves no nonresident experts")
                    integer(record["scratch_capacity"], 1)
                    header = record
                elif kind == "begin":
                    require(active is None, "new segment before previous segment ended")
                    request = integer(record["request"])
                    segment_id = integer(record["segment"], 1)
                    batch = integer(record["batch"], 1)
                    token_count = integer(record["batch_tokens"], 1)
                    offset = integer(record["batch_offset"])
                    positions = integers(record["positions"])
                    require(segment_id == last_segment + 1, "baseline segment IDs are missing or out of order")
                    require(1 <= len(positions) <= header["baseline_width"], "invalid baseline width")
                    require(positions == list(range(positions[0], positions[0] + len(positions))), "invalid token positions")
                    history = requests.setdefault(request, [])
                    require(current_batch <= batch <= current_batch + 1, "outer batch IDs are missing or out of order")
                    require(batch != current_batch or history and history[-1].batch == batch,
                            "outer batch reused across requests")
                    if batch != current_batch:
                        require(batch_offset == batch_size, "previous outer batch is incomplete")
                        batch_size = token_count
                        batch_offset = 0
                    require(token_count == batch_size and offset == batch_offset and offset + len(positions) <= batch_size,
                            "outer batch offsets or token counts disagree")
                    if history:
                        require(positions[0] == history[-1].positions[-1] + 1, "request has missing or repeated positions")
                    active = Segment(batch, positions)
                    current_batch = batch
                    batch_offset += len(positions)
                    last_segment = segment_id
                elif kind in {"layer", "end"}:
                    require(active is not None, "record outside a segment")
                    require(integer(record["request"]) == request and integer(record["segment"]) == last_segment,
                            "record belongs to the wrong segment")
                    if kind == "layer":
                        layer = integer(record["layer"])
                        require(layer in layers and layer not in active.layers, "unexpected or duplicate layer")
                        active.layers[layer] = parse_layer(record, active, header, not requests[request])
                        layer_rows += 1
                    else:
                        require(record["success"] is True and set(active.layers) == set(layers), "failed or incomplete segment")
                        requests[request].append(active)
                        active = None
                        segments += 1
                elif kind == "complete":
                    require(active is None and integer(record["segments"], 1) == segments,
                            "incomplete segment count")
                    require(integer(record["rows"], 1) == layer_rows, "incomplete layer row count")
                    require(batch_offset == batch_size, "last outer batch is incomplete")
                    completed = True
                else:
                    raise ValueError(f"unexpected record type {kind!r}")
            except (TypeError, KeyError, ValueError) as error:
                raise ValueError(f"{path}:{last_line}: {error}") from error
    require(completed, f"{path}:{last_line}: missing completion record")
    return requests


def replay_window(tokens, resident, scratch):
    # Group cold experts by first use. Protect every scratch hit for its wave before filling misses.
    cold = list(dict.fromkeys(expert for row in tokens for expert in row if expert not in resident))
    capacity = len(scratch)
    uploads = 0
    for start in range(0, len(cold), capacity):
        wave = cold[start:start + capacity]
        needed = set(wave)
        misses = [expert for expert in wave if expert not in scratch]
        victims = [slot for slot, expert in enumerate(scratch) if expert == -1]
        victims.extend(slot for slot, expert in enumerate(scratch) if expert != -1 and expert not in needed)
        for slot, expert in zip(victims, misses):
            scratch[slot] = expert
        uploads += len(misses)
    return uploads, len(cold)


def windows(segments, width):
    if width == 2:
        for segment in segments:
            yield [(segment, offset) for offset in range(len(segment.positions))]
        return
    tokens = []
    batch = None
    for segment in segments:
        if batch is not None and segment.batch != batch and tokens:
            yield tokens
            tokens = []
        batch = segment.batch
        for offset in range(len(segment.positions)):
            tokens.append((segment, offset))
            if len(tokens) == width:
                yield tokens
                tokens = []
    if tokens:
        yield tokens


@dataclass
class Totals:
    tokens: int = 0
    windows: int = 0
    full_windows: int = 0
    tail_tokens: int = 0
    max_width: int = 0
    uploads: int = 0
    bytes: int = 0

    def add(self, other):
        self.tokens += other.tokens
        self.windows += other.windows
        self.full_windows += other.full_windows
        self.tail_tokens += other.tail_tokens
        self.max_width = max(self.max_width, other.max_width)
        self.uploads += other.uploads
        self.bytes += other.bytes


@dataclass
class LayerTotals:
    uploads: int = 0
    bytes: int = 0
    cold_distinct_sum: int = 0
    cold_routes: int = 0
    routes: int = 0


def replay(segments, width):
    first_layers = segments[0].layers
    scratch = {layer: row.scratch.copy() for layer, row in first_layers.items()}
    result = Totals()
    layer_totals = {layer: LayerTotals() for layer in first_layers}
    for window in windows(segments, width):
        size = len(window)
        result.tokens += size
        result.windows += 1
        result.full_windows += size == width
        result.tail_tokens += size if size < width else 0
        result.max_width = max(result.max_width, size)
        for layer, first in first_layers.items():
            rows = [segment.layers[layer] for segment, _ in window]
            require(all(row.strides == first.strides and len(row.selected[0]) == len(first.selected[0])
                        for row in rows), f"layer {layer}: weights or routing width changed during request")
            selected = [row.selected[offset] for row, (_, offset) in zip(rows, window)]
            uploads, cold_distinct = replay_window(selected, first.resident, scratch[layer])
            uploaded_bytes = uploads * sum(first.strides)
            result.uploads += uploads
            result.bytes += uploaded_bytes
            layer_result = layer_totals[layer]
            layer_result.uploads += uploads
            layer_result.bytes += uploaded_bytes
            layer_result.cold_distinct_sum += cold_distinct
            layer_result.cold_routes += sum(expert not in first.resident for row in selected for expert in row)
            layer_result.routes += sum(map(len, selected))
    return result, layer_totals


def render(requests, per_layer):
    results = []
    layer_results = []
    total_observed = Totals()
    total_modeled = {width: Totals() for width in WIDTHS}
    for request, segments in requests.items():
        observed = Totals()
        observed_layers = {layer: LayerTotals() for layer in segments[0].layers}
        for segment in segments:
            for layer, row in segment.layers.items():
                observed.uploads += row.uploads
                observed.bytes += row.bytes
                observed_layers[layer].uploads += row.uploads
                observed_layers[layer].bytes += row.bytes
        modeled = {}
        modeled_layers = {}
        for width in WIDTHS:
            modeled[width], modeled_layers[width] = replay(segments, width)
        layer_results.append((request, observed_layers, modeled_layers))
        total_observed.add(observed)
        for width, result in modeled.items():
            total_modeled[width].add(result)
        results.append((request, observed, modeled))
    results.append(("total", total_observed, total_modeled))
    print("Only captured prompt-only tokens are replayed. Each request starts from its first observed cache state.")
    print("Permanent residency stays fixed. The observed scratch capacity, weight strides and outer batch boundaries apply.")
    print("Scratch survives across windows within each request. Width 2 follows the observed microbatch boundaries.")
    print("Deltas are modeled minus reference. Compare against modeled width 2 to isolate the effect of width.")
    print("Width-2 delta_vs_observed shows the gap to real execution, including dynamic residency. No timing is estimated.")
    print("Observed bytes count Expert weight uploads only. Promotion, writeback and other traffic are outside the replay.")
    print("A width with no full_width_windows is represented by shorter tails only.")
    print("\t".join(("request", "width", "tokens", "windows", "full_width_windows", "tail_tokens", "max_width",
                     "observed_uploads", "observed_bytes", "modeled_uploads", "modeled_bytes",
                     "delta_uploads_vs_width2", "delta_bytes_vs_width2",
                     "delta_uploads_vs_observed", "delta_bytes_vs_observed")))
    for request, observed, modeled in results:
        baseline = modeled[2]
        for width, result in modeled.items():
            columns = (request, width, result.tokens, result.windows, result.full_windows, result.tail_tokens,
                       result.max_width, observed.uploads, observed.bytes, result.uploads, result.bytes,
                       f"{result.uploads - baseline.uploads:+d}", f"{result.bytes - baseline.bytes:+d}",
                       f"{result.uploads - observed.uploads:+d}", f"{result.bytes - observed.bytes:+d}")
            print("\t".join(map(str, columns)))
    if per_layer:
        print("\nCold expert counts use fixed residency. Cold distinct is summed across windows.")
        print("\t".join(("request", "layer", "width", "observed_uploads", "observed_bytes",
                         "modeled_uploads", "modeled_bytes", "delta_uploads_vs_width2", "delta_bytes_vs_width2",
                         "delta_uploads_vs_observed", "delta_bytes_vs_observed",
                         "cold_distinct_sum", "cold_routes", "selected_routes", "cold_route_fraction")))
        for request, observed_layers, modeled_layers in layer_results:
            for layer, baseline in sorted(modeled_layers[2].items()):
                observed = observed_layers[layer]
                for width in WIDTHS:
                    modeled = modeled_layers[width][layer]
                    columns = (request, layer, width, observed.uploads, observed.bytes, modeled.uploads, modeled.bytes,
                               f"{modeled.uploads - baseline.uploads:+d}", f"{modeled.bytes - baseline.bytes:+d}",
                               f"{modeled.uploads - observed.uploads:+d}", f"{modeled.bytes - observed.bytes:+d}",
                               modeled.cold_distinct_sum, modeled.cold_routes, modeled.routes,
                               f"{modeled.cold_routes / modeled.routes:.6f}")
                    print("\t".join(map(str, columns)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="successful dialogue-prefill-trace run directory")
    parser.add_argument("--per-layer", action="store_true", help="also print per-request and per-layer replay details")
    args = parser.parse_args()
    try:
        require((args.run_directory / "exit.txt").read_text(encoding="utf-8-sig").strip() == "exit_code=0",
                "launcher did not record a successful exit")
        render(read_trace(args.run_directory / "expert-prefill.jsonl"), args.per_layer)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
