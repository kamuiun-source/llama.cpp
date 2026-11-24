#!/usr/bin/env python3
"""Rank ROCm kernel durations within ROCTx target verification windows."""

import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import json
from math import ceil
from pathlib import Path
import re
import statistics


MARKER = re.compile(r"mtp3:(verify_submit|verify_sync)_(\d+):(\d+):(begin|end)")
ROUTED_MMVQ = re.compile(r"\bmul_mat_vec_q_moe<\(ggml_type\)(\d+),\s*(\d+),\s*(true|false),")
ROUTED_QUANT_NAMES = {18: "IQ3_XXS", 20: "IQ4_NL", 22: "IQ2_S", 23: "IQ4_XS", 42: "Q2_0"}
MAX_CLOCK_RESIDUAL_NS = 1_000_000


def rows(path):
    with path.open(newline="") as stream:
        yield from csv.DictReader(stream)


def load_windows(run, marker_file, width):
    host = {}
    for row in rows(run / "host-marks.csv"):
        label = row["label"]
        match = MARKER.fullmatch(label)
        if match and int(match[2]) == width:
            lo, hi = int(row["ro_lo"]), int(row["ro_hi"])
            if lo > hi or label in host:
                raise ValueError(f"invalid or duplicate host mark: {label}")
            host[label] = lo, hi
    if not host:
        raise ValueError(f"no width-{width} verification marks in host-marks.csv")

    markers = {}
    for row in rows(marker_file):
        label = row["Function"]
        match = MARKER.fullmatch(label)
        if not match or int(match[2]) != width:
            continue
        if label in markers or label not in host:
            raise ValueError(f"duplicate or unmatched ROCm mark: {label}")
        clock = int(row["Start_Timestamp"])
        markers[label] = (clock, row["Thread_Id"], *match.groups())
    if host.keys() != markers.keys():
        raise ValueError(f"missing {len(host.keys() - markers.keys())} ROCm verification marks")

    # The host brackets each roctxMarkA call. Kernel and marker CSVs share the ROCm clock.
    shifts = [(lo + hi) // 2 - markers[label][0] for label, (lo, hi) in host.items()]
    offset = int(statistics.median(shifts))
    residual = max(max(lo - (markers[label][0] + offset),
                       (markers[label][0] + offset) - hi, 0) for label, (lo, hi) in host.items())
    if residual > MAX_CLOCK_RESIDUAL_NS:
        raise ValueError(f"ROCm/host clock mismatch: {residual} ns outside host mark brackets")

    ordered = sorted(markers.values(), key=lambda item: (item[0], int(item[4]), item[5] == "end"))
    if len(ordered) % 4:
        raise ValueError("incomplete verification marker sequence")
    windows = []
    for i in range(0, len(ordered), 4):
        submit_begin, submit_end, sync_begin, sync_end = ordered[i:i + 4]
        expected = [("verify_submit", "begin"), ("verify_submit", "end"),
                    ("verify_sync", "begin"), ("verify_sync", "end")]
        if [(row[2], row[5]) for row in ordered[i:i + 4]] != expected or (
            submit_begin[4] != submit_end[4] or sync_begin[4] != sync_end[4]
            or len({row[1] for row in ordered[i:i + 4]}) != 1
            or not submit_begin[0] < sync_end[0]
            or windows and submit_begin[0] < windows[-1][1]
        ):
            raise ValueError(f"invalid or overlapping verify submit/sync pair near {submit_begin}")
        windows.append((submit_begin[0], sync_end[0]))
    return windows, {"matched_marks": len(markers), "host_minus_rocm_ns": offset,
                     "max_outside_host_bracket_ns": residual}


def union_duration(spans):
    total = end = 0
    for left, right in sorted(spans):
        total += max(0, right - max(left, end))
        end = max(end, right)
    return total


def shape(row, key):
    return tuple(int(row[f"{key}_Size_{axis}"]) for axis in "XYZ")


def classify(name, grid, workgroup):
    if "copyBuffer" in name:
        return "runtime_copy_kernel"
    if "quantize" in name and re.search(r"q8_1(?:\W|$)", name):
        return "q8_1_quantize"
    if "mul_mat_vec_iq3_xxs_moe_" in name:
        return "expert_gate_up_swiglu"
    match = ROUTED_MMVQ.search(name)
    if match:
        rows_per_block = int(match[2])
        if match[3] == "true" and grid[0] == workgroup[0] * ceil(640 / rows_per_block):
            return "expert_gate_up_swiglu"
        if match[3] == "false" and grid[0] == workgroup[0] * ceil(2560 / rows_per_block):
            return "expert_down"
    return "other_compute"


def stats(values, verifies, denominator):
    ordered = sorted(values)
    total = sum(ordered)
    return {
        "count": len(ordered), "total_ms": total / 1e6,
        "ms_per_verify": total / verifies / 1e6,
        "p50_ms": statistics.median(ordered) / 1e6,
        "p95_ms": ordered[ceil(0.95 * len(ordered)) - 1] / 1e6,
        "max_ms": ordered[-1] / 1e6,
        "share_dispatch_time_pct": 100 * total / denominator if denominator else 0,
    }


def ranked(samples, verifies, denominator):
    return [(key, stats(values, verifies, denominator))
            for key, values in sorted(samples.items(), key=lambda item: (-sum(item[1]), item[0]))]


def analyze(run, kernel_file, marker_file, agents_file, copy_file, width, agent):
    windows, calibration = load_windows(run, marker_file, width)
    ends = [end for _, end in windows]

    def overlap(row):
        start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
        if end < start:
            raise ValueError(f"reversed activity span: {start} > {end}")
        pieces = []
        for index in range(bisect_right(ends, start), len(windows)):
            left, right = windows[index]
            if end <= left:
                break
            lo, hi = max(start, left), min(end, right)
            if lo < hi:
                pieces.append((index, lo, hi))
        return pieces, end - start

    if agent is None:
        if agents_file is None:
            found = {row["Agent_Id"] for row in rows(kernel_file)}
            if len(found) != 1:
                raise ValueError("kernel trace has multiple or zero agents; use --agent")
            agent = found.pop()
        else:
            mi50 = [f"Agent {row['Logical_Node_Id']}" for row in rows(agents_file)
                    if row["Agent_Type"] == "GPU" and row["Name"] == "gfx906"]
            if len(mi50) != 1:
                raise ValueError("expected one gfx906 agent; use --agent to select the MI50")
            agent = mi50[0]

    kernels, variants, groups, copies = (defaultdict(list) for _ in range(4))
    categories = defaultdict(set)
    group_spans, copy_spans = defaultdict(list), defaultdict(list)
    covered = set()
    clipped_kernels = clipped_copies = 0
    for row in rows(kernel_file):
        if row["Agent_Id"] != agent:
            continue
        pieces, duration = overlap(row)
        if not pieces:
            continue
        spans = [(lo, hi) for _, lo, hi in pieces]
        clipped = sum(hi - lo for lo, hi in spans)
        clipped_kernels += clipped != duration
        name = row["Kernel_Name"]
        grid, workgroup = shape(row, "Grid"), shape(row, "Workgroup")
        group = classify(name, grid, workgroup)
        if group == "runtime_copy_kernel":
            copies[group].append(clipped)
            copy_spans[group].extend(spans)
            continue
        covered.update(index for index, _, _ in pieces)
        categories[name].add(group)
        kernels[name].append(clipped)
        variants[(name, grid, workgroup, int(row["LDS_Block_Size"]), group)].append(clipped)
        groups[group].append(clipped)
        group_spans[group].extend(spans)

    if len(covered) != len(windows):
        raise ValueError(f"MI50 compute only recorded in {len(covered)}/{len(windows)} verify windows for {agent}")
    if copy_file is not None:
        for row in rows(copy_file):
            if agent not in (row["Source_Agent_Id"], row["Destination_Agent_Id"]):
                continue
            pieces, duration = overlap(row)
            if not pieces:
                continue
            spans = [(lo, hi) for _, lo, hi in pieces]
            clipped = sum(hi - lo for lo, hi in spans)
            clipped_copies += clipped != duration
            direction = row["Direction"]
            copies[direction].append(clipped)
            copy_spans[direction].extend(spans)

    wall = union_duration(windows)
    compute_spans = [span for spans in group_spans.values() for span in spans]
    all_copy_spans = [span for spans in copy_spans.values() for span in spans]
    compute = union_duration(compute_spans)
    all_activity = union_duration(compute_spans + all_copy_spans)
    total = sum(map(sum, kernels.values()))
    verifies = len(windows)
    return {
        "width": width, "verifies": verifies, "agent": agent, "calibration": calibration,
        "clipped_kernel_records": clipped_kernels, "clipped_memory_copy_records": clipped_copies,
        "verify_wall_ms": wall / 1e6, "compute_union_ms": compute / 1e6,
        "outside_compute_ms": (wall - compute) / 1e6, "copy_union_ms": union_duration(all_copy_spans) / 1e6,
        "compute_or_copy_union_ms": all_activity / 1e6, "outside_observed_activity_ms": (wall - all_activity) / 1e6,
        "memory_copy_trace_present": copy_file is not None,
        "groups": [{"group": name, **values, "union_ms": union_duration(group_spans[name]) / 1e6}
                   for name, values in ranked(groups, verifies, total)],
        "kernels": [{"kernel": name, "groups": sorted(categories[name]), **values}
                    for name, values in ranked(kernels, verifies, total)],
        "geometry_variants": [
            {"kernel": name, "grid": grid, "workgroup": workgroup, "lds_bytes": lds, "group": group, **values}
            for (name, grid, workgroup, lds, group), values in ranked(variants, verifies, total)],
        "copies": [{"kind": name, **values, "union_ms": union_duration(copy_spans[name]) / 1e6}
                   for name, values in ranked(copies, verifies, sum(map(sum, copies.values())))],
    }


def print_report(report, top):
    print(f"MI50 {report['agent']} | width {report['width']} | {report['verifies']} verifies | "
          f"max clock residual {report['calibration']['max_outside_host_bracket_ns']} ns")
    print(f"Verify wall {report['verify_wall_ms']:.3f} ms | compute union {report['compute_union_ms']:.3f} ms | "
          f"outside compute {report['outside_compute_ms']:.3f} ms")
    print(f"Copy union {report['copy_union_ms']:.3f} ms | compute or copy union "
          f"{report['compute_or_copy_union_ms']:.3f} ms | outside observed activity "
          f"{report['outside_observed_activity_ms']:.3f} ms")
    if not report["memory_copy_trace_present"]:
        print("SDMA memory copy trace absent. Copy activity covers runtime copy kernels only.")
    print(f"Boundary-clipped records: {report['clipped_kernel_records']} kernels, "
          f"{report['clipped_memory_copy_records']} memory copies")

    def table(items, key):
        print(f"{'n':>7} {'sum ms':>10} {'ms/verify':>10} {'p50 ms':>9} {'p95 ms':>9} "
              f"{'max ms':>9} {'share %':>8}  {key}")
        for row in items:
            label = row[key]
            if key == "kernel":
                # Keep the typed template name, omit the long argument list in console output.
                label = label.split(">(")[0] + ">" if ">(" in label else label.split("(")[0]
                match = ROUTED_MMVQ.search(label)
                if match and int(match[1]) in ROUTED_QUANT_NAMES:
                    label += f" [{ROUTED_QUANT_NAMES[int(match[1])]}]"
            print(f"{row['count']:7} {row['total_ms']:10.3f} {row['ms_per_verify']:10.3f} "
                  f"{row['p50_ms']:9.4f} {row['p95_ms']:9.4f} {row['max_ms']:9.4f} "
                  f"{row['share_dispatch_time_pct']:8.2f}  {label}")

    print("\nCompute groups (share of summed compute durations)")
    table(report["groups"], "group")
    print("\nTop kernels (share of summed compute durations)")
    table(report["kernels"][:top], "kernel")
    if report["copies"]:
        print("\nCopy timings (share of summed copy records)")
        table(report["copies"], "kind")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="run directory with host-marks.csv and rocprof CSV files")
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--agent", help='override GPU selection, e.g. "Agent 1"')
    parser.add_argument("--top", type=int, default=12, help="number of kernel rows to print")
    parser.add_argument("--json", type=Path, help="write complete report, including raw kernel names and geometry")
    args = parser.parse_args()
    if args.width < 1 or args.top < 0:
        parser.error("--width must be positive and --top must be nonnegative")

    def trace_file(suffix, required=True):
        paths = sorted(args.run.rglob(f"*_{suffix}.csv"))
        if len(paths) > 1 or (required and not paths):
            raise ValueError(f"expected {'one' if required else 'at most one'} {suffix} CSV, found {len(paths)}")
        return paths[0] if paths else None

    try:
        report = analyze(args.run, trace_file("kernel_trace"), trace_file("marker_api_trace"),
                         trace_file("agent_info", False), trace_file("memory_copy_trace", False),
                         args.width, args.agent)
        print_report(report, args.top)
        if args.json:
            args.json.write_text(json.dumps(report, indent=2) + "\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
