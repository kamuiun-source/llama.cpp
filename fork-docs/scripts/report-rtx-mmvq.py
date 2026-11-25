#!/usr/bin/env python3
"""Attribute RTX MMVQ kernel times to exact NVTX launches and CUDA Graph nodes."""
import argparse
import csv
import json
import re
import shlex
import sqlite3
import statistics
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

VERIFY = re.compile(r"mtp3:verify_(submit|sync)_3:(\d+):(begin|end)")
TYPE = {"q8_0": 8, "q6_k": 14}
TEMPLATE = re.compile(r"\bmul_mat_vec_q<\s*\(ggml_type\)(\d+),\s*(?:\(int\))?(\d+)")


def tables(db):
    return {name for name, in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def decode_fields(text):
    return dict(word.split("=", 1) for word in shlex.split(text)[1:])


def phase_windows(events, host_file, session_file):
    with session_file.open() as stream:
        turns = json.load(stream)["turns"]

    def clock(record):
        return record.get("mono_ns", record.get("monotonic_ns"))

    responses = [(clock(t["started"]), clock(t["finished"])) for t in turns]
    hosts = {}
    with host_file.open() as stream:
        for row in csv.DictReader(stream):
            label = row["label"]
            if not VERIFY.fullmatch(label):
                continue
            a, b = int(row["nv_lo"]), int(row["nv_hi"])
            if any(lo <= a <= b <= hi for lo, hi in responses):
                if label in hosts:
                    raise ValueError(f"duplicate host mark {label}")
                hosts[label] = (a, b)
    picked = {}
    for stamp, _, _, label in events:
        if label in hosts:
            if label in picked:
                raise ValueError(f"duplicate verification NVTX mark {label}")
            picked[label] = stamp
    if set(picked) != set(hosts):
        raise ValueError("missing verification NVTX marks")
    if not picked or len(picked) % 4:
        raise ValueError("no complete verification marks")
    offset = round(statistics.median((hosts[k][0] + hosts[k][1]) / 2 - t for k, t in picked.items()))
    residual = max(max(hosts[k][0] - t - offset, t + offset - hosts[k][1], 0) for k, t in picked.items())
    if residual > 1_000_000:
        raise ValueError(f"CUDA/host timestamp residual {residual} ns")
    ordered = sorted((stamp, int(ordinal), edge, phase) for label, stamp in picked.items()
                     for phase, ordinal, edge in [VERIFY.fullmatch(label).groups()])
    windows = []
    expected = [("submit", "begin"), ("submit", "end"), ("sync", "begin"), ("sync", "end")]
    for index in range(0, len(ordered), 4):
        marks = ordered[index:index + 4]
        start, _, _, end = marks
        if [(phase, edge) for _, _, edge, phase in marks] != expected or (
                start[1] != marks[1][1] or marks[2][1] != end[1] or start[0] >= end[0]):
            raise ValueError(f"invalid verification event sequence: {marks}")
        windows.append((start[0], end[0]))
    windows.sort()
    if any(previous[1] > current[0] for previous, current in zip(windows, windows[1:])):
        raise ValueError("overlapping verify windows")
    per_turn = [sum(lo <= a + offset < b + offset <= hi for a, b in windows) for lo, hi in responses]
    if sum(per_turn) != len(windows):
        raise ValueError("verification windows outside responses")
    return windows, per_turn, residual


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("sqlite", type=Path)
    cli.add_argument("host_marks", type=Path)
    cli.add_argument("session", type=Path)
    args = cli.parse_args()
    with sqlite3.connect(f"file:{args.sqlite.resolve()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Nsight SQLite integrity check failed")
        nf = ("SELECT e.start,e.end,e.globalTid,COALESCE(e.text,s.value) "
            "FROM NVTX_EVENTS e LEFT JOIN StringIds s ON s.id=e.textId")
        events = list(db.execute(nf))
        metadata = {}
        ranges = defaultdict(list)
        for start, end, tid, label in events:
            if not label:
                continue
            if label.startswith("mmvq_graph_node "):
                fields = decode_fields(label)
                node = int(fields.pop("graphNodeId"))
                if node in metadata and metadata[node] != fields:
                    raise ValueError(f"conflicting metadata for graph node {node}")
                metadata[node] = fields
            elif label.startswith("mmvq_launch ") and end is not None:
                ranges[tid].append((start, end, decode_fields(label)))
        windows, by_prompt, residual = phase_windows(events, args.host_marks, args.session)
        if not metadata:
            raise ValueError("no capture-node metadata")
        edges = defaultdict(set)
        table_names = tables(db)
        if "CUDA_GRAPH_NODE_EVENTS" in table_names:
            for clone, source in db.execute("SELECT graphNodeId,originalGraphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
                                           "WHERE graphNodeId>0 AND originalGraphNodeId>0 AND graphNodeId!=originalGraphNodeId"):
                edges[clone].add(source)
        for timed in ranges.values():
            timed.sort(key=lambda r:r[0])
        starts_by_thread = {k:[r[0] for r in v] for k, v in ranges.items()}
        eager = {}
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table not in table_names:
                continue
            for a, b, thread, cid, name in db.execute(f"SELECT a.start,a.end,a.globalTid,a.correlationId,s.value "
                                                  f"FROM {table} a JOIN StringIds s ON a.nameId=s.id "
                                                  f"WHERE s.value LIKE '%LaunchKernel%'"):
                if thread not in ranges:
                    continue
                near = bisect_right(starts_by_thread[thread], a)-1
                if near < 0:
                    continue
                start, end, fields = ranges[thread][near]
                if a >= start and b <= end:
                    if cid in eager and eager[cid] != fields:
                        raise ValueError(f"conflicting CUDA API correlation {cid}")
                    eager[cid] = fields
        resolved = {}

        def candidates(node, seen):
            if node in seen:
                return []
            if node in metadata:
                return [metadata[node]]
            if node in resolved:
                return resolved[node]
            seen.add(node)
            result = []
            for parent in edges[node]:
                for entry in candidates(parent, seen):
                    if entry not in result:
                        result.append(entry)
            seen.remove(node)
            resolved[node] = result
            return result

        entries = defaultdict(list)
        starts = [v[0] for v in windows]
        counts = Counter()
        unknown = Counter()
        sql = ("SELECT k.start,k.end,k.graphNodeId,k.correlationId,k.gridX,k.gridY,k.gridZ,"
             "k.blockX,k.blockY,k.blockZ,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
             "JOIN StringIds t ON k.shortName=t.id JOIN StringIds s ON k.demangledName=s.id "
             "WHERE k.deviceId=0 AND t.value='mul_mat_vec_q'")
        for start, end, graph, cid, gx, gy, gz, bx, by, bz, name in db.execute(sql):
            idx = bisect_right(starts, start)-1
            if idx < 0 or not start < windows[idx][1]:
                continue
            elapsed = min(end, windows[idx][1])-start
            counts["total"] += 1
            fields = None
            label_source = "graph" if graph else "eager"
            if graph:
                found = candidates(graph, set())
                if len(found) == 1:
                    fields = found[0]
                elif len(found) > 1:
                    counts["ambiguous_graph"] += 1
            else:
                fields = eager.get(cid)
            if fields is None:
                counts["unmatched"] += 1
                unknown[name.split("(", 1)[0]] += elapsed
                continue
            match = TEMPLATE.search(name)
            if match and (int(match[2]) != int(fields["cols"]) or
                          fields["quant"].lower() in TYPE and int(match[1]) != TYPE[fields["quant"].lower()]):
                raise ValueError(f"kernel template does not match exact metadata: {fields}, {name}")
            if fields["grid"] != f"{gx}x{gy}x{gz}" or fields["block"] != f"{bx}x{by}x{bz}":
                raise ValueError(f"kernel geometry does not match exact metadata: {fields}")
            counts[label_source] += 1
            weight = re.sub(r"^blk\.\d+\.", "", fields["weight"])
            key = weight, fields["quant"], fields["weight_ne"], fields["cols"], fields["grid"], fields["block"]
            entries[key].append(elapsed)

    def stats(values):
        return {"calls":len(values), "ms_per_verify":sum(values)/len(windows)/1e6,
                "median_us":statistics.median(values)/1e3}

    totals = defaultdict(list)
    for (weight, quant, *shape), values in entries.items():
        totals[(weight, quant)].extend(values)
    weighted = sorted(totals.items(), key=lambda item:sum(item[1]), reverse=True)
    slots = sorted(entries.items(), key=lambda item:sum(item[1]), reverse=True)
    unmatched_ms = sum(unknown.values())/len(windows)/1e6
    result = dict(verifications=len(windows), verify_by_prompt=by_prompt,
                clock_residual_ns=residual, graph_labels=len(metadata), api_labels=len(eager),
                counts = dict(counts), unmatched_ms_per_verify=unmatched_ms,
                all_mmvq_ms_per_verify=sum(sum(v) for v in entries.values())/len(windows)/1e6+unmatched_ms,
                weights = [dict(weight=k[0], quant=k[1], **stats(v)) for k, v in weighted],
                shapes = [dict(zip(("weight", "quant", "weight_ne", "cols", "grid", "block"), k), **stats(v)) for k, v in slots])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
