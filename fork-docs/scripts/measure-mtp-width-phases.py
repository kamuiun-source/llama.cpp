#!/usr/bin/env python3
"""Compare warm fixed-2 and fast-3 MTP phase telemetry, or summarize an existing archive."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import runpy
from uuid import uuid4


HERE = Path(__file__).resolve().parent
WARM = runpy.run_path(str(HERE / "compare-mtp-warm.py"))
COMMON = WARM["COMMON"]
require = COMMON["require"]
ARMS = ("width-2", "width-3")
STAGES = ("warmup", "measure-1", "measure-2")
RUN_TOTALS = {
    "draft_hidden_rows": "MTP device pending:",
    "routed_upload_overlap": "packed expert Vulkan upload overlap:",
    "routed_promotion_overlap": "packed expert Vulkan promotion overlap:",
    "shared_expert_overlap": "shared expert overlap:",
    "width3_vulkan": "mtp fast-three Vulkan:",
    "width3_expert_layers": "MTP 3-position expert:",
    "width3_waves": "MTP 3-position waves:",
    "width3_overlap_host": "MTP 3-position overlap:",
    "width3_overlap_vulkan": "mtp overlap-three Vulkan:",
}
SCOPE = (
    "Warm speed, unrestricted output. Separate forced-token CLI archive is for numerical and rollback alignment. "
    "CUDA stream spans, Vulkan compute segments, CPU waits and phase wall can overlap. "
    "GPU spans are sampled intervals, not kernel totals. Report timestamps only with their sample, pending and drop counts. "
    "Process totals include prefill and warmup; they are not attributed to individual responses or verify rounds. "
    "Source currently enables direct CUDA convolution history for one or two positions only. "
    "A true width-3 verify uses the generic history path; existing telemetry cannot isolate its wall time."
)


def fields(line):
    result = {}
    for key, value in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)(?=\s|$)", line):
        result[key] = float(value) if "." in value else int(value)
    return result


def responses(log):
    blocks, active = [], None
    seen = set()
    for line in log.splitlines():
        if "decode_response_begin:" in line:
            require(active is None, "nested response telemetry")
            metadata = fields(line.split("decode_response_begin:", 1)[1])
            require("response" in metadata, "missing response ID")
            active = dict(response=metadata, phases={})
        elif "decode_response_end:" in line:
            end = fields(line.split("decode_response_end:", 1)[1]).get("response")
            require(active is not None and end == active["response"]["response"] and end not in seen,
                    "misaligned response telemetry")
            seen.add(end)
            blocks.append(active)
            active = None
        elif active is not None and "decode_telemetry:" in line:
            payload = line.split("decode_telemetry:", 1)[1]
            phase = re.search(r"\bphase=(\w+)", payload)
            if not phase or re.search(r"\b(?:cuda_component|vk_cause)=", payload):
                continue
            values = fields(payload)
            totals = active["phases"].setdefault(phase[1], {})
            require(not totals.keys() & values.keys(), f"duplicate phase metrics in {phase[1]}")
            totals.update(values)
    require(active is None, "incomplete response telemetry")
    return blocks


def normalized(totals, decode_tokens):
    measured = {key: value for key, value in totals.items()
                if key.endswith(("_ms", "_bytes", "_submits")) or key in ("graphs", "scheduled_splits", "expert_requests",
                                                                         "expert_request_misses", "expert_promotions")}
    return {
        "timestamp_samples": {
            "cuda": {key: totals.get(f"cuda_gpu_stream_span_{key}") for key in ("samples", "pending", "dropped")},
            "vulkan": {key: totals.get(f"vk_gpu_compute_segment_{key}") for key in ("samples", "dropped")},
        },
        "per_call": {key: value / totals["calls"] for key, value in measured.items()} if totals.get("calls") else {},
        "per_decode_token": {key: value / decode_tokens for key, value in measured.items()} if decode_tokens else {},
        "per_target_position": {key: value / totals["target_tokens"] for key, value in measured.items()}
        if totals.get("target_tokens") else {},
    }


def summarize_arm(case, metrics, strict=False):
    paths = [case / name for name in ("server.log", "llama.log") if (case / name).is_file()]
    require(len(paths) == 1, f"expected one server or CLI log in {case}")
    path = paths[0]
    log = path.read_text(encoding="utf-8", errors="replace")
    blocks = responses(log)
    data = dict(log=str(path.resolve()), telemetry_present=bool(blocks), process_totals={}, requests=[])
    for key, prefix in RUN_TOTALS.items():
        rows = [fields(line.split(prefix, 1)[1]) for line in log.splitlines() if prefix in line]
        if rows:
            data["process_totals"][key] = rows
    if "rollback_validation" in metrics:
        data["forced_rollback"] = metrics["rollback_validation"]
    if "warmup" not in metrics:
        data["cli_result"] = {key: metrics[key] for key in ("decode_ms", "generated_tokens", "mtp_three", "fast_three", "overlap_three") if key in metrics}
        data["cli_phase_telemetry"] = blocks
        if blocks:
            require(len(blocks) == 1 and blocks[0]["response"]["generated_tokens"] == metrics["generated_tokens"],
                    f"forced CLI telemetry does not align with completed generation in {case}")
            phases = blocks[0]["phases"]
            data["cli_normalized"] = {phase: normalized(totals, metrics["generated_tokens"] - 1)
                                      for phase, totals in phases.items()}
        return data
    requests = [metrics["warmup"], *metrics["measured"]]
    require((not strict and not blocks) or len(blocks) == len(requests) == len(STAGES),
            f"expected three complete aligned telemetry responses in {case}")
    for index, (stage, request) in enumerate(zip(STAGES, requests)):
        row = {key: request[key] for key in ("request_id", "decode_ms", "decode_tokens", "generated_tokens", "request_wall_ms",
                                             "decode_tok_s", "draft_proposed", "draft_accepted", "draft_wall_ms", "two_draft")}
        row["stage"] = stage
        if blocks:
            block = blocks[index]
            metadata = block["response"]
            require(metadata["task"] == row["request_id"] and metadata["generated_tokens"] == row["generated_tokens"],
                    f"response/task alignment mismatch for {case.name} {stage}")
            row["phases"] = block["phases"]
            verify = row["phases"].get("target_verify", {})
            require(verify.get("calls", 0) > 0 and verify.get("target_tokens", 0) >= 2 * verify["calls"],
                    f"no valid target verification telemetry for {case.name} {stage}")
            row["normalized"] = {phase: normalized(totals, row["decode_tokens"])
                                 for phase, totals in row["phases"].items()}
        data["requests"].append(row)
    data["pooled_decode_tok_s"] = metrics["pooled_decode_tok_s"]
    data["pooled_request_wall_tok_s"] = metrics["pooled_request_wall_tok_s"]
    if blocks:
        first, second = data["requests"][1:]
        decode_tokens = first["decode_tokens"] + second["decode_tokens"]
        measured = {}
        for phase in sorted(first["phases"].keys() & second["phases"].keys()):
            left, right = (request["phases"][phase] for request in (first, second))
            totals = {key: left[key] + right[key] for key in left.keys() & right.keys()
                      if key not in ("max_unique",)}
            if "max_unique" in left and "max_unique" in right:
                totals["max_unique"] = max(left["max_unique"], right["max_unique"])
            measured[phase] = dict(totals=totals, **normalized(totals, decode_tokens))
        data["measured_phases"] = measured
    return data


def archive_report(directory, strict=False):
    cases = {mode: directory / mode for mode in ARMS}
    overlap = (directory / "width-3-off").is_dir()
    if overlap:
        cases = {mode: directory / mode for mode in ("width-2", "width-3-off", "width-3-on")}
        if not cases["width-2"].is_dir():
            run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
            cases["width-2"] = Path(run["reference_archive"]) / "width-2"
    metrics = {mode: json.loads((case / "metrics.json").read_text(encoding="utf-8")) for mode, case in cases.items()}
    report = dict(directory=str(directory.resolve()), scope=SCOPE,
                  arms={mode: summarize_arm(case, metrics[mode], strict) for mode, case in cases.items()})
    left, right = (metrics[mode] for mode in (("width-3-off", "width-3-on") if overlap else ARMS))
    if "measured" in left and "measured" in right:
        report["measured_token_ids_equal"] = [
            a["token_ids"] == b["token_ids"] for a, b in zip(left["measured"], right["measured"])
        ]
        report["pooled_decode_delta_tok_s"] = right["pooled_decode_tok_s"] - left["pooled_decode_tok_s"]
    else:
        comparison = directory / "comparison.json"
        if comparison.is_file():
            saved = json.loads(comparison.read_text(encoding="utf-8")).get("logit_diagnostic", {})
            report["forced_logit_summary"] = {key: value for key, value in saved.items()
                                               if isinstance(value, (int, float, bool))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, default=Path("G:/AI/build-prefill32"))
    parser.add_argument("--three-first", action="store_true", help="run fast-3 before fixed-2")
    parser.add_argument("--archive", type=Path, help="report an existing warm or forced-token archive without loading a model")
    args = parser.parse_args()
    if args.archive:
        print(json.dumps(archive_report(args.archive), indent=2))
        return
    require(os.name == "nt", "warm MI50 phase collection requires Windows")
    library = args.build_directory.resolve() / "bin/llama-server-impl.dll"
    before = COMMON["binary_hashes"](library)
    baseline = COMMON["build_environment"]()
    baseline.update({
        "LLAMA_EXPERT_PREFILL_512": "1", "LLAMA_EXPERT_PREFILL_512_POLICY": "auto",
        "LLAMA_EXPERT_PREFILL_LOGICAL_TICKS": "1", "LLAMA_EXPERT_PREFILL_OVERLAP_TEN": "1",
        "LLAMA_DECODE_TELEMETRY": "1", "LLAMA_DECODE_TELEMETRY_PER_RESPONSE": "1", "LLAMA_EXPERT_MISS_HISTOGRAM": "1",
        "GGML_CUDA_GPU_TIMESTAMPS_DEVICE": "CUDA0", "GGML_VK_GPU_TIMESTAMPS_DEVICE": "Vulkan1",
        "GGML_CUDA_DISABLE_GRAPHS": None, "GGML_CUDA_PROFILE_COMPONENTS": None,
        "GGML_VK_PERF_LOGGER": None, "GGML_VK_PERF_LOGGER_CONCURRENT": None, "GGML_VK_SERIALIZE_SUBMISSIONS": None,
        "LLAMA_DECODE_PROFILE": None, "LLAMA_DECODE_PROFILE_CUDA": None, "LLAMA_EXPERT_PROFILE_RAM_COPY": None,
        "LLAMA_MTP_DRAFT_VOCAB": None, "LLAMA_MTP_DRAFT_VOCAB_STATS": "1", "LLAMA_MTP_REUSE_DRAFT_ANCHOR": None,
        "LLAMA_MTP_LOGIT_TRACE": None, "LLAMA_MTP_LOGIT_REFERENCE": None, "LLAMA_MTP_LOGIT_MATCH_DRAFTS": None,
        "LLAMA_MTP_PAIR_EXPERT_OVERLAP": None, "LLAMA_MTP_ROUTED_DISPATCH_STATS": None,
        "GGML_SCHED_SHARED_INPUT_PRODUCER_COPY": "0", "GGML_VK_SKIP_EMPTY_POOL_RESET": "0",
        "GGML_VK_MTP_ROUTED_REDUCE": "1", "GGML_VK_MTP_ROUTED_WEIGHTED": "1", "GGML_VK_MTP_ROUTED_GATE_UP": "1",
        "GGML_VK_MTP_ROUTED_GATE_UP_PAIR": "0", "GGML_VK_MTP_GATE_UP_PRECOMPUTE": "0",
        "GGML_VK_MTP_GATE_UP_COMPACT": "0",
        "GGML_VK_MTP_GATE_UP_TIMING": "0", "GGML_VK_MTP_GATE_UP_PROOF": "0",
    })
    baseline.update(dict.fromkeys(WARM["OLDER_HC"]))
    baseline.update(dict.fromkeys(WARM["HC_PROJECTIONS"], "1"))
    order = ARMS[::-1] if args.three_first else ARMS
    directory = COMMON["LOG_ROOT"] / f"mtp-width-phases-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    directory.mkdir()
    _, message = COMMON["prepare_inputs"](directory)
    COMMON["write_json"](directory / "run.json", dict(order=order, build_sha256=before, scope=SCOPE))
    print(f"Log directory: {directory}", flush=True)
    tokens = None
    for mode in order:
        environment = dict(baseline, LLAMA_MTP_VERIFY_THREE_FAST="1" if mode == "width-3" else "0")
        _, tokens = WARM["run_arm"](directory / mode, library, mode, environment, message, tokens)
        log = (directory / mode / "server.log").read_text(encoding="utf-8", errors="replace")
        WARM["hc_activation"](log, True)
    require(before == COMMON["binary_hashes"](library), "binaries changed during collection")
    report = archive_report(directory, strict=True)
    COMMON["write_json"](directory / "phase-comparison.json", report)
    for mode, arm in report["arms"].items():
        verify = arm["measured_phases"]["target_verify"]
        print(f"{mode} verify: {verify['totals']['calls']} calls, {verify['per_call']['wall_ms']:.3f} ms/call, "
              f"{verify['per_decode_token']['wall_ms']:.3f} ms/decode token")
    print(directory / "phase-comparison.json")


if __name__ == "__main__":
    main()
