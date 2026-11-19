"""Compare shared-RAM RTX-to-MI50 producer-stream3-input copy OFF/ON; fixed prompt and MTP generation."""

import argparse
import json
import os
import re
import runpy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

COMMON = runpy.run_path(str(Path(__file__).with_name("compare-prefill512.py")))
OVERLAP_METRICS = runpy.run_path(str(Path(__file__).with_name("compare-prefill-overlap-ten.py")))["overlap_metrics"]
require = COMMON["require"]
FLAG = "GGML_SCHED_SHARED_INPUT_PRODUCER_COPY"


def producer_metrics(log, enabled):
    rows = re.findall(r"shared host producer copy: ([^\r\n]+)", log)
    require(rows, "missing producer-copy path summary; use the experimental build")
    totals = dict(eligible=0, batches=0, tensors=0, bytes=0, waits=0, fallbacks=0,
                  verify_shared=0, other_shared=0, verify_bytes=0, other_bytes=0, optimized_verify=0, optimized_other=0)
    for row in rows:
        require("scope=mtp_verify" in row, "producer build lacks MTP verification scope")
        fields = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)\b", row)}
        require(set(totals) | {"enabled"} <= fields.keys() and fields["enabled"] == int(enabled),
                "producer-copy setting or summary differs")
        require(fields["eligible"] == fields["batches"] + fields["fallbacks"]
                and fields["tensors"] == fields["batches"] * 3 and fields["waits"] == fields["batches"],
                "inconsistent producer-copy counters")
        for key in totals:
            totals[key] += fields[key]
    shared = re.findall(r"shared host expert input: batches=(\d+) bytes=(\d+)", log)
    require(len(shared) == len(rows), "shared-input and producer summaries differ")
    shared_batches = sum(int(batch) for batch, _ in shared)
    shared_bytes = sum(int(size) for _, size in shared)
    require(shared_batches > 0 and shared_bytes > 0 and totals["verify_shared"] > 0,
            "MTP target verification shared-RAM3-input path did not execute")
    require(totals["verify_shared"] + totals["other_shared"] == shared_batches
            and totals["verify_bytes"] + totals["other_bytes"] == shared_bytes
            and totals["optimized_other"] == 0 and totals["optimized_verify"] == totals["batches"],
            "producer-copy phase exclusion or shared-input accounting failed")
    if enabled:
        require(totals["batches"] > 0 and totals["bytes"] > 0 and totals["eligible"] == totals["verify_shared"]
                and totals["bytes"] <= totals["verify_bytes"], "MTP verification producer-stream copy did not execute or scopes differ")
    else:
        require(not any(totals[key] for key in ("eligible", "batches", "tensors", "bytes", "waits", "fallbacks")),
                "OFF run executed producer-stream copy")
    return {"scope": "MTP target verification only; other shared-input phases counted separately; no nested telemetry sum",
            **totals, "shared_input_batches": shared_batches, "shared_input_bytes": shared_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--mode", choices=("speed", "telemetry"), default="speed")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--on-first", action="store_true")
    args = parser.parse_args()
    require(1 <= args.tokens <= 512, "--tokens must be 1..512 for the fixed context")
    if os.name != "nt":
        parser.error("Windows required")
    directory = COMMON["LOG_ROOT"] / f"shared-input-producer-{args.mode}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    try:
        cli = args.build_directory.resolve() / "bin/llama-cli.exe"
        require(cli.is_file(), f"missing {cli}")
        baseline = COMMON["build_environment"]()
        baseline.update({
            "LLAMA_EXPERT_PREFILL_512": "1", "LLAMA_EXPERT_PREFILL_512_POLICY": "auto",
            "LLAMA_EXPERT_PREFILL_LOGICAL_TICKS": "1", "LLAMA_EXPERT_PREFILL_OVERLAP_TEN": "1",
            "LLAMA_EXPERT_EARLY_DECODE": "1",
            "LLAMA_DECODE_TELEMETRY": "1" if args.mode == "telemetry" else None,
            "LLAMA_DECODE_TELEMETRY_PER_RESPONSE": None,
            "GGML_VK_GPU_TIMESTAMPS_DEVICE": None,
            "GGML_VK_SKIP_EMPTY_POOL_RESET": None,
            "GGML_SCHED_TRACE_EXPERT_HANDOFF": None,
        })
        hashes = COMMON["binary_hashes"](cli)
        directory.mkdir()
        COMMON["write_json"](directory / "run.json", {
            "mode": args.mode, "generated_limit": args.tokens, "fixed_prompt_tokens": COMMON["PROMPT_TOKENS"],
            "source_run": str(COMMON["SOURCE_RUN"]), "build_artifact_sha256": hashes,
            "order": ["on", "off"] if args.on_first else ["off", "on"],
            "fence_reuse": "automatic", "shared_input_producer_copy": "OFF vs ON", "verify_width": 2, "prefill_width": 512, "logical_ticks": True, "prefill_policy": "auto80",
        })
        print(f"Log directory: {directory}", flush=True)
        results, answers = {}, {}
        with COMMON["memory_monitor"](True) as (memory, failure):
            for enabled in ([True, False] if args.on_first else [False, True]):
                label = "on" if enabled else "off"
                case = directory / label
                case.mkdir()
                stdin, message = COMMON["prepare_inputs"](case)
                settings = dict(baseline, **{FLAG: "1" if enabled else "0"})
                options = list(COMMON["read_json"](COMMON["SOURCE_RUN"] / "arguments.json"))
                for key, value in {"--log-file": case / "llama.log", "-b": COMMON["OUTER_BATCH"], "-ub": 2,
                                   "-n": args.tokens, "--spec-draft-n-max": 1, "--spec-draft-n-ubatch": 1}.items():
                    options[COMMON["argument_index"](options, key)] = str(value)
                require(options[COMMON["argument_index"](options, "-np")] == "1"
                        and options[COMMON["argument_index"](options, "--spec-type")] == "draft-mtp",
                        "expected single-slot MTP arguments")
                for option in ("--single-turn", "--ignore-eos"):
                    if option not in options:
                        options.append(option)
                if "--output-file" not in options:
                    options += ["--output-file", str(case / "transcript.txt")]
                else:
                    options[COMMON["argument_index"](options, "--output-file")] = str(case / "transcript.txt")
                COMMON["write_json"](case / "settings.json", {"arguments": options, "environment": settings})
                vram = COMMON["monitor_process"](cli, options, settings, case, case, stdin, memory,
                                                 optional_vram=True, vram_failure=failure)
                result, answer = COMMON["parse_results"](case, message, 512, decode_expert_window=True, last_use=True)
                result.update(exit_code=0, local_vram=vram)
                log = (case / "llama.log").read_text(encoding="utf-8", errors="replace")
                result["prefill_overlap10"] = OVERLAP_METRICS(log, True)
                require(re.search(r"ggml_vulkan: Vulkan1 final graph fence reused\b", log),
                        "target Vulkan final graph fence reuse did not execute")
                result["final_graph_fence_reuse_observed"] = True
                result["producer_copy"] = producer_metrics(log, enabled)
                COMMON["write_json"](case / "metrics.json", result)
                (case / "answer.txt").write_bytes(answer)
                results[label], answers[label] = result, answer
        require(hashes == COMMON["binary_hashes"](cli), "binaries changed during comparison")
        report = COMMON["comparison_report"](results["off"], answers["off"], results["on"], answers["on"], False)
        report["producer_copy_off"] = report.pop("width_2")
        report["producer_copy_on"] = report.pop("width_512")
        delta = report["comparison"]
        delta["decode_ms_on_minus_off"] = round(results["on"]["decode_ms"] - results["off"]["decode_ms"], 2)
        delta["decode_expert_window"] = COMMON["decode_window_comparison"](results["off"], results["on"], delta)
        for key in list(delta):
            if "wide_minus_baseline" in key:
                delta[key.replace("wide_minus_baseline", "on_minus_off")] = delta.pop(key)
        report["mode"] = args.mode
        report["timing_scope"] = "speed has no added per-copy clocks; telemetry is diagnostic and separate from speed; cumulative handoffs are not summed with nested phase spans"
        COMMON["write_json"](directory / "comparison.json", report)
        print(json.dumps(delta, indent=2))
        for label, result in results.items():
            print(f"{label}: decode {result['decode_ms']:.2f} ms; producer copy {result['producer_copy']}")
        require(delta["generated_text_matches"] and delta["generated_token_counts_match"] and delta["draft_counts_match"]
                and delta["decode_expert_window"]["comparable"], "generation or MTP scopes differ; inspect comparison.json")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"error: {error}\nLog directory: {directory}\n")


if __name__ == "__main__":
    main()
