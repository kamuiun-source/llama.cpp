"""Compare synchronous 20-slot and asynchronous 10+10 Prefill512 using one isolated build."""

import argparse
import json
import os
import re
import runpy
from datetime import datetime
from pathlib import Path
from uuid import uuid4


COMMON = runpy.run_path(str(Path(__file__).with_name("compare-prefill512.py")))
require = COMMON["require"]
OVERLAP_ENV = "LLAMA_EXPERT_PREFILL_OVERLAP_TEN"
MARKER = "packed expert prefill overlap10:"


def overlap_metrics(log, enabled):
    rows = re.findall(re.escape(MARKER) + r" ([^\r\n]+)", log)
    if not enabled:
        require(not rows, "OFF run unexpectedly activated overlap10")
        return {"enabled": 0}
    require(len(rows) == 1, "expected exactly one overlap10 activation summary; use the isolated experimental build")
    fields = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)\b", rows[0])}
    for key in ("enabled", "job", "scratch_slots", "bank_slots", "transfer_submits", "transfer_bytes",
                "compute_windows", "pending_windows", "fallback_groups", "fallback_layers"):
        require(key in fields, f"overlap10 summary lacks {key}")
    require(fields["enabled"] == fields["job"] == 1 and fields["scratch_slots"] == 20
            and fields["bank_slots"] == 10, "overlap10 transfer capability did not activate")
    require(fields["transfer_submits"] > 0 and fields["transfer_bytes"] > 0
            and 0 < fields["compute_windows"] < fields["transfer_submits"]
            and fields["pending_windows"] <= fields["compute_windows"], "invalid overlap10 execution counters")
    fields["pending_observed"] = fields["pending_windows"] > 0
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, default=Path("G:/AI/build-prefill-overlap-ten"))
    parser.add_argument("--on-first", action="store_true", help="reverse fresh subprocess order for a repeat")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("this comparison requires Windows")
    directory = COMMON["LOG_ROOT"] / f"compare-prefill-overlap-ten-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    try:
        cli = args.build_directory.resolve() / "bin" / "llama-cli.exe"
        require(cli.is_file(), f"missing {cli}")
        baseline = COMMON["build_environment"]()
        baseline.update({"LLAMA_EXPERT_PREFILL_512_POLICY": "512", OVERLAP_ENV: None})
        directory.mkdir()
        stdin_path, message = COMMON["prepare_inputs"](directory)
        hashes = COMMON["binary_hashes"](cli)
        COMMON["write_json"](directory / "run.json", {
            "source_run": str(COMMON["SOURCE_RUN"]), "build_directory": str(args.build_directory.resolve()),
            "candidate_build_artifact_sha256": hashes, "stdin_sha256": COMMON["sha256"](stdin_path.read_bytes()),
            "expected_user_message_sha256": COMMON["sha256"](message), "prompt_tokens": COMMON["PROMPT_TOKENS"],
            "width": 512, "logical_ticks": True, "verify_width": 2, "outer_batch": COMMON["OUTER_BATCH"],
            "order": ["on", "off"] if args.on_first else ["off", "on"],
        })
        print(f"Log directory: {directory}", flush=True)
        results, answers = {}, {}
        with COMMON["memory_monitor"](True) as (memory, failure):
            for enabled in ([True, False] if args.on_first else [False, True]):
                label = "on" if enabled else "off"
                arm = directory / label
                arm.mkdir()
                arm_stdin, arm_message = COMMON["prepare_inputs"](arm)
                require(arm_message == message, "fixed input changed between arms")
                settings = dict(baseline, **{OVERLAP_ENV: "1" if enabled else None})
                result, answer = COMMON["run_case"](
                    cli, arm, arm_stdin, message, settings, memory, 512,
                    decode_expert_window=True, last_use=True, vram_failure=failure)
                case = arm / "width-512-last-use-on"
                result["overlap10"] = overlap_metrics((case / "llama.log").read_text(encoding="utf-8", errors="replace"), enabled)
                COMMON["write_json"](case / "metrics.json", result)
                results[label], answers[label] = result, answer
        require(hashes == COMMON["binary_hashes"](cli), "build artifacts changed during comparison")
        report = COMMON["comparison_report"](results["off"], answers["off"], results["on"], answers["on"], False)
        report["overlap_off"] = report.pop("width_2")
        report["overlap_on"] = report.pop("width_512")
        delta = report["comparison"]
        delta["decode_expert_window"] = COMMON["decode_window_comparison"](results["off"], results["on"], delta)
        for key in list(delta):
            if "wide_minus_baseline" in key:
                delta[key.replace("wide_minus_baseline", "on_minus_off")] = delta.pop(key)
        report["overlap_scope"] = "pending_windows samples an unfinished transfer fence after compute submission; this is not GPU duration or proof of speedup"
        COMMON["write_json"](directory / "comparison.json", report)
        for label, result in results.items():
            upload = result["prompt_uploads"]
            print(f"{label.upper()}: prompt {result['prompt_ms']:.2f} ms; {upload['tokens']} prompt tokens; "
                  f"{upload['uploads']} uploads / {upload['bytes']} bytes; overlap={result['overlap10']}")
        print(json.dumps(delta, indent=2))
        require(delta["decode_expert_window"]["comparable"], "decode window scopes differ; inspect comparison.json")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"error: {error}\nLog directory: {directory}\n")


if __name__ == "__main__":
    main()
