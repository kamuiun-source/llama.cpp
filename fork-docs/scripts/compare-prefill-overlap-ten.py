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
CASE_DIRECTORY = "width-512-last-use-on"


def overlap_metrics(log, enabled):
    rows = re.findall(re.escape(MARKER) + r" ([^\r\n]+)", log)
    if not enabled:
        require(not rows, "OFF run unexpectedly activated overlap10")
        return {"enabled": 0}
    require(len(rows) == 1, "expected exactly one overlap10 summary; use the isolated experimental build")
    fields = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)\b", rows[0])}
    for key in ("enabled", "job", "scratch_slots", "bank_slots", "transfer_submits", "transfer_bytes",
                "compute_windows", "pending_windows", "fallback_groups", "fallback_layers"):
        require(key in fields, f"overlap10 summary lacks {key}")
    require(fields["enabled"] == 1 and fields["job"] in (0, 1) and fields["scratch_slots"] == 20
            and fields["bank_slots"] == 10, "overlap10 transfer capability did not activate")
    # The job may already be released when the cumulative counters are logged.
    require(fields["transfer_submits"] > 0 and fields["transfer_bytes"] > 0,
            f"no overlap10 transfers were submitted: {rows[0]}")
    require(0 <= fields["pending_windows"] <= fields["compute_windows"] < fields["transfer_submits"],
            "invalid overlap10 execution counters")
    fields["pending_observed"] = fields["pending_windows"] > 0
    return fields


def recover_arms(directory, provenance):
    require(provenance["order"] in (["off", "on"], ["on", "off"])
            and provenance["width"] == 512 and provenance["verify_width"] == 2
            and provenance["logical_ticks"] is True and provenance["outer_batch"] == COMMON["OUTER_BATCH"]
            and provenance["prompt_tokens"] == COMMON["PROMPT_TOKENS"], "unexpected overlap10 archive configuration")
    hashes = provenance["candidate_build_artifact_sha256"]
    require("llama-cli.exe" in hashes and all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values()),
            "missing or invalid archived build hashes")
    cli = (Path(provenance["build_directory"]) / "bin" / "llama-cli.exe").resolve()
    message = (directory / "expected-user-message.txt").read_bytes()
    stdin = (directory / "stdin.txt").read_bytes()
    require(COMMON["sha256"](message) == provenance["expected_user_message_sha256"]
            and COMMON["sha256"](stdin) == provenance["stdin_sha256"], "archived benchmark input hashes differ")
    results, answers = {}, {}
    reference_settings = None
    for label in provenance["order"]:
        enabled = label == "on"
        arm = directory / label
        case = arm / CASE_DIRECTORY
        require((arm / "expected-user-message.txt").read_bytes() == message
                and (arm / "stdin.txt").read_bytes() == stdin, f"{label} archived inputs differ")
        require((case / "exit.txt").read_text(encoding="utf-8-sig").strip() == "exit_code=0",
                f"{label} subprocess did not exit successfully")
        settings = COMMON["read_json"](case / "settings.json")
        require(Path(settings["cli"]).resolve() == cli, f"{label} CLI differs from the recorded build")
        args = settings["arguments"]
        for option in ("--log-file", "--output-file"):
            args[COMMON["argument_index"](args, option)] = option
        overlap_setting = settings["environment"].pop(OVERLAP_ENV, None)
        require(overlap_setting == "1" if enabled else overlap_setting in (None, "0"),
                f"{label} archived overlap setting differs")
        if reference_settings is None:
            reference_settings = settings
        else:
            require(settings == reference_settings, "archived arms use different benchmark settings")
        parsed, answer = COMMON["parse_results"](case, message, 512, decode_expert_window=True, last_use=True)
        result = COMMON["read_json"](case / "metrics.json")
        require(result["exit_code"] == 0
                and all(result[key] == value for key, value in json.loads(json.dumps(parsed)).items()),
                f"{label} archived metrics differ from the log or transcript")
        require((case / "answer.txt").read_bytes() == answer, f"{label} archived answer differs from the transcript")
        result["overlap10"] = overlap_metrics((case / "llama.log").read_text(encoding="utf-8", errors="replace"), enabled)
        result["recovered_from"] = str(case)
        results[label], answers[label] = result, answer
    return results, answers


def write_report(directory, provenance, results, answers):
    report = COMMON["comparison_report"](results["off"], answers["off"], results["on"], answers["on"], False)
    report["overlap_off"] = report.pop("width_2")
    report["overlap_on"] = report.pop("width_512")
    report["run_provenance"] = provenance
    delta = report["comparison"]
    delta["decode_expert_window"] = COMMON["decode_window_comparison"](results["off"], results["on"], delta)
    for key in list(delta):
        if "wide_minus_baseline" in key:
            delta[key.replace("wide_minus_baseline", "on_minus_off")] = delta.pop(key)
    report["overlap_scope"] = ("transfer_submits and transfer_bytes count successful async transfers. "
                               "fallback_groups and fallback_layers count synchronous fallback at group and layer scope. "
                               "pending_windows samples an unfinished transfer fence after compute submission; "
                               "this is not GPU duration or proof of speedup")
    COMMON["write_json"](directory / "comparison.json", report)
    for label, result in results.items():
        upload = result["prompt_uploads"]
        print(f"{label.upper()}: prompt {result['prompt_ms']:.2f} ms; {upload['tokens']} prompt tokens; "
              f"{upload['uploads']} uploads / {upload['bytes']} bytes; overlap={result['overlap10']}")
    print(json.dumps(delta, indent=2))
    require(delta["decode_expert_window"]["comparable"], "decode window scopes differ; inspect comparison.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, default=Path("G:/AI/build-prefill-overlap-ten"))
    parser.add_argument("--on-first", action="store_true", help="reverse fresh subprocess order for a repeat")
    parser.add_argument("--recover-report", type=Path, metavar="DIRECTORY",
                        help="regenerate comparison.json from two completed archived arms")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("this comparison requires Windows")
    directory = args.recover_report or (
        COMMON["LOG_ROOT"] / f"compare-prefill-overlap-ten-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}")
    try:
        if args.recover_report:
            directory = directory.resolve()
            provenance = COMMON["read_json"](directory / "run.json")
            results, answers = recover_arms(directory, provenance)
            write_report(directory, provenance, results, answers)
            return
        cli = args.build_directory.resolve() / "bin" / "llama-cli.exe"
        require(cli.is_file(), f"missing {cli}")
        baseline = COMMON["build_environment"]()
        baseline.update({"LLAMA_EXPERT_PREFILL_512_POLICY": "512", OVERLAP_ENV: "0"})
        directory.mkdir()
        stdin_path, message = COMMON["prepare_inputs"](directory)
        hashes = COMMON["binary_hashes"](cli)
        provenance = {
            "source_run": str(COMMON["SOURCE_RUN"]), "build_directory": str(args.build_directory.resolve()),
            "candidate_build_artifact_sha256": hashes, "stdin_sha256": COMMON["sha256"](stdin_path.read_bytes()),
            "expected_user_message_sha256": COMMON["sha256"](message), "prompt_tokens": COMMON["PROMPT_TOKENS"],
            "width": 512, "logical_ticks": True, "verify_width": 2, "outer_batch": COMMON["OUTER_BATCH"],
            "order": ["on", "off"] if args.on_first else ["off", "on"],
        }
        COMMON["write_json"](directory / "run.json", provenance)
        print(f"Log directory: {directory}", flush=True)
        results, answers = {}, {}
        with COMMON["memory_monitor"](True) as (memory, failure):
            for enabled in ([True, False] if args.on_first else [False, True]):
                label = "on" if enabled else "off"
                arm = directory / label
                arm.mkdir()
                arm_stdin, arm_message = COMMON["prepare_inputs"](arm)
                require(arm_message == message, "fixed input changed between arms")
                settings = dict(baseline, **{OVERLAP_ENV: "1" if enabled else "0"})
                result, answer = COMMON["run_case"](
                    cli, arm, arm_stdin, message, settings, memory, 512,
                    decode_expert_window=True, last_use=True, vram_failure=failure)
                case = arm / CASE_DIRECTORY
                result["overlap10"] = overlap_metrics((case / "llama.log").read_text(encoding="utf-8", errors="replace"), enabled)
                COMMON["write_json"](case / "metrics.json", result)
                results[label], answers[label] = result, answer
        require(hashes == COMMON["binary_hashes"](cli), "build artifacts changed during comparison")
        write_report(directory, provenance, results, answers)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"error: {error}\nLog directory: {directory}\n")


if __name__ == "__main__":
    main()
