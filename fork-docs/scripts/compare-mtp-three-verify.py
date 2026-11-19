"""Compare fixed one-draft, fixed two-draft and adaptive MTP."""

import argparse
import json
import math
import os
import re
import runpy
from datetime import datetime
from pathlib import Path
from uuid import uuid4


COMMON = runpy.run_path(str(Path(__file__).with_name("compare-prefill512.py")))
OVERLAP = runpy.run_path(str(Path(__file__).with_name("compare-prefill-overlap-ten.py")))["overlap_metrics"]
require = COMMON["require"]
VERIFY = "MTP 3-position verify:"
EXPERT = "MTP 3-position expert:"
EXPERT_LAYERS = 48
ADAPTIVE = "MTP adaptive depth:"


def verify_metrics(log, enabled, require_three=True):
    verify_rows = re.findall(re.escape(VERIFY) + r" draft2_proposed=([0-9]+) first_accepted=([0-9]+) both_accepted=([0-9]+)", log)
    expert_rows = re.findall(re.escape(EXPERT) + r" layers=([0-9]+) overflow_layers=([0-9]+) max_nonresident=([0-9]+)", log)
    if not enabled:
        require(VERIFY not in log and EXPERT not in log, "baseline unexpectedly used three-position verification")
        return dict(draft2_proposed=0, first_accepted=0, both_accepted=0, layers=0, overflow_layers=0,
                    max_nonresident=0, overflow_fraction=None)

    require(len(verify_rows) == len(expert_rows) == 1, "missing or ambiguous three-position verification counters")
    proposed, first, both = map(int, verify_rows[0])
    layers, overflow, max_cold = map(int, expert_rows[0])
    require(0 <= both <= first <= proposed and (proposed > 0 or not require_three),
            "invalid two-draft acceptance counters")
    require(layers == proposed * EXPERT_LAYERS and 0 <= overflow <= layers and 0 <= max_cold <= 30
            and (overflow > 0) == (max_cold > 20) and (layers > 0 or max_cold == 0),
            "three-position Expert layer counters disagree with the verified rounds")
    return dict(
        draft2_proposed=proposed, first_accepted=first, both_accepted=both,
        first_given_draft2_proposed=first / proposed if proposed else None,
        both_given_draft2_proposed=both / proposed if proposed else None,
        second_given_first_accepted=both / first if first else None,
        layers=layers, overflow_layers=overflow, max_nonresident=max_cold,
        overflow_fraction=overflow / layers if layers else None,
    )


def fast_three_metrics(log, layers, overflow):
    waves = re.findall(r"MTP 3-position waves: second_passes=(\d+) empty_no_copy=(\d+)", log)
    vk = re.findall(r"mtp fast-three Vulkan: gate_up=(\d+) weighted=(\d+) reduce=(\d+)", log)
    require(len(waves) == 1 and vk, "missing fast width-3 runtime evidence")
    second_passes, empty_no_copy = map(int, waves[0])
    gate_up, weighted, reduce = map(sum, zip(*(map(int, row) for row in vk)))
    require(second_passes == overflow and empty_no_copy == layers - overflow,
            "width-3 second waves or no-copy carry disagree with unique nonresident Expert counts")
    require(gate_up > 0 and weighted > 0 and gate_up <= 3 * (layers + overflow) and weighted + reduce <= layers,
            "fast width-3 Vulkan activations missing or inconsistent")
    return dict(second_passes=second_passes, empty_no_copy=empty_no_copy,
                gate_up_dispatches=gate_up, weighted_dispatches=weighted, reduce_dispatches=reduce)


def adaptive_metrics(log, enabled, verify):
    rows = re.findall(re.escape(ADAPTIVE) + r" ([^\r\n]+)", log)
    if not enabled:
        require(not rows, "fixed-depth arm emitted adaptive counters")
        return None
    require(len(rows) == 1, "missing or ambiguous adaptive counters")
    fields = dict(re.findall(r"(\w+)=([0-9]+(?:\.[0-9]+)?)", rows[0]))
    integers = ("decisions", "explore", "depth2", "chain", "proposed2", "first", "chain_first", "proposed2_first", "both")
    result = {key: int(fields[key]) for key in integers}
    decisions, explore, two, chain, proposed, first, chain_first, proposed_first, both = (result[key] for key in integers)
    require(decisions == two + chain and decisions > 0 and min(decisions, 8) <= explore <= decisions
            and 0 <= both <= proposed_first <= min(proposed, chain_first)
            and 0 <= chain_first <= min(first, chain) and 0 <= first <= decisions
            and both <= proposed <= chain and proposed == verify["draft2_proposed"]
            and proposed_first == verify["first_accepted"] and both == verify["both_accepted"], "inconsistent adaptive decision and acceptance counters")
    result["second_given_first_after_chain_attempt"] = both / chain_first if chain_first else None
    for width, count in ((2, two), (3, chain)):
        draft = float(fields[f"draft{width}_us"])
        cycle = float(fields[f"round{width}_us"])
        require(0 <= draft <= cycle and (cycle > 0) == (count > 0), "invalid adaptive cost")
        result.update({f"draft{width}_us": draft, f"round{width}_us": cycle,
                       f"after_draft{width}_us": round(cycle - draft, 1)})
    return result


def pair_report(results, answers, baseline, candidate):
    report = COMMON["comparison_report"](results[baseline], answers[baseline], results[candidate], answers[candidate], False)
    delta = report["comparison"]
    delta["decode_ms_candidate_minus_baseline"] = round(results[candidate]["decode_ms"] - results[baseline]["decode_ms"], 2)
    delta["decode_tokens_per_second_candidate_minus_baseline"] = (
        results[candidate]["decode_tokens_per_second"] - results[baseline]["decode_tokens_per_second"])
    for key in list(delta):
        if "wide_minus_baseline" in key:
            delta[key.replace("wide_minus_baseline", "candidate_minus_baseline")] = delta.pop(key)
    return delta


def read_logit_trace(log, count, reference=None):
    rows, verifies, resets, checkpoints = [], {}, {}, {}
    ubatch = None
    for match in re.finditer(r"mtp logits (ubatch|row|verify|draft reset|checkpoint): ([^\r\n]+)", log):
        kind, payload = match.groups()
        fields = {key: int(value) for key, value in re.findall(r"(\w+)=(-?\d+)(?= |$)", payload)
                  if key not in ("top_logit", "second_logit", "margin", "chosen_logit")}
        if kind == "ubatch":
            ubatch = fields
            require(ubatch["to"] - ubatch["from"] + 1 == ubatch["size"], "noncontiguous target microbatch")
        elif kind == "verify":
            require(fields["round"] not in verifies, "duplicate finalized verification round")
            verifies[fields["round"]] = fields
        elif kind == "draft reset":
            resets[fields["index"]] = fields
        elif kind == "checkpoint":
            checkpoints[fields["round"]] = fields
        else:
            index = len(rows)
            for key in ("top_logit", "second_logit", "margin", "chosen_logit"):
                fields[key] = float(re.search(rf"(?:^| ){key}=(\S+)", payload)[1])
            require(ubatch is not None and ubatch["from"] <= fields["pos"] <= ubatch["to"] and ubatch["outputs"] > 0,
                    f"logit row {index} does not belong to its executed target microbatch")
            require(fields["index"] == index and fields["top"] != fields["second"] and fields["margin"] >= 0
                    and all(math.isfinite(fields[key]) for key in ("top_logit", "second_logit", "margin", "chosen_logit"))
                    and math.isclose(fields["margin"], fields["top_logit"] - fields["second_logit"], abs_tol=1e-5),
                    f"invalid commitment index or logits at row {index}")
            require(fields["round"] == index or (index > 0 and fields["round"] == rows[-1]["round"]),
                    f"commitment is not in its recorded verification round at row {index}")
            require(0 <= fields["row"] < fields["width"], f"invalid server batch row {index}")
            require(bool(ubatch["prefill"]) == (index == 0), f"unexpected target prompt/decode mode at row {index}")
            if index:
                require(fields["pos"] == rows[-1]["pos"] + 1 and fields["input"] == rows[-1]["chosen"],
                        f"committed prefix is misaligned at row {index}")
                require(ubatch["size"] == fields["width"] and fields["pos"] == ubatch["from"] + fields["row"],
                        f"output row {index} does not align with actual nonprompt microbatch")
            if reference is None:
                require(fields["chosen"] == fields["natural"], f"reference arm was forced at row {index}")
            else:
                require(index < len(reference) and fields["chosen"] == reference[index],
                        f"reference token ID differs at row {index}")
            rows.append(dict(fields, ubatch=ubatch))
    require(len(rows) == count, f"expected {count} committed logit rows, got {len(rows)}")
    for round_index, event in verifies.items():
        proposed = event["proposed"]
        require(0 <= round_index < count and 1 <= proposed <= 2 and 0 <= event["accepted"] <= proposed
                and round_index + event["accepted"] < count
                and rows[round_index]["round"] == round_index and rows[round_index]["width"] == proposed + 1,
                f"verify record is misaligned at round {round_index}")
        for index in range(event["accepted"]):
            require(rows[round_index + index]["chosen"] == event[f"draft{index}"],
                    f"accepted draft disagrees with committed token at round {round_index}")
        expected_tail = rows[round_index]["pos"] + event["accepted"]
        event.update(expected_tail=expected_tail, rollback_start_matches=event["from"] == expected_tail + 1,
                     target_tail_matches=event["target_after"] == expected_tail,
                     draft_tail_matches=event["draft_after"] == expected_tail)
    previous_rejection = None
    for row in rows:
        event = verifies.get(row["round"])
        if not row["ubatch"]["prefill"] and row["width"] > 1:
            require(event is not None and (row["replay"] or row["ubatch"]["verify"] == row["width"]),
                    f"missing executed fixed-width MTP round at commitment {row['index']}")
        if row["index"] == row["round"]:
            prior = previous_rejection
            if event and event["accepted"] < event["proposed"]:
                previous_rejection = event
        row.update(verification=event, prior_rejection=prior, draft_reset=resets.get(row["round"]),
                   checkpoint=checkpoints.get(row["round"]))
    return rows


def logit_comparison(reference, candidate, control=None):
    require(len(reference) == len(candidate) and (control is None or len(reference) == len(control)),
            "different logit trace lengths")
    differences, top_changes, by_shape = [], [], {}
    max_delta = 0.0
    for left, right in zip(reference, candidate):
        require((left["pos"], left["chosen"], left["input"]) == (right["pos"], right["chosen"], right["input"]),
                f"target position or fixed token mismatch at commitment {left['index']}")
        delta = right["chosen_logit"] - left["chosen_logit"]
        max_delta = max(max_delta, abs(delta))
        shape = f"{left['ubatch']['size']}:{left['row']}->{right['ubatch']['size']}:{right['row']}"
        counts = by_shape.setdefault(shape, {"rows": 0, "differences": 0})
        counts["rows"] += 1
        counts["differences"] += left["natural"] != right["natural"]
        if left["natural"] != right["natural"]:
            differences.append(dict(width_2=left, width_3=right, chosen_logit_delta=delta))
        if left["top"] != right["top"]:
            top_changes.append(dict(width_2=left, width_3=right, chosen_logit_delta=delta))
    first = differences[0] if differences else None
    if control is not None:
        for difference in (first, top_changes[0] if top_changes else None):
            if difference is not None:
                right = difference["width_3"]
                comparison = control[right["index"]]
                require((comparison["pos"], comparison["input"]) == (right["pos"], right["input"]),
                        "no-rejection control position is misaligned")
                difference["no_rejection_control"] = comparison
                difference["control_matches_width3_shape"] = (comparison["ubatch"]["size"], comparison["ubatch"]["verify"], comparison["row"]) == (
                    right["ubatch"]["size"], right["ubatch"]["verify"], right["row"])
    return dict(committed_tokens=len(reference), natural_sample_differences=len(differences),
                raw_top1_differences=len(top_changes), first_natural_difference=first,
                first_raw_top1_difference=top_changes[0] if top_changes else None,
                max_abs_chosen_logit_delta=max_delta, natural_differences_by_batch_and_row=by_shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--logit-diagnostic", action="store_true", help="record greedy width-2 token IDs and force the same IDs through fixed width 3")
    modes.add_argument("--two-vs-auto", action="store_true", help="compare fixed width 2 and adaptive only, reporting text differences")
    parser.add_argument("--no-rejection-control", action="store_true", help="add a width-3 diagnostic arm whose proposals are replaced by the reference IDs")
    order_options = parser.add_mutually_exclusive_group()
    order_options.add_argument("--three-first", action="store_true", help="run fixed three, then fixed two, then adaptive")
    order_options.add_argument("--auto-first", action="store_true", help="run adaptive first")
    parser.add_argument("--fast-three", action="store_true", help="test the fast width-3 path with the accepted width-2 baseline")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("this comparison requires Windows")
    if args.fast_three and not args.logit_diagnostic:
        parser.error("use --fast-three with --logit-diagnostic, or compare-mtp-warm.py --fast-three for speed")
    if not 8 <= args.tokens <= 512:
        parser.error("--tokens must be 8..512 for the fixed context")
    if args.no_rejection_control and not args.logit_diagnostic:
        parser.error("--no-rejection-control requires --logit-diagnostic")
    if args.logit_diagnostic and (args.three_first or args.auto_first):
        parser.error("the logit diagnostic records the width-2 reference first")
    if args.two_vs_auto and args.three_first:
        parser.error("--three-first requires a fixed width-3 arm")

    run_name = "mtp-logit-diagnostic" if args.logit_diagnostic else "mtp-three-verify"
    if args.two_vs_auto:
        run_name = "mtp-two-vs-auto"
    if args.fast_three:
        run_name = "mtp-fast-three-logit"
    directory = COMMON["LOG_ROOT"] / f"{run_name}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    try:
        cli = args.build_directory.resolve() / "bin/llama-cli.exe"
        require(cli.is_file(), f"missing {cli}")
        order = ["width-2", "auto", "width-3"]
        if args.three_first:
            order = ["width-3", "width-2", "auto"]
        if args.auto_first:
            order = ["auto", "width-3", "width-2"]
        if args.two_vs_auto:
            order = ["auto", "width-2"] if args.auto_first else ["width-2", "auto"]
        elif args.logit_diagnostic:
            order = ["width-2", "width-3"]
            if args.no_rejection_control:
                order.append("width-3-no-rejection")
        baseline = COMMON["build_environment"]()
        baseline.update({
            "LLAMA_EXPERT_PREFILL_512": "1",
            "LLAMA_EXPERT_PREFILL_512_POLICY": "auto",
            "LLAMA_EXPERT_PREFILL_LOGICAL_TICKS": "1",
            "LLAMA_EXPERT_PREFILL_OVERLAP_TEN": "1",
            "LLAMA_MTP_REUSE_DRAFT_ANCHOR": None,
            "LLAMA_MTP_VERIFY_THREE_FAST": "0",
            "LLAMA_MTP_DRAFT_VOCAB": None,
            "LLAMA_MTP_LOGIT_TRACE": "1" if args.logit_diagnostic else None,
            "LLAMA_MTP_LOGIT_REFERENCE": None,
            "LLAMA_MTP_LOGIT_MATCH_DRAFTS": None,
            "GGML_SCHED_SHARED_INPUT_PRODUCER_COPY": "0",
            "GGML_VK_GPU_TIMESTAMPS_DEVICE": None,
            "GGML_VK_SKIP_EMPTY_POOL_RESET": None,
            "GGML_SCHED_TRACE_EXPERT_HANDOFF": None,
        })
        if args.fast_three:
            baseline.update({
                "GGML_CUDA_HC_DOWN_REUSE": "1", "GGML_CUDA_HC_UP_REUSE": "1", "GGML_CUDA_HC_INJECT_SPLIT": "1",
                "GGML_CUDA_HC_MIX_FUSE": None, "GGML_CUDA_HC_SCALE_SILU_FUSE": None, "GGML_CUDA_HC_UP_PAIR": None,
                "GGML_CUDA_DISABLE_GRAPHS": None, "GGML_VK_PERF_LOGGER": None, "GGML_VK_PERF_LOGGER_CONCURRENT": None,
                "GGML_VK_SKIP_EMPTY_POOL_RESET": "0", "GGML_VK_MTP_ROUTED_REDUCE": "1", "GGML_VK_MTP_ROUTED_WEIGHTED": "1",
                "GGML_VK_MTP_ROUTED_GATE_UP": "1", "GGML_VK_MTP_ROUTED_GATE_UP_PAIR": "0", "LLAMA_MTP_PAIR_EXPERT_OVERLAP": None,
            })
        options = COMMON["read_json"](COMMON["SOURCE_RUN"] / "arguments.json")
        require(options[COMMON["argument_index"](options, "-np")] == "1"
                and options[COMMON["argument_index"](options, "--spec-type")] == "draft-mtp",
                "expected single-slot MTP arguments")
        hashes = COMMON["binary_hashes"](cli)
        directory.mkdir()
        stdin, message = COMMON["prepare_inputs"](directory)
        provenance = {
            "source_run": str(COMMON["SOURCE_RUN"]), "build_directory": str(args.build_directory.resolve()),
            "build_artifact_sha256": hashes, "stdin_sha256": COMMON["sha256"](stdin.read_bytes()),
            "expected_user_message_sha256": COMMON["sha256"](message),
            "generated_limit": args.tokens, "prompt_tokens": COMMON["PROMPT_TOKENS"], "order": order,
            "timing_scope": ("Diagnostic timings include logit reads and logging."
                             if args.logit_diagnostic else
                             "One fresh CLI process and uncached prompt per arm. Prompt/load timing is separate. "
                             "Decode timing includes first-use graph effects and adaptive cold-start exploration."),
        }
        COMMON["write_json"](directory / "run.json", provenance)
        print(f"Log directory: {directory}", flush=True)
        results, answers = {}, {}
        traces = {}
        reference_path = directory / "committed-token-ids.txt"
        reference = None
        with COMMON["memory_monitor"](True) as (memory, failure):
            for mode in order:
                width = 2 if mode == "width-2" else 3
                case = directory / mode
                case.mkdir()
                arm_stdin, arm_message = COMMON["prepare_inputs"](case)
                require(arm_message == message and arm_stdin.read_bytes() == stdin.read_bytes(), "benchmark input changed")
                verify_mode = {"width-2": None, "width-3": "1", "width-3-no-rejection": "1", "auto": "auto"}[mode]
                settings = dict(baseline, LLAMA_MTP_VERIFY_THREE=verify_mode)
                if args.fast_three and width == 3:
                    settings["LLAMA_MTP_VERIFY_THREE_FAST"] = "1"
                if args.logit_diagnostic and reference is not None:
                    settings["LLAMA_MTP_LOGIT_REFERENCE"] = str(reference_path)
                    settings["LLAMA_MTP_LOGIT_MATCH_DRAFTS"] = "1" if mode == "width-3-no-rejection" else None
                arguments = list(options)
                for key, value in {
                    "--log-file": case / "llama.log", "-b": COMMON["OUTER_BATCH"], "-ub": 2, "-n": args.tokens,
                    "--spec-draft-n-max": width - 1, "--spec-draft-n-ubatch": 1,
                }.items():
                    arguments[COMMON["argument_index"](arguments, key)] = str(value)
                for option in ("--single-turn", "--ignore-eos"):
                    if option not in arguments:
                        arguments.append(option)
                if "--output-file" in arguments:
                    arguments[COMMON["argument_index"](arguments, "--output-file")] = str(case / "transcript.txt")
                else:
                    arguments += ["--output-file", str(case / "transcript.txt")]
                COMMON["write_json"](case / "settings.json", {"cli": str(cli), "arguments": arguments, "environment": settings})
                vram = COMMON["monitor_process"](cli, arguments, settings, case, case, arm_stdin, memory,
                                                 optional_vram=True, vram_failure=failure)
                log = (case / "llama.log").read_text(encoding="utf-8", errors="replace")
                verify = verify_metrics(log, width == 3, require_three=mode != "auto")
                actual_width = 3 if verify["draft2_proposed"] else 2
                result, answer = COMMON["parse_results"](case, message, 512, last_use=True, verify_width=width,
                                                         actual_verify_width=actual_width)
                result.update(exit_code=0, local_vram=vram, prefill_overlap10=OVERLAP(log, True),
                              mtp_three=verify, adaptive=adaptive_metrics(log, mode == "auto", verify))
                require(result["mtp_three"]["draft2_proposed"] * 2 <= result["draft_generated"],
                        "two-draft rounds exceed total MTP proposals")
                result["decode_tokens_per_second"] = 1000 * result["generated_tokens"] / result["decode_ms"]
                if args.fast_three and width == 3:
                    result["fast_three"] = fast_three_metrics(log, verify["layers"], verify["overflow_layers"])
                if args.logit_diagnostic:
                    trace = read_logit_trace(log, args.tokens, reference)
                    require(trace[0]["pos"] + 1 == COMMON["PROMPT_TOKENS"], "unexpected first committed target position")
                    if reference is None:
                        reference = [row["chosen"] for row in trace]
                        reference_path.write_text("\n".join(map(str, reference)) + "\n", encoding="ascii")
                    if mode == "width-3-no-rejection":
                        require(not any(row["prior_rejection"] or row["replay"] or
                                        (row["verification"] and row["verification"]["accepted"] < row["verification"]["proposed"])
                                        for row in trace), "no-rejection control rejected a reference proposal")
                    if args.fast_three:
                        finalized = [row["verification"] for row in trace if row["index"] == row["round"] and row["verification"]]
                        result["rollback_validation"] = {
                            "finalized_rounds": len(finalized),
                            "rejected_rounds": sum(event["accepted"] < event["proposed"] for event in finalized),
                            "replayed_rounds": sum(bool(event["replay"]) for event in finalized),
                            "mismatches": [event for event in finalized if not all(
                                event[key] for key in ("rollback_start_matches", "target_tail_matches", "draft_tail_matches"))],
                        }
                    traces[mode] = trace
                    COMMON["write_json"](case / "logits.json", trace)
                COMMON["write_json"](case / "metrics.json", result)
                (case / "answer.txt").write_bytes(answer)
                results[mode], answers[mode] = result, answer
        require(hashes == COMMON["binary_hashes"](cli), "binaries changed during comparison")
        pairs = (("width-2", "width-3"), ("width-2", "auto"), ("width-3", "auto"))
        if args.two_vs_auto:
            pairs = (("width-2", "auto"),)
        elif args.logit_diagnostic:
            pairs = (("width-2", "width-3"),)
        comparison = {
            f"{candidate}_vs_{baseline}": pair_report(results, answers, baseline, candidate)
            for baseline, candidate in pairs
        }
        report = {"runs": results, "comparison": comparison, "run_provenance": provenance}
        if args.logit_diagnostic:
            report["logit_diagnostic"] = logit_comparison(traces["width-2"], traces["width-3"], traces.get("width-3-no-rejection"))
            report["committed_token_ids_sha256"] = COMMON["sha256"](reference_path.read_bytes())
            report["diagnostic_scope"] = ("Same committed token IDs, actual target logits before sampling, and normal MTP "
                                          "acceptance and rollback on those IDs. The logit row position is the input "
                                          "position predicting the next committed token. Optional control replaces proposals "
                                          "before verification. Compare the same microbatch width and row when "
                                          "interpreting the control. Diagnostic timings include tracing overhead.")
        COMMON["write_json"](directory / "comparison.json", report)
        for mode, result in results.items():
            print(f"{mode}: {result['decode_ms']:.2f} ms, {result['decode_tokens_per_second']:.2f} tok/s")
        print(json.dumps(report["logit_diagnostic"] if args.logit_diagnostic else
                         {"comparison": comparison, "adaptive": results["auto"]["adaptive"]}, indent=2))
        if args.fast_three:
            require(not any(result["rollback_validation"]["mismatches"] for result in results.values()),
                    "committed verification rollback ranges or tails differ; inspect comparison.json")
        require(all(delta["generated_token_counts_match"] for delta in comparison.values()),
                "generated token counts differ; inspect comparison.json")
        if not args.two_vs_auto:
            require(all(delta["generated_text_matches"] for delta in comparison.values()),
                    "generated text differs; inspect the answers and comparison.json")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"error: {error}\nLog directory: {directory}\n")


if __name__ == "__main__":
    main()
