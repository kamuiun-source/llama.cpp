"""Compare MTP depth and optional CUDA or Vulkan optimizations after warmup."""

import argparse
import json
import mmap
import os
import re
import runpy
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4


HERE = Path(__file__).resolve().parent
SHORT = runpy.run_path(str(HERE / "compare-short-prefill.py"))
MTP = runpy.run_path(str(HERE / "compare-mtp-three-verify.py"))
COMMON = SHORT["COMMON"]
require = COMMON["require"]
write_json = COMMON["write_json"]
MODES = ("width-2", "auto", "width-3")
GENERATED = 256
WARMUP = 128
HC_PROJECTIONS = {
    "GGML_CUDA_HC_DOWN_REUSE": "CUDA HC down reuse applied",
    "GGML_CUDA_HC_UP_REUSE": "CUDA HC up reuse applied",
    "GGML_CUDA_HC_INJECT_SPLIT": "CUDA HC inject split applied",
}
OLDER_HC = ("GGML_CUDA_HC_MIX_FUSE", "GGML_CUDA_HC_SCALE_SILU_FUSE", "GGML_CUDA_HC_UP_PAIR")
EMPTY_RESET = "GGML_VK_SKIP_EMPTY_POOL_RESET"
ROUTED_STATS = "LLAMA_MTP_ROUTED_DISPATCH_STATS"
ROUTED_REDUCE = "GGML_VK_MTP_ROUTED_REDUCE"
ROUTED_WEIGHTED = "GGML_VK_MTP_ROUTED_WEIGHTED"
GATE_UP = "GGML_VK_MTP_ROUTED_GATE_UP"
GATE_UP_PAIR = "GGML_VK_MTP_ROUTED_GATE_UP_PAIR"


def arm_order(rotation, reverse):
    order = MODES[rotation:] + MODES[:rotation]
    return order[::-1] if reverse else order


def prepare_token_ids(port, key, message, directory):
    rendered = SHORT["http"](port, "/apply-template", key, {
        "messages": [{"role": "user", "content": message.decode("utf-8")}],
        "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False},
    })["prompt"]
    tokens = SHORT["http"](port, "/tokenize", key, {
        "content": rendered, "add_special": True, "parse_special": True,
    })["tokens"]
    require(isinstance(tokens, list) and 512 <= len(tokens) < 2048 - GENERATED
            and all(type(token) is int and token >= 0 for token in tokens),
            "invalid rendered prompt or insufficient context for 256 generated tokens")
    (directory / "rendered-prompt.txt").write_bytes(rendered.encode("utf-8"))
    write_json(directory / "prompt-token-ids.json", tokens)
    print(f"Prompt tokens: {len(tokens)} (historical CLI: {COMMON['PROMPT_TOKENS']})", flush=True)
    return tokens


def read_completion(response, prompt_count, token_count, wall_ms):
    timing = response["timings"]
    require(response["stop"] is True and not response["truncated"] and response["stop_type"] == "limit"
            and response["tokens_predicted"] == timing["predicted_n"] == token_count, "incomplete generation")
    require(timing["cache_n"] == 0 and response["tokens_evaluated"] == timing["prompt_n"] == prompt_count,
            "prompt cache reuse or prompt token count mismatch")
    tokens = response["tokens"]
    require(isinstance(tokens, list) and len(tokens) == token_count
            and all(type(token) is int and token >= 0 for token in tokens), "missing generated token IDs")
    require(0 <= timing["draft_n_accepted"] <= timing["draft_n"] and timing["draft_n"] > 0,
            "missing genuine MTP activity")
    decode_ms = timing["predicted_ms"]
    require(0 < decode_ms <= wall_ms and timing["prompt_ms"] > 0, "invalid response timing")
    return dict(prompt_tokens=prompt_count, generated_tokens=token_count, decode_tokens=token_count - 1,
                prompt_ms=timing["prompt_ms"], decode_ms=decode_ms, request_wall_ms=wall_ms,
                decode_tok_s=1000 * (token_count - 1) / decode_ms,
                generated_per_request_wall_tok_s=1000 * token_count / wall_ms,
                draft_proposed=timing["draft_n"], draft_accepted=timing["draft_n_accepted"],
                answer=response["content"], token_ids=tokens)


def attach_runtime_counters(log, mode, requests):
    snapshots = 1 if mode == "width-2" else 2
    for proof in ("using device CUDA0 (NVIDIA GeForce RTX 4070 Ti)", "using device Vulkan1 (AMD Radeon Pro VII)",
                  f"packed expert batch capacity = 2, n_ubatch = 2, n_rs_seq = {snapshots}",
                  "20 scratch slots per layer"):
        require(proof in log, f"{mode} missing hardware proof: {proof}")
    require("mtp_draft_vocab: enabled=0" in log and "mtp_draft_vocab: enabled=1" not in log,
            "full MTP draft vocabulary was not confirmed")
    require("mtp logits row:" not in log, "logit diagnostic was active")
    prompt = COMMON["first_match"](r"packed expert prompt uploads: ([^\r\n]+)", log, "full prompt and width summary")
    fields = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", prompt)}
    require(fields["tokens"] == sum(item["prompt_tokens"] for item in requests)
            and fields["capacity"] == fields["max_tokens"] == 512, "run-wide prompt accounting or prefill width differs")

    raw_verify = re.findall(r"MTP 3-position verify: draft2_proposed=(\d+) first_accepted=(\d+) both_accepted=(\d+)", log)
    raw_auto = re.findall(r"MTP adaptive depth: [^\r\n]+", log)
    raw_draft = re.findall(r"mtp_vocab_request: request=(\d+) draft_wall_us=(\d+) drafted=(\d+) accepted=(\d+) generated=(\d+) single_slot=1", log)
    require(len(raw_verify) == (0 if mode == "width-2" else len(requests))
            and len(raw_auto) == (len(requests) if mode == "auto" else 0)
            and len(raw_draft) == len({row[0] for row in raw_draft}) == len(requests),
            "missing or unexpected per-request MTP summaries")
    for index, request in enumerate(requests):
        request_id, draft_wall_us, drafted, accepted, generated = map(int, raw_draft[index])
        require((drafted, accepted, generated) ==
                (request["draft_proposed"], request["draft_accepted"], request["generated_tokens"])
                and draft_wall_us > 0, "per-request HTTP and MTP draft accounting differ")
        request.update(request_id=request_id, draft_wall_ms=draft_wall_us / 1000)
        proposed, first, both = map(int, raw_verify[index]) if raw_verify else (0, 0, 0)
        require(0 <= both <= first <= proposed and 2 * proposed <= request["draft_proposed"]
                and first + both <= request["draft_accepted"], "per-request acceptance counters disagree")
        require(mode != "width-3" or proposed > 0, "fixed width 3 did not verify two proposals")
        request["two_draft"] = dict(draft2_proposed=proposed, first_accepted=first, both_accepted=both)
        if raw_auto:
            request["adaptive"] = MTP["adaptive_metrics"](raw_auto[index], True, request["two_draft"])
            require(request["adaptive"]["decisions"] > 8, "adaptive request did not get beyond its startup probes")

    expert = re.findall(r"MTP 3-position expert: layers=(\d+) overflow_layers=(\d+) max_nonresident=(\d+)", log)
    require(len(expert) == (0 if mode == "width-2" else 1), "unexpected run-wide width-3 Expert summary")
    layers, overflow, max_cold = map(int, expert[0]) if expert else (0, 0, 0)
    all_proposed = sum(item["two_draft"]["draft2_proposed"] for item in requests)
    require(layers == all_proposed * MTP["EXPERT_LAYERS"] and 0 <= overflow <= layers and 0 <= max_cold <= 30
            and (overflow > 0) == (max_cold > 20) and (layers > 0 or max_cold == 0)
            and fields["decode_max_tokens"] == (3 if all_proposed else 2), "actual target width or Expert counts disagree")
    return dict(layers=layers, overflow_layers=overflow, max_nonresident=max_cold,
                overflow_fraction=overflow / layers if layers else None)


def run_arm(case, library, mode, baseline, message, token_ids):
    case.mkdir()
    port, key = SHORT["local_port"](), uuid4().hex
    arguments = SHORT["server_args"](case, port, key)
    for option, value in (("-n", GENERATED), ("-ub", 2), ("--spec-draft-n-max", 1 if mode == "width-2" else 2),
                          ("--spec-draft-n-ubatch", 1), ("-np", 1), ("--spec-type", "draft-mtp"), ("--temp", 0)):
        arguments[COMMON["argument_index"](arguments, option)] = str(value)
    arguments += ["--spec-draft-sampling", "greedy", "--spec-draft-p-min", "0"]
    environment = dict(baseline, LLAMA_MTP_VERIFY_THREE={"width-2": None, "auto": "auto", "width-3": "1"}[mode])
    saved_args = arguments.copy()
    saved_args[COMMON["argument_index"](saved_args, "--api-key")] = "[ephemeral]"
    write_json(case / "settings.json", {"library": str(library), "arguments": saved_args, "environment": environment})
    requests = []
    print(f"{case.name}: loading server and warming with {WARMUP} tokens", flush=True)
    with SHORT["server"](case, {"library": str(library), "arguments": arguments}, environment, port, key):
        if token_ids is None:
            token_ids = prepare_token_ids(port, key, message, case.parent)
        for stage, count in (("warmup", WARMUP), ("measure-1", GENERATED), ("measure-2", GENERATED)):
            payload = {"prompt": token_ids, "n_predict": count, "cache_prompt": False, "temperature": 0.0, "seed": 1234,
                       "ignore_eos": True, "stream": False, "return_tokens": True}
            write_json(case / f"{stage}-request.json", payload)
            start = time.perf_counter()
            response = SHORT["http"](port, "/completion", key, payload, timeout=300)
            wall_ms = 1000 * (time.perf_counter() - start)
            write_json(case / f"{stage}-response.json", response)
            result = read_completion(response, len(token_ids), count, wall_ms)
            requests.append(result)
            print(f"{case.name} {stage}: {result['decode_tok_s']:.2f} tok/s, {result['decode_ms']:.2f} ms decode, "
                  f"{wall_ms:.2f} ms request", flush=True)
    log = (case / "server.log").read_text(encoding="utf-8", errors="replace")
    expert = attach_runtime_counters(log, mode, requests)
    if mode == "width-3" and environment.get("LLAMA_MTP_VERIFY_THREE_FAST") == "1":
        expert.update(MTP["fast_three_metrics"](log, expert["layers"], expert["overflow_layers"]))
    measured = requests[1:]
    result = dict(warmup=requests[0], measured=measured, expert_including_warmup=expert,
                  repeat_token_ids_match=measured[0]["token_ids"] == measured[1]["token_ids"],
                  repeat_text_matches=measured[0]["answer"] == measured[1]["answer"],
                  pooled_decode_tok_s=1000 * sum(rep["decode_tokens"] for rep in measured) / sum(rep["decode_ms"] for rep in measured),
                  pooled_request_wall_tok_s=1000 * sum(rep["generated_tokens"] for rep in measured) / sum(rep["request_wall_ms"] for rep in measured))
    write_json(case / "metrics.json", result)
    return result, token_ids


def checked_producer_metrics(log, enabled, result, parse):
    counters = parse(log, enabled)
    requests = [result["warmup"], *result["measured"]]
    verify_batches = MTP["EXPERT_LAYERS"] * sum(request["draft_proposed"] for request in requests)
    # A two-position batch carries one F32 activation, ten F32 weights and ten I32 IDs per position.
    bytes_per_batch = 2 * 4 * (2560 + 10 + 10)
    require(counters["verify_shared"] == verify_batches
            and counters["verify_bytes"] == verify_batches * bytes_per_batch,
            "producer verification totals do not match the warmup and measured fixed-width-2 requests")
    if enabled:
        require(counters["fallbacks"] == 0 and counters["batches"] == verify_batches
                and counters["bytes"] == counters["verify_bytes"],
                "producer copy did not handle every verification batch with no fallbacks")
    return counters


def paired_comparison(off, on):
    def compare_request(left, right):
        return dict(token_ids_match=left["token_ids"] == right["token_ids"], text_matches=left["answer"] == right["answer"],
                    generated_tokens_match=left["generated_tokens"] == right["generated_tokens"],
                    draft_counts_match=(left["draft_proposed"], left["draft_accepted"]) ==
                    (right["draft_proposed"], right["draft_accepted"]))

    warmup = compare_request(off["warmup"], on["warmup"])
    reps = [compare_request(left, right) for left, right in zip(off["measured"], on["measured"])]
    shared_scopes = ("verify_shared", "verify_bytes", "other_shared", "other_bytes")
    shared_input_match = all(off["producer_including_warmup"][key] == on["producer_including_warmup"][key]
                             for key in shared_scopes)
    observed_match = all(all(row.values()) for row in (warmup, *reps)) and shared_input_match
    return dict(warmup=warmup, measured=reps, cumulative_shared_input_match=shared_input_match,
                observed_workload_matches=observed_match,
                measured_decode_tok_s_on_minus_off=[
                    right["decode_tok_s"] - left["decode_tok_s"] for left, right in zip(off["measured"], on["measured"])],
                pooled_decode_tok_s_on_minus_off=on["pooled_decode_tok_s"] - off["pooled_decode_tok_s"],
                pooled_request_wall_tok_s_on_minus_off=on["pooled_request_wall_tok_s"] - off["pooled_request_wall_tok_s"],
                interpretation=("The checked output, draft and shared-input counts match."
                                if observed_match else
                                "Output, draft counts or shared-input scope differs. Treat rates as separate observed workloads."))


def require_binary_strings(library, strings):
    with library.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        missing = [item for item in strings if data.find(item.encode("ascii")) < 0]
    require(not missing, f"{library} lacks experiment support or runtime markers: {', '.join(missing)}")


def hc_activation(log, enabled):
    hits = {flag: log.count(marker) for flag, marker in HC_PROJECTIONS.items()}
    require(all(count == int(enabled) for count in hits.values()),
            f"HC launch markers do not match the selected arm: {hits}")
    require(not any(marker in log for marker in ("CUDA HC mix fusion applied", "CUDA HC scale SiLU fusion applied",
                                               "CUDA HC up pair applied")), "an older HC path was active")
    return dict(first_launch_markers=hits,
                scope="One-time markers confirm each selected kernel launched; they do not count graph replays.")


def empty_reset_activation(log, enabled):
    values = re.findall(r"ggml_vulkan: device=Vulkan1 empty_pool_reset_skip=(\d) fence_reuse=automatic", log)
    require(values and all(int(value) == int(enabled) for value in values),
            "missing or wrong Vulkan1 empty-reset configuration")
    require(re.search(r"ggml_vulkan: Vulkan1 final graph fence reused\b", log), "Vulkan1 fence reuse did not execute")
    require("decode_pool_reset:" not in log, "pool reset telemetry was present in the speed run")
    return dict(configured=enabled, fence_reuse_observed=True, empty_skips=None,
                scope="Configuration and fence reuse verified. Empty pool skips are not counted in the speed run.")


def routed_dispatch_metrics(log, result, required_fusions=()):
    schedule_keys = ("rounds", "vulkan1_aggregate_add", "other_backend_aggregate_add", "other_aggregate_op")
    vulkan_keys = ("add_dispatches", "multi_add_dispatches", "other_fused_groups", "recorded_adds", "aggregates",
                   "aggregate_mask", "full_hot", "scratch_copies", "scratch_bytes", "scratch_mask", "scratch_unknown_layer")

    def totals(prefix, keys, optional_keys=()):
        rows = re.findall(re.escape(prefix) + r" ([^\r\n]+)", log)
        require(rows, f"missing {prefix} counters; check the diagnostic build")
        result = dict.fromkeys((*keys, *optional_keys), 0)
        for row in rows:
            fields = dict(re.findall(r"(\w+)=([0-9a-f]+)(?= |$)", row))
            for key in (*keys, *optional_keys):
                if key in optional_keys and key not in fields:
                    continue
                if key.endswith("_mask"):
                    result[key] |= int(fields[key], 16)
                else:
                    result[key] += int(fields[key])
        return result

    schedule = totals("mtp routed schedule:", schedule_keys)
    fusions = ("reduce_dispatches", "weighted_dispatches")
    vk = totals("mtp routed Vulkan: device=Vulkan1", (*vulkan_keys, *required_fusions),
                tuple(key for key in fusions if key not in required_fusions))
    expected_rounds = sum(request["draft_proposed"] for request in (result["warmup"], *result["measured"]))
    expected_layers = MTP["EXPERT_LAYERS"] * expected_rounds
    classified = sum(schedule[key] for key in schedule_keys[1:])
    exact_fusions = sum(vk[key] for key in fusions)
    groups = exact_fusions + sum(vk[key] for key in ("add_dispatches", "multi_add_dispatches", "other_fused_groups"))
    require(expected_rounds > 0 and schedule["rounds"] == expected_rounds and classified <= expected_layers
            and 0 <= vk["aggregates"] <= schedule["vulkan1_aggregate_add"]
            and groups <= vk["recorded_adds"] <= 9 * expected_layers
            and exact_fusions <= vk["aggregates"] <= vk["recorded_adds"]
            and groups + 8 * exact_fusions <= vk["recorded_adds"],
            "scheduled, recorded, or per-request routed reduction counts disagree")
    require(0 <= vk["scratch_unknown_layer"] <= vk["scratch_copies"] <= vk["full_hot"] <= expected_layers
            and vk["scratch_bytes"] == vk["scratch_copies"] * (4 * 2560 * 10 * 2)
            and (vk["scratch_mask"] | vk["aggregate_mask"]) < (1 << MTP["EXPERT_LAYERS"]),
            "full-hot down-copy counts, dimensions or layer mask disagree")
    for label in ("aggregate", "scratch"):
        vk[f"{label}_layer_ids"] = [layer for layer in range(MTP["EXPERT_LAYERS"])
                                    if vk[f"{label}_mask"] & (1 << layer)]
        vk[f"distinct_{label}_layers"] = len(vk[f"{label}_layer_ids"])
    return dict(schedule=schedule, vulkan1=vk, expected_layer_invocations=expected_layers,
                recorded_dispatch_groups=groups,
                unclassified_layer_invocations=expected_layers - classified,
                unobserved_vulkan1_aggregate_adds=schedule["vulkan1_aggregate_add"] - vk["aggregates"],
                scratch_copy_fraction_of_all_layers=vk["scratch_copies"] / expected_layers,
                scratch_copy_fraction_of_full_hot=vk["scratch_copies"] / vk["full_hot"] if vk["full_hot"] else None,
                scope="Fixed two-position target MTP verification across warmup and both measured requests. "
                      "ADD, MULTI_ADD, MTP_ROUTED_REDUCE and MTP_ROUTED_WEIGHTED dispatches are counted at MI50 command recording. Other fused groups are counted separately. "
                      "Scratch bytes are same-device copy payload. "
                      "Other backend/op counts are logical scheduled layer endpoints and do not imply dispatch counts.")


def gate_up_metrics(log, routed, require_pair=False):
    keys = ("dispatches", "ordinary", "full_hot", "iq2_s", "iq3_xxs", "iq4_xs")
    rows = re.findall(r"mtp gate-up Vulkan: device=Vulkan1 ([^\r\n]+)", log)
    require(rows, "missing Vulkan1 gate/up counters")
    pair_keys = ("pair_dispatches", "pair_slots")
    counts = dict.fromkeys((*keys, *pair_keys), 0)
    for row in rows:
        fields = dict(re.findall(r"(\w+)=(\d+)", row))
        for key in keys:
            counts[key] += int(fields[key])
        for key in pair_keys:
            if require_pair or key in fields:
                counts[key] += int(fields[key])
    layers = counts["ordinary"] + counts["full_hot"]
    pair_dispatches = counts["pair_dispatches"]
    equivalent_single_dispatches = counts["dispatches"] + pair_dispatches
    require(layers == sum(counts[key] for key in keys[3:])
            and 2 * layers <= equivalent_single_dispatches <= 2 * layers + 2 * counts["full_hot"]
            and pair_dispatches <= counts["dispatches"] and 2 * pair_dispatches <= counts["pair_slots"] <= 20 * pair_dispatches
            and counts["full_hot"] <= routed["vulkan1"]["full_hot"] and layers <= routed["expected_layer_invocations"],
            "gate/up dispatch and layer counts disagree")
    return dict(counts, layers=layers, other_layers=routed["expected_layer_invocations"] - layers,
                equivalent_single_dispatches=equivalent_single_dispatches,
                pair_scope="Pair slots are submitted lanes, including unmatched lanes that compute singly.")


def gate_up_timing_metrics(log):
    require("mtp gate-up host IDs: callback=available" in log, "missing native gate/up host-ID callback")
    groups = {}
    fields = ("dispatches", "timed", "dropped", "ns", "phases", "proven", "matched", "shared")
    expected = {(quant, phase) for quant in ("IQ2_S", "IQ3_XXS")
                for phase in ("ordinary", "resident", "nonresident")}
    for quant, phase, payload in re.findall(r"mtp gate-up timing: quant=(\w+) phase=(\w+) ([^\r\n]+)", log):
        key = quant, phase
        require(key in expected, f"unexpected gate/up timing group {key}")
        group = groups.setdefault(key, dict.fromkeys(fields, 0))
        values = dict(re.findall(r"(\w+)=(\d+)", payload))
        for field in fields:
            group[field] += int(values[field])
    require(groups.keys() == expected, "missing Vulkan1 gate/up timestamp evidence")
    for key, group in groups.items():
        require(group["timed"] + group["dropped"] == group["dispatches"]
                and group["phases"] <= group["dispatches"] <= 2 * group["phases"]
                and 0 <= group["shared"] <= group["matched"] <= 10 * group["proven"]
                and group["proven"] <= group["phases"], f"inconsistent gate/up timing counters for {key}")
        group["gpu_ms"] = group["ns"] / 1e6
        group["mean_gpu_ns_per_dispatch"] = group["ns"] / group["timed"] if group["timed"] else None
        group["mean_gpu_ns_per_phase"] = group["ns"] / group["phases"] if group["phases"] and not group["dropped"] else None
        group["timing_coverage"] = group["timed"] / group["dispatches"] if group["dispatches"] else None
        group["host_id_coverage"] = group["proven"] / group["phases"] if group["phases"] else None
    require(sum(group["timed"] for group in groups.values()) > 0, "no MI50 gate/up timestamps completed")
    return {f"{quant}/{phase}": group for (quant, phase), group in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, default=Path("G:/AI/build-prefill32"))
    parser.add_argument("--rotation", type=int, choices=(0, 1, 2), default=0, help="rotate the starting arm")
    parser.add_argument("--reverse", action="store_true", help="reverse the selected order")
    pair_options = parser.add_mutually_exclusive_group()
    pair_options.add_argument("--producer-copy", action="store_true", help="compare producer copy OFF/ON at fixed width 2")
    pair_options.add_argument("--hc-all", action="store_true", help="compare HC down/up/inject all OFF versus all ON at fixed width 2")
    pair_options.add_argument("--empty-reset", action="store_true", help="compare Vulkan empty pool reset skipping OFF/ON at fixed width 2 with HC all ON")
    pair_options.add_argument("--routed-dispatch", action="store_true", help="count MI50 routed reductions and full-hot scratch copies at fixed width 2")
    pair_options.add_argument("--routed-reduce", action="store_true", help="compare exact Vulkan1 ten-expert reduction OFF/ON at fixed width 2 with HC all ON")
    pair_options.add_argument("--routed-weighted", action="store_true", help="compare fused Vulkan1 router weighting OFF/ON with exact reduction ON in both arms")
    pair_options.add_argument("--routed-gate-up", action="store_true", help="compare Vulkan1 gate/up/SwiGLU OFF/ON with accepted weighting and reduction")
    pair_options.add_argument("--routed-gate-up-pair", action="store_true", help="compare shared expert weight loads/decode with gate/up enabled in both arms")
    pair_options.add_argument("--fast-three", action="store_true", help="compare adopted fixed width 2 against fast fixed width 3")
    pair_options.add_argument("--routed-gate-up-pair-timing", action="store_true", help="diagnose MI50 gate/up pair dispatch GPU time and shared experts")
    parser.add_argument("--on-first", action="store_true", help="run the experiment's ON arm before OFF")
    args = parser.parse_args()
    pair_timing = args.routed_gate_up_pair_timing
    args.routed_gate_up_pair |= pair_timing
    if os.name != "nt":
        parser.error("this benchmark requires Windows")
    gate_up = args.routed_gate_up or args.routed_gate_up_pair
    routed_pair = args.routed_reduce or args.routed_weighted or gate_up
    paired = args.producer_copy or args.hc_all or args.empty_reset or routed_pair
    routed_stats = args.routed_dispatch or routed_pair
    accepted_hc = args.empty_reset or routed_stats or args.fast_three
    if paired and (args.reverse or args.rotation):
        parser.error("use --on-first to reverse the OFF/ON comparison")
    if args.on_first and not (paired or args.fast_three):
        parser.error("--on-first requires a two-arm comparison")
    if args.fast_three and (args.rotation or args.reverse):
        parser.error("use --on-first to run fast width 3 first")
    if args.routed_dispatch and (args.reverse or args.rotation):
        parser.error("--routed-dispatch runs one fixed-width-2 diagnostic arm")
    prefix = "shared-input-producer-warm" if args.producer_copy else "hc-all-warm" if args.hc_all else "mtp-warm"
    if args.empty_reset:
        prefix = "vulkan-empty-reset-warm"
    if args.routed_dispatch:
        prefix = "mtp-routed-dispatch"
    if args.routed_reduce:
        prefix = "mtp-routed-reduce-warm"
    if args.routed_weighted:
        prefix = "mtp-routed-weighted-warm"
    if gate_up:
        prefix = "mtp-routed-gate-up-pair-warm" if args.routed_gate_up_pair else "mtp-routed-gate-up-warm"
    if args.fast_three:
        prefix = "mtp-fast-three-warm"
    if pair_timing:
        prefix = "mtp-routed-gate-up-pair-timing"
    directory = COMMON["LOG_ROOT"] / f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    try:
        library = args.build_directory.resolve() / "bin/llama-server-impl.dll"
        require(library.is_file(), f"missing {library}")
        if args.hc_all or accepted_hc:
            require_binary_strings(library.with_name("ggml-cuda.dll"), (*HC_PROJECTIONS, *HC_PROJECTIONS.values()))
        if args.empty_reset:
            require_binary_strings(library.with_name("ggml-vulkan.dll"),
                                   (EMPTY_RESET, "empty_pool_reset_skip=", "final graph fence reused"))
        if routed_stats:
            require_binary_strings(library.with_name("ggml-vulkan.dll"), ("mtp routed Vulkan:",))
            require_binary_strings(library.with_name("ggml-base.dll"), ("mtp routed schedule:",))
        if routed_pair:
            require_binary_strings(library.with_name("ggml-vulkan.dll"), (ROUTED_REDUCE, "reduce_dispatches="))
        if args.routed_weighted or gate_up:
            require_binary_strings(library.with_name("ggml-vulkan.dll"), (ROUTED_WEIGHTED, "weighted_dispatches="))
        if gate_up:
            require_binary_strings(library.with_name("ggml-vulkan.dll"), (GATE_UP, "mtp gate-up Vulkan:"))
        if args.routed_gate_up_pair:
            require_binary_strings(library.with_name("ggml-vulkan.dll"), (GATE_UP_PAIR, "pair_dispatches="))
        if args.fast_three:
            require_binary_strings(library.with_name("ggml-vulkan.dll"), ("mtp fast-three Vulkan:", GATE_UP, ROUTED_WEIGHTED))
        if pair_timing:
            require_binary_strings(library.with_name("ggml-vulkan.dll"), ("mtp gate-up timing:",))
            require_binary_strings(library.with_name("llama.dll"), ("mtp gate-up host IDs:",))
        baseline = COMMON["build_environment"]()
        baseline.update({"LLAMA_EXPERT_PREFILL_512": "1", "LLAMA_EXPERT_PREFILL_512_POLICY": "auto",
                         "LLAMA_EXPERT_PREFILL_LOGICAL_TICKS": "1", "LLAMA_EXPERT_PREFILL_OVERLAP_TEN": "1",
                         "LLAMA_MTP_DRAFT_VOCAB": None, "LLAMA_MTP_DRAFT_VOCAB_STATS": "1",
                         "LLAMA_MTP_REUSE_DRAFT_ANCHOR": None, "LLAMA_MTP_LOGIT_TRACE": None,
                         "LLAMA_MTP_LOGIT_REFERENCE": None, "LLAMA_MTP_LOGIT_MATCH_DRAFTS": None,
                         "GGML_SCHED_SHARED_INPUT_PRODUCER_COPY": "0", "GGML_SCHED_TRACE_EXPERT_HANDOFF": None,
                         EMPTY_RESET: "0", "GGML_VK_GPU_TIMESTAMPS_DEVICE": None, ROUTED_STATS: None,
                         ROUTED_REDUCE: "1" if args.routed_weighted or gate_up else "0",
                         ROUTED_WEIGHTED: "1" if gate_up else "0", GATE_UP: "1" if args.routed_gate_up_pair else "0",
                         GATE_UP_PAIR: "0", "LLAMA_MTP_VERIFY_THREE_FAST": "0",
                         "GGML_VK_MTP_GATE_UP_TIMING": "1" if pair_timing else "0"})
        if gate_up or args.fast_three:
            baseline["LLAMA_MTP_PAIR_EXPERT_OVERLAP"] = None
        if args.fast_three:
            baseline.update({ROUTED_REDUCE: "1", ROUTED_WEIGHTED: "1", GATE_UP: "1"})
        if args.hc_all or accepted_hc:
            baseline.update(dict.fromkeys(OLDER_HC))
        if accepted_hc:
            baseline.update(dict.fromkeys(HC_PROJECTIONS, "1"))
            baseline.update(dict.fromkeys(("LLAMA_DECODE_TELEMETRY", "LLAMA_DECODE_TELEMETRY_PER_RESPONSE",
                                          "GGML_VK_PERF_LOGGER", "GGML_VK_PERF_LOGGER_CONCURRENT",
                                          "GGML_CUDA_DISABLE_GRAPHS")))
        if routed_stats:
            baseline[ROUTED_STATS] = "1"
        hashes = COMMON["binary_hashes"](library)
        directory.mkdir()
        _, message = COMMON["prepare_inputs"](directory)
        order = arm_order(args.rotation, args.reverse)
        if paired:
            order = ("on", "off") if args.on_first else ("off", "on")
            producer = runpy.run_path(str(HERE / "compare-shared-input-producer.py"))
        elif args.routed_dispatch:
            order = ("width-2",)
        elif args.fast_three:
            order = ("width-3", "width-2") if args.on_first else ("width-2", "width-3")
        metadata = dict(order=order, build_directory=str(args.build_directory.resolve()), build_sha256=hashes,
                        source_run=str(COMMON["SOURCE_RUN"]), user_message_sha256=COMMON["sha256"](message),
                        warmup_tokens=WARMUP, measured_tokens=GENERATED, measured_requests_per_arm=2,
                        startup_scope="One process per arm. A generated warmup precedes the two measured requests. "
                                      "The server resets the adaptive policy at each request; hardware stays warm.",
                        decode_scope="Server generation wall time. The token sampled from prefill contributes to total output "
                                     "but is excluded from the decode throughput numerator. HTTP wall includes prefill.")
        if paired:
            metadata.update(verify_width=2, draft_max=1,
                            startup_scope="One process per arm. A generated warmup precedes the two measured requests.",
                            producer_counter_scope="Process-total counters include warmup and both measured requests.")
        if args.hc_all:
            metadata.update(hc_flags=list(HC_PROJECTIONS), producer_copy=False,
                            hc_marker_scope="One first-launch marker per enabled component, across the process including warmup.")
        if args.empty_reset:
            metadata.update(hc_all=True, producer_copy=False, graph_fence_reuse="automatic",
                            empty_reset_scope="Configured flag and actual Vulkan1 fence reuse required in both arms. "
                                              "The uninstrumented speed run does not count skipped resets.")
        if args.routed_dispatch:
            metadata.update(verify_width=2, draft_max=1, hc_all=True, producer_copy=False, empty_reset=False,
                            startup_scope="One fixed-width-2 process with one warmup and two measured requests.",
                            decode_scope="Diagnostic timings include CPU operations needed to count recorded dispatches.")
        if routed_pair:
            metadata.update(hc_all=True, producer_copy=False, empty_reset=False,
                            routed_reduce_scope="Same CPU command-recording counters in both arms, including warmup. CUDA Graphs stay enabled.")
        if args.routed_weighted:
            metadata.update(routed_reduce=True, toggled_flag=ROUTED_WEIGHTED)
        if gate_up:
            metadata.update(routed_reduce=True, routed_weighted=True,
                            toggled_flag=GATE_UP_PAIR if args.routed_gate_up_pair else GATE_UP)
        if args.fast_three:
            metadata.update(width_3_fast=True, hc_all=True, router_weighted=True, routed_reduce=True, gate_up=True,
                            output_scope="Reports exact token-ID agreement and per-arm MTP acceptance. Depth may change generated IDs.")
        if pair_timing:
            metadata["gpu_timing_scope"] = (
                "Only MI50 gate/up+SwiGLU dispatch timestamps, including warmup. Resident/nonresident are "
                "the two overlapped phases; resident includes scratch slots already present. Ordinary dispatches "
                "can mix residency. Matched counts use original Router IDs, slot map, and active masks; shared counts "
                "require that the pair shader actually recorded. Dropped samples and unmatched host rows remain visible. "
                "Wall times include diagnostic overhead.")
        write_json(directory / "run.json", metadata)
        print(f"Log directory: {directory}", flush=True)
        results, token_ids = {}, None
        for label in order:
            case = directory / label
            if paired:
                enabled = label == "on"
                toggles = (HC_PROJECTIONS if args.hc_all else (EMPTY_RESET,) if args.empty_reset
                           else (ROUTED_REDUCE,) if args.routed_reduce else (ROUTED_WEIGHTED,) if args.routed_weighted
                           else (GATE_UP,) if args.routed_gate_up else (GATE_UP_PAIR,) if args.routed_gate_up_pair
                           else (producer["FLAG"],))
                environment = dict(baseline, **dict.fromkeys(toggles, "1" if enabled else "0"))
                result, token_ids = run_arm(case, library, "width-2", environment, message, token_ids)
                log = (case / "server.log").read_text(encoding="utf-8", errors="replace")
                result["producer_including_warmup"] = checked_producer_metrics(
                    log, args.producer_copy and enabled, result, producer["producer_metrics"])
                if args.hc_all or accepted_hc:
                    result["hc_including_warmup"] = hc_activation(log, accepted_hc or enabled)
                if args.empty_reset:
                    result["empty_reset"] = empty_reset_activation(log, enabled)
                if routed_pair:
                    required = ("reduce_dispatches", "weighted_dispatches") if args.routed_weighted or gate_up else ("reduce_dispatches",)
                    result["routed_dispatch"] = routed_dispatch_metrics(log, result, required)
                if gate_up:
                    result["gate_up"] = gate_up_metrics(log, result["routed_dispatch"], require_pair=args.routed_gate_up_pair)
                if pair_timing:
                    result["gate_up_timing"] = gate_up_timing_metrics(log)
                write_json(case / "metrics.json", result)
            else:
                environment = dict(baseline, LLAMA_MTP_VERIFY_THREE_FAST="1") if args.fast_three and label == "width-3" else baseline
                result, token_ids = run_arm(case, library, label, environment, message, token_ids)
                if args.fast_three:
                    log = (case / "server.log").read_text(encoding="utf-8", errors="replace")
                    result["hc_including_warmup"] = hc_activation(log, True)
                    write_json(case / "metrics.json", result)
                if args.routed_dispatch:
                    log = (case / "server.log").read_text(encoding="utf-8", errors="replace")
                    result["hc_including_warmup"] = hc_activation(log, True)
                    result["routed_dispatch"] = routed_dispatch_metrics(log, result)
                    write_json(case / "metrics.json", result)
            results[label] = result
        require(hashes == COMMON["binary_hashes"](library), "binaries changed during comparison")
        if paired:
            comparisons = {"on_vs_off": paired_comparison(results["off"], results["on"])}
            if routed_pair:
                off, on = (results[mode]["routed_dispatch"] for mode in ("off", "on"))
                comparisons["on_vs_off"].update(
                    routed_reduce_off=off["vulkan1"]["reduce_dispatches"],
                    routed_reduce_on=on["vulkan1"]["reduce_dispatches"],
                    routed_weighted_off=off["vulkan1"]["weighted_dispatches"],
                    routed_weighted_on=on["vulkan1"]["weighted_dispatches"],
                    routed_aggregates_off=off["vulkan1"]["aggregates"],
                    routed_aggregates_on=on["vulkan1"]["aggregates"],
                    routed_dispatch_groups_off=off["recorded_dispatch_groups"],
                    routed_dispatch_groups_on=on["recorded_dispatch_groups"],
                )
            if gate_up:
                comparisons["on_vs_off"]["gate_up"] = {mode: results[mode]["gate_up"] for mode in ("off", "on")}
            if pair_timing:
                off, on = (results[mode]["gate_up_timing"] for mode in ("off", "on"))
                timing_comparison = {}
                for group, left in off.items():
                    right = on[group]
                    comparable = (0 < left["phases"] == right["phases"]
                                  and left["proven"] == right["proven"] == left["phases"]
                                  and left["matched"] == right["matched"]
                                  and not (left["dropped"] or right["dropped"]))
                    delta = right["mean_gpu_ns_per_phase"] - left["mean_gpu_ns_per_phase"] if comparable else None
                    timing_comparison[group] = dict(off=left, on=right, comparable=comparable, on_minus_off_ns_per_phase=delta)
                comparisons["on_vs_off"]["gate_up_timing"] = timing_comparison
        else:
            comparisons = {
                mode: {
                    "token_ids_match_width2": [a["token_ids"] == b["token_ids"] for a, b in zip(results["width-2"]["measured"], result["measured"])],
                    "text_matches_width2": [a["answer"] == b["answer"] for a, b in zip(results["width-2"]["measured"], result["measured"])],
                    "pooled_decode_tok_s_delta_vs_width2": result["pooled_decode_tok_s"] - results["width-2"]["pooled_decode_tok_s"],
                } for mode, result in results.items() if mode != "width-2"
            }
        prompt_path = directory / "prompt-token-ids.json"
        write_json(directory / "comparison.json", {"run": metadata, "runs": results, "comparison": comparisons,
                    "prompt_tokens": len(token_ids), "historical_cli_prompt_tokens": COMMON["PROMPT_TOKENS"],
                    "prompt_token_ids_sha256": COMMON["sha256"](prompt_path.read_bytes())})
        for mode, result in results.items():
            print(f"{mode}: pooled {result['pooled_decode_tok_s']:.2f} decode tok/s")
        print(json.dumps(results["width-2"]["routed_dispatch"] if args.routed_dispatch else comparisons, indent=2))
        if pair_timing:
            off, on = (results[mode]["gate_up_timing"] for mode in ("off", "on"))
            require(sum(group["shared"] for group in off.values()) == 0
                    and sum(group["shared"] for group in on.values()) > 0,
                    "no matching Expert pairs were proven in recorded ON dispatches")
        if routed_pair:
            outcome = comparisons["on_vs_off"]
            if gate_up:
                off, on = outcome["gate_up"]["off"], outcome["gate_up"]["on"]
                if args.routed_gate_up_pair:
                    require(off["dispatches"] > 0 and off["pair_dispatches"] == 0 and on["pair_dispatches"] > 0,
                            "gate/up pair did not activate exclusively in ON")
                    keys = ("ordinary", "full_hot", "iq2_s", "iq3_xxs", "iq4_xs", "equivalent_single_dispatches")
                    require(all(off[key] == on[key] for key in keys), "gate/up pair changed projection coverage")
                else:
                    require(off["dispatches"] == 0 and on["dispatches"] > 0,
                            "gate/up fusion did not activate exclusively in ON")
                require(outcome["routed_weighted_off"] > 0
                        and outcome["routed_weighted_off"] == outcome["routed_weighted_on"]
                        and outcome["routed_reduce_off"] == outcome["routed_reduce_on"]
                        and outcome["routed_aggregates_off"] == outcome["routed_aggregates_on"],
                        "accepted routed weighting/reduction counters differ between arms")
            elif args.routed_weighted:
                require(outcome["routed_reduce_off"] > 0 and outcome["routed_weighted_off"] == 0
                        and outcome["routed_weighted_on"] > 0,
                        "expected ordered reduction in OFF and fused router weighting in ON")
                require(outcome["routed_aggregates_off"] == outcome["routed_aggregates_on"]
                        and outcome["routed_reduce_off"] == outcome["routed_reduce_on"] + outcome["routed_weighted_on"],
                        "fused router weighting did not replace the same ordered reductions")
            else:
                require(outcome["routed_reduce_off"] == 0 and outcome["routed_reduce_on"] > 0,
                        "Vulkan1 reduction did not activate exclusively in the ON arm")
            require(outcome["observed_workload_matches"], "Vulkan1 reduction comparison changed tokens or observed workload")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"error: {error}\nLog directory: {directory}\n")


if __name__ == "__main__":
    main()
