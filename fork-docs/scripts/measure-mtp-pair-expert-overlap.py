"""Measure selected-expert overlap across each fixed-width-2 target MTP pair."""

import argparse
import json
import os
import re
import runpy
from datetime import datetime
from pathlib import Path
from uuid import uuid4


HERE = Path(__file__).resolve().parent
WARM = runpy.run_path(str(HERE / "compare-mtp-warm.py"))
COMMON = WARM["COMMON"]
require = WARM["require"]
write_json = WARM["write_json"]
FLAG = "LLAMA_MTP_PAIR_EXPERT_OVERLAP"
GATE_FUSION = "GGML_VK_MTP_ROUTED_GATE_UP"
EXPERTS_PER_POSITION = 10
EXPERT_LAYERS = WARM["MTP"]["EXPERT_LAYERS"]
MARKER = "MTP pair expert overlap:"


def histogram_metrics(histogram):
    samples = sum(histogram)
    shared = sum(overlap * count for overlap, count in enumerate(histogram))
    selections = 2 * EXPERTS_PER_POSITION * samples
    return dict(samples=samples, histogram=histogram, common_selected_experts=shared,
                mean_common_selected_experts=shared / samples,
                mean_distinct_selected_experts=(selections - shared) / samples,
                potential_reuse_fraction_of_selections=shared / selections)


def overlap_metrics(log, expected_pairs):
    by_layer = {}
    require(expected_pairs > 0, "missing width-2 target MTP pairs")
    for row in re.findall(re.escape(MARKER) + r" ([^\r\n]+)", log):
        fields = re.fullmatch(r"layer=(\d+) samples=(\d+) bins=(\d+(?:,\d+){10})", row)
        require(fields is not None, f"malformed overlap histogram: {row}")
        layer, samples = map(int, fields.groups()[:2])
        histogram = [int(value) for value in fields[3].split(",")]
        require(layer not in by_layer, f"duplicate expert overlap histogram for layer {layer}")
        require(sum(histogram) == samples == expected_pairs,
                f"layer {layer} has {samples} reported samples, expected {expected_pairs}, bins sum to {sum(histogram)}")
        by_layer[layer] = histogram_metrics(histogram)
    require(set(by_layer) == set(range(EXPERT_LAYERS)), "missing or unexpected packed Expert layers in overlap histograms")
    total = [sum(counts) for counts in zip(*(layer["histogram"] for layer in by_layer.values()))]
    return dict(scope="Target MTP verification at fixed width 2. One sample is a pair in one routed layer. "
                      "Counts include the warmup and both generated requests.",
                potential_reuse_basis="One shared expert can replace one of the 20 original per-pair weight selections. "
                                      "The ratio uses expert counts. It describes neither cache hits nor transfer or memory traffic.",
                verification_pairs=expected_pairs, experts_per_position=EXPERTS_PER_POSITION,
                total=histogram_metrics(total), per_layer=dict(sorted(by_layer.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, default=Path("G:/AI/build-prefill32"))
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("this benchmark requires Windows")
    directory = COMMON["LOG_ROOT"] / f"mtp-pair-expert-overlap-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    try:
        library = args.build_directory.resolve() / "bin/llama-server-impl.dll"
        require(library.is_file(), f"missing {library}")
        check_binary = WARM["require_binary_strings"]
        check_binary(library.with_name("llama.dll"), (FLAG, MARKER))
        check_binary(library.with_name("ggml-cuda.dll"), (*WARM["HC_PROJECTIONS"], *WARM["HC_PROJECTIONS"].values()))
        check_binary(library.with_name("ggml-vulkan.dll"),
                     (WARM["ROUTED_REDUCE"], WARM["ROUTED_WEIGHTED"], "mtp routed Vulkan:", "weighted_dispatches="))
        check_binary(library.with_name("ggml-base.dll"), ("mtp routed schedule:",))
        environment = COMMON["build_environment"]()
        environment.update(dict.fromkeys((*WARM["OLDER_HC"], "LLAMA_MTP_DRAFT_VOCAB", "LLAMA_MTP_REUSE_DRAFT_ANCHOR",
                                         "LLAMA_MTP_LOGIT_TRACE", "LLAMA_MTP_LOGIT_REFERENCE", "LLAMA_MTP_LOGIT_MATCH_DRAFTS",
                                         "GGML_SCHED_TRACE_EXPERT_HANDOFF", "GGML_VK_GPU_TIMESTAMPS_DEVICE",
                                         "GGML_VK_PERF_LOGGER", "GGML_VK_PERF_LOGGER_CONCURRENT", "GGML_CUDA_DISABLE_GRAPHS")))
        environment.update(dict.fromkeys(WARM["HC_PROJECTIONS"], "1"))
        environment.update({"LLAMA_EXPERT_PREFILL_512": "1", "LLAMA_EXPERT_PREFILL_512_POLICY": "auto",
                            "LLAMA_EXPERT_PREFILL_LOGICAL_TICKS": "1", "LLAMA_EXPERT_PREFILL_OVERLAP_TEN": "1",
                            "LLAMA_MTP_DRAFT_VOCAB_STATS": "1", "LLAMA_EXPERT_RTX_COLD_SLOTS": "0",
                            "GGML_SCHED_SHARED_INPUT_PRODUCER_COPY": "0", WARM["EMPTY_RESET"]: "0",
                            WARM["ROUTED_STATS"]: "1", WARM["ROUTED_REDUCE"]: "1", WARM["ROUTED_WEIGHTED"]: "1",
                            GATE_FUSION: "0", FLAG: "1"})
        hashes = COMMON["binary_hashes"](library)
        directory.mkdir()
        _, message = COMMON["prepare_inputs"](directory)
        write_json(directory / "run.json",
                   dict(build_directory=str(args.build_directory.resolve()), build_sha256=hashes,
                        source_run=str(COMMON["SOURCE_RUN"]), user_message_sha256=COMMON["sha256"](message),
                        warmup_tokens=WARM["WARMUP"], generated_tokens=WARM["GENERATED"], generated_requests=2,
                        verify_width=2, draft_max=1))
        print(f"Log directory: {directory}", flush=True)
        case = directory / "width-2"
        result, _ = WARM["run_arm"](case, library, "width-2", environment, message, None)
        log = (case / "server.log").read_text(encoding="utf-8", errors="replace")
        result["hc_including_warmup"] = WARM["hc_activation"](log, True)
        result["empty_reset"] = WARM["empty_reset_activation"](log, False)
        producer = runpy.run_path(str(HERE / "compare-shared-input-producer.py"))
        result["producer_including_warmup"] = WARM["checked_producer_metrics"](log, False, result, producer["producer_metrics"])
        dispatch = WARM["routed_dispatch_metrics"](log, result, ("reduce_dispatches", "weighted_dispatches"))
        require(dispatch["vulkan1"]["weighted_dispatches"] > 0, "fused router weighting and ordered reduction did not dispatch")
        result["routed_dispatch"] = dispatch
        overlap = overlap_metrics(log, dispatch["schedule"]["rounds"])
        result["expert_overlap_including_warmup"] = overlap
        require(hashes == COMMON["binary_hashes"](library), "binaries changed during measurement")
        write_json(case / "metrics.json", result)
        print(json.dumps({key: value for key, value in overlap.items() if key != "per_layer"}, indent=2))
        print(f"Per-layer and total histograms: {case / 'metrics.json'}")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"error: {error}\nLog directory: {directory}\n")


if __name__ == "__main__":
    main()
