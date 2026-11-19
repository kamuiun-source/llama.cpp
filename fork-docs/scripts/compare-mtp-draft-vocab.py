"""Paired Japanese/code draft-vocabulary runs: fixed MTP2, two identical in-process warmups, greedy target."""

import argparse
import csv
import hashlib
import json
import os
import re
import runpy
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
SHORT = runpy.run_path(str(HERE / "compare-short-prefill.py"))
COMMON = SHORT["COMMON"]
PREP = runpy.run_path(str(HERE / "prepare-mtp-draft-vocab.py"))
require = COMMON["require"]
write_json = COMMON["write_json"]
PROMPTS = {
    "japanese": "\u65e5\u672c\u8a9e\u3060\u3051\u3067\u8aac\u660e\u3057\u3066\u304f\u3060\u3055\u3044\u3002\u6b21\u306e\u5bb6\u5ead\u5185\u30cd\u30c3\u30c8\u30ef\u30fc\u30af\u3092\u8abf\u67fb\u3057\u307e\u3059\u3002\u5bb6\u65cf\u306f\u56db\u4eba\u3067\u3001\u591c\u306f\u30c6\u30ec\u30d3\u306e\u52d5\u753b\u914d\u4fe1\u3001\u4e8c\u53f0\u306e\u30b9\u30de\u30fc\u30c8\u30d5\u30a9\u30f3\u306e\u901a\u8a71\u3001\u30ce\u30fc\u30c8\u30d1\u30bd\u30b3\u30f3\u306e\u30aa\u30f3\u30e9\u30a4\u30f3\u6388\u696d\u304c\u540c\u6642\u306b\u52d5\u304d\u307e\u3059\u3002\u30eb\u30fc\u30bf\u30fc\u306f\u7384\u95a2\u306e\u68da\u306b\u7f6e\u304b\u308c\u3001\u96a3\u5ba4\u3068\u306e\u9593\u306b\u58c1\u304c\u4e8c\u679a\u3042\u308a\u307e\u3059\u3002\u663c\u306f\u5feb\u9069\u3067\u3059\u304c\u3001\u591c\u306f\u52d5\u753b\u304c\u6b62\u307e\u308a\u3001\u901a\u8a71\u306e\u97f3\u58f0\u304c\u9014\u5207\u308c\u307e\u3059\u3002\u6709\u7dda\u63a5\u7d9a\u306e\u30d1\u30bd\u30b3\u30f3\u3067\u306f\u540c\u3058\u6642\u9593\u5e2f\u3067\u3082\u5b89\u5b9a\u3057\u3066\u3044\u307e\u3059\u3002\u539f\u56e0\u3092\u4e09\u3064\u306b\u5206\u3051\u3001\u305d\u308c\u305e\u308c\u89b3\u5bdf\u3067\u304d\u308b\u75c7\u72b6\u3068\u5b89\u5168\u306a\u78ba\u8a8d\u624b\u9806\u3092\u5177\u4f53\u7684\u306b\u8aac\u660e\u3057\u3066\u304f\u3060\u3055\u3044\u3002\u8a2d\u5b9a\u5909\u66f4\u306e\u524d\u306b\u6bd4\u8f03\u3059\u308b\u6761\u4ef6\u3082\u793a\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
    "code": "Write a complete Python CSV aggregation function and deterministic tests. Input rows contain customer, amount and currency. Group valid USD amounts by customer using Decimal, preserve row numbers of malformed records, reject negative or nonfinite values, and report other currencies separately. Return data instead of printing. Explain duplicate customer handling and missing columns.\nExample CSV:\ncustomer,amount,currency\nalice,10.50,USD\nbob,20.00,USD\nalice,3.25,USD\ncarol,broken,USD\nbob,-2.00,USD\ndave,NaN,USD\nalice,8.00,EUR\n,1.00,USD\ncarol,0.00,USD\nInclude tests for this exact example and a header-only file. Keep the implementation runnable using only the Python standard library."
}


@contextmanager
def server(case, library, arguments, environment, port, api_key):
    config = case / "server-arguments.json"
    write_json(config, {"library": str(library), "arguments": arguments})
    with (case / "stdout.log").open("wb") as stdout, (case / "stderr.log").open("wb") as stderr:
        with subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--server-worker", str(config)],
                              stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                              env=COMMON["child_environment"](environment), creationflags=subprocess.CREATE_NO_WINDOW) as process:
            try:
                SHORT["wait_ready"](process, port, api_key)
                yield process
            finally:
                if process.poll() is None:
                    process.stdin.write(b"stop\n")
                    process.stdin.flush()
                try:
                    code = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait()
                    raise RuntimeError("server failed graceful shutdown")
                (case / "exit.txt").write_text(f"exit_code={code}\n", encoding="utf-8")
                require(code == 0, f"server failed with exit{code}: {case}")


@contextmanager
def vram_sampling(case, memory, pid, stage, failure):
    stop = threading.Event()
    samples, errors = [], []
    def sample():
        try:
            with memory.process_handle(pid) as handle:
                while not stop.is_set():
                    item = memory.sample(handle)
                    samples.append((stage[0], time.monotonic(), item["rtx"], item["mi50"]))
                    stop.wait(0.2)
        except Exception as error:
            errors.append(str(error))
    worker = threading.Thread(target=sample, daemon=True) if memory is not None else None
    if worker:
        worker.start()
    try:
        yield samples, errors
    finally:
        stop.set()
        if worker:
            worker.join(timeout=5)
            if worker.is_alive():
                errors.append("sampling thread did not finish")
        with (case / "local-vram.csv").open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerows([("phase", "monotonic_seconds", "rtx_bytes", "mi50_bytes"), *samples])
        write_json(case / "vram-status.json", {"reason": failure, "errors": errors})


def read_result(case, response, enabled, count, samples):
    log = (case / "server.log").read_text(encoding="utf-8", errors="replace")
    for marker in ("using device CUDA0 (NVIDIA GeForce RTX 4070 Ti)", "using device Vulkan1 (AMD Radeon Pro VII)",
                   "packed expert batch capacity = 2, n_ubatch = 2, n_rs_seq = 1",
                   "20 scratch slots per layer", "Vulkan1 final graph fence reused"):
        require(marker in log, f"missing runtime proof: {marker}")
    require(not re.search(r"mtp_(?:three|adaptive)", log), "adaptive/three-width experiment appeared")
    native = COMMON["first_match"](r"packed expert prompt uploads: ([^\r\n]+)", log, "actual verification width")
    require(re.search(r"decode_max_tokens=2\b", native), "actual target verification width was not2")
    head = COMMON["first_match"](r"mtp_draft_vocab: ([^\r\n]+)", log, "draft head allocation")
    allocation = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)\b", head)}
    require(allocation["enabled"] == int(enabled) and allocation["full_head_retained"] == allocation["target_full_vocab"] == 1,
            "head activation differs")
    require((allocation["subset_rows"] > 0) == enabled and (allocation["buffer_bytes"] > 0) == enabled,
            "subset allocation differs")
    require(not enabled or re.search(r"backend=CUDA0\b", head), "subset head is not on draft CUDA0")
    rows = re.findall(r"mtp_vocab_request: request=(\d+) draft_wall_us=(\d+) drafted=(\d+) accepted=(\d+) generated=(\d+) single_slot=1", log)
    require(len(rows) == 3 and [int(row[4]) for row in rows[:2]] == [16, 16], "missing equal in-process warmups")
    request, draft_us, proposed, accepted, generated = map(int, rows[-1])
    timing = response["timings"]
    require(response["stop"] and not response["truncated"] and response["tokens_predicted"] == timing["predicted_n"] == generated == count
            and timing["cache_n"] == 0 and timing["predicted_ms"] > 0, "generation/prompt accounting differs")
    require(proposed == timing["draft_n"] > 0 and accepted == timing["draft_n_accepted"] <= proposed and draft_us > 0,
            "per-request MTP draft timing/count proof missing")
    tokens = response.get("tokens")
    require(isinstance(tokens, list) and len(tokens) == count, "missing generated token IDs")
    measured = [row for row in samples if row[0] == "measure"]
    return {"request": request, "prompt_tokens": timing["prompt_n"], "generated_tokens": generated,
            "decode_ms": timing["predicted_ms"], "decode_tps": timing["predicted_per_second"],
            "draft_wall_ms": draft_us / 1000, "draft_proposed": proposed, "draft_accepted": accepted,
            "acceptance_rate": accepted / proposed, "head": allocation, "answer": response["content"], "token_ids": tokens,
            "observed_peak_bytes": {role: max((row[index] for row in measured), default=None)
                                    for role, index in (("rtx", 2), ("mi50", 3))},
            "vram_scope": "sampled process local VRAM during measured request; unavailable if no samples"}


def run_case(case, library, baseline, vocab, prompt, count, memory, failure):
    case.mkdir()
    enabled = vocab is not None
    environment = dict(baseline, LLAMA_MTP_DRAFT_VOCAB=str(vocab) if enabled else None)
    port, key = SHORT["local_port"](), uuid4().hex
    arguments = SHORT["server_args"](case, port, key)
    for option, value in (("-n", count), ("-ub", 2), ("--spec-draft-n-max", 1), ("--spec-draft-n-ubatch", 1)):
        arguments[COMMON["argument_index"](arguments, option)] = str(value)
    for option, value in (("--spec-type", "draft-mtp"), ("-np", "1"), ("--spec-draft-sampling", "greedy"), ("--spec-draft-p-min", "0")):
        if option in arguments:
            arguments[COMMON["argument_index"](arguments, option)] = value
        else:
            arguments.extend((option, value))
    safe = arguments.copy()
    safe[COMMON["argument_index"](safe, "--api-key")] = "[ephemeral]"
    write_json(case / "settings.json", {"arguments": safe, "environment": environment, "automatic_warmup": False,
                                      "warmup_requests": 2, "warmup_generated_tokens": 16})
    responses = []
    with server(case, library, arguments, environment, port, key) as process:
        stage = ["warmup"]
        with vram_sampling(case, memory, process.pid, stage, failure) as (samples, errors):
            for index, n in enumerate((16, 16, count)):
                stage[0] = "measure" if index == 2 else "warmup"
                payload = {"prompt": prompt, "n_predict": n, "cache_prompt": False, "temperature": 0.0,
                           "seed": 1234, "ignore_eos": True, "stream": False, "return_tokens": True}
                write_json(case / f"request-{index}.json", payload)
                response = SHORT["http"](port, "/completion", key, payload, timeout=300)
                write_json(case / f"completion-{index}.json", response)
                responses.append(response)
    result = read_result(case, responses[-1], enabled, count, samples)
    result["warmup_tokens"] = [response.get("tokens") for response in responses[:2]]
    result["vram_errors"] = errors
    write_json(case / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, default=Path("G:/AI/build-prefill32"))
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--on-first", action="store_true")
    parser.add_argument("--server-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.server_worker:
        return SHORT["dll_server"](args.server_worker)
    if os.name != "nt" or args.vocab is None or not 1 <= args.tokens <= 512:
        parser.error("requires Windows, --vocab, and tokens1..512")
    directory = COMMON["LOG_ROOT"] / f"mtp-draft-vocab-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    try:
        library = args.build_directory.resolve() / "bin/llama-server-impl.dll"
        require(library.is_file(), f"missing {library}")
        manifest = COMMON["read_json"](args.vocab.with_suffix(args.vocab.suffix + ".json"))
        require(hashlib.sha256(args.vocab.read_bytes()).hexdigest() == manifest["list_sha256"], "vocabulary SHA mismatch")
        source_arguments = COMMON["read_json"](COMMON["SOURCE_RUN"] / "arguments.json")
        for option in ("-m", "-md"):
            _, _, digest, _, _ = PREP["read_tokenizer"](Path(source_arguments[COMMON["argument_index"](source_arguments, option)]),
                                                       library.parent / "ggml-base.dll")
            require(digest == manifest["tokenizer_sha256"], "vocabulary does not match target/draft global token IDs")
        baseline = COMMON["build_environment"]()
        baseline.update({"LLAMA_EXPERT_PREFILL_512": "1", "LLAMA_EXPERT_PREFILL_LOGICAL_TICKS": "1",
                         "LLAMA_EXPERT_PREFILL_512_POLICY": "auto", "LLAMA_EXPERT_PREFILL_OVERLAP_TEN": "1",
                         "GGML_VK_SKIP_EMPTY_POOL_RESET": "0", "GGML_SCHED_SHARED_INPUT_PRODUCER_COPY": "0",
                         "LLAMA_MTP_VERIFY_THREE": "0", "LLAMA_MTP_REUSE_DRAFT_ANCHOR": None,
                         "LLAMA_MTP_LOGIT_TRACE": None, "LLAMA_MTP_LOGIT_REFERENCE": None,
                         "LLAMA_MTP_LOGIT_MATCH_DRAFTS": None, "GGML_SCHED_TRACE_EXPERT_HANDOFF": None,
                         "GGML_VK_GPU_TIMESTAMPS_DEVICE": None, "LLAMA_MTP_DRAFT_VOCAB_STATS": "1",
                         "LLAMA_DECODE_TELEMETRY": None, "LLAMA_EXPERT_EARLY_DECODE": None})
        directory.mkdir()
        hashes = COMMON["binary_hashes"](library)
        write_json(directory / "run.json", {"build_sha256": hashes, "manifest": manifest, "prompts": PROMPTS,
                                           "target_temperature": 0, "verify_width": 2, "draft_max": 1,
                                           "adaptive": False, "on_first": args.on_first})
        print(f"Log directory: {directory}", flush=True)
        results, comparisons = {}, {}
        with COMMON["memory_monitor"](True) as (memory, failure):
            for language, prompt in PROMPTS.items():
                pair = {}
                for enabled in ([True, False] if args.on_first else [False, True]):
                    label = "subset" if enabled else "full"
                    pair[label] = run_case(directory / f"{language}-{label}", library, baseline,
                                           args.vocab.resolve() if enabled else None, prompt, args.tokens, memory, failure)
                    head = pair[label]["head"]
                    require(head["full_rows"] == manifest["vocab_size"] and head["subset_rows"] == (manifest["subset_size"] if enabled else 0),
                            "runtime head dimensions do not match the validated list")
                    if enabled:
                        require(head["subset_head_bytes"] * head["full_rows"] == head["full_head_bytes"] * head["subset_rows"]
                                and head["buffer_bytes"] >= head["subset_head_bytes"] + 8 * head["subset_rows"],
                                "head row storage accounting differs")
                results[language] = pair
                comparisons[language] = {
                    "target_tokens_equal": pair["full"]["token_ids"] == pair["subset"]["token_ids"],
                    "warmup_tokens_equal": pair["full"]["warmup_tokens"] == pair["subset"]["warmup_tokens"],
                    "prompt_tokens_equal": pair["full"]["prompt_tokens"] == pair["subset"]["prompt_tokens"],
                    "decode_ms_subset_minus_full": pair["subset"]["decode_ms"] - pair["full"]["decode_ms"],
                    "draft_wall_ms_subset_minus_full": pair["subset"]["draft_wall_ms"] - pair["full"]["draft_wall_ms"],
                    "acceptance_full": pair["full"]["acceptance_rate"], "acceptance_subset": pair["subset"]["acceptance_rate"],
                }
        require(hashes == COMMON["binary_hashes"](library), "binaries changed during run")
        write_json(directory / "comparison.json", {"results": results, "comparison": comparisons,
                    "memory_note": "Original full draft head remains allocated. Subset buffer is additional; use observed VRAM and reported buffer bytes.",
                    "draft_timing_scope": "Existing MTP draft() CPU wall timer per measured request; excludes process/catchup and two warmup requests.",
                    "warmup_scope": "Same two in-process requests/tokens per arm; candidate-dependent cache state may differ intrinsically."})
        print(json.dumps(comparisons, indent=2))
        require(all(row["target_tokens_equal"] and row["warmup_tokens_equal"] and row["prompt_tokens_equal"]
                    for row in comparisons.values()), "greedy target outputs or warmup differ; inspect comparison.json")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"error: {error}\nLog directory: {directory}\n")


if __name__ == "__main__":
    raise SystemExit(main())
