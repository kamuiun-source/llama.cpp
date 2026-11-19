"""Compare native packed Expert prefill widths 2 and 512 on the saved 1,378-token workload."""

import argparse
import csv
import hashlib
import json
import os
import re
import runpy
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from windows_gpu_memory import DedicatedMemory, ProcessTerminating


LOG_ROOT = Path("G:/AI/llama-hetero-moe-local/logs")
SOURCE_RUN = LOG_ROOT / "dialogue-prefill-trace-20261119-125254-d8976f7f"
INPUT_PATH = "fork-docs/research/dialogue-benchmark-prompts.md"
QUESTION = "\u3053\u306e\u8cc7\u6599\u3067\u60f3\u5b9a\u3057\u3066\u3044\u308bWi-Fi\u306e\u554f\u984c\u3092\u3001\u65e5\u672c\u8a9e\u3067\u4e00\u6587\u3060\u3051\u8981\u7d04\u3057\u3066\u304f\u3060\u3055\u3044\u3002"
PROMPT_TOKENS = 1378
OUTER_BATCH = 512
SAMPLE_SECONDS = 0.2
ADAPTERS = ("rtx", "mi50")
UPLOAD_MARKER = "packed expert prompt uploads:"
EARLY_DECODE_ENV = "LLAMA_EXPERT_EARLY_DECODE"
LAST_USE_ENV = "LLAMA_EXPERT_PREFILL_LOGICAL_TICKS"
EARLY_DECODE_TOKENS = 32
DECODE_EXPERT_COUNTERS = ("misses", "uploads", "bytes", "promotions", "writeback_bytes")


def require(test, message):
    if not test:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def argument_index(args, key):
    require(args.count(key) == 1, f"saved arguments must contain exactly one {key}")
    index = args.index(key) + 1
    require(index < len(args), f"saved {key} has no value")
    return index


def prepare_inputs(run_directory):
    require((SOURCE_RUN / "exit.txt").read_text(encoding="utf-8-sig").strip() == "exit_code=0",
            "historical run did not exit successfully")
    source = SOURCE_RUN / Path(INPUT_PATH).name
    attachment = source.read_bytes()
    staged = run_directory / INPUT_PATH
    staged.parent.mkdir(parents=True)
    staged.write_bytes(attachment)
    input_path = run_directory / "stdin.txt"
    input_path.write_bytes((f"/read {INPUT_PATH}\n" * 2 + QUESTION + "\n").encode("utf-8"))

    # /read appends a file label and the raw bytes. The submitted question has its last newline removed.
    message = (f"--- File: {INPUT_PATH} ---\n".encode() + attachment) * 2 + QUESTION.encode("utf-8")
    (run_directory / "expected-user-message.txt").write_bytes(message)
    return input_path, message


def build_environment():
    snapshot = read_json(SOURCE_RUN / "settings.json")["Environment"]
    require(snapshot["LLAMA_EXPERT_DEVICE"] == "Vulkan1" and snapshot["LLAMA_EXPERT_BATCH_TOKENS"] == "2",
            "saved environment does not match Vulkan1 scratch allocation at width 2")
    require(Path(snapshot["LLAMA_EXPERT_PLACEMENT"]).is_file(), "packed Expert placement profile is missing")
    settings = dict(snapshot)
    settings.update({
        "LLAMA_EXPERT_PREFILL_TRACE": None,
        "LLAMA_MOE_CACHE_TRACE": None,
        "LLAMA_TRACE": None,
        "LLAMA_EXPERT_PREFILL_32": None,
        "LLAMA_EXPERT_PREFILL_ACTIVE_WAVES": None,
        "LLAMA_EXPERT_PROFILE_RAM_COPY": None,
        "LLAMA_DECODE_TELEMETRY": None,
        "LLAMA_DECODE_TELEMETRY_PER_RESPONSE": None,
        "LLAMA_EXPERT_PREFILL_STATS": "1",
    })
    return settings


def child_environment(settings):
    result = os.environ.copy()
    result.pop(EARLY_DECODE_ENV, None)
    result.pop(LAST_USE_ENV, None)
    for key, value in settings.items():
        if value:
            result[key] = value
        else:
            result.pop(key, None)
    if result.get("CUDA_PATH"):
        result["PATH"] = str(Path(result["CUDA_PATH"]) / "bin" / "x64") + os.pathsep + result.get("PATH", "")
    return result


def first_match(pattern, text, description):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    require(len(matches) == 1, f"expected one {description}, found {len(matches)}")
    return matches[0]


def decode_fields(line, fields, signed=()):
    values = {}
    for item in line.split():
        key, separator, value = item.partition("=")
        require(separator and key not in values and value, f"malformed decode diagnostic: {line}")
        values[key] = value
    require(all(field in values for field in fields), f"missing decode diagnostic fields: {line}")
    require(all(re.fullmatch(r"-?[0-9]+" if field in signed else r"[0-9]+", values[field]) for field in fields),
            f"invalid decode diagnostic counters: {line}")
    return {field: int(values[field]) for field in fields}, values


def decode_expert_metrics(log, result):
    marker = re.compile(r"expert_decode_(initial|round|summary): ([^\r\n]+)")
    events = marker.findall(log)
    require(len(events) >= 3 and events[0][0] == "initial" and events[-1][0] == "summary"
            and all(kind == "round" for kind, _ in events[1:-1]), "missing, duplicate or misplaced target decode diagnostics")
    aggregate_keys = ("requests", *DECODE_EXPERT_COUNTERS)
    positions = ("position_begin", "position_end")
    initial, _ = decode_fields(events[0][1], ("request", "before", "after", "count", *positions))
    terminal, _ = decode_fields(events[-1][1], ("request", "rounds", "initial", "generated", *aggregate_keys, "supported"))
    require(terminal["supported"] == 1, "target decode diagnostics reported an unsupported request")
    require(terminal["rounds"] == len(events) - 2, "incomplete target decode rounds")
    require(initial["request"] == terminal["request"] and initial["before"] == 0
            and initial["after"] == initial["count"] == terminal["initial"] == 1
            and initial["position_begin"] == initial["position_end"], "unexpected first prompt-sampled token")
    require(terminal["generated"] == result["generated_tokens"] > EARLY_DECODE_TOKENS,
            "target decode diagnostic disagrees with total generated tokens or has no later window")

    ranges = ("early", "crosses_boundary", "later", "without_generated_positions")
    windows = {
        name: dict(rounds=0, generated_tokens=0, **dict.fromkeys(DECODE_EXPERT_COUNTERS, 0))
        for name in ranges
    }
    rounds = []
    alignment = []
    previous = terminal["initial"]
    last_position = initial["position_end"]
    for ordinal, (_, text) in enumerate(events[1:-1], 1):
        keys = ("request", "round", "before", "after", *positions, "proposed", "accepted", "emitted_candidates",
                "emitted", "target_calls", "target_tokens", *aggregate_keys, "permanent_hits", "scratch_hits")
        row, values = decode_fields(text, keys, positions)
        kind = values.get("kind")
        status = values.get("status")
        require(kind in {"decode", "verify"} and status in {"accepted", "stopped", "deferred", "retry"},
                f"unsupported kind or status in target round {ordinal}: {text}")
        require(row["request"] == terminal["request"] and row["round"] == ordinal, "missing or out-of-order target decode round")
        require(row["before"] == previous and previous <= row["after"] <= terminal["generated"],
                "target decode rounds do not cover a continuous generated-token range")
        emitted = row["after"] - previous
        require(row["emitted"] == emitted and row["emitted_candidates"] == min(row["accepted"], emitted),
                f"inconsistent emitted token counts in round {ordinal}")
        require(0 <= row["accepted"] <= row["proposed"] <= 1 and (kind == "verify" or row["proposed"] == 0),
                f"invalid one-candidate MTP counters in round {ordinal}")
        require(emitted <= row["accepted"] + 1 and (not emitted or status in {"accepted", "stopped"}),
                f"unexpected generated tokens in round {ordinal}")
        require(row["requests"] == row["misses"] + row["permanent_hits"] + row["scratch_hits"]
                and row["misses"] >= row["uploads"]
                and (row["uploads"] == 0) == (row["bytes"] == 0)
                and 1 <= row["target_calls"] <= row["target_tokens"] <= 2 * row["target_calls"],
                f"inconsistent target work or Expert counters in round {ordinal}")
        if emitted:
            require(row["position_begin"] == last_position + 1 and row["position_end"] == last_position + emitted,
                    f"noncontiguous generated positions in round {ordinal}")
            last_position = row["position_end"]
        else:
            require(row["position_begin"] == row["position_end"] == -1, f"unexpected positions in empty round {ordinal}")
        if emitted == 0:
            window = "without_generated_positions"
        elif row["after"] <= EARLY_DECODE_TOKENS:
            window = "early"
        elif previous >= EARLY_DECODE_TOKENS:
            window = "later"
        else:
            window = "crosses_boundary"
        row["kind"] = kind
        row["status"] = status
        row["window"] = window
        windows[window]["rounds"] += 1
        windows[window]["generated_tokens"] += emitted
        for counter in DECODE_EXPERT_COUNTERS:
            windows[window][counter] += row[counter]
        rounds.append(row)
        alignment.append((kind, status, row["before"], row["after"], row["proposed"], row["accepted"],
                          row["position_begin"], row["position_end"], row["target_calls"], row["target_tokens"]))
        previous = row["after"]
    require(previous == terminal["generated"] and all(sum(row[key] for row in rounds) == terminal[key]
                                                       for key in aggregate_keys),
            "target decode rounds disagree with generated-token or Expert totals")
    accepted_rounds = [row for row in rounds if row["status"] in {"accepted", "stopped"}]
    require(sum(row["proposed"] for row in accepted_rounds) == result["draft_generated"]
            and sum(row["accepted"] for row in accepted_rounds) == result["draft_accepted"],
            "target decode rounds disagree with MTP timing")
    require(terminal["requests"] > 0, "target decode rounds recorded no Expert routing activity")
    require(windows["early"]["rounds"] and windows["later"]["rounds"], "missing a complete early or later target round")
    for values in windows.values():
        count = values["generated_tokens"]
        values["per_generated_token"] = {key: values[key] / count for key in DECODE_EXPERT_COUNTERS} if count else None
    totals = {
        "uploads": int(first_match(r"packed expert slots: [^\r\n]* ram_uploads=(\d+)", log, "run-wide upload count")),
        "bytes": int(first_match(r"packed expert dynamic residency: promotions=[^\r\n]* uploaded_bytes=(\d+)", log,
                                 "run-wide uploaded bytes")),
    }
    remainder = {key: total - result["prompt_uploads"][key] - terminal[key] for key, total in totals.items()}
    require(all(value >= 0 for value in remainder.values()), "prompt and target round uploads exceed run-wide totals")
    first_promotion = next((row for row in rounds if row["promotions"]), None)
    return {
        "scope": "Target decode/verify Expert route-request misses and weight uploads by round. "
                 "before and after count server-generated token positions; "
                 "the stop token can be counted even when it contributes no visible text. "
                 "Each round covers positions before+1 through after and includes work for rejected candidates.",
        "first_prompt_sample_excluded": initial,
        "boundary_after_generated_token": EARLY_DECODE_TOKENS,
        "run_wide_reconciliation": {"matches": not any(remainder.values()), "unattributed": remainder,
                                    "run_wide": totals},
        "windows": windows,
        "rounds": rounds,
        "first_promotion_round": first_promotion,
        "round_alignment": alignment,
    }


def decode_window_comparison(narrow, wide, delta):
    before = narrow["decode_expert_window"]
    after = wide["decode_expert_window"]
    aligned = before["round_alignment"] == after["round_alignment"]
    old_account = before["run_wide_reconciliation"]
    new_account = after["run_wide_reconciliation"]
    absolute_reconciled = old_account["matches"] and new_account["matches"]
    remainder_equal = old_account["unattributed"] == new_account["unattributed"]
    decode_deltas = {
        key: sum(window[key] for window in after["windows"].values()) -
             sum(window[key] for window in before["windows"].values())
        for key in ("uploads", "bytes")
    }
    run_wide_less_prompt_deltas = {
        key: new_account["run_wide"][key] - old_account["run_wide"][key]
             - (wide["prompt_uploads"][key] - narrow["prompt_uploads"][key])
        for key in decode_deltas
    }
    delta_reconciled = remainder_equal and decode_deltas == run_wide_less_prompt_deltas
    comparable = (aligned and delta["generated_text_matches"] and delta["generated_token_counts_match"]
                  and delta["draft_counts_match"] and delta_reconciled)
    return {
        "comparable": comparable,
        "rounds_aligned": aligned,
        "run_wide_uploads_reconciled": absolute_reconciled,
        "unattributed_run_wide_uploads_equal": remainder_equal,
        "run_wide_difference_reconciled": delta_reconciled,
        "target_decode_difference": decode_deltas if comparable else None,
        "run_wide_less_prompt_difference": run_wide_less_prompt_deltas if comparable else None,
        "reconciliation_note": (
            "Neither absolute total reconciles. The two unattributed remainders match exactly; their origins are unknown."
            if delta_reconciled and not absolute_reconciled else None
        ),
        "boundary_after_generated_token": EARLY_DECODE_TOKENS,
        "wide_minus_baseline": {
            name: {counter: values[counter] - before["windows"][name][counter]
                   for counter in DECODE_EXPERT_COUNTERS}
            for name, values in after["windows"].items()
        } if comparable else None,
    }


def parse_last_use_ticks(log, enabled, prompt, *, require_age_change=True):
    marker = first_match(r"packed expert prefill ticks: ([^\r\n]+)", log, "Prefill tick activation")
    fields, _ = decode_fields(marker, ("enabled", "layer_steps", "tokens", "advanced_ticks", "min_delta",
                                      "max_delta", "native_steps", "native_ticks", "last_use_age_sum"))
    steps = fields["layer_steps"]
    require(fields["enabled"] == int(enabled), "Prefill logical-tick activation does not match the requested mode")
    require(steps > 0 and steps + fields["native_steps"] == prompt["layer_steps"]
            and fields["native_ticks"] == fields["native_steps"] and 3 * steps <= fields["tokens"] <= 512 * steps,
            "Prefill tick row disagrees with compact and native layer coverage")
    if enabled:
        require(2 <= fields["min_delta"] <= fields["max_delta"] == (prompt["max_tokens"] + 1) // 2
                and 0 <= 2 * fields["advanced_ticks"] - fields["tokens"] <= steps
                and (not require_age_change or fields["last_use_age_sum"] > 0),
                "Prefill logical ticks or token-order Expert ages did not activate")
    else:
        require(fields["min_delta"] == fields["max_delta"] == 1
                and fields["advanced_ticks"] == steps and fields["last_use_age_sum"] == 0,
                "Prefill OFF did not keep the original per-batch ticks")
    return fields


def parse_results(case_directory, expected_message, width, *, archived=False, decode_expert_window=False, last_use=None):
    log = (case_directory / "llama.log").read_text(encoding="utf-8", errors="replace")
    for marker in (
        "using device CUDA0 (NVIDIA GeForce RTX 4070 Ti)",
        "using device Vulkan1 (AMD Radeon Pro VII)",
        "packed expert batch capacity = 2, n_ubatch = 2, n_rs_seq = 1",
        "20 scratch slots per layer",
    ):
        require(marker in log, f"missing expected benchmark configuration: {marker}")
    prompt_ms, prompt_tokens = first_match(r"prompt eval time =\s+([0-9.]+) ms /\s+(\d+) tokens", log, "prompt timing")
    require(int(prompt_tokens) == PROMPT_TOKENS and float(prompt_ms) > 0,
            f"expected {PROMPT_TOKENS} processed prompt tokens, got {prompt_tokens}")
    draft_accepted, draft_generated = map(int, first_match(
        r"draft acceptance = [0-9.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)", log, "MTP draft timing"))
    require(0 <= draft_accepted <= draft_generated and draft_generated > 0, "MTP verification did not run")
    decode_ms, generated_tokens = first_match(r"\|\s+eval time =\s+([0-9.]+) ms /\s+(\d+) tokens", log, "generation timing")
    generated_tokens = int(generated_tokens)
    require(generated_tokens > 0 and float(decode_ms) > 0, "missing generated tokens or decode timing")
    raw_stats = first_match(re.escape(UPLOAD_MARKER) + r" ([^\r\n]+)", log, "prompt upload summary")
    fields = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)\b", raw_stats)}
    for field in ("capacity", "tokens", "ubatches", "max_tokens", "decode_max_tokens", "layer_steps", "uploads", "bytes",
                  "compact_chunks", "workspace_bytes"):
        require(field in fields, f"prompt upload summary lacks {field}")
    require(fields["capacity"] == fields["max_tokens"] == width, f"expected live prompt width {width}: {raw_stats}")
    require(fields["decode_max_tokens"] == 2, f"expected actual target nonprompt width 2: {raw_stats}")
    require(fields["tokens"] == PROMPT_TOKENS, f"expected prompt-only coverage of {PROMPT_TOKENS} tokens: {raw_stats}")
    require(0 < fields["ubatches"] <= fields["tokens"] <= width * fields["ubatches"] and fields["layer_steps"] > 0,
            f"invalid prompt coverage: {raw_stats}")
    if fields["uploads"] == 0 and fields["bytes"] > 0 and archived and width == 2:
        fields["reported_uploads"] = 0
        fields["uploads"] = None
        fields["upload_count_source"] = "unavailable: archived counter reported zero despite positive bytes"
    else:
        require((fields["uploads"] == 0) == (fields["bytes"] == 0), f"inconsistent upload summary: {raw_stats}")
        fields["upload_count_source"] = "measured in this run"
    require((fields["compact_chunks"] > 0) == (width == 512)
            and (fields["workspace_bytes"] > 0) == (width == 512), f"compact prefill activation mismatch: {raw_stats}")
    transcript = (case_directory / "transcript.txt").read_bytes().replace(b"\r\n", b"\n")
    prefix = (f"User: Added text file: {INPUT_PATH}\n" * 2 + "User:\n").encode("utf-8")
    before_response = prefix + expected_message + b"\n\nAssistant:\n"
    require(transcript.startswith(before_response), "CLI did not record the exact reconstructed user message")
    answer = transcript[len(before_response):]
    require(bool(answer.strip()), "CLI transcript is missing generated text")
    result = {
        "prompt_ms": float(prompt_ms),
        "prompt_tokens": int(prompt_tokens),
        "draft_accepted": draft_accepted,
        "draft_generated": draft_generated,
        "generated_tokens": generated_tokens,
        "decode_ms": float(decode_ms),
        "user_message_sha256": sha256(expected_message),
        "answer_sha256": sha256(answer),
        "prompt_uploads": fields,
    }
    if decode_expert_window:
        result["decode_expert_window"] = decode_expert_metrics(log, result)
    if last_use is not None:
        result["last_use_ticks"] = parse_last_use_ticks(log, last_use, fields)
    return result, answer


def summarize_vram(path):
    readings = {role: [] for role in ADAPTERS}
    times = []
    with path.open(newline="", encoding="utf-8-sig") as sample_file:
        rows = csv.DictReader(sample_file)
        require(rows.fieldnames == ["seconds_since_launch", "rtx_bytes", "mi50_bytes"], "invalid VRAM CSV header")
        for row in rows:
            require(None not in row and all(value is not None for value in row.values()), "invalid VRAM CSV row")
            elapsed = float(row["seconds_since_launch"])
            require(elapsed >= 0 and (not times or elapsed > times[-1]) and elapsed != float("inf"),
                    "invalid VRAM sample timestamps")
            times.append(elapsed)
            for role in ADAPTERS:
                raw = row[f"{role}_bytes"]
                if raw != "":
                    require(raw.isdecimal(), f"invalid VRAM sample for {role}")
                    readings[role].append(int(raw))
    for role, values in readings.items():
        require(len(values) >= 2 and max(values) > 0, f"no usable local VRAM samples for {role}, see {path}")
    return {
        "scope": "llama-cli process local WDDM memory over recorded live samples, including load, prefill and generation",
        "target_interval_ms": int(SAMPLE_SECONDS * 1000),
        "max_poll_gap_ms": round(1000 * max(right - left for left, right in zip(times, times[1:]))),
        "valid_samples": {role: len(values) for role, values in readings.items()},
        "peak_bytes": {role: max(values) for role, values in readings.items()},
    }


def vram_metrics(path, *, optional=False, failure=None):
    try:
        result = summarize_vram(path)
    except (OSError, ValueError, RuntimeError, csv.Error) as error:
        if not optional:
            raise
        result = {"scope": "live llama-cli process local WDDM memory",
                  "peak_bytes": dict.fromkeys(ADAPTERS)}
        failure = failure or str(error) or type(error).__name__
    if failure:
        result["observed_peak_bytes"] = result["peak_bytes"]
        result["peak_bytes"] = dict.fromkeys(ADAPTERS)
        result["status"] = "degraded" if any(result["observed_peak_bytes"].values()) else "unavailable"
        result["reason"] = failure
    else:
        result["status"] = "sampled"
    return result


@contextmanager
def memory_monitor(optional):
    with ExitStack() as stack:
        try:
            memory = stack.enter_context(DedicatedMemory())
        except Exception as error:
            if not optional:
                raise
            yield None, str(error) or type(error).__name__
        else:
            try:
                yield memory, None
            finally:
                try:
                    stack.close()
                except Exception as error:
                    if not optional:
                        raise
                    print(f"VRAM monitor cleanup failed: {str(error) or type(error).__name__}", file=sys.stderr)


def monitor_process(cli, args, settings, run_directory, case_directory, stdin_path, memory, *,
                    optional_vram=False, vram_failure=None):
    env = child_environment(settings)
    sample_path = case_directory / "local-vram.csv"
    with (stdin_path.open("rb") as stdin, (case_directory / "stdout.log").open("wb") as stdout,
          (case_directory / "stderr.log").open("wb") as stderr):
        start = time.monotonic()
        with subprocess.Popen([str(cli), *args], cwd=run_directory, stdin=stdin, stdout=stdout, stderr=stderr, env=env) as process:
            try:
                if memory is not None:
                    try:
                        with (sample_path.open("w", newline="", encoding="utf-8") as sample_file,
                              memory.process_handle(process.pid) as handle):
                            writer = csv.writer(sample_file)
                            writer.writerow(("seconds_since_launch", "rtx_bytes", "mi50_bytes"))
                            while process.poll() is None:
                                elapsed = time.monotonic() - start
                                try:
                                    sample = memory.sample(handle)
                                except ProcessTerminating:
                                    if optional_vram:
                                        try:
                                            process.wait(timeout=SAMPLE_SECONDS)
                                        except subprocess.TimeoutExpired:
                                            vram_failure = "VRAM monitoring stopped before the child exited"
                                    break
                                writer.writerow((f"{elapsed:.6f}", *(sample[role] for role in ADAPTERS)))
                                sample_file.flush()
                                try:
                                    process.wait(timeout=SAMPLE_SECONDS)
                                except subprocess.TimeoutExpired:
                                    pass
                    except Exception as error:
                        if not optional_vram:
                            raise
                        vram_failure = str(error) or type(error).__name__
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                raise
            finally:
                code = process.wait()
                (case_directory / "exit.txt").write_text(f"exit_code={code}\n", encoding="utf-8")
    require(code == 0, f"llama-cli exited with {code}, see {case_directory / 'stderr.log'}")
    return vram_metrics(sample_path, optional=optional_vram, failure=vram_failure)


def run_case(cli, run_directory, stdin_path, message, baseline, memory, width, archived_args=None, *,
             decode_expert_window=False, last_use=None, vram_failure=None):
    suffix = "" if last_use is None else "-last-use-on" if last_use else "-last-use-off"
    case_directory = run_directory / f"width-{width}{suffix}"
    case_directory.mkdir()
    settings = dict(baseline, LLAMA_EXPERT_PREFILL_512="1" if width == 512 else None)
    if decode_expert_window:
        settings[EARLY_DECODE_ENV] = "1"
    if last_use is not None:
        settings[LAST_USE_ENV] = "1" if last_use else None
    args = list(archived_args) if archived_args is not None else read_json(SOURCE_RUN / "arguments.json")
    require(isinstance(args, list) and all(isinstance(arg, str) for arg in args), "saved arguments must be strings")
    for key, value in {
        "--log-file": case_directory / "llama.log", "-b": OUTER_BATCH,
        "-ub": 2, "--spec-draft-n-max": 1, "--spec-draft-n-ubatch": 1,
    }.items():
        args[argument_index(args, key)] = str(value)
    require(args[argument_index(args, "-dev")] == "CUDA0,Vulkan1"
            and args[argument_index(args, "--spec-type")] == "draft-mtp"
            and args[argument_index(args, "-np")] == "1", "saved arguments do not match the single-slot MTP setup")
    for key in ("-m", "-md"):
        require(Path(args[argument_index(args, key)]).is_file(), f"missing model for {key}")
    transcript = str(case_directory / "transcript.txt")
    if archived_args is None:
        args += ["--single-turn", "--output-file", transcript]
    else:
        args[argument_index(args, "--output-file")] = transcript
    write_json(case_directory / "settings.json", {"cli": str(cli), "arguments": args, "environment": settings})
    print(f"Running prefill width {width}{suffix}", flush=True)
    if last_use is None:
        vram = monitor_process(cli, args, settings, run_directory, case_directory, stdin_path, memory)
    else:
        vram = monitor_process(cli, args, settings, run_directory, case_directory, stdin_path, memory,
                               optional_vram=True, vram_failure=vram_failure)
    result, answer = parse_results(case_directory, message, width, decode_expert_window=decode_expert_window, last_use=last_use)
    result["exit_code"] = 0
    (case_directory / "answer.txt").write_bytes(answer)
    result["local_vram"] = vram
    write_json(case_directory / "metrics.json", result)
    return result, answer


def benchmark_signature(log):
    patterns = {
        "upload_count_all_phases": r"packed expert slots: [^\r\n]* ram_uploads=(\d+)",
        "requests_and_misses": r"packed expert requests: total=(\d+) misses=(\d+)",
        "promotion_counts_and_bytes": r"packed expert dynamic residency: promotions=(\d+) evictions=(\d+) writeback_bytes=(\d+)",
        "upload_bytes_all_phases": r"packed expert dynamic residency: promotions=[^\r\n]* uploaded_bytes=(\d+)",
        "draft_accepted_and_proposed": r"draft acceptance = [0-9.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)",
        "generated_tokens": r"\|\s+eval time =\s+[0-9.]+ ms /\s*(\d+) tokens",
    }
    return {key: first_match(pattern, log, key) for key, pattern in patterns.items()}


def historical_upload_reference(archived_case, result):
    source = SOURCE_RUN / "expert-prefill.jsonl"
    try:
        prior = (SOURCE_RUN / "dialogue.log").read_text(encoding="utf-8")
        archived = (archived_case / "llama.log").read_text(encoding="utf-8")
        prior_signature = benchmark_signature(prior)
        require(prior_signature == benchmark_signature(archived), "run-wide expert and draft signatures differ")
        read_trace = runpy.run_path(str(Path(__file__).with_name("compare-expert-prefill.py")))["read_trace"]
        requests = read_trace(source)
        require(len(requests) == 1, "expected one historical trace request")
        segments = next(iter(requests.values()))
        tokens = sum(len(segment.positions) for segment in segments)
        require(tokens == result["prompt_tokens"]
                and len(segments) == result["prompt_uploads"]["ubatches"], "historical prompt coverage differs")
        rows = [row for segment in segments for row in segment.layers.values()]
        uploads = sum(row.uploads for row in rows)
        total_bytes = sum(row.bytes for row in rows)
        require(total_bytes == result["prompt_uploads"]["bytes"], "historical prompt bytes differ")
        return {
            "source": str(source),
            "historical_prompt_uploads": uploads,
            "historical_prompt_upload_bytes": total_bytes,
            "quantized_expert_bytes": sorted({sum(row.strides) for row in rows}),
            "matching_run_wide_counters": prior_signature,
            "note": "Historical trace records the same reconstructed workload. Matching bytes and run-wide counters do not uniquely determine the archived prompt upload count.",
        }
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        return {"source": str(source), "historical_prompt_uploads": None, "note": f"Reference unavailable: {error}"}


def recover_baseline(directory, expected_environment, *, archived=True, decode_expert_window=False, last_use_off=False):
    case = directory / ("width-512-last-use-off" if last_use_off else "width-2")
    provenance = read_json(directory / "run.json")
    require(not last_use_off or (provenance.get("compare_prefill_last_use") is True
                                and provenance.get("decode_expert_window") is True),
            "archive does not contain a last-use OFF diagnostic run")
    require(Path(provenance["source_run"]).resolve() == SOURCE_RUN.resolve(), "archive points to a different historical input")
    require(provenance["outer_batch"] == OUTER_BATCH, "archive uses a different outer batch size")
    require((case / "exit.txt").read_text(encoding="utf-8-sig").strip() == "exit_code=0",
            "archived baseline subprocess did not exit successfully")
    stdin = (directory / "stdin.txt").read_bytes()
    message = (directory / "expected-user-message.txt").read_bytes()
    attachment = (directory / INPUT_PATH).read_bytes()
    require(sha256(stdin) == provenance["stdin_sha256"] and sha256(message) == provenance["expected_user_message_sha256"],
            "archived stdin or expected message hash differs")
    require(attachment == (SOURCE_RUN / Path(INPUT_PATH).name).read_bytes(), "archived attachment differs from the historical file")
    require(stdin == (f"/read {INPUT_PATH}\n" * 2 + QUESTION + "\n").encode("utf-8"), "archived stdin differs from the benchmark input")
    require(message == (f"--- File: {INPUT_PATH} ---\n".encode() + attachment) * 2 + QUESTION.encode("utf-8"),
            "archived message differs from the reconstructed benchmark input")
    settings = read_json(case / "settings.json")
    environment = settings["environment"]
    required_environment = dict(expected_environment, LLAMA_EXPERT_PREFILL_512="1" if last_use_off else None)
    if last_use_off:
        required_environment.update({LAST_USE_ENV: None, EARLY_DECODE_ENV: "1"})
    require(environment == required_environment, "archived baseline environment differs from the expected benchmark settings")
    args = settings["arguments"]
    require(isinstance(args, list) and all(isinstance(arg, str) for arg in args), "archived arguments must be strings")
    expected_args = read_json(SOURCE_RUN / "arguments.json")
    expected_args[argument_index(expected_args, "--log-file")] = str(case / "llama.log")
    expected_args[argument_index(expected_args, "-b")] = str(OUTER_BATCH)
    expected_args += ["--single-turn", "--output-file", str(case / "transcript.txt")]
    require(args == expected_args, "archived arguments diverge from the expected benchmark settings")
    result, answer = parse_results(case, message, 512 if last_use_off else 2, archived=archived and not last_use_off,
                                   decode_expert_window=decode_expert_window or last_use_off,
                                   last_use=False if last_use_off else None)
    result["exit_code"] = 0
    if last_use_off:
        metrics_path = case / "metrics.json"
        try:
            saved_vram = read_json(metrics_path)["local_vram"]
            failure = None if saved_vram["status"] == "sampled" else saved_vram["reason"]
        except (OSError, ValueError, KeyError, TypeError):
            failure = "archived OFF run did not complete VRAM measurement reporting"
        result["local_vram"] = vram_metrics(case / "local-vram.csv", optional=True, failure=failure)
    else:
        result["local_vram"] = summarize_vram(case / "local-vram.csv")
    result["recovered_from"] = str(case)
    if result["prompt_uploads"]["uploads"] is None:
        result["historical_upload_reference"] = historical_upload_reference(case, result)
    return result, answer, args, provenance


def binary_hashes(cli):
    names = {"llama-cli-impl.dll", "llama-server-impl.dll", "llama.dll", "llama-common.dll"}
    artifacts = [cli, *(path for path in cli.parent.glob("*.dll") if path.name in names or path.name.startswith("ggml"))]
    result = {}
    for path in artifacts:
        with path.open("rb") as file:
            result[path.name] = hashlib.file_digest(file, "sha256").hexdigest()
    return result


def comparison_report(narrow, narrow_answer, wide, wide_answer, resumed):
    observed = narrow["prompt_uploads"]
    candidate = wide["prompt_uploads"]
    require(observed["tokens"] == candidate["tokens"],
            f"prompt-only scopes differ ({observed['tokens']} vs {candidate['tokens']} tokens)")
    return {
        "status": "exit unverified" if wide["exit_code"] is None else "subprocess exits verified",
        "upload_scope": "prompt-only Expert weight uploads, counted by Expert across all weight planes",
        "answer_scope": "assistant text written by the CLI transcript",
        "build_scope": ("Archived baseline and current candidate use separate build provenance. "
                        "The archived executable hash does not establish linked DLL identity."
                        if resumed else "Both runs use the same recorded build."),
        "width_2": narrow,
        "width_512": wide,
        "comparison": {
            "generated_text_matches": narrow_answer == wide_answer,
            "generated_token_counts_match": narrow["generated_tokens"] == wide["generated_tokens"],
            "draft_counts_match": (narrow["draft_accepted"], narrow["draft_generated"]) ==
                                  (wide["draft_accepted"], wide["draft_generated"]),
            "prompt_ms_wide_minus_baseline": round(wide["prompt_ms"] - narrow["prompt_ms"], 2),
            "upload_count_wide_minus_baseline": (
                candidate["uploads"] - observed["uploads"] if observed["uploads"] is not None else None
            ),
            "upload_bytes_wide_minus_baseline": candidate["bytes"] - observed["bytes"],
            "peak_local_vram_bytes_wide_minus_baseline": {
                role: (wide["local_vram"]["peak_bytes"][role] - narrow["local_vram"]["peak_bytes"][role]
                       if wide["local_vram"]["peak_bytes"][role] is not None
                       and narrow["local_vram"]["peak_bytes"][role] is not None else None)
                for role in ADAPTERS
            },
        },
    }


def recover_report(directory):
    directory = directory.resolve()
    provenance = read_json(directory / "run.json")
    resumed = bool(provenance.get("baseline_archive"))
    decode_expert_window = provenance.get("decode_expert_window", False)
    require(type(decode_expert_window) is bool and not (resumed and decode_expert_window),
            "invalid decode diagnostic run provenance")
    archive = Path(provenance["baseline_archive"]).resolve() if resumed else directory
    environment = build_environment()
    if decode_expert_window:
        environment[EARLY_DECODE_ENV] = "1"
    narrow, narrow_answer, args, archived_provenance = recover_baseline(
        archive, environment, archived=resumed, decode_expert_window=decode_expert_window)
    message = (directory / "expected-user-message.txt").read_bytes()
    if resumed:
        require(provenance["baseline_archived_cli_sha256"] == archived_provenance["cli_sha256"],
                "recovered baseline executable provenance differs")
    require(sha256(message) == provenance["expected_user_message_sha256"] == narrow["user_message_sha256"]
            and sha256((directory / "stdin.txt").read_bytes()) == provenance["stdin_sha256"] ==
                archived_provenance["stdin_sha256"], "recovered candidate provenance or input differs from the baseline")
    case = directory / "width-512"
    candidate_settings = read_json(case / "settings.json")
    for option, filename in (("--log-file", "llama.log"), ("--output-file", "transcript.txt")):
        args[argument_index(args, option)] = str(case / filename)
    require(candidate_settings["arguments"] == args
            and candidate_settings["environment"] == dict(environment, LLAMA_EXPERT_PREFILL_512="1"),
            "recovered width-512 arguments or environment differ")
    exit_status = case / "exit.txt"
    has_exit_status = exit_status.is_file()
    if has_exit_status:
        require(exit_status.read_text(encoding="utf-8-sig").strip() == "exit_code=0", "width-512 exit was unsuccessful")
    else:
        require(not decode_expert_window, "missing width-512 exit status for decode diagnostic comparison")
    wide, wide_answer = parse_results(case, message, 512, decode_expert_window=decode_expert_window)
    wide["exit_code"] = 0 if has_exit_status else None
    wide["local_vram"] = summarize_vram(case / "local-vram.csv")
    wide["recovered_from"] = str(case)
    report = comparison_report(narrow, narrow_answer, wide, wide_answer, resumed)
    if decode_expert_window:
        delta = report["comparison"]
        delta["decode_expert_window"] = decode_window_comparison(narrow, wide, delta)
    report["run_provenance"] = provenance
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, default=Path("G:/AI/build-prefill32"),
                        help="Windows build directory containing bin/llama-cli.exe")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--resume-baseline", type=Path, help="reuse the completed width-2 subprocess from this run directory")
    operation.add_argument("--recover-report", type=Path, help="read a past run's logs and print JSON without running inference")
    operation.add_argument("--compare-prefill-last-use", action="store_true",
                           help="compare two fresh width-512 runs with Prefill logical ticks off and on")
    operation.add_argument("--resume-last-use-off", type=Path,
                           help="reuse a completed width-512 OFF run and measure logical ticks ON only")
    parser.add_argument("--decode-expert-window", action="store_true",
                        help="capture target Expert counters for both fresh runs, with a boundary after generated token 32")
    args = parser.parse_args()
    if args.decode_expert_window and (args.resume_baseline or args.recover_report or args.resume_last_use_off):
        parser.error("--decode-expert-window requires two fresh runs")
    last_use_comparison = args.compare_prefill_last_use or args.resume_last_use_off is not None
    decode_expert_window = args.decode_expert_window or last_use_comparison
    if args.recover_report:
        try:
            report = recover_report(args.recover_report)
            print(json.dumps(report, indent=2))
            decode = report["comparison"].get("decode_expert_window")
            require(decode is None or decode["comparable"], "recovered target decode comparison is not comparable")
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            parser.exit(1, f"error: {error}\nLog directory: {args.recover_report}\n")
        return
    if os.name != "nt":
        parser.error("this comparison requires Windows")
    cli = args.build_directory.resolve() / "bin" / "llama-cli.exe"
    suffix = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    kind = ("compare-prefill512-last-use" if last_use_comparison else
            "compare-prefill512-resume" if args.resume_baseline else
            "compare-prefill512-decode-experts" if decode_expert_window else "compare-prefill512")
    run_directory = LOG_ROOT / f"{kind}-{suffix}"
    try:
        require(cli.is_file(), f"missing {cli}")
        baseline = build_environment()
        resume = args.resume_baseline or args.resume_last_use_off
        resume = resume.resolve() if resume else None
        hashes = binary_hashes(cli)
        archived_args = None
        archive_provenance = None
        if resume:
            print(f"Recovering archived baseline: {resume}", flush=True)
            narrow, narrow_answer, archived_args, archive_provenance = recover_baseline(
                resume, baseline, last_use_off=last_use_comparison)
            if last_use_comparison:
                require(hashes == archive_provenance["candidate_build_artifact_sha256"],
                        "use the exact same executable and DLLs as the archived OFF run")
        with memory_monitor(last_use_comparison) as (memory, vram_failure):
            if resume and memory and archive_provenance.get("adapters"):
                for role in ADAPTERS:
                    old = archive_provenance["adapters"][role]
                    new = memory.adapters[role]
                    require(old["name"] == new["name"] and old["dedicated_capacity_bytes"] == new["dedicated_capacity_bytes"],
                            f"the measured {role} differs from the archived adapter")
            run_directory.mkdir()
            stdin_path, message = prepare_inputs(run_directory)
            write_json(run_directory / "run.json", {
                "source_run": str(SOURCE_RUN), "cli_sha256": hashes[cli.name], "candidate_build_artifact_sha256": hashes,
                "baseline_archive": str(resume) if resume and not last_use_comparison else None,
                "last_use_off_archive": str(resume) if resume and last_use_comparison else None,
                "baseline_archived_cli_sha256": archive_provenance["cli_sha256"] if resume else None,
                "stdin_sha256": sha256(stdin_path.read_bytes()), "expected_user_message_sha256": sha256(message),
                "decode_expert_window": decode_expert_window,
                "compare_prefill_last_use": last_use_comparison,
                "adapters": memory.adapters if memory else None, "outer_batch": OUTER_BATCH,
            })
            print(f"Log directory: {run_directory}", flush=True)
            if resume:
                require(sha256(message) == narrow["user_message_sha256"], "candidate input differs from archived baseline")
                archive_label = "width-512-last-use-off" if last_use_comparison else "width-2"
                write_json(run_directory / f"recovered-{archive_label}.json", narrow)
                (run_directory / f"recovered-{archive_label}-answer.txt").write_bytes(narrow_answer)
            else:
                narrow, narrow_answer = run_case(cli, run_directory, stdin_path, message, baseline, memory,
                                                 512 if last_use_comparison else 2, decode_expert_window=decode_expert_window,
                                                 last_use=False if last_use_comparison else None, vram_failure=vram_failure)
            wide, wide_answer = run_case(cli, run_directory, stdin_path, message, baseline, memory, 512, archived_args,
                                         decode_expert_window=decode_expert_window,
                                         last_use=True if last_use_comparison else None, vram_failure=vram_failure)
        report = comparison_report(narrow, narrow_answer, wide, wide_answer, bool(resume) and not last_use_comparison)
        delta = report["comparison"]
        if decode_expert_window:
            delta["decode_expert_window"] = decode_window_comparison(narrow, wide, delta)
        if last_use_comparison:
            report["width_512_last_use_off"] = report.pop("width_2")
            report["width_512_last_use_on"] = report.pop("width_512")
            for key in list(delta):
                if "wide_minus_baseline" in key:
                    delta[key.replace("wide_minus_baseline", "on_minus_off")] = delta.pop(key)
            decode = delta["decode_expert_window"]
            decode["on_minus_off"] = decode.pop("wide_minus_baseline")
            delta["decode_ms_on_minus_off"] = round(wide["decode_ms"] - narrow["decode_ms"], 2) if decode["comparable"] else None
        write_json(run_directory / "comparison.json", report)
        labels = ("Width 512 last-use OFF", "Width 512 last-use ON") if last_use_comparison else ("Width 2", "Width 512")
        for label, result in zip(labels, (narrow, wide)):
            uploads = result["prompt_uploads"]
            peak = result["local_vram"]["peak_bytes"]
            count = uploads["uploads"] if uploads["uploads"] is not None else "unknown"
            vram = (f"sampled process local VRAM RTX {peak['rtx'] / 2**20:.1f} MiB / MI50 {peak['mi50'] / 2**20:.1f} MiB"
                    if all(peak[role] is not None for role in ADAPTERS) else
                    f"VRAM {result['local_vram']['status']}: {result['local_vram']['reason']}")
            print(f"{label}: prompt {result['prompt_ms']:.2f} ms, decode {result['decode_ms']:.2f} ms, "
                  f"{count} prompt uploads / {uploads['bytes']} bytes, {vram}")
            if decode_expert_window:
                expert = result["decode_expert_window"]
                if last_use_comparison:
                    first = expert["first_promotion_round"]
                    label = f"{first['round']} ({first['window']})" if first else "none"
                    print(f"  first decode promotion round: {label}")
                for name, window in expert["windows"].items():
                    if window["rounds"]:
                        print(f"  {name}: {window['rounds']} target rounds, {window['generated_tokens']} generated tokens, "
                              f"{window['misses']} misses, {window['uploads']} uploads / {window['bytes']} bytes, "
                              f"{window['promotions']} promotions / {window['writeback_bytes']} writeback bytes")
        if resume and not last_use_comparison:
            print("Comparison uses an archived baseline and the current build. Deltas include any build differences.")
        if "historical_upload_reference" in narrow:
            reference = narrow["historical_upload_reference"]["historical_prompt_uploads"]
            if reference is not None:
                print(f"Historical trace reference only: {reference} prompt uploads. Archived prompt count remains unknown.")
        print(json.dumps(delta, indent=2))
        if not delta["draft_counts_match"]:
            print("MTP draft acceptance or proposal counts differed. See comparison.json.")
        require(delta["generated_text_matches"] and delta["generated_token_counts_match"],
                "generated text or token counts differ, see comparison.json before interpreting performance")
        if decode_expert_window:
            note = delta["decode_expert_window"]["reconciliation_note"]
            if note:
                print(note)
            require(delta["decode_expert_window"]["comparable"],
                    "target decode rounds, MTP counts or run-wide uploads do not reconcile across the two cases. "
                    "See both measurements in comparison.json.")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        location = f"\nLog directory: {run_directory}" if run_directory.is_dir() else ""
        parser.exit(1, f"error: {error}{location}\n")


if __name__ == "__main__":
    main()
