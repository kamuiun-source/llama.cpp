"""Measure exact raw-token prompt lengths using the current fork's native completion server."""

import argparse
import ctypes
import json
import os
import runpy
import socket
import statistics
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4


SIZES = (8, 16, 32, 64)
GENERATE = 8
SOURCE_TEXT = (
    "Explain why a home's Wi-Fi connection might get slower at night while a wired connection stays fast. "
    "Start with one likely cause, describe what a person could observe, and suggest one simple check. "
    "Consider what changes when more nearby devices are active, what happens when the router is in another room, "
    "and whether the same device behaves differently when it is close to the router. Use everyday language."
)
OPENER = build_opener(ProxyHandler({}))
COMMON = runpy.run_path(str(Path(__file__).with_name("compare-prefill512.py")))
LOG_ROOT = COMMON["LOG_ROOT"]
SOURCE_RUN = COMMON["SOURCE_RUN"]
require = COMMON["require"]
write_json = COMMON["write_json"]
read_json = COMMON["read_json"]


def dll_server(config_path):
    """Own the exported server entry point and graceful shutdown function in a child process."""
    config = read_json(config_path)
    dll = Path(config["library"])
    with ExitStack() as directories:
        directories.enter_context(os.add_dll_directory(str(dll.parent)))
        if os.environ.get("CUDA_PATH"):
            cuda = Path(os.environ["CUDA_PATH"]) / "bin" / "x64"
            if cuda.is_dir():
                directories.enter_context(os.add_dll_directory(str(cuda)))
        library = ctypes.CDLL(str(dll))
        start = getattr(library, "?llama_server@@YAHHPEAPEAD@Z")
        stop = getattr(library, "?llama_server_terminate@@YAXXZ")
        start.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        start.restype = ctypes.c_int
        stop.argtypes = []
        stop.restype = None

        def listen_for_stop():
            if sys.stdin.buffer.readline().strip() == b"stop":
                stop()

        threading.Thread(target=listen_for_stop, daemon=True).start()
        values = [b"llama-server", *(arg.encode("utf-8") for arg in config["arguments"])]
        arguments = (ctypes.c_char_p * (len(values) + 1))(*values, None)
        return start(len(values), arguments)


def http(port, endpoint, api_key=None, payload=None, timeout=60):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = Request(f"http://127.0.0.1:{port}{endpoint}", data=data, headers=headers)
    with OPENER.open(request, timeout=timeout) as response:
        return json.load(response)


def local_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(process, port, api_key):
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        require(process.poll() is None, f"server exited before it became ready: {process.returncode}")
        try:
            health = http(port, "/health", timeout=1)
        except (URLError, TimeoutError) as error:
            if getattr(getattr(error, "reason", error), "winerror", None) == 10013:
                raise RuntimeError("Windows denied the loopback request; run the benchmark from a PowerShell outside the restricted sandbox") from error
            time.sleep(0.25)
            continue
        if health.get("status") == "ok":
            try:
                http(port, "/props", timeout=1)
            except HTTPError as error:
                require(error.code == 401, f"unexpected HTTP {error.code} on unauthenticated /props")
            else:
                raise ValueError("loopback port answered without the private benchmark API key")
            props = http(port, "/props", api_key, timeout=2)
            require(process.poll() is None and isinstance(props, dict), "private benchmark server did not respond")
            return
        time.sleep(0.25)
    raise RuntimeError("server did not become ready within five minutes")


@contextmanager
def server(case, config, environment, port, api_key):
    config_path = case / "server-arguments.json"
    write_json(config_path, config)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--server-worker", str(config_path)]
    with ((case / "stdout.log").open("wb") as stdout, (case / "stderr.log").open("wb") as stderr,
          subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                           env=COMMON["child_environment"](environment), creationflags=subprocess.CREATE_NO_WINDOW) as process):
        try:
            wait_ready(process, port, api_key)
            yield
        finally:
            if process.poll() is None:
                try:
                    process.stdin.write(b"stop\n")
                    process.stdin.flush()
                except BrokenPipeError:
                    pass
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                code = process.wait()
                raise RuntimeError("server did not shut down gracefully; its final counters cannot be used")
            finally:
                config_path.unlink(missing_ok=True)
                if process.returncode is not None:
                    (case / "exit.txt").write_text(f"exit_code={process.returncode}\n", encoding="utf-8")
        require(code == 0, f"server exit code {code}")


def server_args(case, port, api_key):
    args = read_json(SOURCE_RUN / "arguments.json")
    args.remove("--simple-io")
    color = COMMON["argument_index"](args, "--color")
    del args[color - 1:color + 1]
    options = {"--log-file": case / "server.log", "-b": 512, "-ub": 2, "-n": GENERATE}
    for name, value in options.items():
        args[COMMON["argument_index"](args, name)] = str(value)
    return args + ["--host", "127.0.0.1", "--port", str(port), "--api-key", api_key, "--no-ui",
                   "--no-warmup", "--no-cache-prompt", "--cache-ram", "0"]


def native_shapes(size, batches):
    ones = 2 * batches - size
    twos = size - batches
    require(ones >= 0 and twos >= 0, "invalid width-1/2 prompt coverage")
    return [1] * ones + [2] * twos


def read_case(case, response, size, width):
    log = (case / "server.log").read_text(encoding="utf-8", errors="replace")
    for marker in ("using device CUDA0 (NVIDIA GeForce RTX 4070 Ti)", "using device Vulkan1 (AMD Radeon Pro VII)",
                   "packed expert batch capacity = 2, n_ubatch = 2, n_rs_seq = 1", "20 scratch slots per layer"):
        require(marker in log, f"missing expected configuration: {marker}")
    timings = response["timings"]
    require(response["stop"] is True and response["truncated"] is False
            and response["tokens_evaluated"] == timings["prompt_n"] == size
            and timings["cache_n"] == 0 and timings["prompt_ms"] > 0,
            "HTTP completion did not evaluate exactly the uncached requested prompt")
    require(response["tokens_predicted"] == timings["predicted_n"] == GENERATE,
            "the fixed follow-up generation did not complete")
    first = COMMON["first_match"]
    acceptance = first(r"draft acceptance = [0-9.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)", log, "MTP timing")
    require(0 <= int(acceptance[0]) <= int(acceptance[1]) and int(acceptance[1]) > 0, "MTP did not run")
    marker = first(r"packed expert prompt uploads: ([^\r\n]+)", log, "prompt-only Expert counters")
    fields, _ = COMMON["decode_fields"](marker, ("capacity", "tokens", "ubatches", "max_tokens", "decode_max_tokens",
                                                "layer_steps", "uploads", "bytes", "compact_chunks", "workspace_bytes"))
    count, maximum = fields["ubatches"], fields["max_tokens"]
    require(fields["capacity"] == width and fields["tokens"] == size
            and 1 <= count <= size <= count * maximum and 1 <= maximum <= min(width, size)
            and fields["decode_max_tokens"] == 2 and fields["layer_steps"] == 48 * count
            and ((fields["uploads"] == 0) == (fields["bytes"] == 0)), "invalid prompt-only Expert coverage")
    if width == 2 or maximum <= 2:
        require(fields["compact_chunks"] == 0, "compact Prefill unexpectedly ran at native width")
        shapes = native_shapes(size, count)
    else:
        require(fields["compact_chunks"] > 0 and fields["workspace_bytes"] > 0, "compact Prefill did not activate")
        ticks = COMMON["parse_last_use_ticks"](log, True, fields, require_age_change=False)
        require(ticks["tokens"] % 48 == ticks["native_steps"] % 48 == ticks["layer_steps"] % 48 == 0,
                "compact tick accounting differs among layers")
        compact_tokens, compact_batches = ticks["tokens"] // 48, ticks["layer_steps"] // 48
        native = native_shapes(size - compact_tokens, ticks["native_steps"] // 48)
        if compact_batches <= 2:
            compact = [maximum] if compact_batches == 1 else [maximum, compact_tokens - maximum]
            require(sum(compact) == compact_tokens and all(3 <= item <= maximum for item in compact),
                    "inconsistent compact ubatch sizes")
            shapes = sorted(native + compact)
        else:
            shapes = None
    if shapes is not None:
        require(len(shapes) == count and sum(shapes) == size and max(shapes) == maximum, "ubatch shapes disagree with totals")
    return {"tokens": size, "width": width, "prompt_ms": timings["prompt_ms"],
            "prompt_stats": fields, "ubatch_sizes_unordered": shapes,
            "draft_accepted": int(acceptance[0]), "draft_generated": int(acceptance[1]),
            "generated_ids": response["tokens"]}


def run_case(case, library, base_environment, size, width, tokens):
    case.mkdir()
    port, api_key = local_port(), uuid4().hex
    args = server_args(case, port, api_key)
    environment = dict(base_environment, LLAMA_EXPERT_PREFILL_512="1" if width == 512 else None)
    environment["LLAMA_EXPERT_PREFILL_512_POLICY"] = str(width)
    environment[COMMON["LAST_USE_ENV"]] = "1" if width == 512 else None
    args_for_log = args.copy()
    args_for_log[COMMON["argument_index"](args_for_log, "--api-key")] = "[ephemeral]"
    write_json(case / "settings.json", {"library": str(library), "arguments": args_for_log, "environment": environment})
    with server(case, {"library": str(library), "arguments": args}, environment, port, api_key):
        if tokens is None:
            tokenized = http(port, "/tokenize", api_key, {"content": SOURCE_TEXT, "add_special": True, "parse_special": False})
            tokens = tokenized["tokens"]
            require(isinstance(tokens, list) and len(tokens) >= max(SIZES)
                    and all(type(value) is int and value >= 0 for value in tokens), "invalid fixed source tokenization")
        payload = {"prompt": tokens[:size], "n_predict": GENERATE, "stream": False, "cache_prompt": False,
                   "temperature": 0.0, "seed": 1234, "ignore_eos": True, "return_tokens": True}
        write_json(case / "request.json", payload)
        response = http(port, "/completion", api_key, payload)
        write_json(case / "completion.json", response)
    measured = read_case(case, response, size, width)
    write_json(case / "metrics.json", measured)
    return measured, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, default=Path("G:/AI/build-prefill32"))
    parser.add_argument("--repetitions", type=int, default=1, help="number of paired runs at each prompt length")
    parser.add_argument("--server-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.server_worker:
        return dll_server(args.server_worker)
    if os.name != "nt" or args.repetitions < 1:
        parser.error("requires Windows and at least one repetition")
    library = args.build_directory.resolve() / "bin" / "llama-server-impl.dll"
    directory = LOG_ROOT / f"compare-short-prefill-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    try:
        require(library.is_file(), f"missing {library}")
        environment = COMMON["build_environment"]()
        directory.mkdir()
        write_json(directory / "run.json", {
            "source_text": SOURCE_TEXT, "sizes": SIZES, "repetitions": args.repetitions,
            "build_sha256": COMMON["binary_hashes"](library.parent / "llama-cli.exe"),
            "placement_sha256": COMMON["sha256"](Path(environment["LLAMA_EXPERT_PLACEMENT"]).read_bytes()),
            "method": "Raw /completion token ID prefixes from /tokenize with model specials enabled. No chat template. "
                      "Fresh server, same placement profile, no model warmup or KV prompt reuse. "
                      "Timing includes cold graph/pipeline execution under the default checkpoint policy. "
                      "Follow-up generation confirms MTP activity and is excluded from prompt time.",
        })
        print(f"Log directory: {directory}", flush=True)
        rows = []
        tokens = None
        for repeat in range(args.repetitions):
            for index, size in enumerate(SIZES):
                order = (512, 2) if (repeat + index) % 2 else (2, 512)
                pair = {}
                for width in order:
                    print(f"{size} tokens, repetition {repeat + 1}/{args.repetitions}, width {width}", flush=True)
                    case = directory / f"n{size:02d}-rep{repeat + 1:02d}-w{width}"
                    pair[width], tokens = run_case(case, library, environment, size, width, tokens)
                rows.append({"tokens": size, "repetition": repeat + 1, "order": order, "width_2": pair[2],
                             "width_512": pair[512], "wide_minus_narrow_ms": pair[512]["prompt_ms"] - pair[2]["prompt_ms"]})
                write_json(directory / "pairs.json", rows)
        for size in SIZES:
            pairs = [row for row in rows if row["tokens"] == size]
            median = statistics.median(row["wide_minus_narrow_ms"] for row in pairs)
            for row in pairs:
                narrow, wide = row["width_2"], row["width_512"]
                print(f"{size} tokens #{row['repetition']}: width 2 {narrow['prompt_ms']:.2f} ms "
                      f"(max {narrow['prompt_stats']['max_tokens']}), width 512 {wide['prompt_ms']:.2f} ms "
                      f"(max {wide['prompt_stats']['max_tokens']}), delta {row['wide_minus_narrow_ms']:+.2f} ms")
            if len(pairs) > 1:
                print(f"{size} tokens median paired delta: {median:+.2f} ms")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"error: {error}\nLog directory: {directory}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
