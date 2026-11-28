"""Linux Docker memory telemetry for a benchmark runner.

Use SwapTelemetry(out / "swap-telemetry.jsonl", container_name) as a context
manager. Call attach(container_pid, run) on launch, sample("response-start",
turn=i) before stamping request start, sample("response-finish", turn=i,
rollup=True) after stamping completion, and detach() after server teardown.
The host and response timestamps use time.monotonic_ns(). Pass saved smaps
text to summarize_smaps() for per-mapping swap attribution.
"""

import contextlib
import json
import re
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path


PROC = Path("/proc")
CGROUP = Path("/sys/fs/cgroup")
STATUS_KEYS = {"VmSwap", "VmLck", "VmPin", "VmRSS", "RssAnon", "RssFile", "RssShmem"}
MEMINFO_KEYS = {"MemAvailable", "MemFree", "SwapFree", "SwapTotal", "SwapCached",
                "Active(anon)", "Inactive(anon)", "Active(file)", "Inactive(file)",
                "Mlocked", "Unevictable", "Dirty", "Writeback", "Shmem", "Cached"}
VMSTAT_KEYS = {"pswpin", "pswpout", "pgmajfault", "pgpgin", "pgpgout", "oom_kill",
               "workingset_refault_anon", "workingset_refault_file",
               "pgscan_direct", "pgscan_kswapd", "pgsteal_direct", "pgsteal_kswapd"}
SMAPS_KEYS = {"Size", "Rss", "Pss", "Anonymous", "Locked", "Swap", "SwapPss"}
MAP_HEADER = re.compile(r"^[0-9a-f]+-[0-9a-f]+\s")


def kv(raw, keys=None):
    values = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0].rstrip(":")
            if (keys is None or key in keys) and parts[1].isdigit():
                values[key] = int(parts[1])
    return values


def summarize_smaps(raw):
    mappings = []
    totals = defaultdict(lambda: defaultdict(int))
    for block in re.split(r"(?m)(?=^[0-9a-f]+-[0-9a-f]+\s)", raw):
        if not MAP_HEADER.match(block):
            continue
        header = block.splitlines()[0]
        fields = header.split(maxsplit=5)
        path = fields[5] if len(fields) == 6 else ""
        if path.startswith(("/dev/shm/", "/memfd:", "/SYSV", "[anon_shmem:")):
            kind = "shmem"
        elif path.startswith("/dev/"):
            kind = "device"
        elif path.startswith("/"):
            kind = "file_mapping"
        else:
            kind = "anonymous"
        values = kv(block, SMAPS_KEYS)
        for key, value in values.items():
            totals[kind][key] += value
        if values.get("Swap", 0):
            mappings.append({"header": header, "kind": kind, "kB": values})
    return {"totals_kB": totals, "swapped_mappings": mappings}


def disk_io():
    devices = {}
    for line in (PROC / "diskstats").read_text().splitlines():
        fields = line.split()
        major, minor, device = fields[:3]
        if device not in {"sda", "nvme1n1"} and not device.startswith("dm-"):
            continue
        devices[device] = {"major_minor": f"{major}:{minor}", "reads": int(fields[3]),
                           "sectors_read": int(fields[5]), "writes": int(fields[7]),
                           "sectors_written": int(fields[9]), "in_flight": int(fields[11]),
                           "io_ms": int(fields[12]), "weighted_io_ms": int(fields[13])}
    return devices


def stat_fields(pid):
    # The command in parentheses can contain spaces or closing parentheses.
    fields = (PROC / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return {"minflt": int(fields[7]), "majflt": int(fields[9]),
            "utime_ticks": int(fields[11]), "stime_ticks": int(fields[12]),
            "start_ticks": int(fields[19])}


def read_field(record, key, reader):
    try:
        record[key] = reader()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        record.setdefault("errors", {})[key] = str(exc)


class SwapTelemetry:
    def __init__(self, output, container, period=0.75):
        self.output = output
        self.container = container
        self.period = period
        self.server = None
        self.cgroup = None
        self.stop = threading.Event()
        self.write_lock = threading.Lock()

    def __enter__(self):
        self.stream = self.output.open("w", buffering=1)
        self.thread = threading.Thread(target=self.periodic)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.stream.close()

    def periodic(self):
        while not self.stop.is_set():
            started = time.monotonic()
            self.sample()
            self.stop.wait(max(0, self.period - (time.monotonic() - started)))

    def attach(self, container_pid, run):
        if self.server is not None:
            return
        try:
            top = subprocess.run(["docker", "top", self.container, "-eo", "pid,comm"],
                                 check=True, capture_output=True, text=True, timeout=3)
            for line in top.stdout.splitlines()[1:]:
                fields = line.split()
                if len(fields) != 2 or fields[1] != "llama-server":
                    continue
                host_pid = int(fields[0])
                proc = PROC / str(host_pid)
                status = (proc / "status").read_text()
                namespace_pids = next(line.split()[1:] for line in status.splitlines()
                                      if line.startswith("NSpid:"))
                if int(namespace_pids[-1]) != container_pid:
                    continue
                cgroup = next(line[3:] for line in (proc / "cgroup").read_text().splitlines()
                              if line.startswith("0::"))
                self.cgroup = CGROUP / cgroup.lstrip("/")
                self.server = (host_pid, container_pid, stat_fields(host_pid)["start_ticks"], run)
                self.sample("server-attached")
                return
            error = "docker top has no llama-server matching the container PID"
        except (OSError, ValueError, StopIteration, subprocess.SubprocessError) as exc:
            error = str(exc)
        self.emit({"kind": "attach-error", "mono_ns": time.monotonic_ns(),
                   "server_pid_container": container_pid, "run": run, "error": error})

    def detach(self):
        self.server = None

    def emit(self, record):
        with self.write_lock:
            self.stream.write(json.dumps(record) + "\n")

    def sample(self, kind="periodic", *, turn=None, rollup=False):
        record = {"kind": kind, "mono_ns": time.monotonic_ns(), "wall_ns": time.time_ns()}
        if turn is not None:
            record["turn"] = turn
        read_field(record, "host_meminfo_kB", lambda: kv((PROC / "meminfo").read_text(), MEMINFO_KEYS))
        read_field(record, "host_vmstat", lambda: kv((PROC / "vmstat").read_text(), VMSTAT_KEYS))
        read_field(record, "host_cpu_iowait_ticks", lambda: int((PROC / "stat").read_text().splitlines()[0].split()[5]))
        read_field(record, "host_disks", disk_io)
        for source in ("memory", "io"):
            read_field(record, "host_pressure_" + source,
                       lambda: (PROC / "pressure" / source).read_text().strip())
        cgroup = self.cgroup
        if cgroup is not None:
            record["cgroup"] = str(cgroup)
            for name in ("memory.current", "memory.swap.current"):
                read_field(record, name, lambda: int((cgroup / name).read_text()))
            for name in ("memory.stat", "memory.events"):
                read_field(record, name, lambda: kv((cgroup / name).read_text()))
            for name in ("memory.pressure", "io.pressure", "io.stat"):
                read_field(record, name, lambda: (cgroup / name).read_text().strip())
        server = self.server
        if server is not None:
            self.sample_process(record, server, rollup)
        record["finished_mono_ns"] = time.monotonic_ns()
        self.emit(record)

    def sample_process(self, record, server, rollup):
        host_pid, container_pid, start, label = server
        record.update(run=label, server_pid_host=host_pid, server_pid_container=container_pid)
        proc = PROC / str(host_pid)
        try:
            stat = stat_fields(host_pid)
            if stat["start_ticks"] != start:
                raise ProcessLookupError("server PID was reused")
            record["server_stat"] = stat
        except OSError as exc:
            record.setdefault("errors", {})["server_stat"] = str(exc)
            return
        read_field(record, "server_status_kB", lambda: kv((proc / "status").read_text(), STATUS_KEYS))
        read_field(record, "server_io", lambda: kv((proc / "io").read_text()))
        if rollup:
            read_field(record, "server_smaps_rollup_kB", lambda: kv((proc / "smaps_rollup").read_text()))
        # Docker is only needed for ptrace-gated files, outside measured requests.
        fallback = []
        if record["kind"] != "periodic" and "server_io" not in record:
            fallback.append(("io", "server_io"))
        if rollup and "server_smaps_rollup_kB" not in record:
            fallback.append(("smaps_rollup", "server_smaps_rollup_kB"))
        if fallback:
            def docker_proc():
                files = [f"/proc/{container_pid}/{name}" for name, _ in fallback]
                result = subprocess.run(
                    ["docker", "exec", self.container, "bash", "-c",
                     'for path; do echo "@@@"; cat "$path" 2>&1; done', "proc", *files],
                    check=True, capture_output=True, text=True, timeout=5)
                for (_, key), raw in zip(fallback, result.stdout.split("@@@\n")[1:]):
                    values = kv(raw)
                    record.setdefault("docker_fallback", {})[key] = values or {"error": raw.strip()}
            try:
                docker_proc()
            except (OSError, subprocess.SubprocessError) as exc:
                record.setdefault("errors", {})["docker_fallback"] = str(exc)
        with contextlib.suppress(OSError):
            if stat_fields(host_pid)["start_ticks"] == start:
                return
        for key in ("server_stat", "server_status_kB", "server_io",
                    "server_smaps_rollup_kB", "docker_fallback"):
            record.pop(key, None)
        record.setdefault("errors", {})["server_stat"] = "server exited or changed during collection"
