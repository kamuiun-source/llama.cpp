"""Sample a Windows child process's local video memory on the benchmark adapters."""

import ctypes as ct
import struct
from contextlib import contextmanager
from uuid import UUID


class ProcessTerminating(RuntimeError):
    """The sampled process began exiting during a memory query."""


class AdapterDescription(ct.Structure):
    _fields_ = [
        ("name", ct.c_wchar * 128),
        ("vendor_id", ct.c_uint32),
        ("device_id", ct.c_uint32),
        ("subsystem_id", ct.c_uint32),
        ("revision", ct.c_uint32),
        ("video_memory", ct.c_size_t),
        ("dedicated_system_memory", ct.c_size_t),
        ("shared_system_memory", ct.c_size_t),
        ("luid_low", ct.c_uint32),
        ("luid_high", ct.c_uint32),
        ("flags", ct.c_uint32),
    ]


class OpenAdapter(ct.Structure):
    _fields_ = [("luid_low", ct.c_uint32), ("luid_high", ct.c_uint32), ("handle", ct.c_uint32)]


class VideoMemoryInfo(ct.Structure):
    _fields_ = [
        ("process", ct.c_void_p),
        ("adapter", ct.c_uint32),
        ("segment_group", ct.c_uint32),
        ("budget", ct.c_uint64),
        ("usage", ct.c_uint64),
        ("reservation", ct.c_uint64),
        ("available", ct.c_uint64),
        ("physical_adapter", ct.c_uint32),
    ]


def com_method(pointer, slot, result, *arguments):
    table = ct.cast(pointer, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
    return ct.WINFUNCTYPE(result, ct.c_void_p, *arguments)(table[slot])


def check_status(operation, status):
    if status:
        raise RuntimeError(f"{operation} failed (0x{status & 0xffffffff:08x})")


def cuda_luid():
    cuda = ct.WinDLL("nvcuda")
    device = ct.c_int()
    luid = ct.create_string_buffer(8)
    mask = ct.c_uint32()
    check_status("cuInit", cuda.cuInit(0))
    check_status("cuDeviceGet(CUDA0)", cuda.cuDeviceGet(ct.byref(device), 0))
    check_status("cuDeviceGetLuid(CUDA0)", cuda.cuDeviceGetLuid(luid, ct.byref(mask), device))
    if mask.value != 1:
        raise RuntimeError("CUDA0 is not the expected single-node display adapter")
    return struct.unpack("<II", luid.raw)


def discrete_adapters():
    target_cuda = cuda_luid()
    factory = ct.c_void_p()
    iid = (ct.c_ubyte * 16).from_buffer_copy(UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le)
    create = ct.WinDLL("dxgi").CreateDXGIFactory1
    create.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
    create.restype = ct.c_int32
    check_status("CreateDXGIFactory1", create(iid, ct.byref(factory)))
    adapters = {}
    try:
        # These slots follow the inherited IDXGIFactory and IDXGIAdapter methods.
        enum_adapter = com_method(factory, 12, ct.c_int32, ct.c_uint32, ct.POINTER(ct.c_void_p))
        index = 0
        while True:
            adapter = ct.c_void_p()
            status = enum_adapter(factory, index, ct.byref(adapter))
            if status & 0xffffffff == 0x887a0002:  # DXGI_ERROR_NOT_FOUND
                break
            check_status("IDXGIFactory1::EnumAdapters1", status)
            try:
                description = AdapterDescription()
                get_description = com_method(adapter, 10, ct.c_int32, ct.POINTER(AdapterDescription))
                check_status("IDXGIAdapter1::GetDesc1", get_description(adapter, ct.byref(description)))
                luid = (description.luid_low, description.luid_high)
                name = description.name.lower()
                if luid == target_cuda and "4070 ti" in name and description.vendor_id == 0x10de:
                    role = "rtx"
                elif "radeon pro vii" in name and description.vendor_id == 0x1002:
                    role = "mi50"
                else:
                    role = None
                if role:
                    if role in adapters or not description.video_memory:
                        raise RuntimeError(f"Ambiguous or invalid discrete adapter for {role}")
                    adapters[role] = {"name": description.name, "luid": luid,
                                      "dedicated_capacity_bytes": description.video_memory}
            finally:
                com_method(adapter, 2, ct.c_uint32)(adapter)
            index += 1
    finally:
        com_method(factory, 2, ct.c_uint32)(factory)
    if set(adapters) != {"rtx", "mi50"}:
        raise RuntimeError("DXGI did not identify both CUDA0 RTX 4070 Ti and Radeon Pro VII")
    return adapters


class DedicatedMemory:
    def __enter__(self):
        self.adapters = discrete_adapters()
        self.gdi = ct.WinDLL("gdi32")
        self.kernel = ct.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [ct.c_uint32, ct.c_int32, ct.c_uint32]
        self.kernel.OpenProcess.restype = ct.c_void_p
        self.kernel.CloseHandle.argtypes = [ct.c_void_p]
        self.kernel.CloseHandle.restype = ct.c_int32
        self.handles = {}
        try:
            for role, adapter in self.adapters.items():
                request = OpenAdapter(*adapter["luid"], 0)
                status = self.gdi.D3DKMTOpenAdapterFromLuid(ct.byref(request))
                check_status(f"D3DKMTOpenAdapterFromLuid ({role})", status)
                self.handles[role] = request.handle
            self.sample(None)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_):
        for adapter in self.handles.values():
            self.gdi.D3DKMTCloseAdapter(ct.byref(ct.c_uint32(adapter)))

    @contextmanager
    def process_handle(self, pid):
        handle = self.kernel.OpenProcess(0x0400, False, pid)  # PROCESS_QUERY_INFORMATION
        if not handle:
            raise ct.WinError(ct.get_last_error())
        try:
            yield handle
        finally:
            self.kernel.CloseHandle(handle)

    def sample(self, process):
        result = {}
        for role, adapter in self.handles.items():
            request = VideoMemoryInfo(process=process, adapter=adapter, segment_group=0, physical_adapter=0)
            status = self.gdi.D3DKMTQueryVideoMemoryInfo(ct.byref(request))
            if process is not None and status & 0xffffffff == 0xc000010a:
                raise ProcessTerminating()
            if process is not None and status & 0xffffffff == 0xc000000d:
                # The child may not have registered with this adapter yet.
                result[role] = None
                continue
            check_status(f"D3DKMTQueryVideoMemoryInfo ({role})", status)
            result[role] = request.usage
        return result
