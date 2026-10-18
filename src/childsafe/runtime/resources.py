"""Process peak memory only; this is not per-model or per-camera memory usage."""
import sys


def peak_memory_bytes():
    try:
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
                           ('peak_working_set','working_set','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
            kernel=ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype=wintypes.HANDLE
            psapi=ctypes.WinDLL('psapi'); psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
            counters=Counters(); counters.cb=ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
                return None
            return counters.peak_working_set
        import resource
        value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform=='darwin' else value*1024)
    except (AttributeError,ImportError,OSError):
        return None
