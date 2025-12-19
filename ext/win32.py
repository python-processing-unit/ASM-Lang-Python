"""ASM-Lang extension exposing common Win32 interactions.

This file provides a generic `WIN_CALL` operator for invoking arbitrary
functions from Win32 DLLs using `ctypes`, plus a few convenience helpers:

- WIN_CALL(lib, func, arg_types, ret_type, ...args)
  - `lib`: string, e.g. "kernel32" or "user32.dll"
  - `func`: string function name
  - `arg_types`: comma-separated type codes (I=Int, S=Str(wide), P=Ptr, B=Bytes, V=void)
  - `ret_type`: single type code (same codes as above; use V for void)
  - remaining args are passed to the function

- WIN_MESSAGE_BOX(text, title="") -> INT (MessageBox result)
- WIN_SLEEP(ms) -> INT (0)
- WIN_LAST_ERROR() -> INT (value of GetLastError)

This extension requires Windows; it will raise at import time on other OSes.
"""

from __future__ import annotations

import ctypes
import sys
from typing import List

from extensions import ASMExtensionError, ExtensionAPI

ASM_LANG_EXTENSION_NAME = "win32"
ASM_LANG_EXTENSION_API_VERSION = 1


def _ensure_windows() -> None:
    if sys.platform != "win32":
        raise ASMExtensionError("win32 extension only supported on Windows")


def _ctype_for(code: str):
    """Map single-letter type code to ctypes type/class.

    Supported codes:
    - I: integer (mapped to c_longlong)
    - S: wide string (c_wchar_p)
    - s: narrow bytes string (c_char_p)
    - P: void pointer (c_void_p)
    - B: bytes buffer (will pass pointer to buffer)
    - V: void / no value
    """
    mapping = {
        "I": ctypes.c_longlong,
        "S": ctypes.c_wchar_p,
        "s": ctypes.c_char_p,
        "P": ctypes.c_void_p,
        "B": ctypes.c_void_p,
        "V": None,
    }
    return mapping.get(code)


def _convert_arg(code: str, val):
    """Convert a Python value to a ctypes-compatible value based on type code."""
    if code == "I":
        return int(val)
    if code == "S":
        # Accept Python string; convert to wide char pointer
        if val is None:
            return None
        return str(val)
    if code == "s":
        if val is None:
            return None
        if isinstance(val, bytes):
            return val
        return str(val).encode("utf-8")
    if code == "P":
        # accept integer addresses or None
        if val is None:
            return None
        return int(val)
    if code == "B":
        # bytes buffer - create a c_char buffer and return pointer
        if val is None:
            return None
        if isinstance(val, (bytes, bytearray)):
            buf = ctypes.create_string_buffer(bytes(val))
            return ctypes.cast(buf, ctypes.c_void_p)
        raise ASMExtensionError("B argument must be bytes or bytearray")
    raise ASMExtensionError(f"Unsupported arg type code: {code}")


def asm_lang_register(ext: ExtensionAPI) -> None:
    """Register operators into the runtime services."""
    _ensure_windows()

    def _win_call(interpreter, args: List[object], arg_nodes, env, location):
        # Lazy imports to avoid import-time surprises
        from interpreter import Value, TYPE_INT, TYPE_STR
        if len(args) < 4:
            raise ASMExtensionError("WIN_CALL requires at least 4 arguments")

        lib_name = args[0].value if hasattr(args[0], "value") else args[0]
        func_name = args[1].value if hasattr(args[1], "value") else args[1]
        arg_types_raw = args[2].value if hasattr(args[2], "value") else args[2]
        ret_type = args[3].value if hasattr(args[3], "value") else args[3]

        if not isinstance(lib_name, str):
            raise ASMExtensionError("WIN_CALL: lib must be a string")
        if not isinstance(func_name, str):
            raise ASMExtensionError("WIN_CALL: func must be a string")

        # Normalize library name
        dll_name = lib_name if lib_name.lower().endswith(".dll") else lib_name + ".dll"
        try:
            dll = ctypes.WinDLL(dll_name)
        except Exception:
            # fallback to windll attribute style (kernel32, user32, etc.)
            try:
                dll = getattr(ctypes.windll, lib_name)
            except Exception as exc:
                raise ASMExtensionError(f"Failed to load DLL {dll_name}: {exc}")

        try:
            func = getattr(dll, func_name)
        except AttributeError as exc:
            raise ASMExtensionError(f"Function {func_name} not found in {dll_name}: {exc}")

        # Parse arg types
        arg_codes = [c.strip() for c in str(arg_types_raw).split(",") if c.strip()]

        py_args = []
        for i, code in enumerate(arg_codes):
            if 4 + i >= len(args) + 4:  # guard; not strictly necessary
                break
            # arguments beyond the 4th are at index 4+i
            raw = args[4 + i].value if hasattr(args[4 + i], "value") else args[4 + i]
            py_args.append(_convert_arg(code, raw))

        # Set ctypes argtypes
        c_argtypes = []
        for code in arg_codes:
            ct = _ctype_for(code)
            if ct is None:
                c_argtypes.append(ct)
            else:
                c_argtypes.append(ct)
        try:
            func.argtypes = c_argtypes
        except Exception:
            # some functions don't like argtypes set; ignore
            pass

        # Set restype
        ret_code = str(ret_type)
        rest = _ctype_for(ret_code)
        if rest is None:
            func.restype = None
        else:
            func.restype = rest

        # Call
        result = func(*py_args)

        # Marshal result
        if rest is None:
            return Value(TYPE_INT, 0)
        if rest is ctypes.c_wchar_p:
            return Value(TYPE_STR, result if result is not None else "")
        if rest is ctypes.c_char_p:
            return Value(TYPE_STR, result.decode("utf-8") if result else "")
        if rest is ctypes.c_void_p:
            return Value(TYPE_INT, int(result) if result is not None else 0)
        # treat numeric c_longlong etc. as INT
        try:
            return Value(TYPE_INT, int(result))
        except Exception:
            return Value(TYPE_INT, 0)

    def _message_box(interpreter, args, arg_nodes, env, location):
        from interpreter import Value, TYPE_INT, TYPE_STR
        # Args: text[, title]
        if not (1 <= len(args) <= 2):
            raise ASMExtensionError("WIN_MESSAGE_BOX requires 1 or 2 args")
        text = args[0].value if hasattr(args[0], "value") else args[0]
        title = args[1].value if len(args) == 2 and hasattr(args[1], "value") else (args[1] if len(args) == 2 else "")
        user32 = ctypes.WinDLL("user32.dll")
        user32.MessageBoxW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        user32.MessageBoxW.restype = ctypes.c_int
        res = user32.MessageBoxW(None, str(text), str(title), 0)
        return Value(TYPE_INT, int(res))

    def _sleep(interpreter, args, arg_nodes, env, location):
        from interpreter import Value, TYPE_INT
        if len(args) < 1:
            raise ASMExtensionError("WIN_SLEEP requires 1 arg (milliseconds)")
        ms = int(args[0].value if hasattr(args[0], "value") else args[0])
        kernel32 = ctypes.WinDLL("kernel32.dll")
        kernel32.Sleep.argtypes = [ctypes.c_uint]
        kernel32.Sleep.restype = None
        kernel32.Sleep(ms)
        return Value(TYPE_INT, 0)

    def _last_error(interpreter, args, arg_nodes, env, location):
        from interpreter import Value, TYPE_INT
        kernel32 = ctypes.WinDLL("kernel32.dll")
        kernel32.GetLastError.argtypes = []
        kernel32.GetLastError.restype = ctypes.c_ulong
        return Value(TYPE_INT, int(kernel32.GetLastError()))

    # Register operators
    ext.register_operator("WIN_CALL", 4, None, _win_call, doc="Call a Win32 API function")
    ext.register_operator("WIN_MESSAGE_BOX", 1, 2, _message_box, doc="Show a simple Windows message box")
    ext.register_operator("WIN_SLEEP", 1, 1, _sleep, doc="Sleep (ms)")
    ext.register_operator("WIN_LAST_ERROR", 0, 0, _last_error, doc="GetLastError")

    # Convenience helpers for common Win32 tasks
    def _load_library(interpreter, args, arg_nodes, env, location):
        from interpreter import Value, TYPE_INT
        if len(args) < 1:
            raise ASMExtensionError("WIN_LOAD_LIBRARY requires 1 arg")
        name = args[0].value if hasattr(args[0], "value") else args[0]
        dll_name = name if name.lower().endswith(".dll") else name + ".dll"
        try:
            h = ctypes.WinDLL(dll_name)
            # return Python id of dll object as handle (opaque)
            return Value(TYPE_INT, id(h))
        except Exception as exc:
            raise ASMExtensionError(f"LoadLibrary failed: {exc}")

    def _free_library(interpreter, args, arg_nodes, env, location):
        from interpreter import Value, TYPE_INT
        if len(args) < 1:
            raise ASMExtensionError("WIN_FREE_LIBRARY requires 1 arg (handle id)")
        # We only have the id; can't reliably free Python object. Provide FreeLibrary for handles returned by LoadLibrary via kernel32.
        handle = int(args[0].value if hasattr(args[0], "value") else args[0])
        # try calling kernel32.FreeLibrary on handle if it's a real HMODULE
        kernel32 = ctypes.WinDLL("kernel32.dll")
        try:
            kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
            kernel32.FreeLibrary.restype = ctypes.c_int
            res = kernel32.FreeLibrary(ctypes.c_void_p(handle))
            return Value(TYPE_INT, int(res))
        except Exception as exc:
            raise ASMExtensionError(f"FreeLibrary failed: {exc}")

    def _get_proc_address(interpreter, args, arg_nodes, env, location):
        from interpreter import Value, TYPE_INT
        if len(args) < 2:
            raise ASMExtensionError("WIN_GET_PROC_ADDRESS requires 2 args (module_handle, proc_name)")
        mod = int(args[0].value if hasattr(args[0], "value") else args[0])
        name = args[1].value if hasattr(args[1], "value") else args[1]
        kernel32 = ctypes.WinDLL("kernel32.dll")
        kernel32.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        kernel32.GetProcAddress.restype = ctypes.c_void_p
        try:
            res = kernel32.GetProcAddress(ctypes.c_void_p(mod), str(name).encode("ascii"))
            return Value(TYPE_INT, int(res) if res is not None else 0)
        except Exception as exc:
            raise ASMExtensionError(f"GetProcAddress failed: {exc}")

    def _create_file(interpreter, args, arg_nodes, env, location):
        from interpreter import Value, TYPE_INT
        if len(args) < 5:
            raise ASMExtensionError("WIN_CREATE_FILE requires 5 args (path, access, share, creation, flags)")
        path = args[0].value if hasattr(args[0], "value") else args[0]
        access = int(args[1].value if hasattr(args[1], "value") else args[1])
        share = int(args[2].value if hasattr(args[2], "value") else args[2])
        creation = int(args[3].value if hasattr(args[3], "value") else args[3])
        flags = int(args[4].value if hasattr(args[4], "value") else args[4])
        kernel32 = ctypes.WinDLL("kernel32.dll")
        kernel32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
        kernel32.CreateFileW.restype = ctypes.c_void_p
        h = kernel32.CreateFileW(str(path), ctypes.c_uint(access), ctypes.c_uint(share), None, ctypes.c_uint(creation), ctypes.c_uint(flags), None)
        return Value(TYPE_INT, int(h) if h is not None else 0)

    def _read_file(interpreter, args, arg_nodes, env, location):
        from interpreter import Value, TYPE_STR, TYPE_INT
        if len(args) < 2:
            raise ASMExtensionError("WIN_READ_FILE requires 2 args (handle, length)")
        handle = int(args[0].value if hasattr(args[0], "value") else args[0])
        length = int(args[1].value if hasattr(args[1], "value") else args[1])
        kernel32 = ctypes.WinDLL("kernel32.dll")
        buf = ctypes.create_string_buffer(length)
        read = ctypes.c_uint(0)
        res = kernel32.ReadFile(ctypes.c_void_p(handle), buf, ctypes.c_uint(length), ctypes.byref(read), None)
        if res == 0:
            # error
            return Value(TYPE_STR, "")
        data = buf.raw[: read.value]
        # return as latin1 string to preserve bytes
        return Value(TYPE_STR, data.decode("latin1"))

    def _write_file(interpreter, args, arg_nodes, env, location):
        from interpreter import Value, TYPE_INT
        if len(args) < 2:
            raise ASMExtensionError("WIN_WRITE_FILE requires 2 args (handle, data)")
        handle = int(args[0].value if hasattr(args[0], "value") else args[0])
        data = args[1].value if hasattr(args[1], "value") else args[1]
        if isinstance(data, str):
            b = data.encode("latin1")
        else:
            b = bytes(data)
        kernel32 = ctypes.WinDLL("kernel32.dll")
        written = ctypes.c_uint(0)
        res = kernel32.WriteFile(ctypes.c_void_p(handle), ctypes.c_char_p(b), ctypes.c_uint(len(b)), ctypes.byref(written), None)
        if res == 0:
            return Value(TYPE_INT, 0)
        return Value(TYPE_INT, int(written.value))

    def _close_handle(interpreter, args, arg_nodes, env, location):
        from interpreter import Value, TYPE_INT
        if len(args) < 1:
            raise ASMExtensionError("WIN_CLOSE_HANDLE requires 1 arg (handle)")
        handle = int(args[0].value if hasattr(args[0], "value") else args[0])
        kernel32 = ctypes.WinDLL("kernel32.dll")
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        res = kernel32.CloseHandle(ctypes.c_void_p(handle))
        return Value(TYPE_INT, int(res))

    def _virtual_alloc(interpreter, args, arg_nodes, env, location):
        from interpreter import Value, TYPE_INT
        if len(args) < 3:
            raise ASMExtensionError("WIN_VIRTUAL_ALLOC requires 3 args (size, alloc_type, protect)")
        size = int(args[0].value if hasattr(args[0], "value") else args[0])
        alloc_type = int(args[1].value if hasattr(args[1], "value") else args[1])
        protect = int(args[2].value if hasattr(args[2], "value") else args[2])
        kernel32 = ctypes.WinDLL("kernel32.dll")
        kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_uint]
        kernel32.VirtualAlloc.restype = ctypes.c_void_p
        addr = kernel32.VirtualAlloc(None, ctypes.c_size_t(size), ctypes.c_uint(alloc_type), ctypes.c_uint(protect))
        return Value(TYPE_INT, int(addr) if addr is not None else 0)

    def _virtual_free(interpreter, args, arg_nodes, env, location):
        from interpreter import Value, TYPE_INT
        if len(args) < 3:
            raise ASMExtensionError("WIN_VIRTUAL_FREE requires 3 args (address, size, free_type)")
        addr = int(args[0].value if hasattr(args[0], "value") else args[0])
        size = int(args[1].value if hasattr(args[1], "value") else args[1])
        free_type = int(args[2].value if hasattr(args[2], "value") else args[2])
        kernel32 = ctypes.WinDLL("kernel32.dll")
        kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        kernel32.VirtualFree.restype = ctypes.c_int
        res = kernel32.VirtualFree(ctypes.c_void_p(addr), ctypes.c_size_t(size), ctypes.c_uint(free_type))
        return Value(TYPE_INT, int(res))

    def _format_message(interpreter, args, arg_nodes, env, location):
        from interpreter import Value, TYPE_STR, TYPE_INT
        if len(args) < 1:
            code = int(ctypes.WinDLL("kernel32.dll").GetLastError())
        else:
            code = int(args[0].value if hasattr(args[0], "value") else args[0])
        FORMAT_MESSAGE_FROM_SYSTEM = 0x00001000
        buf = ctypes.create_unicode_buffer(1024)
        windll = ctypes.WinDLL("kernel32.dll")
        windll.FormatMessageW.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_void_p]
        windll.FormatMessageW.restype = ctypes.c_uint
        windll.FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM, None, ctypes.c_uint(code), 0, buf, ctypes.c_uint(len(buf)), None)
        return Value(TYPE_STR, buf.value.rstrip())

    ext.register_operator("WIN_LOAD_LIBRARY", 1, 1, _load_library, doc="Load a DLL and return handle id")
    ext.register_operator("WIN_FREE_LIBRARY", 1, 1, _free_library, doc="Free a loaded library by handle")
    ext.register_operator("WIN_GET_PROC_ADDRESS", 2, 2, _get_proc_address, doc="GetProcAddress")
    ext.register_operator("WIN_CREATE_FILE", 5, 5, _create_file, doc="CreateFileW wrapper")
    ext.register_operator("WIN_READ_FILE", 2, 2, _read_file, doc="ReadFile wrapper")
    ext.register_operator("WIN_WRITE_FILE", 2, 2, _write_file, doc="WriteFile wrapper")
    ext.register_operator("WIN_CLOSE_HANDLE", 1, 1, _close_handle, doc="CloseHandle wrapper")
    ext.register_operator("WIN_VIRTUAL_ALLOC", 3, 3, _virtual_alloc, doc="VirtualAlloc wrapper")
    ext.register_operator("WIN_VIRTUAL_FREE", 3, 3, _virtual_free, doc="VirtualFree wrapper")
    ext.register_operator("WIN_FORMAT_MESSAGE", 0, 1, _format_message, doc="FormatMessage for error codes")
