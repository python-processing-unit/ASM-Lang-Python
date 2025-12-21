"""ASM-Lang extension: real WASAPI binding (stdlib-only) via COM + ctypes.

This module intentionally has **no third-party dependencies**. It binds the
Windows Core Audio APIs (WASAPI / MMDevice API) directly using `ctypes`.

Scope (practical core):
- Device enumeration (default render/capture endpoint) and friendly names
- Endpoint master volume (scalar) and mute via IAudioEndpointVolume

The operators are prefixed with `WASAPI_` so it's obvious they belong to this
extension.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from extensions import ASMExtensionError


ASM_LANG_EXTENSION_API_VERSION = 1
ASM_LANG_EXTENSION_NAME = "wasapi"
ASM_LANG_EXTENSION_ASMODULE = True


if sys.platform != "win32":
    def asm_lang_register(api: Any) -> None:  # pragma: no cover
        raise ASMExtensionError("WASAPI extension is Windows-only")


# ---- Interpreter value helpers (no cyclic import at load time) ----

def _val_int(interpreter: Any, n: int) -> Any:
    return interpreter.Value(interpreter.TYPE_INT, int(n)) if hasattr(interpreter, "Value") else None


def _val_str(interpreter: Any, s: str) -> Any:
    return interpreter.Value(interpreter.TYPE_STR, str(s)) if hasattr(interpreter, "Value") else None


# In ASM-Lang, Value/TYPE_* live in interpreter.py. We can safely import them
# at runtime (operators are called after interpreter is loaded).
def _make_int(n: int) -> "Value":
    from interpreter import TYPE_INT, Value

    return Value(TYPE_INT, int(n))


def _make_str(s: str) -> "Value":
    from interpreter import TYPE_STR, Value

    return Value(TYPE_STR, str(s))


# ---- COM / WASAPI ctypes binding ----

HRESULT = ctypes.c_long
DWORD = ctypes.c_ulong
UINT = ctypes.c_uint
ULONG = ctypes.c_ulong
WORD = ctypes.c_ushort
LPVOID = ctypes.c_void_p
LPCWSTR = ctypes.c_wchar_p


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, s: str) -> "GUID":
        # Accept formats like {xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx} or without braces.
        ss = s.strip()
        if ss.startswith("{") and ss.endswith("}"):
            ss = ss[1:-1]
        parts = ss.split("-")
        if len(parts) != 5:
            raise ValueError(f"Invalid GUID: {s}")
        d1 = int(parts[0], 16)
        d2 = int(parts[1], 16)
        d3 = int(parts[2], 16)
        d4a = bytes.fromhex(parts[3])
        d4b = bytes.fromhex(parts[4])
        d4 = (ctypes.c_ubyte * 8)(*(list(d4a + d4b)))
        return cls(d1, d2, d3, d4)


def _hresult_ok(hr: int) -> bool:
    return hr >= 0


def _check_hr(hr: int, msg: str) -> None:
    if not _hresult_ok(int(hr)):
        raise RuntimeError(f"{msg} (HRESULT=0x{int(hr) & 0xFFFFFFFF:08X})")


COINIT_MULTITHREADED = 0x0
COINIT_APARTMENTTHREADED = 0x2
RPC_E_CHANGED_MODE = 0x80010106

CLSCTX_INPROC_SERVER = 0x1

EDataFlow_eRender = 0
EDataFlow_eCapture = 1

ERole_eConsole = 0

STGM_READ = 0x0

VT_LPWSTR = 31
VT_BSTR = 8


class PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", GUID), ("pid", DWORD)]


# PKEY_Device_FriendlyName
PKEY_Device_FriendlyName = PROPERTYKEY(GUID.from_string("{A45C254E-DF1C-4EFD-8020-67D146A850E0}"), DWORD(14))


class PROPVARIANT_UNION(ctypes.Union):
    _fields_ = [
        ("pwszVal", ctypes.c_wchar_p),
        ("bstrVal", ctypes.c_wchar_p),
        ("ulVal", DWORD),
        ("punkVal", LPVOID),
    ]


class PROPVARIANT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("vt", WORD),
        ("wReserved1", WORD),
        ("wReserved2", WORD),
        ("wReserved3", WORD),
        ("u", PROPVARIANT_UNION),
    ]


ole32 = ctypes.OleDLL("ole32")

ole32.CoInitializeEx.argtypes = [LPVOID, DWORD]
ole32.CoInitializeEx.restype = HRESULT

ole32.CoUninitialize.argtypes = []
ole32.CoUninitialize.restype = None

ole32.CoCreateInstance.argtypes = [ctypes.POINTER(GUID), LPVOID, DWORD, ctypes.POINTER(GUID), ctypes.POINTER(LPVOID)]
ole32.CoCreateInstance.restype = HRESULT

ole32.CoTaskMemFree.argtypes = [LPVOID]
ole32.CoTaskMemFree.restype = None

try:
    ole32.PropVariantClear.argtypes = [ctypes.POINTER(PROPVARIANT)]
    ole32.PropVariantClear.restype = HRESULT
    _prop_variant_clear = ole32.PropVariantClear
except AttributeError:  # pragma: no cover
    oleaut32 = ctypes.OleDLL("oleaut32")
    oleaut32.PropVariantClear.argtypes = [ctypes.POINTER(PROPVARIANT)]
    oleaut32.PropVariantClear.restype = HRESULT
    _prop_variant_clear = oleaut32.PropVariantClear


@dataclass
class ComObject:
    ptr: LPVOID

    def _vtable(self) -> ctypes.POINTER(LPVOID):
        return ctypes.cast(self.ptr, ctypes.POINTER(ctypes.POINTER(LPVOID))).contents

    def call(self, index: int, restype: Any, argtypes: List[Any], *args: Any) -> Any:
        vt = self._vtable()
        fn_ptr = vt[index]
        prototype = ctypes.WINFUNCTYPE(restype, LPVOID, *argtypes)
        fn = prototype(fn_ptr)
        return fn(self.ptr, *args)

    def add_ref(self) -> int:
        return int(self.call(1, ULONG, [],))

    def release(self) -> int:
        return int(self.call(2, ULONG, [],))


_tls = threading.local()


def _com_ensure_initialized() -> None:
    # Keep a per-thread init flag.
    if getattr(_tls, "com_init", False):
        return
    hr = int(ole32.CoInitializeEx(None, COINIT_MULTITHREADED))
    if hr == RPC_E_CHANGED_MODE:
        hr = int(ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED))
    _check_hr(hr, "CoInitializeEx failed")
    _tls.com_init = True


def _com_create_instance(clsid: GUID, iid: GUID) -> ComObject:
    _com_ensure_initialized()
    out = LPVOID()
    hr = int(ole32.CoCreateInstance(ctypes.byref(clsid), None, CLSCTX_INPROC_SERVER, ctypes.byref(iid), ctypes.byref(out)))
    _check_hr(hr, "CoCreateInstance failed")
    return ComObject(out)


# CLSIDs/IIDs
CLSID_MMDeviceEnumerator = GUID.from_string("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
IID_IMMDeviceEnumerator = GUID.from_string("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
IID_IMMDevice = GUID.from_string("{D666063F-1587-4E43-81F1-B948E807363F}")
IID_IPropertyStore = GUID.from_string("{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}")
IID_IAudioEndpointVolume = GUID.from_string("{5CDF2C82-841E-4546-9722-0CF74078229A}")


def _mmdevice_enumerator() -> ComObject:
    return _com_create_instance(CLSID_MMDeviceEnumerator, IID_IMMDeviceEnumerator)


def _enum_get_default_device(flow: int) -> ComObject:
    enum = _mmdevice_enumerator()
    try:
        out = LPVOID()
        # IMMDeviceEnumerator::GetDefaultAudioEndpoint at vtbl index 4
        hr = int(enum.call(4, HRESULT, [ctypes.c_int, ctypes.c_int, ctypes.POINTER(LPVOID)], int(flow), int(ERole_eConsole), ctypes.byref(out)))
        _check_hr(hr, "GetDefaultAudioEndpoint failed")
        return ComObject(out)
    finally:
        enum.release()


def _device_get_id(dev: ComObject) -> str:
    out = ctypes.c_wchar_p()
    # IMMDevice::GetId at vtbl index 5
    hr = int(dev.call(5, HRESULT, [ctypes.POINTER(ctypes.c_wchar_p)], ctypes.byref(out)))
    _check_hr(hr, "IMMDevice.GetId failed")
    try:
        return str(out.value or "")
    finally:
        if out.value is not None:
            ole32.CoTaskMemFree(ctypes.cast(out, LPVOID))


def _device_open_property_store(dev: ComObject) -> ComObject:
    out = LPVOID()
    # IMMDevice::OpenPropertyStore at vtbl index 4
    hr = int(dev.call(4, HRESULT, [DWORD, ctypes.POINTER(LPVOID)], DWORD(STGM_READ), ctypes.byref(out)))
    _check_hr(hr, "IMMDevice.OpenPropertyStore failed")
    return ComObject(out)


def _propstore_get_string(store: ComObject, key: PROPERTYKEY) -> str:
    pv = PROPVARIANT()
    try:
        # IPropertyStore::GetValue at vtbl index 5
        hr = int(store.call(5, HRESULT, [ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(PROPVARIANT)], ctypes.byref(key), ctypes.byref(pv)))
        _check_hr(hr, "IPropertyStore.GetValue failed")
        if int(pv.vt) == VT_LPWSTR:
            return str(pv.pwszVal or "")
        if int(pv.vt) == VT_BSTR:
            return str(pv.bstrVal or "")
        # Unknown type; return empty.
        return ""
    finally:
        _prop_variant_clear(ctypes.byref(pv))


def _device_get_friendly_name(dev: ComObject) -> str:
    store = _device_open_property_store(dev)
    try:
        return _propstore_get_string(store, PKEY_Device_FriendlyName)
    finally:
        store.release()


def _device_activate_endpoint_volume(dev: ComObject) -> ComObject:
    out = LPVOID()
    # IMMDevice::Activate at vtbl index 3
    # signature: Activate(REFIID iid, DWORD clsctx, PROPVARIANT* activationParams, void** ppInterface)
    hr = int(
        dev.call(
            3,
            HRESULT,
            [ctypes.POINTER(GUID), DWORD, LPVOID, ctypes.POINTER(LPVOID)],
            ctypes.byref(IID_IAudioEndpointVolume),
            DWORD(CLSCTX_INPROC_SERVER),
            None,
            ctypes.byref(out),
        )
    )
    _check_hr(hr, "IMMDevice.Activate(IAudioEndpointVolume) failed")
    return ComObject(out)


def _endpoint_get_master_scalar(vol: ComObject) -> float:
    out = ctypes.c_float()
    # IAudioEndpointVolume::GetMasterVolumeLevelScalar at vtbl index 9
    hr = int(vol.call(9, HRESULT, [ctypes.POINTER(ctypes.c_float)], ctypes.byref(out)))
    _check_hr(hr, "IAudioEndpointVolume.GetMasterVolumeLevelScalar failed")
    return float(out.value)


def _endpoint_set_master_scalar(vol: ComObject, scalar: float) -> None:
    # IAudioEndpointVolume::SetMasterVolumeLevelScalar at vtbl index 7
    # signature: (float level, LPCGUID eventContext)
    hr = int(vol.call(7, HRESULT, [ctypes.c_float, LPVOID], ctypes.c_float(float(scalar)), None))
    _check_hr(hr, "IAudioEndpointVolume.SetMasterVolumeLevelScalar failed")


def _endpoint_get_mute(vol: ComObject) -> int:
    out = ctypes.c_int()
    # IAudioEndpointVolume::GetMute at vtbl index 15
    hr = int(vol.call(15, HRESULT, [ctypes.POINTER(ctypes.c_int)], ctypes.byref(out)))
    _check_hr(hr, "IAudioEndpointVolume.GetMute failed")
    return 1 if int(out.value) != 0 else 0


def _endpoint_set_mute(vol: ComObject, mute: int) -> None:
    # IAudioEndpointVolume::SetMute at vtbl index 14
    hr = int(vol.call(14, HRESULT, [ctypes.c_int, LPVOID], int(1 if mute else 0), None))
    _check_hr(hr, "IAudioEndpointVolume.SetMute failed")


# ---- Handle registry ----

_next_handle = 1
_handles: Dict[int, Tuple[str, ComObject]] = {}
_handles_lock = threading.Lock()


def _new_handle(kind: str, obj: ComObject) -> int:
    global _next_handle
    with _handles_lock:
        h = _next_handle
        _next_handle += 1
        _handles[h] = (kind, obj)
        return h


def _get_handle(h: int, expected_kind: str) -> ComObject:
    with _handles_lock:
        item = _handles.get(int(h))
    if item is None:
        raise KeyError(f"Invalid WASAPI handle: {h}")
    kind, obj = item
    if kind != expected_kind:
        raise TypeError(f"WASAPI handle {h} is {kind}, expected {expected_kind}")
    return obj


def _release_handle(h: int) -> None:
    with _handles_lock:
        item = _handles.pop(int(h), None)
    if item is None:
        return
    _kind, obj = item
    try:
        obj.release()
    except Exception:
        # Best-effort.
        pass


# ---- Operator implementations (BuiltinImpl) ----

def _expect_int(interpreter: Any, value: Any, rule: str, loc: Any) -> int:
    return int(interpreter.builtins._expect_int(value, rule, loc))


def _wasapi_default_render_device(interpreter: Any, _args: List[Any], _arg_nodes: List[Any], _env: Any, _loc: Any) -> Any:
    dev = _enum_get_default_device(EDataFlow_eRender)
    h = _new_handle("IMMDevice", dev)
    return _make_int(h)


def _wasapi_default_capture_device(interpreter: Any, _args: List[Any], _arg_nodes: List[Any], _env: Any, _loc: Any) -> Any:
    dev = _enum_get_default_device(EDataFlow_eCapture)
    h = _new_handle("IMMDevice", dev)
    return _make_int(h)


def _wasapi_device_release(interpreter: Any, args: List[Any], _arg_nodes: List[Any], _env: Any, loc: Any) -> Any:
    h = _expect_int(interpreter, args[0], "WASAPI_DEVICE_RELEASE", loc)
    _release_handle(h)
    return _make_int(0)


def _wasapi_device_get_id(interpreter: Any, args: List[Any], _arg_nodes: List[Any], _env: Any, loc: Any) -> Any:
    h = _expect_int(interpreter, args[0], "WASAPI_DEVICE_GET_ID", loc)
    dev = _get_handle(h, "IMMDevice")
    return _make_str(_device_get_id(dev))


def _wasapi_device_get_name(interpreter: Any, args: List[Any], _arg_nodes: List[Any], _env: Any, loc: Any) -> Any:
    h = _expect_int(interpreter, args[0], "WASAPI_DEVICE_GET_NAME", loc)
    dev = _get_handle(h, "IMMDevice")
    return _make_str(_device_get_friendly_name(dev))


def _wasapi_device_get_endpoint_volume(interpreter: Any, args: List[Any], _arg_nodes: List[Any], _env: Any, loc: Any) -> Any:
    h = _expect_int(interpreter, args[0], "WASAPI_DEVICE_GET_ENDPOINT_VOLUME", loc)
    dev = _get_handle(h, "IMMDevice")
    vol = _device_activate_endpoint_volume(dev)
    vh = _new_handle("IAudioEndpointVolume", vol)
    return _make_int(vh)


def _wasapi_endpoint_volume_release(interpreter: Any, args: List[Any], _arg_nodes: List[Any], _env: Any, loc: Any) -> Any:
    h = _expect_int(interpreter, args[0], "WASAPI_ENDPOINT_VOLUME_RELEASE", loc)
    _release_handle(h)
    return _make_int(0)


def _wasapi_endpoint_get_master_volume(interpreter: Any, args: List[Any], _arg_nodes: List[Any], _env: Any, loc: Any) -> Any:
    h = _expect_int(interpreter, args[0], "WASAPI_ENDPOINT_GET_MASTER_VOLUME", loc)
    vol = _get_handle(h, "IAudioEndpointVolume")
    scalar = _endpoint_get_master_scalar(vol)
    pct = int(round(max(0.0, min(1.0, scalar)) * 100.0))
    return _make_int(pct)


def _wasapi_endpoint_set_master_volume(interpreter: Any, args: List[Any], _arg_nodes: List[Any], _env: Any, loc: Any) -> Any:
    h = _expect_int(interpreter, args[0], "WASAPI_ENDPOINT_SET_MASTER_VOLUME", loc)
    pct = _expect_int(interpreter, args[1], "WASAPI_ENDPOINT_SET_MASTER_VOLUME", loc)
    pct = max(0, min(100, int(pct)))
    vol = _get_handle(h, "IAudioEndpointVolume")
    _endpoint_set_master_scalar(vol, pct / 100.0)
    return _make_int(pct)


def _wasapi_endpoint_get_mute(interpreter: Any, args: List[Any], _arg_nodes: List[Any], _env: Any, loc: Any) -> Any:
    h = _expect_int(interpreter, args[0], "WASAPI_ENDPOINT_GET_MUTE", loc)
    vol = _get_handle(h, "IAudioEndpointVolume")
    return _make_int(_endpoint_get_mute(vol))


def _wasapi_endpoint_set_mute(interpreter: Any, args: List[Any], _arg_nodes: List[Any], _env: Any, loc: Any) -> Any:
    h = _expect_int(interpreter, args[0], "WASAPI_ENDPOINT_SET_MUTE", loc)
    mute = _expect_int(interpreter, args[1], "WASAPI_ENDPOINT_SET_MUTE", loc)
    vol = _get_handle(h, "IAudioEndpointVolume")
    _endpoint_set_mute(vol, 1 if mute else 0)
    return _make_int(1 if mute else 0)


# Convenience ops on the default render endpoint
def _wasapi_get_master_volume(interpreter: Any, _args: List[Any], _arg_nodes: List[Any], _env: Any, _loc: Any) -> Any:
    dev = _enum_get_default_device(EDataFlow_eRender)
    try:
        vol = _device_activate_endpoint_volume(dev)
        try:
            pct = int(round(max(0.0, min(1.0, _endpoint_get_master_scalar(vol))) * 100.0))
            return _make_int(pct)
        finally:
            vol.release()
    finally:
        dev.release()


def _wasapi_set_master_volume(interpreter: Any, args: List[Any], _arg_nodes: List[Any], _env: Any, loc: Any) -> Any:
    pct = _expect_int(interpreter, args[0], "WASAPI_SET_MASTER_VOLUME", loc)
    pct = max(0, min(100, int(pct)))
    dev = _enum_get_default_device(EDataFlow_eRender)
    try:
        vol = _device_activate_endpoint_volume(dev)
        try:
            _endpoint_set_master_scalar(vol, pct / 100.0)
            return _make_int(pct)
        finally:
            vol.release()
    finally:
        dev.release()


def _wasapi_get_mute(interpreter: Any, _args: List[Any], _arg_nodes: List[Any], _env: Any, _loc: Any) -> Any:
    dev = _enum_get_default_device(EDataFlow_eRender)
    try:
        vol = _device_activate_endpoint_volume(dev)
        try:
            return _make_int(_endpoint_get_mute(vol))
        finally:
            vol.release()
    finally:
        dev.release()


def _wasapi_set_mute(interpreter: Any, args: List[Any], _arg_nodes: List[Any], _env: Any, loc: Any) -> Any:
    mute = _expect_int(interpreter, args[0], "WASAPI_SET_MUTE", loc)
    dev = _enum_get_default_device(EDataFlow_eRender)
    try:
        vol = _device_activate_endpoint_volume(dev)
        try:
            _endpoint_set_mute(vol, 1 if mute else 0)
            return _make_int(1 if mute else 0)
        finally:
            vol.release()
    finally:
        dev.release()


def _wasapi_toggle_mute(interpreter: Any, _args: List[Any], _arg_nodes: List[Any], _env: Any, _loc: Any) -> Any:
    dev = _enum_get_default_device(EDataFlow_eRender)
    try:
        vol = _device_activate_endpoint_volume(dev)
        try:
            cur = _endpoint_get_mute(vol)
            new = 0 if cur else 1
            _endpoint_set_mute(vol, new)
            return _make_int(new)
        finally:
            vol.release()
    finally:
        dev.release()


def _wasapi_get_default_device_name(interpreter: Any, _args: List[Any], _arg_nodes: List[Any], _env: Any, _loc: Any) -> Any:
    dev = _enum_get_default_device(EDataFlow_eRender)
    try:
        return _make_str(_device_get_friendly_name(dev))
    finally:
        dev.release()


def asm_lang_register(api: Any) -> None:
    # Device handles
    api.register_operator("WASAPI_DEFAULT_RENDER_DEVICE", 0, 0, _wasapi_default_render_device, doc="Return handle to default render IMMDevice")
    api.register_operator("WASAPI_DEFAULT_CAPTURE_DEVICE", 0, 0, _wasapi_default_capture_device, doc="Return handle to default capture IMMDevice")
    api.register_operator("WASAPI_DEVICE_RELEASE", 1, 1, _wasapi_device_release, doc="Release an IMMDevice handle")
    api.register_operator("WASAPI_DEVICE_GET_ID", 1, 1, _wasapi_device_get_id, doc="Get device id string for IMMDevice handle")
    api.register_operator("WASAPI_DEVICE_GET_NAME", 1, 1, _wasapi_device_get_name, doc="Get friendly name for IMMDevice handle")
    api.register_operator("WASAPI_DEVICE_GET_ENDPOINT_VOLUME", 1, 1, _wasapi_device_get_endpoint_volume, doc="Activate IAudioEndpointVolume, return handle")

    # Endpoint volume handles
    api.register_operator("WASAPI_ENDPOINT_VOLUME_RELEASE", 1, 1, _wasapi_endpoint_volume_release, doc="Release an IAudioEndpointVolume handle")
    api.register_operator("WASAPI_ENDPOINT_GET_MASTER_VOLUME", 1, 1, _wasapi_endpoint_get_master_volume, doc="Get endpoint master volume percent (0..100)")
    api.register_operator("WASAPI_ENDPOINT_SET_MASTER_VOLUME", 2, 2, _wasapi_endpoint_set_master_volume, doc="Set endpoint master volume percent (0..100)")
    api.register_operator("WASAPI_ENDPOINT_GET_MUTE", 1, 1, _wasapi_endpoint_get_mute, doc="Get endpoint mute (0/1)")
    api.register_operator("WASAPI_ENDPOINT_SET_MUTE", 2, 2, _wasapi_endpoint_set_mute, doc="Set endpoint mute (0/1)")

    # Convenience ops on default render endpoint
    api.register_operator("WASAPI_GET_MASTER_VOLUME", 0, 0, _wasapi_get_master_volume, doc="Get default render master volume percent (0..100)")
    api.register_operator("WASAPI_SET_MASTER_VOLUME", 1, 1, _wasapi_set_master_volume, doc="Set default render master volume percent (0..100)")
    api.register_operator("WASAPI_GET_MUTE", 0, 0, _wasapi_get_mute, doc="Get default render mute (0/1)")
    api.register_operator("WASAPI_SET_MUTE", 1, 1, _wasapi_set_mute, doc="Set default render mute (0/1)")
    api.register_operator("WASAPI_TOGGLE_MUTE", 0, 0, _wasapi_toggle_mute, doc="Toggle default render mute")
    api.register_operator("WASAPI_GET_DEFAULT_DEVICE_NAME", 0, 0, _wasapi_get_default_device_name, doc="Get default render device friendly name")
