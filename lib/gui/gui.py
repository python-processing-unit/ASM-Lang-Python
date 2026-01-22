"""ASM-Lang extension: lightweight GUI windows for image tensors.

This module exposes three operators:
- GUI_CREATE_WINDOW([type, width, height, title, scale_to_fit]) -> INT handle
- GUI_SHOW_IMAGE(handle, TNS image) -> INT (1 on success)
- GUI_CLOSE_WINDOW(handle) -> INT (1 on success)

Images are expected to be ASM-Lang "image"-compatible tensors shaped
[width][height][4] with INT channels (RGBA). The default window type is a
resizable, scaled window with the standard title-bar buttons enabled.
"""

from __future__ import annotations

import base64
import tempfile
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

import tkinter as tk

from extensions import ASMExtensionError, ExtensionAPI

ASM_LANG_EXTENSION_NAME = "gui"
ASM_LANG_EXTENSION_API_VERSION = 1
ASM_LANG_EXTENSION_ASMODULE = True


@dataclass
class _GuiWindow:
    top: "tk.Toplevel"
    label: "tk.Label"
    kind: str
    scale_to_fit: bool
    last_image: Optional[np.ndarray] = None
    last_size: Tuple[int, int] = (0, 0)
    photo: Optional["tk.PhotoImage"] = None


@dataclass
class _GuiState:
    root: "tk.Tk"
    windows: Dict[int, _GuiWindow]
    next_id: int = 1


def _state(interpreter) -> _GuiState:
    st = getattr(interpreter, "_gui_state", None)
    if st is None:
        root = tk.Tk()
        root.withdraw()
        st = _GuiState(root=root, windows={})
        setattr(interpreter, "_gui_state", st)
    return st


def _expect_type_str(interpreter, arg, default: str, rule: str, location) -> str:
    if arg is None:
        return default
    return interpreter._expect_str(arg, rule, location)


def _expect_type_int(interpreter, arg, default: int, rule: str, location) -> int:
    if arg is None:
        return default
    return interpreter._expect_int(arg, rule, location)


def _tns_to_int_image(interpreter, value, rule: str, location) -> np.ndarray:
    from interpreter import ASMRuntimeError

    tns = interpreter._expect_tns(value, rule, location)
    if len(tns.shape) != 3 or tns.shape[2] not in (3, 4):
        raise ASMRuntimeError(
            f"{rule} expects an image tensor shaped [w][h][3|4]",
            location=location,
            rewrite_rule=rule,
        )
    interpreter.builtins._ensure_tensor_ints(tns, rule, location)
    arr_view = tns.data.reshape(tuple(tns.shape))
    ints = np.fromiter((int(v.value) for v in arr_view.flat), dtype=np.int32, count=int(np.prod(tns.shape)))
    img = ints.reshape((tns.shape[0], tns.shape[1], tns.shape[2]))
    if img.shape[2] == 3:
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.int32)
        img = np.concatenate([img, alpha], axis=2)
    return img


def _resize_int_image(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    src_w, src_h, _ = img.shape
    if target_w <= 0 or target_h <= 0:
        return img
    if target_w == src_w and target_h == src_h:
        return img.copy()
    xs = ((np.arange(target_w, dtype=float) + 0.5) * (src_w / float(target_w)) - 0.5).round().astype(int)
    ys = ((np.arange(target_h, dtype=float) + 0.5) * (src_h / float(target_h)) - 0.5).round().astype(int)
    xs = np.clip(xs, 0, src_w - 1)
    ys = np.clip(ys, 0, src_h - 1)
    xg, yg = np.meshgrid(xs, ys, indexing="ij")
    return img[xg, yg].astype(np.int32)


def _int_image_to_photo(img: np.ndarray) -> "tk.PhotoImage":
    w, h, _ = img.shape
    alpha = np.clip(img[:, :, 3], 0, 255).astype(np.int32)
    rgb = np.clip(img[:, :, :3], 0, 255).astype(np.int32)
    premul = (rgb * alpha[..., None] + (255 - alpha)[..., None] * 0) // 255
    premul = np.clip(premul, 0, 255).astype(np.uint8)
    buf = bytearray()
    # Tk PhotoImage expects row-major order (y,x)
    for y in range(h):
        row = premul[:, y, :]
        buf.extend(row.flatten())

    header = f"P6\n{w} {h}\n255\n".encode("ascii")

    # Try a base64-in-memory path first (faster than temp files). Some
    # Tcl/Tk builds reject certain binary payloads via `data=`, so fall
    # back to the temp-file approach if that fails.
    try:
        payload = header + bytes(buf)
        b64 = base64.b64encode(payload).decode("ascii")
        try:
            photo = tk.PhotoImage(data=b64)
            return photo
        except Exception:
            # Fall back to writing a temporary PPM file
            pass

        fd, path = tempfile.mkstemp(suffix=".ppm")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(header)
                handle.write(bytes(buf))
            photo = tk.PhotoImage(file=path)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
        return photo
    except Exception as exc:
        # Surface a clearer error to the caller
        raise RuntimeError(f"failed to convert image to PhotoImage: {exc}")


def _render_window(win: _GuiWindow) -> None:
    if win.last_image is None:
        return
    win.top.update_idletasks()
    target_w = win.top.winfo_width()
    target_h = win.top.winfo_height()
    if target_w <= 1 or target_h <= 1:
        target_w, target_h, _ = win.last_image.shape
    else:
        # Keep a small padding margin for borders; best-effort only.
        target_w = max(1, target_w)
        target_h = max(1, target_h)
    if win.scale_to_fit:
        img = _resize_int_image(win.last_image, target_w, target_h)
    else:
        img = win.last_image
    photo = _int_image_to_photo(img)
    # Keep a reference to prevent GC and configure the label
    win.photo = photo
    win.label.configure(image=photo)
    # Force the window widgets to process pending geometry/paint events
    try:
        win.label.update_idletasks()
        win.top.update_idletasks()
        win.top.update()
    except Exception:
        pass
    win.last_size = (target_w, target_h)


def _close_window(state: _GuiState, wid: int) -> None:
    win = state.windows.pop(wid, None)
    if win is None:
        return
    try:
        win.top.destroy()
    finally:
        if not state.windows:
            state.root.withdraw()


def _on_resize(state: _GuiState, wid: int, _event=None) -> None:
    win = state.windows.get(wid)
    if win is None or win.last_image is None or not win.scale_to_fit:
        return
    current = (win.top.winfo_width(), win.top.winfo_height())
    if current == win.last_size:
        return
    _render_window(win)


def _apply_window_kind(top: "tk.Toplevel", kind: str, location) -> Tuple[bool, str]:
    from interpreter import ASMRuntimeError

    normalized = kind.strip().lower() or "scaled"
    scale_to_fit = True
    if normalized in ("scaled", "resizable"):
        top.resizable(True, True)
    elif normalized == "fixed":
        top.resizable(False, False)
        scale_to_fit = False
    elif normalized == "fullscreen":
        top.attributes("-fullscreen", True)
    elif normalized == "borderless":
        top.overrideredirect(True)
    else:
        raise ASMRuntimeError(
            f"GUI_CREATE_WINDOW: unknown window type '{kind}'",
            location=location,
            rewrite_rule="GUI_CREATE_WINDOW",
        )
    return scale_to_fit, normalized


def _op_create_window(interpreter, args, _arg_nodes, _env, location):
    from interpreter import TYPE_INT, Value

    st = _state(interpreter)
    kind = _expect_type_str(interpreter, args[0] if len(args) >= 1 else None, "scaled", "GUI_CREATE_WINDOW", location)
    width = _expect_type_int(interpreter, args[1] if len(args) >= 2 else None, 640, "GUI_CREATE_WINDOW", location)
    height = _expect_type_int(interpreter, args[2] if len(args) >= 3 else None, 480, "GUI_CREATE_WINDOW", location)
    title = _expect_type_str(interpreter, args[3] if len(args) >= 4 else None, "ASM-Lang GUI", "GUI_CREATE_WINDOW", location)
    scale_provided = len(args) >= 5
    scale_flag = _expect_type_int(interpreter, args[4] if scale_provided else None, 1, "GUI_CREATE_WINDOW", location)

    top = tk.Toplevel(st.root)
    top.title(title)
    top.geometry(f"{max(1, width)}x{max(1, height)}")

    scale_to_fit_kind, norm_kind = _apply_window_kind(top, kind, location)
    scale_to_fit = bool(scale_flag) if scale_provided else scale_to_fit_kind

    label = tk.Label(top, bd=0)
    label.pack(fill="both", expand=True)

    wid = st.next_id
    st.next_id += 1
    win = _GuiWindow(top=top, label=label, kind=norm_kind, scale_to_fit=scale_to_fit)
    st.windows[wid] = win

    def _on_close(wid_to_close=wid):
        _close_window(st, wid_to_close)

    top.protocol("WM_DELETE_WINDOW", _on_close)
    top.bind("<Configure>", lambda event, wid=wid: _on_resize(st, wid, event))
    st.root.update_idletasks()
    return Value(TYPE_INT, int(wid))


def _op_show_image(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError, TYPE_INT, Value

    if len(args) < 2:
        raise ASMRuntimeError("GUI_SHOW_IMAGE expects 2 arguments", location=location, rewrite_rule="GUI_SHOW_IMAGE")
    wid = interpreter._expect_int(args[0], "GUI_SHOW_IMAGE", location)
    st = _state(interpreter)
    win = st.windows.get(wid)
    if win is None:
        raise ASMRuntimeError("GUI_SHOW_IMAGE: invalid window handle", location=location, rewrite_rule="GUI_SHOW_IMAGE")

    try:
        img = _tns_to_int_image(interpreter, args[1], "GUI_SHOW_IMAGE", location)
        win.last_image = img
        _render_window(win)
        # Ensure the root processes events so the window redraws promptly
        try:
            st.root.update_idletasks()
            st.root.update()
        except Exception:
            pass
    except ASMRuntimeError:
        raise
    except Exception as exc:
        raise ASMRuntimeError(f"GUI_SHOW_IMAGE failed: {exc}", location=location, rewrite_rule="GUI_SHOW_IMAGE")
    return Value(TYPE_INT, 1)


def _op_close_window(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError, TYPE_INT, Value

    if len(args) < 1:
        raise ASMRuntimeError("GUI_CLOSE_WINDOW expects 1 argument", location=location, rewrite_rule="GUI_CLOSE_WINDOW")
    wid = interpreter._expect_int(args[0], "GUI_CLOSE_WINDOW", location)
    st = _state(interpreter)
    if wid not in st.windows:
        raise ASMRuntimeError("GUI_CLOSE_WINDOW: invalid window handle", location=location, rewrite_rule="GUI_CLOSE_WINDOW")
    _close_window(st, wid)
    st.root.update_idletasks()
    return Value(TYPE_INT, 1)


def _op_minimize(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError, TYPE_INT, Value

    if len(args) < 1:
        raise ASMRuntimeError("GUI_MINIMIZE expects 1 argument", location=location, rewrite_rule="GUI_MINIMIZE")
    wid = interpreter._expect_int(args[0], "GUI_MINIMIZE", location)
    st = _state(interpreter)
    win = st.windows.get(wid)
    if win is None:
        raise ASMRuntimeError("GUI_MINIMIZE: invalid window handle", location=location, rewrite_rule="GUI_MINIMIZE")
    try:
        win.top.iconify()
        st.root.update_idletasks()
    except Exception:
        raise ASMRuntimeError("GUI_MINIMIZE failed", location=location, rewrite_rule="GUI_MINIMIZE")
    return Value(TYPE_INT, 1)


def _op_maximize(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError, TYPE_INT, Value

    if len(args) < 1:
        raise ASMRuntimeError("GUI_MAXIMIZE expects 1 argument", location=location, rewrite_rule="GUI_MAXIMIZE")
    wid = interpreter._expect_int(args[0], "GUI_MAXIMIZE", location)
    st = _state(interpreter)
    win = st.windows.get(wid)
    if win is None:
        raise ASMRuntimeError("GUI_MAXIMIZE: invalid window handle", location=location, rewrite_rule="GUI_MAXIMIZE")
    try:
        # Prefer platform-friendly state call; fall back silently if unsupported
        try:
            win.top.state("zoomed")
        except Exception:
            try:
                win.top.attributes("-zoomed", True)
            except Exception:
                # best-effort: resize to screen size
                w = st.root.winfo_screenwidth()
                h = st.root.winfo_screenheight()
                win.top.geometry(f"{w}x{h}")
        st.root.update_idletasks()
    except Exception:
        raise ASMRuntimeError("GUI_MAXIMIZE failed", location=location, rewrite_rule="GUI_MAXIMIZE")
    return Value(TYPE_INT, 1)


def _op_to_front(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError, TYPE_INT, Value

    if len(args) < 1:
        raise ASMRuntimeError("GUI_TO_FRONT expects 1 argument", location=location, rewrite_rule="GUI_TO_FRONT")
    wid = interpreter._expect_int(args[0], "GUI_TO_FRONT", location)
    st = _state(interpreter)
    win = st.windows.get(wid)
    if win is None:
        raise ASMRuntimeError("GUI_TO_FRONT: invalid window handle", location=location, rewrite_rule="GUI_TO_FRONT")
    try:
        win.top.lift()
        # Some platforms may require also setting attributes
        try:
            win.top.attributes("-topmost", True)
            win.top.attributes("-topmost", False)
        except Exception:
            pass
        st.root.update_idletasks()
    except Exception:
        raise ASMRuntimeError("GUI_TO_FRONT failed", location=location, rewrite_rule="GUI_TO_FRONT")
    return Value(TYPE_INT, 1)


def _op_to_back(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError, TYPE_INT, Value

    if len(args) < 1:
        raise ASMRuntimeError("GUI_TO_BACK expects 1 argument", location=location, rewrite_rule="GUI_TO_BACK")
    wid = interpreter._expect_int(args[0], "GUI_TO_BACK", location)
    st = _state(interpreter)
    win = st.windows.get(wid)
    if win is None:
        raise ASMRuntimeError("GUI_TO_BACK: invalid window handle", location=location, rewrite_rule="GUI_TO_BACK")
    try:
        win.top.lower()
        st.root.update_idletasks()
    except Exception:
        raise ASMRuntimeError("GUI_TO_BACK failed", location=location, rewrite_rule="GUI_TO_BACK")
    return Value(TYPE_INT, 1)


def _op_screen(interpreter, args, _arg_nodes, _env, location):
    from interpreter import TYPE_TNS, TYPE_INT, Value, Tensor, ASMRuntimeError

    st = _state(interpreter)
    try:
        # Query the underlying Tk root for the screen size
        width = int(st.root.winfo_screenwidth())
        height = int(st.root.winfo_screenheight())
    except Exception:
        raise ASMRuntimeError("SCREEN: failed to query screen size", location=location, rewrite_rule="SCREEN")

    # Build a 1-D tensor [width, height] with INT channel values
    data = np.array([Value(TYPE_INT, int(width)), Value(TYPE_INT, int(height))], dtype=object)
    return Value(TYPE_TNS, Tensor(shape=[2], data=data))


def _op_window(interpreter, args, _arg_nodes, _env, location):
    from interpreter import TYPE_TNS, TYPE_INT, Value, Tensor, ASMRuntimeError

    if len(args) < 1:
        raise ASMRuntimeError("WINDOW expects 1 argument", location=location, rewrite_rule="WINDOW")
    wid = interpreter._expect_int(args[0], "WINDOW", location)
    st = _state(interpreter)
    win = st.windows.get(wid)
    if win is None:
        raise ASMRuntimeError("WINDOW: invalid window handle", location=location, rewrite_rule="WINDOW")
    try:
        # Ensure geometry is up-to-date
        win.top.update_idletasks()
        width = int(win.top.winfo_width())
        height = int(win.top.winfo_height())
    except Exception:
        raise ASMRuntimeError("WINDOW: failed to query window size", location=location, rewrite_rule="WINDOW")

    data = np.array([Value(TYPE_INT, int(width)), Value(TYPE_INT, int(height))], dtype=object)
    return Value(TYPE_TNS, Tensor(shape=[2], data=data))


def _op_screenshot(interpreter, args, _arg_nodes, _env, location):
    from interpreter import TYPE_TNS, TYPE_INT, Value, Tensor, ASMRuntimeError
    import sys

    # Only implement a reliable screenshot on Windows using GDI.
    if sys.platform != "win32":
        raise ASMRuntimeError("SCREENSHOT is only supported on Windows", location=location, rewrite_rule="SCREENSHOT")

    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        # screen dimensions
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))

        hdc_screen = user32.GetDC(0)
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        hbmp = gdi32.CreateCompatibleBitmap(hdc_screen, width, height)
        old = gdi32.SelectObject(hdc_mem, hbmp)
        SRCCOPY = 0x00CC0020
        if gdi32.BitBlt(hdc_mem, 0, 0, width, height, hdc_screen, 0, 0, SRCCOPY) == 0:
            raise RuntimeError("BitBlt failed")

        # Prepare BITMAPINFO for 32bpp BGRA
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        # Use negative height to request a top-down DIB
        bmi.bmiHeader.biHeight = -height
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0  # BI_RGB

        buf_size = width * height * 4
        buf = (ctypes.c_ubyte * buf_size)()
        bits = gdi32.GetDIBits(hdc_mem, hbmp, 0, height, ctypes.byref(buf), ctypes.byref(bmi), 0)
        if bits == 0:
            raise RuntimeError("GetDIBits failed")

        # Cleanup GDI objects
        gdi32.SelectObject(hdc_mem, old)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)

        # buf contains BGRA bytes in row-major order (top-down)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
        # Convert to interpreter tensor layout: [width, height, channel] and RGBA order
        rgba = arr[:, :, [2, 1, 0, 3]]
        transposed = np.transpose(rgba, (1, 0, 2)).copy()

        # Wrap into Value objects
        _Val = Value
        _TINT = TYPE_INT
        flat = np.array([_Val(_TINT, int(v)) for v in transposed.flatten()], dtype=object)
        return Value(TYPE_TNS, Tensor(shape=[int(width), int(height), 4], data=flat))

    except ASMRuntimeError:
        raise
    except Exception as exc:
        raise ASMRuntimeError(f"SCREENSHOT failed: {exc}", location=location, rewrite_rule="SCREENSHOT")


def asm_lang_register(ext: ExtensionAPI) -> None:
    ext.metadata(name="gui", version="0.1.0")
    ext.register_operator(
        "CREATE_WINDOW",
        0,
        5,
        _op_create_window,
        doc="CREATE_WINDOW([STR: type, INT: width, INT: height, STR: title, INT: scale_to_fit=1]):INT handle",
    )
    ext.register_operator(
        "SHOW_IMAGE",
        2,
        2,
        _op_show_image,
        doc="SHOW_IMAGE(INT: handle, TNS: image):INT (1 on success)",
    )
    ext.register_operator(
        "CLOSE_WINDOW",
        1,
        1,
        _op_close_window,
        doc="CLOSE_WINDOW(INT: handle):INT (1 on success)",
    )
    ext.register_operator(
        "SCREEN",
        0,
        0,
        _op_screen,
        doc="SCREEN():TNS [screen_width, screen_height]",
    )
    ext.register_operator(
        "WINDOW",
        1,
        1,
        _op_window,
        doc="WINDOW(INT: handle):TNS [window_width, window_height]",
    )
    ext.register_operator(
        "SCREENSHOT",
        0,
        0,
        _op_screenshot,
        doc="SCREENSHOT():TNS image of the entire screen (Windows only)",
    )
    ext.register_operator(
        "MINIMIZE",
        1,
        1,
        _op_minimize,
        doc="MINIMIZE(INT: handle):INT (1 on success)",
    )
    ext.register_operator(
        "MAXIMIZE",
        1,
        1,
        _op_maximize,
        doc="MAXIMIZE(INT: handle):INT (1 on success)",
    )
    ext.register_operator(
        "TO_FRONT",
        1,
        1,
        _op_to_front,
        doc="TO_FRONT(INT: handle):INT (1 on success)",
    )
    ext.register_operator(
        "TO_BACK",
        1,
        1,
        _op_to_back,
        doc="TO_BACK(INT: handle):INT (1 on success)",
    )