"""Dependency-free Windows screen, region, and window capture plus RGB image helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any
from uuid import uuid4

from . import __version__ as APP_VERSION
from .vision import RGBImage, encode_png_bytes

DEFAULT_CAPTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "captures"


def format_capture_version_tag(version: str | None = None) -> str:
    """Normalize a version tag for capture archive subfolders (e.g. 'v0.2.0')."""
    raw = APP_VERSION if version is None else str(version)
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw.strip())
    if not cleaned:
        return ""
    return cleaned if cleaned.startswith("v") else f"v{cleaned}"

_SRCCOPY = 0x00CC0020
_CAPTUREBLT = 0x40000000
_PW_RENDERFULLCONTENT = 0x00000002
_GA_ROOT = 2
_DWMWA_EXTENDED_FRAME_BOUNDS = 9
_DWMWA_CLOAKED = 14
_SM_CXSCREEN = 0
_SM_CYSCREEN = 1
_SM_XVIRTUALSCREEN = 76
_SM_YVIRTUALSCREEN = 77
_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79
_SPI_GETWORKAREA = 0x0030
_SW_RESTORE = 9
_SW_MAXIMIZE = 3
_SWP_NOACTIVATE = 0x0010
_SWP_SHOWWINDOW = 0x0040


@dataclass(frozen=True)
class WindowInfo:
    """Metadata for a visible top-level desktop window."""

    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int
    minimized: bool = False

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hwnd": self.hwnd,
            "title": self.title,
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "minimized": self.minimized,
        }


@dataclass(frozen=True)
class CaptureTarget:
    """Target specification for live screen, region, or window capture."""

    kind: str = "fullscreen"
    region: tuple[int, int, int, int] | None = None
    hwnd: int | None = None
    title: str = "Ecrã Inteiro"

    def __post_init__(self) -> None:
        if self.kind not in ("fullscreen", "region", "window"):
            raise ValueError(f"unsupported capture target kind: {self.kind!r}")
        if self.kind == "region":
            if self.region is None or len(self.region) != 4:
                raise ValueError("region capture target requires (x, y, width, height)")
            if self.region[2] <= 0 or self.region[3] <= 0:
                raise ValueError("region width and height must be positive")
        if self.kind == "window" and not self.hwnd:
            raise ValueError("window capture target requires a valid hwnd")

    def describe(self) -> str:
        if self.kind == "region" and self.region is not None:
            x, y, w, h = self.region
            return f"Região {w}×{h} @ ({x}, {y})"
        if self.kind == "window":
            label = self.title.strip() or f"HWND {self.hwnd}"
            if len(label) > 34:
                label = label[:31] + "..."
            return f"Janela: {label}"
        return self.title or "Ecrã Inteiro"


_WIN32_CACHE: Any = None


def _get_win32() -> Any:
    """Lazily initialize typed ctypes bindings for Win32 capture APIs."""
    global _WIN32_CACHE
    if _WIN32_CACHE is not None:
        return _WIN32_CACHE
    if sys.platform != "win32":
        raise RuntimeError("Captura de ecrã nativa suportada apenas em Windows (win32).")

    import ctypes
    from ctypes import wintypes

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

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    try:
        dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
        dwmapi.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long
    except OSError:
        dwmapi = None

    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int
    user32.SystemParametersInfoW.argtypes = [wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT]
    user32.SystemParametersInfoW.restype = wintypes.BOOL
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.EnumWindows.argtypes = [enum_proc, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.IsZoomed.argtypes = [wintypes.HWND]
    user32.IsZoomed.restype = wintypes.BOOL
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    user32.PrintWindow.restype = wintypes.BOOL
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = ctypes.c_short

    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.DeleteDC.restype = wintypes.BOOL
    gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.BitBlt.argtypes = [
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.DWORD,
    ]
    gdi32.BitBlt.restype = wintypes.BOOL
    gdi32.GetDIBits.argtypes = [
        wintypes.HDC,
        wintypes.HBITMAP,
        wintypes.UINT,
        wintypes.UINT,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.UINT,
    ]
    gdi32.GetDIBits.restype = ctypes.c_int

    class _Win32Api:
        pass

    api = _Win32Api()
    api.ctypes = ctypes
    api.wintypes = wintypes
    api.user32 = user32
    api.gdi32 = gdi32
    api.dwmapi = dwmapi
    api.EnumWindowsProc = enum_proc
    api.BITMAPINFOHEADER = BITMAPINFOHEADER
    _WIN32_CACHE = api
    return api


def enable_dpi_awareness() -> None:
    """Enable per-monitor DPI awareness on Windows before creating GUI windows."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    try:
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except AttributeError:
        pass
    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        shcore.SetProcessDpiAwareness.argtypes = [ctypes.c_int]
        shcore.SetProcessDpiAwareness.restype = ctypes.c_long
        if shcore.SetProcessDpiAwareness(2) == 0:
            return
    except (OSError, AttributeError):
        pass
    try:
        user32.SetProcessDPIAware.argtypes = []
        user32.SetProcessDPIAware.restype = wintypes.BOOL
        user32.SetProcessDPIAware()
    except AttributeError:
        pass


def get_primary_screen_bounds() -> tuple[int, int, int, int]:
    """Return (0, 0, width, height) of the primary monitor in physical pixels."""
    api = _get_win32()
    width = int(api.user32.GetSystemMetrics(_SM_CXSCREEN))
    height = int(api.user32.GetSystemMetrics(_SM_CYSCREEN))
    if width <= 0 or height <= 0:
        raise RuntimeError("Não foi possível obter a resolução do monitor principal.")
    return (0, 0, width, height)


def get_work_area_bounds() -> tuple[int, int, int, int]:
    """Return the primary monitor's usable area, excluding the taskbar and docked toolbars."""
    api = _get_win32()
    rect = api.wintypes.RECT()
    if api.user32.SystemParametersInfoW(
        _SPI_GETWORKAREA,
        0,
        api.ctypes.byref(rect),
        0,
    ):
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width > 0 and height > 0:
            return int(rect.left), int(rect.top), width, height
    return get_primary_screen_bounds()


def get_virtual_screen_bounds() -> tuple[int, int, int, int]:
    """Return (x, y, width, height) covering all connected monitors."""
    api = _get_win32()
    vx = int(api.user32.GetSystemMetrics(_SM_XVIRTUALSCREEN))
    vy = int(api.user32.GetSystemMetrics(_SM_YVIRTUALSCREEN))
    vw = int(api.user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN))
    vh = int(api.user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN))
    if vw > 0 and vh > 0:
        return (vx, vy, vw, vh)
    return get_primary_screen_bounds()


def get_window_hwnd_from_tk(widget_id: int) -> int:
    """Resolve the top-level Win32 HWND for a Tkinter widget or window."""
    if sys.platform != "win32":
        return 0
    api = _get_win32()
    root_hwnd = api.user32.GetAncestor(api.wintypes.HWND(widget_id), _GA_ROOT)
    return int(root_hwnd or widget_id)


def get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """Return (left, top, width, height) for a window HWND, or None if unavailable."""
    if sys.platform != "win32" or not hwnd:
        return None
    api = _get_win32()
    try:
        return _read_window_rect(api, int(hwnd), extended=True)
    except RuntimeError:
        return None


def is_window_maximized(hwnd: int) -> bool:
    """Return whether a window is currently maximized."""
    if sys.platform != "win32" or not hwnd:
        return False
    api = _get_win32()
    return bool(api.user32.IsZoomed(api.wintypes.HWND(int(hwnd))))


def set_window_bounds(hwnd: int, rect: tuple[int, int, int, int]) -> None:
    """Restore and position a top-level window at (left, top, width, height) without stealing focus."""
    if sys.platform != "win32" or not hwnd:
        raise RuntimeError("Reposicionamento de janelas suportado apenas em Windows.")
    left, top, width, height = (int(value) for value in rect)
    if width <= 0 or height <= 0:
        raise ValueError("largura e altura da janela têm de ser positivas")
    api = _get_win32()
    handle = api.wintypes.HWND(int(hwnd))
    api.user32.ShowWindow(handle, _SW_RESTORE)
    ok = api.user32.SetWindowPos(
        handle,
        api.wintypes.HWND(0),
        left,
        top,
        width,
        height,
        _SWP_NOACTIVATE | _SWP_SHOWWINDOW,
    )
    if not ok:
        raise RuntimeError(f"Não foi possível ajustar a janela (HWND={hwnd}).")


def maximize_window(hwnd: int) -> None:
    """Maximize a top-level window if it still exists."""
    if sys.platform != "win32" or not hwnd:
        return
    api = _get_win32()
    api.user32.ShowWindow(api.wintypes.HWND(int(hwnd)), _SW_MAXIMIZE)


def mask_screen_rect_on_image(
    image: RGBImage,
    capture_rect: tuple[int, int, int, int],
    mask_rect: tuple[int, int, int, int],
    fill: tuple[int, int, int] = (15, 23, 42),
) -> None:
    """Blank out a screen rectangle (e.g. our own Companion UI) inside a captured RGBImage in-place."""
    cx, cy, cw, ch = capture_rect
    mx, my, mw, mh = mask_rect
    if cw <= 0 or ch <= 0 or mw <= 0 or mh <= 0:
        return
    ix0 = max(cx, mx) - cx
    iy0 = max(cy, my) - cy
    ix1 = min(cx + cw, mx + mw, cx + image.width) - cx
    iy1 = min(cy + ch, my + mh, cy + image.height) - cy
    if ix1 <= ix0 or iy1 <= iy0:
        return
    span_bytes = (ix1 - ix0) * 3
    row_fill = bytes(fill) * (ix1 - ix0)
    stride = image.width * 3
    mv = memoryview(image.pixels)
    for row in range(max(0, iy0), min(image.height, iy1)):
        start = row * stride + ix0 * 3
        mv[start : start + span_bytes] = row_fill


def is_key_pressed(vk_code: int = 0x77) -> bool:
    """Return True if a Windows virtual-key (default VK_F8=0x77) is currently depressed."""
    if sys.platform != "win32":
        return False
    api = _get_win32()
    return bool(int(api.user32.GetAsyncKeyState(vk_code)) & 0x8000)


def _bgra_to_rgb_image(bgra: bytes | bytearray, width: int, height: int) -> RGBImage:
    expected = width * height * 4
    if len(bgra) != expected:
        raise ValueError(f"BGRA buffer length {len(bgra)} does not match {width}x{height} ({expected})")
    rgb = bytearray(width * height * 3)
    rgb[0::3] = bgra[2::4]
    rgb[1::3] = bgra[1::4]
    rgb[2::3] = bgra[0::4]
    return RGBImage(width=width, height=height, pixels=rgb)


def crop_rgb_image(image: RGBImage, left: int, top: int, width: int, height: int) -> RGBImage:
    """Crop an RGBImage cleanly to (left, top, width, height)."""
    if width <= 0 or height <= 0:
        raise ValueError("crop dimensions must be positive")
    x0 = max(0, left)
    y0 = max(0, top)
    x1 = min(image.width, left + width)
    y1 = min(image.height, top + height)
    cw = x1 - x0
    ch = y1 - y0
    if cw <= 0 or ch <= 0:
        raise ValueError("crop rectangle does not overlap the image")
    row_bytes = cw * 3
    src_stride = image.width * 3
    out = bytearray(row_bytes * ch)
    src_mv = memoryview(image.pixels)
    dst_mv = memoryview(out)
    for row in range(ch):
        src_start = (y0 + row) * src_stride + x0 * 3
        dst_start = row * row_bytes
        dst_mv[dst_start : dst_start + row_bytes] = src_mv[src_start : src_start + row_bytes]
    return RGBImage(width=cw, height=ch, pixels=out)


def resize_rgb_image(image: RGBImage, max_width: int, max_height: int) -> RGBImage:
    """Proportionally resize an RGBImage to fit inside (max_width, max_height)."""
    if max_width <= 0 or max_height <= 0:
        raise ValueError("max_width and max_height must be positive")
    if image.width <= max_width and image.height <= max_height:
        return image
    scale = min(max_width / image.width, max_height / image.height)
    dst_w = max(1, round(image.width * scale))
    dst_h = max(1, round(image.height * scale))
    src_w, src_h = image.width, image.height
    src_stride = src_w * 3
    dst_stride = dst_w * 3
    out = bytearray(dst_stride * dst_h)
    src_px = image.pixels
    x_offsets = [(x * src_w // dst_w) * 3 for x in range(dst_w)]
    for dy in range(dst_h):
        sy = dy * src_h // dst_h
        row_base = sy * src_stride
        dst_base = dy * dst_stride
        for dx, ox in enumerate(x_offsets):
            si = row_base + ox
            di = dst_base + dx * 3
            out[di] = src_px[si]
            out[di + 1] = src_px[si + 1]
            out[di + 2] = src_px[si + 2]
    return RGBImage(width=dst_w, height=dst_h, pixels=out)


def image_to_ppm_bytes(image: RGBImage) -> bytes:
    """Encode an RGBImage to binary PPM (P6) bytes for zero-overhead Tkinter PhotoImage display."""
    header = f"P6\n{image.width} {image.height}\n255\n".encode("ascii")
    return header + bytes(image.pixels)


def compute_frame_signature(image: RGBImage, grid_w: int = 96, grid_h: int = 54) -> bytes:
    """Sample an RGB grid signature for fast real-time change detection."""
    if image.width <= 0 or image.height <= 0:
        return b""
    sw = min(grid_w, image.width)
    sh = min(grid_h, image.height)
    sig = bytearray(sw * sh * 3)
    src = image.pixels
    stride = image.width * 3
    xs = [((x * image.width) // sw) * 3 for x in range(sw)]
    idx = 0
    for y in range(sh):
        row_base = ((y * image.height) // sh) * stride
        for ox in xs:
            pos = row_base + ox
            sig[idx] = src[pos]
            sig[idx + 1] = src[pos + 1]
            sig[idx + 2] = src[pos + 2]
            idx += 3
    return bytes(sig)


def frame_difference(sig_a: bytes | None, sig_b: bytes | None, channel_tolerance: int = 16) -> float:
    """Return the fraction [0.0..1.0] of sampled grid pixels that changed beyond tolerance."""
    if not sig_a or not sig_b or len(sig_a) != len(sig_b) or len(sig_a) % 3 != 0:
        return 1.0
    changed = 0
    total = len(sig_a) // 3
    for i in range(0, len(sig_a), 3):
        if (
            abs(sig_a[i] - sig_b[i]) > channel_tolerance
            or abs(sig_a[i + 1] - sig_b[i + 1]) > channel_tolerance
            or abs(sig_a[i + 2] - sig_b[i + 2]) > channel_tolerance
        ):
            changed += 1
    return changed / total


def _read_window_rect(api: Any, hwnd: int, extended: bool = True) -> tuple[int, int, int, int]:
    rect = api.wintypes.RECT()
    if extended and api.dwmapi is not None:
        hr = api.dwmapi.DwmGetWindowAttribute(
            api.wintypes.HWND(hwnd),
            _DWMWA_EXTENDED_FRAME_BOUNDS,
            api.ctypes.byref(rect),
            api.ctypes.sizeof(rect),
        )
        if hr == 0:
            return (int(rect.left), int(rect.top), int(rect.right - rect.left), int(rect.bottom - rect.top))
    if not api.user32.GetWindowRect(api.wintypes.HWND(hwnd), api.ctypes.byref(rect)):
        raise RuntimeError(f"Não foi possível ler as dimensões da janela (HWND={hwnd}).")
    return (int(rect.left), int(rect.top), int(rect.right - rect.left), int(rect.bottom - rect.top))


def list_windows(
    exclude_hwnds: set[int] | None = None,
    min_width: int = 160,
    min_height: int = 120,
) -> list[WindowInfo]:
    """Enumerate visible non-cloaked desktop windows suitable for table capture."""
    api = _get_win32()
    excluded = set(exclude_hwnds or ())
    results: list[WindowInfo] = []

    def _callback(hwnd: int, _lparam: int) -> bool:
        handle = int(hwnd)
        if handle in excluded or not api.user32.IsWindowVisible(hwnd):
            return True
        length = api.user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        if api.dwmapi is not None:
            cloaked = api.wintypes.DWORD(0)
            hr = api.dwmapi.DwmGetWindowAttribute(
                hwnd,
                _DWMWA_CLOAKED,
                api.ctypes.byref(cloaked),
                api.ctypes.sizeof(cloaked),
            )
            if hr == 0 and cloaked.value:
                return True
        title_buf = api.ctypes.create_unicode_buffer(length + 1)
        api.user32.GetWindowTextW(hwnd, title_buf, length + 1)
        title = title_buf.value.strip()
        if not title:
            return True
        minimized = bool(api.user32.IsIconic(hwnd))
        try:
            left, top, width, height = _read_window_rect(api, handle, extended=True)
        except RuntimeError:
            return True
        if not minimized and (width < min_width or height < min_height):
            return True
        results.append(
            WindowInfo(
                hwnd=handle,
                title=title,
                left=left,
                top=top,
                width=max(0, width),
                height=max(0, height),
                minimized=minimized,
            )
        )
        return True

    cb = api.EnumWindowsProc(_callback)
    api.user32.EnumWindows(cb, 0)
    return results


def _grab_dc_region(api: Any, src_dc: Any, src_x: int, src_y: int, width: int, height: int) -> bytearray:
    mem_dc = api.gdi32.CreateCompatibleDC(src_dc)
    if not mem_dc:
        raise RuntimeError("CreateCompatibleDC falhou durante a captura.")
    bmp = None
    old_obj = None
    try:
        bmp = api.gdi32.CreateCompatibleBitmap(src_dc, width, height)
        if not bmp:
            raise RuntimeError("CreateCompatibleBitmap falhou durante a captura.")
        old_obj = api.gdi32.SelectObject(mem_dc, bmp)
        if not api.gdi32.BitBlt(mem_dc, 0, 0, width, height, src_dc, src_x, src_y, _SRCCOPY | _CAPTUREBLT):
            raise RuntimeError("BitBlt falhou ao capturar o ecrã (ecrã bloqueado ou sessão sem compositor).")
        return _read_bitmap_bgra(api, mem_dc, bmp, width, height)
    finally:
        if old_obj is not None:
            api.gdi32.SelectObject(mem_dc, old_obj)
        if bmp is not None:
            api.gdi32.DeleteObject(bmp)
        api.gdi32.DeleteDC(mem_dc)


def _read_bitmap_bgra(api: Any, mem_dc: Any, bmp: Any, width: int, height: int) -> bytearray:
    bmi = api.BITMAPINFOHEADER()
    bmi.biSize = api.ctypes.sizeof(api.BITMAPINFOHEADER)
    bmi.biWidth = width
    bmi.biHeight = -height  # top-down DIB
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0
    buf = (api.ctypes.c_ubyte * (width * height * 4))()
    lines = api.gdi32.GetDIBits(mem_dc, bmp, 0, height, buf, api.ctypes.byref(bmi), 0)
    if lines == 0:
        raise RuntimeError("GetDIBits não devolveu linhas de imagem.")
    return bytearray(buf)


def capture_screen_region(x: int, y: int, width: int, height: int) -> RGBImage:
    """Capture a screen rectangle in virtual-screen coordinates as an RGBImage."""
    if width <= 0 or height <= 0:
        raise ValueError("width e height da região devem ser positivos")
    api = _get_win32()
    hdc_screen = api.user32.GetDC(0)
    if not hdc_screen:
        raise RuntimeError("GetDC(0) falhou ao aceder ao ecrã.")
    try:
        bgra = _grab_dc_region(api, hdc_screen, int(x), int(y), int(width), int(height))
    finally:
        api.user32.ReleaseDC(0, hdc_screen)
    return _bgra_to_rgb_image(bgra, int(width), int(height))


def _has_non_black_pixels(bgra: bytearray) -> bool:
    step = max(4, (len(bgra) // 4 // 256) * 4)
    for i in range(0, len(bgra), step):
        if bgra[i] or bgra[i + 1] or bgra[i + 2]:
            return True
    return any(bgra[i] or bgra[i + 1] or bgra[i + 2] for i in range(0, min(len(bgra), 4096), 4))


def capture_window(hwnd: int) -> tuple[RGBImage, WindowInfo]:
    """Capture a specific desktop window via PrintWindow, falling back to screen BitBlt."""
    api = _get_win32()
    w_hwnd = api.wintypes.HWND(hwnd)
    if not api.user32.IsWindow(w_hwnd):
        raise RuntimeError("A janela selecionada já não existe.")
    if api.user32.IsIconic(w_hwnd):
        raise RuntimeError("A janela selecionada está minimizada; restaura a janela para capturar.")

    length = api.user32.GetWindowTextLengthW(w_hwnd)
    title_buf = api.ctypes.create_unicode_buffer(max(1, length + 1))
    if length > 0:
        api.user32.GetWindowTextW(w_hwnd, title_buf, length + 1)

    win_x, win_y, win_w, win_h = _read_window_rect(api, hwnd, extended=False)
    ext_x, ext_y, ext_w, ext_h = _read_window_rect(api, hwnd, extended=True)
    if win_w <= 0 or win_h <= 0:
        raise RuntimeError("A janela selecionada tem dimensões nulas.")

    hdc_screen = api.user32.GetDC(0)
    if not hdc_screen:
        raise RuntimeError("GetDC(0) falhou ao preparar captura da janela.")

    image: RGBImage | None = None
    try:
        mem_dc = api.gdi32.CreateCompatibleDC(hdc_screen)
        if not mem_dc:
            raise RuntimeError("CreateCompatibleDC falhou na captura da janela.")
        bmp = None
        old_obj = None
        try:
            bmp = api.gdi32.CreateCompatibleBitmap(hdc_screen, win_w, win_h)
            if not bmp:
                raise RuntimeError("CreateCompatibleBitmap falhou na captura da janela.")
            old_obj = api.gdi32.SelectObject(mem_dc, bmp)
            captured_bgra: bytearray | None = None
            for flags in (_PW_RENDERFULLCONTENT, 0):
                if api.user32.PrintWindow(w_hwnd, mem_dc, flags):
                    candidate = _read_bitmap_bgra(api, mem_dc, bmp, win_w, win_h)
                    if _has_non_black_pixels(candidate):
                        captured_bgra = candidate
                        break
            if captured_bgra is not None:
                image = _bgra_to_rgb_image(captured_bgra, win_w, win_h)
                if ext_w > 0 and ext_h > 0 and (ext_x != win_x or ext_y != win_y or ext_w != win_w or ext_h != win_h):
                    crop_x = max(0, ext_x - win_x)
                    crop_y = max(0, ext_y - win_y)
                    crop_w = min(win_w - crop_x, ext_w)
                    crop_h = min(win_h - crop_y, ext_h)
                    if crop_w > 0 and crop_h > 0:
                        image = crop_rgb_image(image, crop_x, crop_y, crop_w, crop_h)
        finally:
            if old_obj is not None:
                api.gdi32.SelectObject(mem_dc, old_obj)
            if bmp is not None:
                api.gdi32.DeleteObject(bmp)
            api.gdi32.DeleteDC(mem_dc)

        if image is None:
            fallback_bgra = _grab_dc_region(api, hdc_screen, ext_x, ext_y, max(1, ext_w), max(1, ext_h))
            image = _bgra_to_rgb_image(fallback_bgra, max(1, ext_w), max(1, ext_h))
    finally:
        api.user32.ReleaseDC(0, hdc_screen)

    info = WindowInfo(
        hwnd=int(hwnd),
        title=title_buf.value.strip() or f"HWND {hwnd}",
        left=ext_x,
        top=ext_y,
        width=image.width,
        height=image.height,
        minimized=False,
    )
    return image, info


def capture_target(target: CaptureTarget) -> tuple[RGBImage, dict[str, Any]]:
    """Capture an RGBImage and metadata from a fullscreen, region, or window CaptureTarget."""
    t0 = time.perf_counter()
    if target.kind == "fullscreen":
        x, y, w, h = get_primary_screen_bounds()
        image = capture_screen_region(x, y, w, h)
        rect = [x, y, w, h]
        title = target.title or f"Ecrã Inteiro ({w}×{h})"
        hwnd = None
    elif target.kind == "region":
        assert target.region is not None
        x, y, w, h = target.region
        image = capture_screen_region(x, y, w, h)
        rect = [x, y, w, h]
        title = target.describe()
        hwnd = None
    else:
        assert target.hwnd is not None
        image, win_info = capture_window(target.hwnd)
        rect = [win_info.left, win_info.top, win_info.width, win_info.height]
        title = win_info.title
        hwnd = win_info.hwnd

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    meta = {
        "kind": target.kind,
        "title": title,
        "hwnd": hwnd,
        "rect": rect,
        "width": image.width,
        "height": image.height,
        "capture_ms": elapsed_ms,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    return image, meta


def save_diagnostic_capture(
    fullscreen_image: RGBImage,
    metadata: dict[str, Any],
    captures_dir: Path = DEFAULT_CAPTURES_DIR,
    max_saved: int = 200,
    version: str | None = None,
) -> tuple[Path, Path]:
    """Save a full-screen PNG capture and diagnostic JSON metadata into a versioned captures subfolder."""
    version_tag = format_capture_version_tag(version)
    target_dir = Path(captures_dir) / version_tag if version_tag else Path(captures_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + f"_{uuid4().hex[:8]}"
    png_path = target_dir / f"cap_{stamp}_full.png"
    json_path = target_dir / f"cap_{stamp}_meta.json"
    meta_out = dict(metadata)
    meta_out.setdefault("app_version", version_tag or f"v{APP_VERSION}")
    png_path.write_bytes(encode_png_bytes(fullscreen_image, compress_level=1))
    json_path.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")

    if max_saved > 0:
        pngs = sorted(target_dir.glob("cap_*_full.png"))
        excess = len(pngs) - max_saved
        for old_png in pngs[: max(0, excess)]:
            old_meta = target_dir / old_png.name.replace("_full.png", "_meta.json")
            try:
                old_png.unlink(missing_ok=True)
                old_meta.unlink(missing_ok=True)
            except OSError:
                pass
    return png_path, json_path
