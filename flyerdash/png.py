"""流式 RGBA PNG 写出器。

按行带(strip)增量写入, 不需要整画布像素缓冲区。
仅支持 8 位 RGBA (color type 6), 每行使用 filter 0。
"""

from __future__ import annotations

import struct
import zlib

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IDAT_FLUSH_THRESHOLD = 1 << 16


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


class PngWriter:
    """增量 PNG 写出器。

    用法:
        w = PngWriter(path, width, height)
        w.write_rows(rgba_bytes)   # 长度为 行数*width*4, 可多次调用
        w.close()
    """

    def __init__(self, path: str, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self._path = path
        self._width = width
        self._height = height
        self._rows_written = 0
        self._file = open(path, "wb")
        self._compressor = zlib.compressobj(6)
        self._pending = bytearray()
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
        self._file.write(_SIGNATURE)
        self._file.write(_chunk(b"IHDR", ihdr))

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def write_rows(self, rgba: bytes) -> None:
        """写入若干完整行的 RGBA8 数据 (长度必须为 width*4 的整数倍)。"""
        row_bytes = self._width * 4
        if len(rgba) % row_bytes != 0:
            raise ValueError("rgba length must be a multiple of width*4")
        rows = len(rgba) // row_bytes
        if self._rows_written + rows > self._height:
            raise ValueError("too many rows written")
        # 每行前插入 filter 字节 0
        filtered = bytearray((self._width * 4 + 1) * rows)
        filtered[0 :: self._width * 4 + 1] = b"\x00" * rows
        for r in range(rows):
            src = r * row_bytes
            dst = r * (row_bytes + 1) + 1
            filtered[dst : dst + row_bytes] = rgba[src : src + row_bytes]
        self._pending += self._compressor.compress(bytes(filtered))
        self._rows_written += rows
        if len(self._pending) >= _IDAT_FLUSH_THRESHOLD:
            self._flush_idat()

    def _flush_idat(self) -> None:
        if self._pending:
            self._file.write(_chunk(b"IDAT", bytes(self._pending)))
            self._pending.clear()

    def close(self) -> None:
        """刷新压缩流并写入 IEND。必须在恰好写满 height 行后调用。"""
        if self._file is None:
            return
        try:
            if self._rows_written != self._height:
                raise ValueError(
                    f"expected {self._height} rows, got {self._rows_written}"
                )
            self._pending += self._compressor.flush()
            self._flush_idat()
            self._file.write(_chunk(b"IEND", b""))
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> "PngWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
