"""Streaming PNG writer (stdlib only) plus a small decoder for tests."""

from __future__ import annotations

import struct
import zlib
from typing import BinaryIO

_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


class PngWriter:
    """Incrementally writes an RGBA8 PNG, one strip of rows at a time."""

    def __init__(self, fileobj: BinaryIO, width: int, height: int):
        self._file = fileobj
        self._width = width
        self._height = height
        self._rows_written = 0
        self._compressor = zlib.compressobj(level=6)
        self._idat_size = 0
        fileobj.write(_SIG)
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
        fileobj.write(_chunk(b"IHDR", ihdr))
        # IDAT length is not known up front when streaming; buffer per
        # strip instead: we emit one IDAT per flush via a length-prefixed
        # backpatch-free approach -> collect compressed data and write
        # IDAT chunks as they become available.
        self._idat = bytearray()

    def write_rows(self, rgba8: bytes) -> None:
        """Append whole rows of RGBA8 data (len == width * 4 * n_rows)."""
        row_len = self._width * 4
        if len(rgba8) % row_len != 0:
            raise ValueError("row data is not a whole number of rows")
        n_rows = len(rgba8) // row_len
        if self._rows_written + n_rows > self._height:
            raise ValueError("too many rows")
        filtered = bytearray()
        for i in range(n_rows):
            filtered.append(0)  # filter type 0 (None)
            start = i * row_len
            filtered += rgba8[start : start + row_len]
        self._idat += self._compressor.compress(bytes(filtered))
        self._rows_written += n_rows
        # keep memory bounded: flush compressed data to disk in 1 MiB pieces
        while len(self._idat) >= 1 << 20:
            piece = bytes(self._idat[: 1 << 20])
            del self._idat[: 1 << 20]
            self._file.write(_chunk(b"IDAT", piece))

    def finish(self) -> None:
        if self._rows_written != self._height:
            raise ValueError(
                f"wrote {self._rows_written} rows, expected {self._height}"
            )
        self._idat += self._compressor.flush()
        if self._idat:
            self._file.write(_chunk(b"IDAT", bytes(self._idat)))
            self._idat.clear()
        self._file.write(_chunk(b"IEND", b""))


def write_png(path: str, width: int, height: int, rgba8: bytes) -> None:
    """One-shot helper: write a complete RGBA8 PNG."""
    with open(path, "wb") as fh:
        writer = PngWriter(fh, width, height)
        writer.write_rows(rgba8)
        writer.finish()


def decode_png(path: str) -> tuple[int, int, bytes]:
    """Decode an 8-bit RGBA PNG (no interlace) back to raw bytes."""
    with open(path, "rb") as fh:
        data = fh.read()
    if not data.startswith(_SIG):
        raise ValueError("not a PNG file")
    pos = len(_SIG)
    width = height = None
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, depth, ctype, comp, filt, inter = struct.unpack(
                ">IIBBBBB", payload
            )
            if (depth, ctype, comp, filt, inter) != (8, 6, 0, 0, 0):
                raise ValueError("only 8-bit RGBA non-interlaced PNG supported")
        elif tag == b"IDAT":
            idat += payload
        elif tag == b"IEND":
            break
    if width is None:
        raise ValueError("missing IHDR")
    raw = zlib.decompress(bytes(idat))
    stride = width * 4
    out = bytearray(width * height * 4)
    prev = bytearray(stride)
    src = 0
    for row in range(height):
        ftype = raw[src]
        src += 1
        line = bytearray(raw[src : src + stride])
        src += stride
        if ftype == 1:  # Sub
            for i in range(4, stride):
                line[i] = (line[i] + line[i - 4]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                left = line[i - 4] if i >= 4 else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = line[i - 4] if i >= 4 else 0
                b = prev[i]
                c = prev[i - 4] if i >= 4 else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 0xFF
        elif ftype != 0:
            raise ValueError(f"unknown filter type {ftype}")
        out[row * stride : (row + 1) * stride] = line
        prev = line
    return width, height, bytes(out)
