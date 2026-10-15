from __future__ import annotations

import asyncio

import pytest

from furrow.web.server import QueueWriter


def test_queue_writer_strips_ansi():
    q: asyncio.Queue = asyncio.Queue()
    w = QueueWriter(q)
    w.write("\x1b[31mred text\x1b[0m\n")
    assert q.get_nowait() == "red text"


def test_queue_writer_normalizes_crlf():
    q: asyncio.Queue = asyncio.Queue()
    w = QueueWriter(q)
    w.write("line1\r\nline2\r\n")
    assert q.get_nowait() == "line1"
    assert q.get_nowait() == "line2"


def test_queue_writer_buffers_partial_lines():
    q: asyncio.Queue = asyncio.Queue()
    w = QueueWriter(q)
    w.write("partial")
    assert q.qsize() == 0
    w.write("-line\n")
    assert q.get_nowait() == "partial-line"


def test_queue_writer_flush_emits_buffer():
    q: asyncio.Queue = asyncio.Queue()
    w = QueueWriter(q)
    w.write("no newline")
    w.flush()
    assert q.get_nowait() == "no newline"


def test_queue_writer_isatty_false():
    w = QueueWriter(asyncio.Queue())
    assert w.isatty() is False


def test_queue_writer_encoding_utf8():
    w = QueueWriter(asyncio.Queue())
    assert w.encoding == "utf-8"