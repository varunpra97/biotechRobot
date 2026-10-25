"""The localizer's wire format, shared by server.py and client.py: numpy and
the standard library only (the client runs inside Isaac's Python, which has
neither open3d nor pyzmq).

A message is: 4-byte big-endian header length, a JSON header, then the raw
bytes of the arrays the header lists ({"name", "dtype", "shape"}), in order.
"""
from __future__ import annotations

import json
import socket
import struct

import numpy as np


def send(sock: socket.socket, header: dict, arrays: dict | None = None):
    arrays = arrays or {}
    header = dict(header, arrays=[{"name": k, "dtype": str(np.asarray(v).dtype), "shape": list(np.shape(v))}
                                  for k, v in arrays.items()])
    h = json.dumps(header).encode()
    parts = [struct.pack(">I", len(h)), h] + [np.ascontiguousarray(v).tobytes() for v in arrays.values()]
    sock.sendall(b"".join(parts))


def _recv_exact(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("localizer connection closed")
        got += k
    return bytes(buf)


def recv(sock: socket.socket):
    """(header dict, {name: array})."""
    (n,) = struct.unpack(">I", _recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, n))
    arrays = {}
    for a in header.pop("arrays", []):
        dt = np.dtype(a["dtype"])
        size = int(np.prod(a["shape"])) * dt.itemsize
        arrays[a["name"]] = np.frombuffer(_recv_exact(sock, size), dt).reshape(a["shape"])
    return header, arrays
