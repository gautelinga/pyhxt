"""Smoke test of libpyhxt.so, with no dependency beyond the standard library: the unit
cube's surface in, a mesh of volume 1 out, every input point kept where it was.
Usage: python3 test.py [path/to/libpyhxt.so]"""
import ctypes
import hashlib
import os
import sys

lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libpyhxt.so'))
u32 = ctypes.c_uint32
xyz = [float(c) for i in range(8) for c in ((i & 1), (i >> 1) & 1, (i >> 2) & 1)]
quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
tris = [n for a, b, c, d in quads for n in (a, b, c, a, c, d)]
arr = lambda t, v: (t * len(v))(*v)  # noqa: E731
mesh = ctypes.c_void_p()
status = lib.pyhxt_tetmesh(
    u32(8), arr(ctypes.c_double, xyz), ctypes.c_uint64(12), arr(u32, tris), arr(u32, [1] * 12),
    u32(1), arr(u32, [0, 1]), arr(u32, [1]), ctypes.c_int(2), ctypes.c_double(0.),
    ctypes.c_double(0.3), ctypes.c_double(3 ** .5), ctypes.c_double(1e-12),
    ctypes.c_double(0.3), ctypes.c_int(1), None, ctypes.byref(mesh))
assert status == 0, status
nv, nt, ntri = u32(), ctypes.c_uint64(), ctypes.c_uint64()
lib.pyhxt_counts(mesh, ctypes.byref(nv), ctypes.byref(nt), ctypes.byref(ntri))
X = (ctypes.c_double * (3 * nv.value))()
T = (u32 * (4 * nt.value))()
col = (u32 * nt.value)()
tri = (u32 * (3 * ntri.value))()
lib.pyhxt_copy(mesh, X, T, col, tri)
lib.pyhxt_free(mesh)
assert list(X[:24]) == xyz


def vol(t):
    a, b, c, d = ([X[3 * n + k] for k in range(3)] for n in t)
    u, v, w = ([p[k] - a[k] for k in range(3)] for p in (b, c, d))
    return (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0])) / 6


tets = [T[4 * i:4 * i + 4] for i in range(nt.value) if col[i] == 0]
total = sum(abs(vol(t)) for t in tets)
assert abs(total - 1) < 1e-12, total
# the mesh itself, to compare builds: a compiler or libm that rounds differently shows here
digest = hashlib.sha256(bytes(X) + bytes(T) + bytes(col)).hexdigest()[:16]
print('OK: {} vertices, {} tets, volume {!r}, sha256 {}'.format(
    nv.value, len(tets), total, digest))
