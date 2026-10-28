"""libpyhxt.so through ctypes: finding it, checking its interface version, declaring its
functions."""
import ctypes
import os

import numpy as np

#: pyhxt_abi() of the library this package calls
ABI = 3

# pyhxt_flags.h
PREBOND, NO_RECOVER_DELAUNAY, SEEDS_FIRST, NONREPRODUCIBLE, CHECK_CENTRES = 1, 2, 4, 8, 16
FAST = PREBOND | NO_RECOVER_DELAUNAY | SEEDS_FIRST

#: int field(double *pts, uint64_t n): pts is n x 4 (x, y, z, size); return 0 for success
FIELD = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_uint64)
#: void message(int level, const char *text)
MESSAGE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)

_lib = None


def library():
    """libpyhxt.so: $PYHXT_LIB, else beside this package, else in the repository's root
    (where build.sh puts it by default)."""
    global _lib
    if _lib is not None:
        return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.environ.get('PYHXT_LIB'), os.path.join(here, 'libpyhxt.so'),
                 os.path.join(here, os.pardir, 'libpyhxt.so')):
        if path and os.path.exists(path):
            break
    else:
        raise OSError('libpyhxt.so not found: pip install pyhxt, run build.sh, '
                      'or set PYHXT_LIB')
    lib = ctypes.CDLL(path)
    if lib.pyhxt_abi() != ABI:
        raise OSError(f'{path} has interface {lib.pyhxt_abi()}, this pyhxt needs {ABI}: '
                      'rebuild it')
    u32, u64, f64, ptr = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_double, ctypes.c_void_p
    U32, F64 = (np.ctypeslib.ndpointer(t, flags='C_CONTIGUOUS') for t in (np.uint32, np.float64))
    lib.pyhxt_tetmesh.argtypes = [
        u32, F64, ptr, u32, F64, u32, u64, U32, U32, u32, U32, U32, ctypes.c_int, f64, f64,
        f64, f64, f64, ctypes.c_int, ctypes.c_int, ptr, ptr, ctypes.c_char_p,
        ctypes.POINTER(ptr)]
    lib.pyhxt_counts.argtypes = [ptr, ctypes.POINTER(u32), ctypes.POINTER(u64),
                                 ctypes.POINTER(u64)]
    lib.pyhxt_copy.argtypes = [ptr, F64, U32, U32, U32]
    lib.pyhxt_copy_triangle_colors.argtypes = [ptr, U32]
    lib.pyhxt_locate.argtypes = [ptr, u32, F64, U32]
    lib.pyhxt_free.argtypes = [ptr]
    _lib = lib
    return lib
