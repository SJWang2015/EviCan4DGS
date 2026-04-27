import os
import sys
from cffi import FFI

ffi = FFI()
ffi.cdef("""
    int bcpd_wrapper(double* X, int n, double* Y, int m, int dim,
                     double* omega, double* sigma2, double* lambda,
                     double* gamma, int w, double* X_new, double* P);
""")

try:
    ws = os.getcwd()
    ws2 = os.path.abspath(os.path.dirname(__file__))
    lib_path = os.path.join(ws2,'_bcpd.cpython-38-x86_64-linux-gnu.so')
    if os.path.exists(lib_path):
        lib = ffi.dlopen(lib_path)
        print("The library was successfully loaded using CFFI.")
except Exception as e:
    print(f"CFFI loading failed: {e}")