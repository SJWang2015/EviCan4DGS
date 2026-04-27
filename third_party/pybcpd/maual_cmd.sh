# Ensure you have gcc and necessary development packages installed

# Execute in the project root directory

# Get the NumPy include path

NUMPY_INCLUDE=$(python -c "import numpy; print(numpy.get_include())")

# Compile
gcc -shared -fPIC -o _bcpd_c.so \
    src/pybcpd/register/main.c \
    src/pybcpd/register/bcpd.c \
    src/pybcpd/register/info.c \
    src/pybcpd/register/norm.c \
    src/pybcpd/base/util.c \
    src/pybcpd/base/misc.c \
    src/pybcpd/base/kdtree.c \
    src/pybcpd/base/kernel.c \
    src/pybcpd/base/sampling.c \
    src/pybcpd/base/sgraph.c \
    src/pybcpd/base/geokdecomp.c \
    -Isrc/register \
    -Isrc/register/base \
    -I$NUMPY_INCLUDE \
    -lm -O3