import os
import time
import numpy as np
import glob
from cffi import FFI
# import importlib

ffi = FFI()

# Define the new C interface corresponding to the bcpd_wrapper function in bcpd_wrapper.c
ffi.cdef("""
int bcpd_wrapper(
    const double *X_ptr, int N, int D,  // target point cloud
    const double *Y_ptr, int M,         // source point cloud
    int random_seed,
    double omega, double lambda, double beta, double gamma, double kappa, // model parameters
    int max_iter, int min_iter, double tolerance,  // iteration parameters
    double *y_out, double *v_out, double *s_out, double *R_out, double *t_out, // output variables
    double *sigma_out, int quiet_mode, // other option parameters
    int accel_flag, int nystrom_J, int nystrom_K, int kd_tree_flag, // acceleration options
    int gauss_kernel_type, double tau, int knn, double radius, // kernel parameters
    char normalization_type, // normalization type: 'e', 'x', 'y', 'n'
    int transformation_type, // 0: Tsrn, 1: Tan, 2: Ta, 3: Tsr, 4: Tr, 5: Tn
    double search_scale, double search_radius, double sigma_threshold,  // KD-tree parameters
    int downsampling_flag, int dwn_X, int dwn_Y, double dwr_X, double dwr_Y, // downsampling parameters
    int save_options // added save options parameter
);
""")

# Load the compiled library
_lib_path = glob.glob(os.path.join(os.path.dirname(__file__), "*.so"))[0]
# print("Trying to load:", _lib_path)
# print("File exists?", os.path.exists(_lib_path))

try:
    _lib = ffi.dlopen(_lib_path)
except OSError:
    raise ImportError(f"Failed to load the BCPD library from {_lib_path}")

# Transformation type enumeration
class TransformationType:
    TSRN = 0  # full transformation (translation, scaling, rotation, nonrigid)
    TAN = 1   # affine + nonrigid
    TA = 2    # affine only
    TSR = 3   # similarity (translation, scaling, rotation)
    TR = 4    # rigid (translation, rotation)
    TN = 5    # nonrigid only

# Save option constants corresponding to the bitmask in the C code
class SaveOptions:
    SAVE_X = 0x0001      # save x
    SAVE_Y = 0x0002      # save y
    SAVE_V = 0x0004      # save v
    SAVE_A = 0x0008      # save a
    SAVE_C = 0x0010      # save c
    SAVE_E = 0x0020      # save e
    SAVE_P = 0x0040      # save P
    SAVE_T = 0x0080      # save T
    SAVE_PATHX = 0x0100  # save X
    SAVE_PATHY = 0x0200  # save Y
    SAVE_PFLOG = 0x0400  # save t
    SAVE_VTIME = 0x0800  # save 0
    SAVE_ALL = 0x1000    # save A

    # Common combinations
    SAVE_YP = 0x0240     # save Y and P (0x0200 | 0x0040)


def bcpd_advanced(X, Y,
                  random_seed=0,
                  omega=0.0, lambda_=20.0, beta=2.0, gamma=10.0, kappa=0.0,
                  max_iter=500, min_iter=30, tolerance=1e-6,
                  quiet=False, accelerate=False,
                  nystrom_J=300, nystrom_K=70, use_kdtree=True,
                  gauss_kernel_type=0, tau=0.0, knn=0, radius=0.0,
                  normalization='e', transformation_type=TransformationType.TSRN,
                  search_scale=7.0, search_radius=0.15, sigma_threshold=0.2,
                  downsample=False, dwn_X=0, dwn_Y=0, dwr_X=0.0, dwr_Y=0.0,
                  save_options=0x0202,  # added save options parameter, default is 0x0240 (save Y and P)
                  return_dict=False):
    """
    Advanced version of the Bayesian Coherent Point Drift algorithm with support for more parameters.

    Parameters:
    -----------
    X : numpy.ndarray
        Target point cloud (N x D)
    Y : numpy.ndarray
        Source point cloud (M x D)
    random_seed : int, default=1
        Random seed for random sampling
    omega : float, default=0.0
        Outlier probability [0,1)
    lambda_ : float, default=2.0
        Regularization weight
    beta : float, default=2.0
        Beta parameter of the kernel function
    gamma : float, default=10.0
        Gamma parameter of the kernel function
    kappa : float, default=0.0
        Iterative optimization parameter
    max_iter : int, default=500
        Maximum number of iterations
    min_iter : int, default=30
        Minimum number of iterations
    tolerance : float, default=1e-6
        Convergence threshold
    quiet : bool, default=False
        Whether to suppress output messages
    accelerate : bool, default=False
        Whether to use acceleration
    nystrom_J : int, default=300
        Nyström method parameter J
    nystrom_K : int, default=70
        Nyström method parameter K
    use_kdtree : bool, default=True
        Whether to use KD-tree for nearest-neighbor acceleration
    gauss_kernel_type : int, default=0
        Gaussian kernel type
    tau : float, default=0.0
        Kernel parameter tau
    knn : int, default=0
        Number of k-nearest neighbors
    radius : float, default=0.0
        Search radius
    normalization : str, default='e'
        Normalization type: 'e' (per point), 'x', 'y', or 'n' (no normalization)
    transformation_type : int, default=0
        Transformation type: 0(TSRN), 1(TAN), 2(TA), 3(TSR), 4(TR), 5(TN)
    search_scale : float, default=7.0
        KD-tree search scaling parameter
    search_radius : float, default=0.15
        KD-tree search radius
    sigma_threshold : float, default=0.2 (Must larger than 0.0)
        Covariance threshold
    downsample : bool, default=False
        Whether to apply downsampling
    dwn_X : int, default=0
        Number of downsampled points for X
    dwn_Y : int, default=0
        Number of downsampled points for Y
    dwr_X : float, default=0.0
        Downsampling rate for X
    dwr_Y : float, default=0.0
        Downsampling rate for Y
    return_dict : bool, default=False
        If True, return a dictionary containing all results

    save_options : int, default=0x0202
        Bitmask of save options. The following values can be combined:
        0x0001: save x, 0x0002: save y, 0x0004: save v, 0x0008: save a
        0x0010: save c, 0x0020: save e, 0x0040: save P, 0x0080: save T
        0x0100: save X, 0x0200: save Y, 0x0400: save t, 0x0800: save 0
        0x1000: save A
        Default value 0x0240 saves Y and P (corresponding to 'YP')

    Returns:
    --------
    dict or tuple
        If return_dict=True, returns a dictionary containing all results
        Otherwise returns (Y_transformed, velocities, correspondence_matrix)
    """
    # Convert inputs to contiguous numpy arrays
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)

    # Get dimensions
    N, D = X.shape
    M, _ = Y.shape

    # Check dimensions
    if Y.shape[1] != D:
        raise ValueError(f"X and Y must have the same number of dimensions, got {D} and {Y.shape[1]}")

    # Prepare output arrays
    y_out = np.zeros((M, D), dtype=np.float64)
    v_out = np.zeros((M, D), dtype=np.float64)
    R_out = np.zeros((D, D), dtype=np.float64)
    t_out = np.zeros(D, dtype=np.float64)

    # Scalar outputs
    s_out = ffi.new("double *")
    sigma_out = ffi.new("double *")

    # Set flags
    quiet_mode = 1 if quiet else 0
    accel_flag = 1 if accelerate else 0
    kd_tree_flag = 1 if use_kdtree else 0
    downsampling_flag = 1 if downsample else 0

    # Ensure normalization is valid
    # # Convert character to bytes
    norm_char = normalization[0].encode('ascii')

    # Call the C function
    ret = _lib.bcpd_wrapper(
        ffi.cast("const double *", X.ctypes.data), N, D,
        ffi.cast("const double *", Y.ctypes.data), M,
        random_seed,
        omega, lambda_, beta, gamma, kappa,
        max_iter, min_iter, tolerance,
        ffi.cast("double *", y_out.ctypes.data),
        ffi.cast("double *", v_out.ctypes.data),
        s_out,
        ffi.cast("double *", R_out.ctypes.data),
        ffi.cast("double *", t_out.ctypes.data),
        sigma_out, quiet_mode,
        accel_flag, nystrom_J, nystrom_K, kd_tree_flag,
        gauss_kernel_type, tau, knn, radius,
        norm_char, transformation_type,  # directly use bytes
        search_scale, search_radius, sigma_threshold,
        downsampling_flag, dwn_X, dwn_Y, dwr_X, dwr_Y,
        save_options  # pass save_options parameter
    )

    if ret < 0:
        raise RuntimeError(f"BCPD computation failed with error code {ret}")

    # Compute correspondence matrix P (optional)
    P = np.zeros((N, M), dtype=np.float64)
    sigma2 = sigma_out[0]

    # Compute correspondence matrix
    if sigma2 > 0:
        for i in range(N):
            for j in range(M):
                dist = np.sum((X[i] - y_out[j]) ** 2)
                P[i, j] = np.exp(-0.5 * dist / sigma2)

    # Prepare result
    if return_dict:
        return {
            'Y_transformed': y_out,
            'velocities': v_out,
            'scale': s_out[0],
            'rotation': R_out,
            'translation': t_out,
            'sigma': sigma_out[0],
            'iterations': ret,
            'correspondence': P
        }
    else:
        return y_out, v_out, P

# Keep the original bcpd function for compatibility, but internally use bcpd_advanced
def bcpd(X, Y,
         omega=0.0, lambda_=20.0, gamma=10.0, beta=2.0, delta=7.0,
         epsilon=1e-6, max_iterations=500, J=300, K=70, rand_seed=1,
         options=0, normalization='e', use_kdtree=True,
         save_options=0x0202,  # added save_options parameter
         return_dict=False):
    """
    Bayesian Coherent Point Drift algorithm.

    Parameters:
    -----------
    X : numpy.ndarray
        Target point cloud (N x D)
    Y : numpy.ndarray
        Source point cloud (M x D)
    fx : numpy.ndarray, optional
        Function values on X (N x Df) - not used in the current version
    fy : numpy.ndarray, optional
        Function values on Y (M x Df) - not used in the current version
    omega : float, default=0.0
        Outlier probability [0,1)
    lambda_ : float, default=20.0
        Regularization weight
    gamma : float, default=10.0
        Gamma parameter of the kernel function
    beta : float, default=2.0
        Beta parameter of the kernel function
    delta : float, default=7.0
        Search parameter
    epsilon : float, default=1e-6
        Convergence threshold
    max_iterations : int, default=500
        Maximum number of iterations
    K : int, default=0
        Number of eigenvectors for low-rank approximation
    options : int, default=0
        Option bit field (see BCPDOptions class)
    normalization : str, default='e'
        Normalization type: 'e' (foreach), 'x', 'y', or 'n' (none)
    save_options : int, default=0x0240
        Bitmask of save options. The following values can be combined:
        0x0001: save x, 0x0002: save y, 0x0004: save v, 0x0008: save a
        0x0010: save c, 0x0020: save e, 0x0040: save P, 0x0080: save T
        0x0100: save X, 0x0200: save Y, 0x0400: save t, 0x0800: save 0
        0x1000: save A
        Default value 0x0240 saves Y and P (corresponding to 'YP')
    return_dict : bool, default=False
        If True, return a dictionary containing all results

    Returns:
    --------
    dict or tuple
        If return_dict=True, returns a dictionary containing all results
        Otherwise returns (Y_transformed, corresponding points, correspondence matrix)
    """
    # Set transformation type
    transformation_type = 0
    if options & BCPDOptions.AFFIN:
        transformation_type = 1
    elif options & BCPDOptions.NONRG:
        if options & BCPDOptions.NOSCL:
            transformation_type = 4  # rigid
        else:
            transformation_type = 3  # similarity

    # print(transformation_type)

    # Set acceleration flags
    accelerate = bool(options & BCPDOptions.ACCEL)
    # use_kdtree = bool(options & BCPDOptions.LOCAL)
    quiet = bool(options & BCPDOptions.QUIET)
    # print(accelerate, use_kdtree, quiet)

    # # Convert character to bytes
    # norm_char = normalization[0].encode('ascii')
    # Call advanced version
    result = bcpd_advanced(
        X, Y,
        random_seed=rand_seed,
        omega=omega, lambda_=lambda_, beta=beta, gamma=gamma,
        max_iter=max_iterations, tolerance=epsilon,
        quiet=quiet, accelerate=accelerate,
        nystrom_J=J, nystrom_K=K, use_kdtree=use_kdtree,
        search_scale=delta,
        normalization=normalization,
        transformation_type=transformation_type,
        save_options=save_options,  # pass save_options parameter
        return_dict=return_dict
    )

    # Process return values to match the original function
    if return_dict:
        # time_start = time.time()
        # Add keys that existed in the original function but not in the new one
        result['X_transformed'] = np.zeros_like(result['Y_transformed'])  # placeholder
        result['weights'] = np.ones(len(result['Y_transformed']), dtype=np.float64)  # placeholder
        result['sigmas'] = np.ones(len(result['Y_transformed']), dtype=np.float64)  # placeholder
        result['point_count'] = len(result['Y_transformed'])
        # # Compute elapsed time
        # elapsed_time = time.time() - time_start
        # print(f"Execution time: {elapsed_time:.4f} s")
        return result
    else:
        # time_start = time.time()
        y_transformed, v_out, P = result
        # Create a placeholder X_transformed (the original function returned this, but the new function does not compute it)
        x_transformed = np.zeros_like(y_transformed)
        # Compute elapsed time
        # elapsed_time = time.time() - time_start
        # print(f"Execution time: {elapsed_time:.4f} s")
        return y_transformed, x_transformed, v_out, P