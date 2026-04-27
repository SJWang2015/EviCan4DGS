/* bcpd_c_api.c - Python C API 包装 BCPD 算法 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

/* 声明BCPD包装函数 */
extern int bcpd_wrapper(
    const double *X_ptr, int N, int D,
    const double *Y_ptr, int M,
    int random_seed,
    double omega, double lambda, double beta, double gamma, double kappa,
    int max_iter, int min_iter, double tolerance,
    double *y_out, double *v_out, double *s_out, double *R_out, double *t_out,
    double *sigma_out, int quiet_mode,
    int accel_flag, int nystrom_J, int nystrom_K, int kd_tree_flag,
    int gauss_kernel_type, double tau, int knn, double radius,
    char normalization_type,
    int transformation_type,
    double search_scale, double search_radius, double sigma_threshold,
    int downsampling_flag, int dwn_X, int dwn_Y, double dwr_X, double dwr_Y,
    int save_options
);

static PyObject *
bcpd_register(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *X_array, *Y_array;
    int random_seed = 0;
    double omega = 0.0, lambda = 2.0, beta = 2.0, gamma = 1.0, kappa = 0.0;
    int max_iter = 50, min_iter = 30;
    double tolerance = 1e-4;
    int quiet_mode = 0, accel_mode = 0, use_kdtree = 1;
    int nystrom_J = 300, nystrom_K = 70;
    int gauss_kernel_type = 0;
    double tau = 0.0;
    int knn = 0;
    double radius = 0.0;
    char normalization_type = 'e';  // 默认值
    PyObject *py_normalization_type = NULL;  // 接收Python参数
    int transformation_type = 0;
    double search_scale = 7.0, search_radius = 0.15, sigma_threshold = 0.2;
    int downsampling_flag = 0, dwn_X = 0, dwn_Y = 0;
    double dwr_X = 0.0, dwr_Y = 0.0;
    int save_options = 0;
    
    static char *kwlist[] = {
        "X", "Y", "random_seed", "omega", "lambda", "beta", "gamma", "kappa",
        "max_iter", "min_iter", "tolerance", "quiet_mode", "accel_mode",
        "nystrom_J", "nystrom_K", "use_kdtree", "gauss_kernel_type", "tau",
        "knn", "radius", "normalization_type", "transformation_type",
        "search_scale", "search_radius", "sigma_threshold", "downsampling_flag",
        "dwn_X", "dwn_Y", "dwr_X", "dwr_Y", "save_options", NULL
    };
    
    // 重要修改：这里使用"s"而不是"O"来接收normalization_type参数
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!|idddddiidiiiiiididOidddiiiddi:bcpd_register", kwlist,
                                      &PyArray_Type, &X_array,
                                      &PyArray_Type, &Y_array,
                                      &random_seed, &omega, &lambda, &beta, &gamma, &kappa,
                                      &max_iter, &min_iter, &tolerance, &quiet_mode, &accel_mode,
                                      &nystrom_J, &nystrom_K, &use_kdtree, &gauss_kernel_type, &tau,
                                      &knn, &radius, &py_normalization_type, &transformation_type,
                                      &search_scale, &search_radius, &sigma_threshold, &downsampling_flag,
                                      &dwn_X, &dwn_Y, &dwr_X, &dwr_Y, &save_options))
        return NULL;
        // printf("In PyObject *bcpd_register nystrom_J=%d\n", nystrom_J);

        /* 处理normalization_type参数 */
        if (py_normalization_type != NULL) {
            if (PyUnicode_Check(py_normalization_type)) {
                /* 如果是Unicode字符串 */
                Py_ssize_t length;
                const char *str = PyUnicode_AsUTF8AndSize(py_normalization_type, &length);
                if (str != NULL && length > 0) {
                    normalization_type = str[0];
                }
            } else if (PyBytes_Check(py_normalization_type)) {
                /* 如果是字节字符串 */
                char *str = PyBytes_AsString(py_normalization_type);
                if (str != NULL && PyBytes_Size(py_normalization_type) > 0) {
                    normalization_type = str[0];
                }
            } else if (PyLong_Check(py_normalization_type)) {
                /* 如果是整数 */
                int val = (int)PyLong_AsLong(py_normalization_type);
                if (val >= 0 && val <= 255) {
                    normalization_type = (char)val;
                } else {
                    PyErr_SetString(PyExc_ValueError, "normalization_type整数值必须在0-255范围内");
                    return NULL;
                }
            } else if (PyFloat_Check(py_normalization_type)) {
                /* 如果是浮点数 */
                double val = PyFloat_AsDouble(py_normalization_type);
                if (val == (int)val && val >= 0 && val <= 255) {
                    normalization_type = (char)(int)val;
                } else {
                    PyErr_SetString(PyExc_TypeError, "normalization_type如果是浮点数，必须是0-255范围内的整数值");
                    return NULL;
                }
            } else {
                PyErr_Format(PyExc_TypeError, 
                            "normalization_type必须是字符串、字节或整数，不能是%s", 
                            Py_TYPE(py_normalization_type)->tp_name);
                return NULL;
            }
        }
   
    /* 验证输入数组 */
    if (PyArray_NDIM(X_array) != 2 || PyArray_NDIM(Y_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "输入点云必须是2维数组 (点数 x 维度)");
        return NULL;
    }
    
    if (PyArray_TYPE(X_array) != NPY_DOUBLE || PyArray_TYPE(Y_array) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "输入数组必须是双精度浮点类型");
        return NULL;
    }
    
    /* 获取维度 */
    int N = (int)PyArray_DIM(X_array, 0);
    int D = (int)PyArray_DIM(X_array, 1);
    int M = (int)PyArray_DIM(Y_array, 0);

    
    if (PyArray_DIM(Y_array, 1) != D) {
        PyErr_SetString(PyExc_ValueError, "源点云和目标点云必须有相同的维度");
        return NULL;
    }
    
    /* 获取数据指针 */
    double *X_ptr = (double *)PyArray_DATA(X_array);
    double *Y_ptr = (double *)PyArray_DATA(Y_array);
    
    /* 创建输出数组 */
    npy_intp y_dims[2] = {M, D};
    PyObject *y_out_obj = PyArray_SimpleNew(2, y_dims, NPY_DOUBLE);
    PyObject *v_out_obj = PyArray_SimpleNew(2, y_dims, NPY_DOUBLE);
    
    npy_intp R_dims[2] = {D, D};
    PyObject *R_out_obj = PyArray_SimpleNew(2, R_dims, NPY_DOUBLE);
    
    npy_intp t_dims[1] = {D};
    PyObject *t_out_obj = PyArray_SimpleNew(1, t_dims, NPY_DOUBLE);
    
    if (!y_out_obj || !v_out_obj || !R_out_obj || !t_out_obj) {
        Py_XDECREF(y_out_obj);
        Py_XDECREF(v_out_obj);
        Py_XDECREF(R_out_obj);
        Py_XDECREF(t_out_obj);
        PyErr_SetString(PyExc_MemoryError, "无法创建输出数组");
        return NULL;
    }
    
    PyArrayObject *y_out = (PyArrayObject *)y_out_obj;
    PyArrayObject *v_out = (PyArrayObject *)v_out_obj;
    PyArrayObject *R_out = (PyArrayObject *)R_out_obj;
    PyArrayObject *t_out = (PyArrayObject *)t_out_obj;
    
    /* 获取输出指针 */
    double *y_ptr = (double *)PyArray_DATA(y_out);
    double *v_ptr = (double *)PyArray_DATA(v_out);
    double *R_ptr = (double *)PyArray_DATA(R_out);
    double *t_ptr = (double *)PyArray_DATA(t_out);
    
    double s_out = 0.0;
    double sigma_out = 0.0;
    
    /* 调用C包装器函数 */
    int iterations = bcpd_wrapper(
        X_ptr, N, D,
        Y_ptr, M,
        random_seed,
        omega, lambda, beta, gamma, kappa,
        max_iter, min_iter, tolerance,
        y_ptr, v_ptr, &s_out, R_ptr, t_ptr,
        &sigma_out, quiet_mode,
        accel_mode, nystrom_J, nystrom_K, use_kdtree,
        gauss_kernel_type, tau, knn, radius,
        normalization_type,
        transformation_type,
        search_scale, search_radius, sigma_threshold,
        downsampling_flag, dwn_X, dwn_Y, dwr_X, dwr_Y,
        save_options
    );
    
    /* 创建返回字典 */
    PyObject *result = PyDict_New();
    if (!result) {
        Py_DECREF(y_out);
        Py_DECREF(v_out);
        Py_DECREF(R_out);
        Py_DECREF(t_out);
        return NULL;
    }
    
    PyDict_SetItemString(result, "iterations", PyLong_FromLong(iterations));
    PyDict_SetItemString(result, "transformed_points", (PyObject *)y_out);
    PyDict_SetItemString(result, "displacement_field", (PyObject *)v_out);
    PyDict_SetItemString(result, "scale", PyFloat_FromDouble(s_out));
    PyDict_SetItemString(result, "rotation", (PyObject *)R_out);
    PyDict_SetItemString(result, "translation", (PyObject *)t_out);
    PyDict_SetItemString(result, "sigma", PyFloat_FromDouble(sigma_out));
    
    return result;
}


/* 模块方法定义 */
static PyMethodDef BCPDMethods[] = {
    {"bcpd", (PyCFunction)bcpd_register, METH_VARARGS | METH_KEYWORDS,
     "执行BCPD点云配准算法"},
    {NULL, NULL, 0, NULL}
};

/* 模块定义 */
static struct PyModuleDef bcpdmodule = {
    PyModuleDef_HEAD_INIT,
    "_bcpd",   /* 模块名称与原pybind11扩展相同 */
    "BCPD算法的纯C接口",
    -1,
    BCPDMethods
};

/* 初始化模块 */
PyMODINIT_FUNC
PyInit__bcpd(void)
{
    import_array();  /* 初始化NumPy */
    return PyModule_Create(&bcpdmodule);
}