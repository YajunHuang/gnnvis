#include "Python.h"
#include "numpy/arrayobject.h"
#include "knn.h"


static PyObject *build_knn_index_numpy(PyObject *self, PyObject *args)
{
    PyArrayObject *v, *result;
    long long n_neighbor = 1;
    int n_tree = -1; 
    int n_propagation = 3;
    int n_thread = 1;
    real perplexity = -1;

    if (!PyArg_ParseTuple(args, "OLiifi", &v, &n_neighbor, &n_tree, &n_propagation, &perplexity, &n_thread))
    {
        std::cout << "Input error!\n";
        return Py_None;
    }
    std::cout << "------------- Build knn index -------------" << std::endl;
    std::cout << "n_neighbor=" << n_neighbor << std::endl;
    std::cout << "n_tree=" << n_tree << std::endl;
    std::cout << "n_propagation=" << n_propagation << std::endl;
    std::cout << "perplexity=" << perplexity << std::endl;
    std::cout << "n_thread=" << n_thread << std::endl;

    int n_samples = v->dimensions[0];
    int n_dim = v->dimensions[1];
    double *vdata = (double *) v->data;
    real *cdata = new real[n_samples * n_dim];

    for (int i = 0; i < n_samples; i++) 
    {
        for (int j = 0; j < n_dim; j++) 
        {
            cdata[i*n_dim + j] = (real) vdata[i*n_dim + j];
        }
    }
    KNN model(n_neighbor, n_thread);
    model.load_data(cdata, n_samples, n_dim);
    model.construct_knn(n_tree, n_propagation, perplexity);

    npy_intp result_dims[1] = {(int) model.edge_from.size()};
    PyArrayObject *edge_from, *edge_to, *edge_weight;
    edge_from = (PyArrayObject *) PyArray_SimpleNew(1, result_dims, NPY_INT);
    edge_to = (PyArrayObject *) PyArray_SimpleNew(1, result_dims, NPY_INT);
    edge_weight = (PyArrayObject *) PyArray_SimpleNew(1, result_dims, NPY_FLOAT);

    int *cedge_from = (int *) edge_from->data;
    int *cedge_to = (int *) edge_to->data;
    real *cedge_weight = (real *) edge_weight->data;

    for (int i = 0; i < model.edge_from.size(); i++) 
    {
        cedge_from[i] = (int) model.edge_from[i];
        cedge_to[i] = (int) model.edge_to[i];
        cedge_weight[i] = (real) model.edge_weight[i];
    }
    std::cout << model.edge_from.size() << " edges in knn graph." << std::endl;
    PyObject *py_list = Py_BuildValue("(OOO)", edge_from, edge_to, edge_weight);

    return py_list;
}

// static PyObject *build_knn_index(PyObject *self, PyObject *args)
// {
//     PyObject *v;
//     long long n_neighbor = 1;
//     int n_tree = -1; 
//     int n_propagation = 3;
//     int n_thread = 1;
//     real perplexity = -1;
//     if (!PyArg_ParseTuple(args, "OLnnfn", &v, &n_neighbor, &n_tree, &n_propagation, &perplexity, &n_thread))
//     {
//         std::cout << "Input error!\n";
//         return Py_None;
//     }
//     long long n_dim = PyList_Size(PyList_GetItem(v, 0));
// 	long long n_samples = PyList_Size(v);

//     std::cout << "data shape = (" << n_samples << ", " << n_dim << ")" << std::endl;
//     real *data = new real[n_samples * n_dim];
//     // std::cout << std::setprecision(2) << std::fixed;
//     for (long long i = 0; i < n_samples; i++)
//     {
//         PyObject *vec = PyList_GetItem(v, i);
//         if (i % 3000 == 0 || i == n_samples - 1)
// 		{
//             std::cout << "Reading feature vectors " << i * 100.0 / n_samples << "%" << std::endl;
// 		}
// 		if (PyList_Size(vec) != n_dim)
// 		{
// 			printf("Input dimension error!\n");
// 			return Py_None;
// 		}
//         for (long long j = 0; j < n_dim; j++)
//         {
//             real x = (real)(PyFloat_AsDouble(PyList_GetItem(vec, j)));
//             data[i * n_dim + j] = x;
//         }
//     }
//     KNN model(n_neighbor, n_propagation);
//     std::cout << "load_from_data" << std::endl;
//     model.load_data(data, n_samples, n_dim);
//     std::cout << "construct_knn" << std::endl;
//     model.construct_knn(n_tree, n_propagation, perplexity);
//     return Py_None;
// }


// Method table
static PyMethodDef PyExtMethods[] = 
{
    {"build_knn_index", build_knn_index_numpy, METH_VARARGS, "build_knn_index(data, n_neigh, n_trees, n_propagation, perplexity, n_thread)"},
    // {"build_knn_index", build_knn_index, METH_VARARGS, "build_knn_index_numpy(data, n_neigh, n_trees, n_propagation, perplexity, n_thread)"},
    { NULL, NULL, 0, NULL }             /* Sentinel - marks the end of this structure */
};

// The method table must be referenced in the module definition structure
static struct PyModuleDef knnmodule = {
    PyModuleDef_HEAD_INIT,
    "knn",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    PyExtMethods
};

// Initialization function
PyMODINIT_FUNC PyInit_KNN(void)
{
    import_array();  // Must be present for NumPy.  Called first after above line.
   // printf("knn successfully imported!\n");
    return PyModule_Create(&knnmodule);
}