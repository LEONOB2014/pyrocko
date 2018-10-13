
#define NPY_NO_DEPRECATED_API 7

#define SQR(a)  ( (a) * (a) )

#include "Python.h"
#include "numpy/arrayobject.h"

/*
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
*/
#if defined(_OPENMP)
    #include <omp.h>
#endif

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state); (void) m;
static struct module_state _state;
#endif

int good_array(PyObject* o, int typenum_want, npy_intp size_want, int ndim_want, npy_intp* shape_want) {
    int i;

    if (!PyArray_Check(o)) {
        PyErr_SetString(PyExc_AttributeError, "not a NumPy array" );
        return 0;
    }

    if (PyArray_TYPE((PyArrayObject*)o) != typenum_want) {
        PyErr_SetString(PyExc_AttributeError, "array of unexpected type");
        return 0;
    }

    if (!PyArray_ISCARRAY((PyArrayObject*)o)) {
        PyErr_SetString(PyExc_AttributeError, "array is not contiguous or not well behaved");
        return 0;
    }

    if (size_want != -1 && size_want != PyArray_SIZE((PyArrayObject*)o)) {
        PyErr_SetString(PyExc_AttributeError, "array is of unexpected size");
        return 0;
    }
    if (ndim_want != -1 && ndim_want != PyArray_NDIM((PyArrayObject*)o)) {
        PyErr_SetString(PyExc_AttributeError, "array is of unexpected ndim");
        return 0;
    }

    if (ndim_want != -1 && shape_want != NULL) {
        for (i=0; i<ndim_want; i++) {
            if (shape_want[i] != -1 && shape_want[i] != PyArray_DIMS((PyArrayObject*)o)[i]) {
                PyErr_SetString(PyExc_AttributeError, "array is of unexpected shape");
                return 0;
            }
        }
    }
    return 1;
}

typedef enum {
    SUCCESS = 0,
    MALLOC_FAILED = 1,
    HEAP_FULL = 2
} eikonal_error_t;

const char* eikonal_error_names[] = {
    "success",
    "memory allocation failed",
    "heap capacity exhausted",
};


typedef struct {
    size_t *indices;
    size_t n;
    size_t nmax;
} heap_t;


static heap_t *heap_new(size_t nmax) {
    heap_t *heap;

    heap = (heap_t*)malloc(sizeof(heap_t));
    if (heap == NULL) {
        return NULL;
    }

    heap->indices = (size_t*)calloc(nmax, sizeof(size_t));
    if (heap->indices == NULL) {
        free(heap);
        return NULL;
    }
    heap->nmax = nmax;
    heap->n = 0;
    return heap;
}


static void heap_delete(heap_t *heap) {
    if (heap == NULL) return;
    if (heap->indices != NULL) free(heap->indices);
    free(heap);
}

static eikonal_error_t heap_push(heap_t *heap, size_t index, double *keys, size_t *backpointers) {
    if (heap->n + 1 > heap->nmax) {
        return HEAP_FULL;
    }

    heap->n += 1;
    heap->indices[heap->n - 1] = index;
    backpointers[index] = heap->n - 1;
    heap_up(heap, heap->n - 1, keys, backpointers);
}

static size_t heap_pop(heap_t *heap, double *keys, size_t *backpointers) {
    size_t index;
    heap->n -= 1;
    swap(&(heap->indices[0]), &(heap->indices[heap->n-1]))
    swap(&(backpointers[heap->indices[0]]), &(backpointers[heap->indices[heap->n-1]]));
    index = heap->indices[n]
}

static eikonal_error_t eikonal_solver_fmm_cartesian(
        double *speeds,
        size_t ndim,
        size_t *shape,
        double *deltas,
        double *times) {

    heap_t *heap;
    size_t idim;
    size_t n;

    n = 1;
    for (idim=0; idim<ndim; idim++) {
        n *= shape[idim];
    }
    
    heap = heap_new(n);
    if (heap == NULL) {
        return MALLOC_FAILED;
    }


    (void) speeds;
    (void) deltas;
    (void) times;

    heap_delete(heap);

    return SUCCESS;
};


static PyObject* w_eikonal_solver_fmm_cartesian(PyObject *m, PyObject *args) {
    eikonal_error_t err;
    PyObject *speeds_arr, *times_arr, *deltas_arr;
    int ndim, i;
    npy_intp shape[3], size;
    size_t size_t_shape[3];
    double *speeds, *deltas, *times;

    struct module_state *st = GETSTATE(m);

    if (!PyArg_ParseTuple(args, "OOO", &speeds_arr, &times_arr, &deltas_arr)) {
        PyErr_SetString(st->error, "usage: eikonal_solver_fmm_cartesian(speeds, times, deltas)" );
        return NULL;
    }

    if (!good_array(speeds_arr, NPY_FLOAT64, -1, -1, NULL)) return NULL;
    ndim = PyArray_NDIM((PyArrayObject*)speeds_arr);
    if (!(2 <= ndim && ndim <= 3)) {
        PyErr_SetString(st->error, "only 2 and 3 dimensional inputs are supported.");
        return NULL;
    }

    size = 1;
    for (i=0; i<ndim; i++) {
        shape[i] = PyArray_DIMS((PyArrayObject*)speeds_arr)[i];
        size *= shape[i];
    }

    if (!good_array(times_arr, NPY_FLOAT64, size, ndim, shape)) return NULL;
    if (!good_array(deltas_arr, NPY_FLOAT64, ndim, 1, NULL)) return NULL;

    speeds = (double*)PyArray_DATA((PyArrayObject*)speeds_arr);
    times = (double*)PyArray_DATA((PyArrayObject*)times_arr);
    deltas = (double*)PyArray_DATA((PyArrayObject*)deltas_arr);

    for (i=0; i<ndim; i++) {
        size_t_shape[i] = shape[i];
    }

    err = eikonal_solver_fmm_cartesian(speeds, (size_t)ndim, size_t_shape, deltas, times);
    if (SUCCESS != err) {
        PyErr_SetString(st->error, eikonal_error_names[err]);
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef eikonal_ext_methods[] = {
    {"eikonal_solver_fmm_cartesian",  w_eikonal_solver_fmm_cartesian, METH_VARARGS,
        "Solve eikonal equation using the fast marching method." },

    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static int eikonal_ext_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int eikonal_ext_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "eikonal_ext",
        NULL,
        sizeof(struct module_state),
        eikonal_ext_methods,
        NULL,
        eikonal_ext_traverse,
        eikonal_ext_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_eikonal_ext(void)

#else
#define INITERROR return

void
initeikonal_ext(void)
#endif

{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("eikonal_ext", eikonal_ext_methods);
#endif
    import_array();

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("pyrocko.gf.eikonal_ext.EikonalExtError", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    Py_INCREF(st->error);
    PyModule_AddObject(module, "EikonalExtError", st->error);

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

