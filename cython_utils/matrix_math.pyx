import cython
import numpy as np


@cython.boundscheck(False)
cpdef int[:] argsort(int[:,:] arr, int[:] out):
    cdef size_t i
    I = arr.shape[0]
    for i in range(I):
        out[i] = get_sort_index(arr[i])
    return out


@cython.boundscheck(False)
cpdef int get_sort_index(int[:] arr) nogil:
    cdef size_t i
    cdef int idx = 0
    cdef int min_val = arr[0]
    I = arr.shape[0]
    for i in range(I):
        if arr[i] < min_val:
            min_val = arr[i]
            idx = i
    return idx


@cython.boundscheck(False)
cpdef int euclidean_distance_2d(int[:, :] array_1, int[:, :] array_2) nogil:
    cdef size_t i, j
    cdef int total = 0
    cdef int dx
    I = array_1.shape[0]
    J = array_1.shape[1]
    for i in range(I):
        for j in range(J):
            dx = array_1[i, j] - array_2[i, j]
            total += dx * dx
    return total


@cython.boundscheck(False)
cpdef int[:, :] euclidean_distance_matrix_2d(int[:, :, :] image_tiles, int[:, :, :] sample_tiles, int[:, :] out):
    cdef size_t i, j
    I = image_tiles.shape[0]
    J = sample_tiles.shape[0]
    for i in range(I):
        for j in range(J):
            out[i, j] = euclidean_distance_2d(sample_tiles[j], image_tiles[i])
    return out


@cython.boundscheck(False)
cpdef int euclidean_distance_1d(int[:] array_1, int[:] array_2) nogil:
    cdef size_t i
    cdef int total = 0
    cdef int dx
    I = array_1.shape[0]
    for i in range(I):
        dx = array_1[i] - array_2[i]
        total += dx * dx
    return total


@cython.boundscheck(False)
cpdef int[:, :] euclidean_distance_matrix_1d(int[:, :] image_tiles, int[:, :] sample_tiles, int[:, :] out):
    cdef size_t i, j
    I = image_tiles.shape[0]
    J = sample_tiles.shape[0]
    for i in range(I):
        for j in range(J):
            out[i, j] = euclidean_distance_1d(sample_tiles[j], image_tiles[i])
    return out

