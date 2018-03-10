import cython
import numpy as np
cimport numpy as cnp

from numpy.math cimport sqrtf

ctypedef cnp.float32_t float32
ctypedef cnp.float64_t float64
ctypedef cnp.int64_t int64

cdef inline float32 sq_l2(float32[:] a, float32[:] b):
  cdef int i
  cdef float32 s = 0.0

  for i in range(a.shape[0]):
    s += (a[i] - b[i]) ** 2

  return s

cdef inline float32 l2(float32[:] a, float32[:] b):
  return sqrtf(l2(a, b))
