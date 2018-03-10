import cython

import numpy as np
cimport numpy as cnp

ctypedef cnp.float32_t float32
ctypedef cnp.float64_t float64
ctypedef cnp.int64_t int64

cdef class SimpleSampler:
  """
    This class is a result of lazyness to read and search for an efficient tree algorithms.
  """
  cdef readonly int64 n_objects

  cdef readonly float32[:] probs
  cdef readonly float64[:] cum_probs

  cdef readonly float64 normalization

  def __init__(self, n_objects):
    self.n_objects = n_objects
    self.probs = np.ones(shape=(self.n_objects, ), dtype='float32')
    self.cum_probs = np.ones(shape=(self.n_objects, ), dtype='float64')

    self.update()

  @cython.nonecheck(False)
  @cython.boundscheck(False)
  @cython.wraparound(False)
  def update(self, ):
    cdef float64 sum = 0.0
    cdef int i

    for i in range(self.cum_probs.shape[0]):
      sum += self.probs[i]
      self.cum_probs[i] = sum

    self.normalization = sum

  @cython.nonecheck(False)
  @cython.boundscheck(False)
  @cython.wraparound(False)
  def find(self, float64 r):
    cdef int64 left = 0, right = self.n_objects - 1, prob = self.n_objects // 2

    if self.cum_probs[left] > r:
      return 0

    while right - left > 1:
      if self.cum_probs[prob] > r:
        right = prob
      else:
        left = prob

      prob = (right + left) // 2

    return right


  @cython.nonecheck(False)
  @cython.boundscheck(False)
  @cython.wraparound(False)
  def sample(self, int num):
    cdef float64[:] rs = np.random.uniform(0, self.normalization, size=num)
    cdef int64[:] indx = np.ndarray(shape=(num, ), dtype='int64')
    cdef float32[:] probs = np.ndarray(shape=(num, ), dtype='float32')
    cdef int i

    for i in range(num):
      indx[i] = self.find(rs[i])
      probs[i] = self.probs[indx[i]] / self.normalization

    return indx, probs

  @cython.nonecheck(False)
  @cython.boundscheck(False)
  @cython.wraparound(False)
  def set_probas(self, float32[:] probs):
    if probs.shape[0] != self.n_objects:
      self.n_objects = probs.shape[0]
      self.cum_probs = np.ndarray(shape=(self.n_objects, ), dtype='float64')

    self.probs = probs
    self.update()
