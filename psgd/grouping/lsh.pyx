import numpy as np
cimport numpy as cnp

ctypedef cnp.int64_t int64
ctypedef cnp.float32_t float32
ctypedef cnp.float64_t float64

import cython

from libc.stdlib cimport rand

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef list _sub_split(
  cnp.ndarray[cnp.float32_t, ndim=2] projections,
  cnp.ndarray[cnp.int64_t, ndim=1] indx,
  split_seq, int split_num=0
):

  if split_num >= len(split_seq):
    return [ indx ]

  cdef int num_groups = split_seq[split_num]
  cdef int group_size = indx.shape[0] // num_groups
  cdef int i

  cdef cnp.ndarray[cnp.int64_t, ndim=1] sub_indx = np.argsort(projections[indx, split_num])

  for i in range(sub_indx.shape[0]):
    sub_indx[i] = indx[sub_indx[i]]

  cdef list groups = list()
  cdef el

  for i in range(num_groups):
    for el in _sub_split(
      projections,
      sub_indx[i * group_size : (i + 1) * group_size],
      split_seq, split_num + 1
    ):
      groups.append(el)

  return groups


@cython.boundscheck(False)
@cython.wraparound(False)
def lsh_random_grid(cnp.ndarray[cnp.float32_t, ndim=2] X, project, tuple split_seq):
  cdef int num_splits = len(split_seq)

  cdef cnp.ndarray[cnp.float32_t, ndim=2] n = \
      np.random.normal(0, 1, size=(num_splits, X.shape[1])).astype('float32')

  cdef int i

  cdef cnp.ndarray[cnp.float32_t, ndim=2] projections = project(X, n)

  return _sub_split(
    projections,
    np.arange(X.shape[0], dtype='int64'),
    split_seq=split_seq,
    split_num=0
  )

@cython.boundscheck(False)
@cython.wraparound(False)
def lsh_random_point_grid(cnp.ndarray[cnp.float32_t, ndim=2] X, project, tuple split_seq, int num_random_points=2):
  cdef int num_splits = len(split_seq)

  cdef cnp.ndarray[cnp.float32_t, ndim=2] n = \
      np.zeros(shape=(num_splits, X.shape[1]), dtype='float32')

  cdef int i, j

  for i in range(num_splits):
    for j in range(num_random_points):
      n[i, :] += X[rand() % X.shape[0]]

  n /= num_random_points

  cdef cnp.ndarray[cnp.float32_t, ndim=2] projections = project(X, n)

  return _sub_split(
    projections,
    np.arange(X.shape[0], dtype='int64'),
    split_seq=split_seq,
    split_num=0
  )


@cython.boundscheck(False)
@cython.wraparound(False)
def lsh_random_tree(cnp.ndarray[cnp.float32_t, ndim=2] X, project, tuple split_seq, int64[:] indx=None):
  if len(split_seq) == 0:
    return [ indx ]

  if indx is None:
    indx = np.arange(X.shape[0], dtype='int64')

  cdef int num_groups = split_seq[0]

  cdef int group_size = indx.shape[0] // num_groups

  cdef float32[:, :] n = np.random.normal(0, 1, size=(1, X.shape[1])).astype('float32')
  cdef int64[:] sub_indx = np.argsort(project(X[indx], n).reshape(-1))

  cdef int i
  for i in range(sub_indx.shape[0]):
    sub_indx[i] = indx[sub_indx[i]]

  cdef list groups = list()

  for i in range(num_groups):
    for el in lsh_random_tree(
      X, project,
      split_seq[1:],
      sub_indx[(i * group_size):(( i + 1) * group_size)],
    ):
      groups.append(el)

  return groups

@cython.boundscheck(False)
@cython.wraparound(False)
def lsh_random_points_tree(cnp.ndarray[cnp.float32_t, ndim=2] X, project, tuple split_seq, int64[:] indx=None, int num_random_points=2):
  if len(split_seq) == 0:
    return [ indx ]

  cdef int i, k

  if indx is None:
    indx = np.arange(X.shape[0], dtype='int64')

  cdef int num_groups = split_seq[0]

  cdef int group_size = indx.shape[0] // num_groups

  cdef cnp.ndarray[cnp.float32_t, ndim=1] n = np.zeros(shape=X.shape[1], dtype='float32')
  for i in range(num_random_points):
    k = indx[rand() % indx.shape[0]]
    n += X[k]

  n /= num_random_points


  cdef int64[:] sub_indx = np.argsort(project(X[indx], n.reshape(1, -1)).reshape(-1))

  for i in range(sub_indx.shape[0]):
    sub_indx[i] = indx[sub_indx[i]]

  cdef list groups = list()

  for i in range(num_groups):
    for el in lsh_random_points_tree(
      X, project,
      split_seq[1:],
      sub_indx[(i * group_size):(( i + 1) * group_size)],
      num_random_points=num_random_points
    ):
      groups.append(el)

  return groups


