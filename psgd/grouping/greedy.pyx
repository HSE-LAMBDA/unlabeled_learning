import numpy as np
cimport numpy as cnp

ctypedef cnp.int64_t int64
ctypedef cnp.float32_t float32
ctypedef cnp.float64_t float64

import cython

from sklearn.neighbors import NearestNeighbors

from libc.math cimport sqrt
from libc.stdlib cimport rand

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float32 l2(float32[:] a, float32[:] b):
  cdef int i
  cdef double s = 0.0

  for i in range(a.shape[0]):
    s += (a[i] - b[i]) ** 2

  return sqrt(s)

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def randomized_greedy_pairing(float32[:, :] X_pos, float32[:, :] X_neg, int max_trials=100, int q=25, int n_restart=100):
  cdef int i, j, k
  cdef int i_pos, i_neg, best_i
  cdef int i_restart
  cdef float32 d

  cdef float32[:] stats = np.ndarray(shape=(max_trials * 10), dtype='float32')

  for k in range(stats.shape[0]):
    i = rand() % X_pos.shape[0]
    j = rand() % X_neg.shape[0]

    stats[k] = l2(X_pos[i], X_neg[j])

  cdef float32 threshold = np.percentile(stats, q=q)

  cdef set free_negatives = set()
  cdef int64[:] left_neg

  for i in range(X_neg.shape[0]):
    free_negatives.add(i)

  i_pos = 0
  i_restart = 0

  for i_pos in range(X_pos.shape[0]):
    if i_restart % n_restart == 0:
      left_neg = np.array(list(free_negatives))

    best_i = -1

    for j in range(max_trials):
      i_neg = left_neg[rand() % left_neg.shape[0]]
      while i_neg not in free_negatives:
        i_neg = left_neg[rand() % left_neg.shape[0]]

      d = l2(X_pos[i_pos], X_neg[i_neg])

      if d < threshold:
        pass

    i_restart += 1

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def greedy_pairing_labels(X, y, n_neighbors=10, *args, **kwargs):
  indx_pos = np.where(y == 1)[0]
  indx_neg = np.where(y == 0)[0]

  cdef int64[:, :] pairs = np.ndarray(shape=(indx_neg.shape[0], 2), dtype='int64')
  cdef int i

  cdef int64[:] indx = greedy_pairing(X[indx_pos], X[indx_neg], n_neighbors, *args, **kwargs)

  for i in range(indx.shape[0]):
    pairs[i, 0] = indx_pos[i]
    pairs[i, 1] = indx_neg[i]

  return pairs


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def greedy_pairing(X_pos, X_neg, n_neighbors=10, *args, **kwargs):
  cdef int64[:] indx_pairs = np.zeros(shape=(X_pos.shape[0], ), dtype='int64')

  nn = NearestNeighbors(n_neighbors=n_neighbors, *args, **kwargs)

  cdef set pos_paired_points = set()
  cdef set pos_unpaired_points = set()

  cdef set neg_paired_points = set()
  cdef set neg_unpaired_points = set()

  cdef int64[:] neg_left
  cdef int64[:] pos_left

  cdef int64[:, :] neighbors
  cdef int64[:] lindx
  cdef int i, j, n

  for i in range(X_pos.shape[0]):
    neg_unpaired_points.add(i)
    pos_unpaired_points.add(i)

  while len(pos_unpaired_points) > 0:
    neg_left = np.array(list(neg_unpaired_points))
    pos_left = np.array(list(pos_unpaired_points))

    nn.fit(X_neg[neg_left, :])

    distance, neighbors = nn.kneighbors(X_pos[pos_left, :], n_neighbors=min(n_neighbors, pos_left.shape[0]), return_distance=True)
    print(np.array(distance))
    print(np.array(neighbors))

    for i in range(neighbors.shape[0]):
      lindx = np.argsort(distance[i, :])

      for j in lindx:
        n = neighbors[i, j]

        if neg_left[n] in neg_unpaired_points:
          neg_paired_points.add(neg_left[n])
          pos_paired_points.add(pos_left[i])

          neg_unpaired_points.remove(neg_left[n])
          pos_unpaired_points.remove(pos_left[i])

          indx_pairs[pos_left[i]] = neg_left[n]
          break


  return indx_pairs