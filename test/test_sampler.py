from psgd.sampling import SimpleSampler
import numpy as np
from scipy import stats

def test_psgd():
  sampler = SimpleSampler(10)

  for i in range(10):
    sampler.set(i, i + 1)

  print(np.array(sampler.probs))
  sampler.update()

  N = 1000
  indx = sampler.sample(N)

  h, _ = np.histogram(indx, bins=10)

  expected = np.array([i + 1 for i in range(10)], dtype='float64')
  expected /= np.sum(expected)
  expected *= N

  print(expected)
  print(h)

  assert stats.chisquare(h, expected)[1] > 0.95
