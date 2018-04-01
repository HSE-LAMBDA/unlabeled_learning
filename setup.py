"""
  p-SGD
"""

from setuptools import setup, find_packages, Extension

from codecs import open
import os
import os.path as osp
import numpy as np
import subprocess as sp

def get_includes():
  env = os.environ

  includes = []

  for k in ['CPATH', 'C_INCLUDE_PATH', 'INCLUDE_PATH']:
    if k in env:
      includes.append(env[k])

  return includes

def get_library_dirs():
  env = os.environ

  libs = []

  for k in ['LD_LIBRARY_PATH']:
    if k in env:
      libs.append(env[k])

  return libs

from Cython.Build import cythonize

here = osp.abspath(osp.dirname(__file__))

with open(osp.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

extra_compile_args=['-std=c++17', '-Ofast', '-D_GLIBCXX_USE_CXX11_ABI=1']

extensions = [
  Extension(
    'psgd.sampling.simple', ['psgd/sampling/simple.pyx'],
    libraries=['stdc++'],
    include_dirs=[np.get_include()] + get_includes(),
    library_dirs=get_library_dirs(),
    language='c++',
    extra_compile_args=extra_compile_args
  ),

  Extension(
    'psgd.grouping.greedy', ['psgd/grouping/greedy.pyx'],
    libraries=['stdc++'],
    include_dirs=[np.get_include()] + get_includes(),
    library_dirs=get_library_dirs(),
    language='c++',
    extra_compile_args=extra_compile_args
  ),

  Extension(
    'psgd.grouping.lsh', ['psgd/grouping/lsh.pyx'],
    libraries=['stdc++'],
    include_dirs=[np.get_include()] + get_includes(),
    library_dirs=get_library_dirs(),
    language='c++',
    extra_compile_args=extra_compile_args
  )
]

setup(
  name='psgd',

  version='0.0.1',

  description="""SGD that is p.""",

  long_description = long_description,

  url='no',

  setup_requires=['pytest-runner',],
  tests_require=['pytest',],

  author='LAMBDA',
  author_email='',

  maintainer = 'LAMBDA',
  maintainer_email = '',

  license='MIT',

  classifiers=[
    'Development Status :: 4 - Beta',

    'Intended Audience :: Science/Research',

    'License :: OSI Approved :: MIT License',

    'Programming Language :: Python :: 3',
  ],

  keywords='SGD',

  packages=find_packages(exclude=['contrib', 'examples', 'docs', 'tests']),

  extras_require={
    'dev': ['check-manifest'],
    'test': ['pytest>=1.3.0'],
  },

  install_requires=[
    'numpy',
    'cython',
    'scikit-learn'
  ],

  ext_modules = cythonize(extensions, gdb_debug=False),
)
