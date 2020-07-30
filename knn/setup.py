from distutils.core import setup, Extension
import os
import numpy as np

os.environ["CC"] = "clang"
os.environ["CXX"] = "clang++"

# os.environ["CC"] = "gcc"
# os.environ["CXX"] = "g++"

KNN = Extension('KNN',
                sources = ['knn.cpp', 'knnmodule.cpp'],
                depends = ['knn.h'],
                include_dirs = ['/usr/local/include', np.get_include()],
                library_dirs = ['/usr/local/lib'],
                libraries = ['gsl', 'gslcblas'],
                extra_compile_args = ['-std=c++1y', '-lm', '-pthread', '-lgsl', '-lgslcblas', '-Ofast', '-march=native', '-ffast-math']
)

setup(name = 'KNN',
        version = '0.1',
        description = 'KNN',
        ext_modules = [KNN])