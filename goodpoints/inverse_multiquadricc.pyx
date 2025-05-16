import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef double inverse_multiquadric_kernel_two_points(const double[:] X1,
                                                      const double[:] X2,
                                                      const double[:] c) noexcept nogil:
    """
    Computes the inverse multiquadratic kernel 
    k(X1, X2) = 1 / sqrt(c + ||X1 - X2||^2)

    Args:
      X1: array of size d
      X2: array of size d
      c: array of size 1, scalar > 0 stored at c[0]
    """
    cdef long d = X1.shape[0]
    cdef double sq_dist = 0.
    cdef double diff
    cdef long j

    for j in range(d):
        diff = X1[j] - X2[j]
        sq_dist += diff * diff
        
    return 1. / sqrt(c[0] + sq_dist)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef double inverse_multiquadric_kernel_one_point(const double[:] X1,
                                                     const double[:] c) noexcept nogil:
    """
    Computes the inverse multiquadratic kernel between X1 and itself:
    k(X1, X1) = 1 / sqrt(c[0])

    Args:
      X1: array of size d
      c: array of size 1, scalar > 0 stored at c[0]
    """
    return 1. / sqrt(c[0])
