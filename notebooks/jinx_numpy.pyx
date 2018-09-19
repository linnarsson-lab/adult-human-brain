import cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport sqrt, fabs, log10
from cython.parallel import threadid, parallel, prange

