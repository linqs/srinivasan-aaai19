## @package wrapper
#Documentation for this module.
#
#


from cython.operator cimport preincrement as inc
import numpy as np
cimport numpy as np

#there is a bug with typing certain variables. To circumvent it we need to define our own type and pass it to the compiler otherwise the function will pass garbage to the c++ function
ctypedef np.uintp_t itype_t


cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        vector()
        void push_back(T&)
        T& operator[](int)
        T& at(int)
        iterator begin()
        iterator end()
        int size()


cdef extern from "saucy_src/fc.h":
    vector[np.intp_t] equitablePartitionSaucyBipartite(const size_t nrows, const size_t ncols, const size_t medges, const size_t data[], const size_t rowind[], const size_t colind[], const size_t b[], const size_t c[], int cIters, int coarsest, int verbose);
    vector[itype_t] equitablePartitionSaucyV2(itype_t mvertices, itype_t medges, double data[], itype_t rown[], itype_t coln[], itype_t b[], int cIters, int coarsest);

## Computes and equiable partition of given matrix,which is passed over as a coordinate matrix
#
#@param A -- the data vector of a scipy coordinate matrix
#@param B -- the row vector of a scipy coordinate matrix
#@param C -- the column vector of a scipy coordinate matrix
#@param Z -- numpy inverted dense matrix
#@param cIters -- TODO
#@param coarsest -- TODO
def epSaucy(
    np.ndarray[np.double_t,ndim=1] A,
    np.ndarray[itype_t,ndim=1] B,
    np.ndarray[itype_t,ndim=1] C,
    np.ndarray[itype_t,ndim=1] Z,
    cIters = 0,
    coarsest = True):
    # Pass references to c++ function to guarantee correct memory access
    cdef vector[size_t] res = equitablePartitionSaucyV2(Z.shape[0], A.shape[0], &A[0], &B[0], &C[0], &Z[0], cIters, 1 if coarsest else 0)
    
    #Get size of result and copy elements into numpy array
    i = res.size()
    cdef np.ndarray result = np.empty(i)
    for x in range(0,i):
     result[x] = res[x]
   
    
    #Print and return resulting numpy array
    return result

## TODO : Description of epSaucyBipartite
#
#@param A The data vector of a scipy coordinate matrix
#@param rows The row vector of a scipy coordinate matrix
#@param cols The column vector of a scipy coordinate matrix
#@param rowcolor TODO
#@param cIters TODO
#@param coarsest TODO
def epSaucyBipartite(
    np.ndarray[itype_t,ndim=1] A,
    np.ndarray[itype_t,ndim=1] rows,
    np.ndarray[itype_t,ndim=1] cols,
    np.ndarray[itype_t,ndim=1] rowcolor,
    np.ndarray[itype_t,ndim=1] colcolor,
    cIters = 0,
    orbits = False,
    verbose = False):

    # Pass references to c++ function to guarantee correct memory access
    cdef vector[int] res = equitablePartitionSaucyBipartite(rowcolor.shape[0], colcolor.shape[0], A.shape[0], &A[0], &rows[0], &cols[0], &rowcolor[0], &colcolor[0], cIters, 0 if orbits else 1, verbose)
    
    #Get size of result and copy elements into numpy array
    i = res.size()
    cdef np.ndarray result = np.empty(i)
    for x in range(0,i):
     result[x] = res[x]
   
    
    #Print and return resulting numpy array
    return result


