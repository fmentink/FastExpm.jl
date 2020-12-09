# fastExpm.jl, a fast exponential for large matrices

This is a fast implementation of exponential matrix for large full/sparse matrices, commonly found in (quantum) physics simulations.
It is significantly faster than Julia's exp implementation for matrices bigger than 16x16.

## Requirements
### Needed
- LinearAlgebra 
- SparseArrays

### Optinal
- MKL.jl (speed computations up)

## Description

    fastExpm(A)
    fastExpm(A; threshold=1e-6)
    fastExpm(A; nonzero_tol=1e-14)
    fastExpm(A; threshold=1e-6, nonzero_tol=1e-14)

 This function efficiently implements matrix exponentiation for sparse and full matrices.
 This code is based on scaling, taylor series and squaring.
 Currently works only on the CPU

 Two optional keyword arguments are used to speed up the computation and preserve sparsity.
- "threshold" determines the threshold for the Taylor series (default 1e-6), e.g. fastExpm(A, threshold=1e-10) 
- "nonzero_tol" strips elements smaller than nonzero_tol at each computation step to preserve sparsity (default 1e-14) ,e.g. fastExpm(A, nonzero_tol=1e-10)

 This code was originally developed by Ilya Kuprov (http://spindynamics.org/) and has been adapted by F. Mentink-Vigier (fmentink@magnet.fsu.edu)
 and Murari Soundararajan (murari@magnet.fsu.edu). If you use this code, please cite
 - H. J. Hogben, M. Krzystyniak, G. T. P. Charnock, P. J. Hore and I. Kuprov, Spinach – A software library for simulation of spin dynamics in large spin systems, J. Magn. Reson., 2011, 208, 179–194.
 - I. Kuprov, Diagonalization-free implementation of spin relaxation theory for large spin systems., J. Magn. Reson., 2011, 209, 31–38.
