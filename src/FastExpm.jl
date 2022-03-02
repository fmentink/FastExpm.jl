module FastExpm

using LinearAlgebra
using SparseArrays
using CUDA
using CUDA.CUSPARSE
export fastExpm

"""
    fastExpm(A)
    fastExpm(A; threshold=1e-6)
    fastExpm(A; nonzero_tol=1e-14)
    fastExpm(A; threshold=1e-6, nonzero_tol=1e-14)

 This function efficiently implements matrix exponential for sparse and full matrices.
 This code is based on scaling, taylor series and squaring.
 Currently works only on the CPU

 Two optional keyword arguments are used to speed up the computation and preserve sparsity.
 [1] threshold determines the threshold for the Taylor series (default 1e-6): e.g. fastExpm(A, threshold=1e-10)
 [2] nonzero_tol strips elements smaller than nonzero_tol at each computation step to preserve sparsity (default 1e-14) ,e.g. fastExpm(A, nonzero_tol=1e-10)
 The code automatically switches from sparse to full if sparsity is below 25% to maintain speed.

 This code was originally developed by Ilya Kuprov (http://spindynamics.org/) and has been adapted by F. Mentink-Vigier (fmentink@magnet.fsu.edu)
 and Murari Soundararajan (murari@magnet.fsu.edu)
 If you use this code, please cite
  - H. J. Hogben, M. Krzystyniak, G. T. P. Charnock, P. J. Hore and I. Kuprov, Spinach – A software library for simulation of spin dynamics in large spin systems, J. Magn. Reson., 2011, 208, 179–194.
  - I. Kuprov, Diagonalization-free implementation of spin relaxation theory for large spin systems., J. Magn. Reson., 2011, 209, 31–38.
"""
function fastExpm(A::AbstractMatrix;threshold=1e-6,nonzero_tol=1e-14)
    mat_norm=norm(A,Inf);
    scaling_factor = nextpow(2,mat_norm); # Native routine, faster
    A = A./scaling_factor;
    delta=1;
    rows = LinearAlgebra.checksquare(A); # Throws exception if not square

    # Run Taylor series procedure on the CPU
    if nnz_ext(A)/(rows^2)>0.25 || rows<64
        if issparse(A)
            A=_dense(A);
        end
        P= one(A);#=_dense((1.0+0*im)*I,(rows,rows));=# next_term=P; n=1;
    else
        A=sparse(A);
        P= _sparse(((1.0+0*im)*I,(rows,rows)),A); next_term=P; n=1;
    end

    #show(P)
    while delta>threshold
        # Compute the next term
        if issparse(next_term)
            next_term=(1/n)*A*next_term;
            #Eliminate small elements
            next_term=droptolerance!(next_term, nonzero_tol);
            if nnz_ext(next_term)/length(next_term)>0.25 && issparse(next_term)
                next_term=_dense(next_term);
            end
        else
            next_term=(1/n)*next_term*A;
        end
        delta=norm(next_term,Inf);
        #Add to the total and increment the counter
        P .+= next_term; n=n+1;
    end
    #show(P)
    #Squaring of P to generate correct P
    for n=1:log2(scaling_factor)
        P=P*P;
        if issparse(P)
            if nnz_ext(P)/length(P)<0.25
                P = droptolerance!(P, nonzero_tol);
            else
                P=_dense(P);
            end
        end
    end
    return P
end

function droptolerance!(A::Union{Matrix,CuArray}, tolerance)
    A .= tolerance*round.((1/tolerance).*A)
end
function droptolerance!(A::SparseMatrixCSC, tolerance)
    droptol!(A, tolerance) # Native routine, faster
end

function nnz_ext(A::Matrix)
    count(x->x>0, abs.(A))
end
function nnz_ext(A::SparseMatrixCSC)
    nnz(A) # Native routine, faster
end
function nnz_ext(A::CuArray)
    sum(abs.(A).>0)
end
function nnz_ext(A::AbstractCuSparseMatrix)
    nnz(A) # Native routine, faster
end

function _dense(A::SparseMatrixCSC)
    Matrix(A)
end
function _dense(A::AbstractCuSparseMatrix)
    cu(A)
end
function _sparse(A,B)
    if isa(B,AbstractCuSparseMatrix)
        cu(sparse(A...))
    else
        sparse(A...)
    end
end
function norm(A::CuSparseMatrixCSC,B) #Assume Inf norm
    A = CuSparseMatrixCSR(A)
    columns(n,A) = A[n,:];
    @CUDA.allowscalar rowsums = sum.(columns.(1:size(A,1),(A,)))
    A = CuSparseMatrixCSC(A)
    maximum(rowsums)
end
end # module
