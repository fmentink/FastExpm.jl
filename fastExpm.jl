using LinearAlgebra
using SparseArrays

"""
    fastExpm(A)
    fastExpm(A; threshold=1e-6, nonzero_tol=1e-14)

 This function efficiently implements matrix exponentiation for sparse and full matrices.
 This code is based on scaling, taylor series and scaling.
 Currently works only on the CPU

 Two optional keyword arguments are used to speed up the computation and preserve sparsity.
 [1] threshold determines the threshold for the Taylor series (default 1e-6)
 [2] nonzero_tol strips elements smaller than nonzero_tol at each computation step to preserve sparsity (default 1e-14)
 The code automatically switches from sparse to full if sparsity is below 25% to maintain speed.

 This code was originally developed by Ilya Kuprov (http://spindynamics.org/) and has been adapted by F. Mentink-Vigier (fmentink@magnet.fsu.edu)
 and Murari Soundararajan (murari@magnet.fsu.edu)
 If you use this code, please cite
  - H. J. Hogben, M. Krzystyniak, G. T. P. Charnock, P. J. Hore and I. Kuprov, Spinach – A software library for simulation of spin dynamics in large spin systems, J. Magn. Reson., 2011, 208, 179–194.
  - I. Kuprov, Diagonalization-free implementation of spin relaxation theory for large spin systems., J. Magn. Reson., 2011, 209, 31–38.
""" 
function fastExpm(A;threshold=1e-6,nonzero_tol=1e-14)
    mat_norm=norm(A,Inf);
    scaling_factor = nextpow(2,mat_norm); # Native routine, faster
    A = A/scaling_factor;
    delta=1;

    # Run Taylor series procedure on the CPU
    P=sparse(1.0I,size(A)); next_term=P; n=1;
    if nonzeros(A)/length(A)>0.25 || length(A)<64^2
        A=Matrix(A);
    else
        A=sparse(A);
    end

    while delta>threshold
        # Compute the next term
        if issparse(next_term)
            next_term=(1/n)*A*next_term;
            #Eliminate small elements
            next_term=droptolerance!(next_term, nonzero_tol);
            if nonzeros(next_term)/length(next_term)>0.25
                next_term=Matrix(next_term);
            end
        else
            next_term=(1/n)*next_term*A;
        end
        delta=norm(next_term,Inf);
        #Add to the total and increment the counter
        P .+= next_term; n=n+1;
    end
    #Squaring of P to generate correct P
    for n=1:log2(scaling_factor)
        P=P*P;
        if issparse(P)
            if nonzeros(P)/length(P)>0.25
                P = droptolerance!(P, nonzero_tol);
            else
                P=Matrix(P);
            end
        end
    end
    return P
end

function droptolerance!(A::Matrix, tolerance)
    A .= tolerance*round.((1/tolerance)*A)
end
function droptolerance!(A::SparseMatrixCSC, tolerance)
    droptol!(A, tolerance) # Native routine, faster
end

function nonzeros(A::Matrix)
    count(x->x>0, abs.(A))
end
function nonzeros(A::SparseMatrixCSC)
    nnz(A) # Native routine, faster
end
