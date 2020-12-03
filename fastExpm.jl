function taylorExpm(A,threshold=1e-6,nonzero_tol=1e-14)
    mat_norm=norm(A,Inf);
    n_squarings=max(0,ceil(log2(mat_norm))); scaling_factor=2^n_squarings;
    A=A*scaling_factor^-1;
    delta=1;

    # Run Taylor series procedure on the CPU
    P=sparse(I,size(A)); next_term=P; n=1;
    if sum(x->x>0, abs.(A))/length(A)>0.25 || length(A)<64^2
        A=Matrix(A);
    else
        A=sparse(A);
    end

    while delta>threshold
        # Compute the next term
        if issparse(next_term)
            next_term=(1/n)*A*next_term;
            #Eliminate small elements
            #next_term=nonzero_tol*round.((1/nonzero_tol)*next_term);
            next_term=droptol!(next_term, nonzero_tol); #julia's internal function
            if sum(x->x>0, abs.(next_term))/length(next_term)>0.25
                next_term=Matrix(next_term);
            end
        else
            next_term=(1/n)*next_term*A;
        end
        delta=norm(next_term,Inf);
        #Add to the total and increment the counter
        P=P+next_term; n=n+1;
    end
    #Squaring of P to generate correct P
    for n=1:n_squarings
        P=P*P;
        if issparse(P)
            if sum(x->x>0, abs.(P))/length(P)>0.25
                #P=nonzero_tol*round.((1/nonzero_tol)*P);
                P=droptol!(P, nonzero_tol);
            else
                P=Matrix(P);
            end
        end
    end
    return P
end