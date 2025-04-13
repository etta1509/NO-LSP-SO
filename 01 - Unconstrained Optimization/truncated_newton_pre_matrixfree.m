function [xk , fk , gradfk_norm , k, xseq , btseq , rate_convergence ] = truncated_newton_pre_matrixfree(x0, f, gradf, kmax, tolgrad, c1, rho, btmax)
% This version solves the system H(x)*p = -grad using a " matrix - free " CG method .
% Here H(x) = a*I + 4*x*x' ( with 'a' defined as above ) and
% the product H(x)*v is computed without forming H explicitly .

xk = x0(:);
n = numel(xk);
xseq = zeros(n, kmax );
btseq = zeros(1, kmax );

fk = f(xk);
gradfk = gradf(xk);
gradfk_norm = norm( gradfk );
k = 0;

armijo_rhs = @(fk , alpha , gradfk , pk) fk + c1 * alpha * ( gradfk' * pk);

while (k < kmax ) && ( gradfk_norm >= tolgrad )
    % Calculate 'a' and define the matrix - free Hessian - vector product
    a = 1/100000 + 2*( sum (xk .^2) -0.25);
    % We do not use a shift here , as truncated CG handles potential indefiniteness
    Afun = @(v) (a*v + 4 * xk * (xk' * v)); % Matrix - free Hessian - vector product : H(x) * v

    % Choose an inner tolerance parameter for CG
    etak = min(0.5, gradfk_norm );
    % Solve H*p = -gradfk using matrix - free truncated CG
    p = cg_truncated_matrix_free(Afun, -gradfk , zeros(n ,1), etak );

    % Backtracking line search ( Armijo condition )
    alpha = 1;
    bt = 0;
    while (bt < btmax )
        xnew = xk + alpha * p;
        fnew = f( xnew );
        if fnew <= armijo_rhs(fk, alpha, gradfk, p)
            break ;
        end
        alpha = rho * alpha ;
        bt = bt + 1;
    end
    if (bt == btmax ) && ( fnew > armijo_rhs (fk , alpha , gradfk , p))
        warning ('Maximum backtracking iterations (%d) reached at iteration %d: Armijo condition not satisfied .', btmax , k +1);
    end

    % Iterate update
    xk = xnew;
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm( gradfk );

    k = k + 1;
    xseq(:, k) = xk;
    btseq(k) = bt;
end
xseq = xseq(:, 1:k);
btseq = btseq(1:k);

% Convergence rate calculation ( vectorized estimation )
x_star = xseq(:, end );
e = vecnorm( xseq - x_star, 2, 1);
valid = (e(1: end -2) > 1e-12) & (e(2: end -1) > 1e-12) & (e(3: end ) > 1e-12) ;
if any ( valid )
    p_est = log(e(3: end)./e(2: end -1)) ./ log(e(2: end -1) ./e(1: end -2));
    rate_convergence = mean(p_est( valid ));
else
    rate_convergence = NaN;
end
end