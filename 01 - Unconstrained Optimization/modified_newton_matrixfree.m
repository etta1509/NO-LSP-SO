function [xk , fk , gradfk_norm , k, xseq , btseq , rate_convergence ] = modified_newton_matrixfree(x0, f, gradf, kmax, tolgrad, c1, rho, btmax)
% This version exploits the fact that the Hessian is:
% H(x) = a*I + 4*x*x', with a = 1/100000 + 2*( sum (x .^2) -0.25)
% If H is not PD ( Positive Definite ), i.e., if a <= 0, a shift tau is added such that a+
% tau > delta .

xk = x0(:);
n = numel(xk); % Number of element in a vector
xseq = zeros(n, kmax);
btseq = zeros(1, kmax);

fk = f(xk);
gradfk = gradf(xk);
gradfk_norm = norm(gradfk);

k = 0;
while (k < kmax) && (gradfk_norm >= tolgrad)
    % Calculation of the parameter "a"
    a = 1/100000 + 2*( sum (xk .^2) -0.25);
    delta = 1e-6; % small offset to ensure PD ( Positive Definiteness )
    tau = 0;
    if a <= delta
        tau = delta - a; % minimum shift to ensure a+ tau = delta > 0
    end
    A = a + tau ; % the scalar part (PD) of the matrix

    % The modified matrix is:
    % B = A*I + 4* xk*xk '
    % We want to solve B*p = -gradfk 
    % We use the Sherman - Morrison formula to solve this system efficiently 
    %
    % Let z = -gradfk . Then , by Sherman - Morrison formula :
    % p = z/A - (4/ A^2 * xk * (xk '*z)) /(1 + 4*( xk '* xk)/A)
    z = -gradfk ;
    denom = 1 + 4*( xk'* xk)/A;
    p = z/A - (4/ A ^2) * xk * (xk'*z) / denom;

    % Backtracking line search ( Armijo condition )
    alpha = 1;
    bt = 0;
    while (bt < btmax )
        xnew = xk + alpha *p;
        fnew = f( xnew );
        if fnew <= fk + c1 * alpha * (gradfk' * p)
            break ;
        end
        alpha = rho * alpha ;
        bt = bt + 1;
    end

% Iterate update
xk = xnew;
fk = fnew;
gradfk = gradf(xk);
gradfk_norm = norm(gradfk);

k = k + 1;
xseq(:, k) = xk;
btseq(k) = bt;
end
xseq = xseq(:, 1:k);
btseq = btseq(1: k);

% Convergence rate calculation ( vectorized estimation )
x_star = xseq (:, end );
e = vecnorm ( xseq - x_star , 2, 1);
valid = (e (1: end -2) > 1e-12) & (e (2: end -1) > 1e-12) & (e (3: end ) > 1e-12) ;
if any ( valid )
    p_est = log (e (3: end )./e (2: end -1) ) ./ log(e(2: end -1) ./e (1: end -2));
    rate_convergence = mean ( p_est ( valid ));
else
    rate_convergence = NaN;
end

end