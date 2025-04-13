function xk = cg_truncated_matrix_free(Afun, b, x0, tol)
% CG_TRUNCATED_MATRIX_FREE Solves the system Afun (x) = b using the matrix - free truncated CG method
% ( without preconditioning ) handling potential negative curvature cases .
% In this Hessian - free version , Afun is a function handle that computes the
% matrix - vector product A*v, WITHOUT explicitly forming the matrix A.
%
% xk = cg_truncated_matrix_free (Afun , b, x0 , tol )
%
% INPUTS :
% Afun - Function handle for the matrix - vector product , Afun (v) = A * v
% where A is the system matrix (e.g., Hessian approximation ).
% This function should compute the product WITHOUT explicitly forming A.
% b - Right - hand side vector ( typically -gradf )
% x0 - Initial vector for the CG iteration
% tol - Tolerance on the relative residual norm ( stopping criterion )
%
% OUTPUT :
% xk - Approximate solution vector ( computed direction )

xk = x0; % Initialize the solution vector with the starting vector
r = b - Afun(xk); % Calculate the initial residual : r = b - Afun (xk)
normb = norm(b); % Calculate the norm of the right - hand side vector b
if normb == 0, normb = 1; end % Avoid division by zero if norm of b is zero
relres = norm(r) / normb ; % Calculate the initial relative residual
p = r; % Initialize the search direction p with the initial residual
maxit = 100; % Maximum number of iterations for CG ( modify if needed )
it = 0; % Initialize iteration counter

while ( relres > tol ) && (it < maxit )
    Ap = Afun(p); % Compute the matrix - vector product Ap = A * p using the provided function handle Afun

    % Negative curvature check : if p '* Ap <= 0, terminate CG loop
    if p' * Ap <= 0
        if it == 0
            % If negative curvature is detected at the first iteration ,
            % proceed with a step calculated along p
            alpha = (r' * p) / max (p' * Ap , eps ); % Step size calculation , ensure
            % denominator is not zero
            xk = xk + alpha * p; % Update solution with step along p
        end
        break ; % Exit CG iteration if negative curvature is detected
    end

alpha = (r' * r) / (p' * Ap); % Calculate step size alpha for CG update
xk = xk + alpha * p; % Update solution vector xk
r_new = r - alpha * Ap; % Calculate the new residual r_new = r - alpha * A*p
beta = (r_new' * r_new ) / (r' * r); % Calculate beta for updating the search direction ( Fletcher - Reeves version )
p = r_new + beta * p; % Update search direction p
r = r_new; % Update residual r to r_new
relres = norm(r) / normb ; % Update relative residual norm
it = it + 1; % Increment iteration counter
end
end