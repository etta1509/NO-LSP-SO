function xk = cg_curvtrun_newt_pre(A_f, b, x0, tol)
% CG_CURVTRUN_NEWT_PRE Solves A*x = b using truncated CG with sparse preconditioning .
% A_f can be a sparse matrix or a handle that , given a point x, returns
% the Hessian matrix ( NOT used here : in truncated_newton_pre we already pass the matrix ).
%
% Preconditioning Strategy :
% - Use ichol as a preconditioner ( Incomplete Cholesky ) if possible .
% - If ichol fails , fall back to a diagonal preconditioner .
% - If negative curvature is detected (p'Ap <= 0) , terminate CG iteration .

persistent L_ichol_factor % Persistent preconditioner factor

% Preconditioner construction only on the first function call
if isempty( L_ichol_factor )
    try
        opts.michol = 'lower';
        A0 = sparse(A_f); % In this version , A_f is already a matrix ( not a handle )
        L_ichol_factor = ichol(A0 , opts );
    catch
        % If ichol fails , use diagonal preconditioning
        d = diag(sparse(A_f(x0)));
        d(d == 0) = 1;
        L_ichol_factor = diag( sqrt(d));
    end
end

L = L_ichol_factor; % Assign the preconditioner factor ( either ichol or diagonal ) to L

% Initialization
xk = x0;
% Calculate the initial residual : r = b - A*xk
r = b - A_f(xk)*xk;
% Apply preconditioner : solve L*z = r and L '* precond_z = z, where preconditioner M = L*L'
z = L \ r; % Solve L*z = r ( forward substitution since L is lower triangular )
p = L' \ z; % Solve L '*p = z ( backward substitution since L' is upper triangular ) - p is
                                                              % the preconditioned residual

normb = norm(b);
if normb == 0, normb = 1; end % Avoid division by zero if norm of b is zero

relres = norm(r) / normb ; % Relative residual
n = numel(x0);
maxit = min(100 , n); % Maximum CG iterations ( capped at 100 or problem dimension )
it = 0; % Iteration counter

while ( relres > tol ) && (it < maxit )
    Ap = A_f(xk) * p; % Matrix - vector product A*p

    % Negative curvature check
    if p' * Ap <= 0
        % If negative curvature is detected , especially at the first iteration , take a step along p
        if it == 0
            alpha = (r' * p) / max (p' * Ap , eps ); % Step size calculation , ensure denominator is not zero
            xk = xk + alpha * p; % Update solution with step along p
        end
        break ; % Exit CG iteration if negative curvature is detected
   end

% Standard Conjugate Gradient updates
alpha = (r' * p) / (p' * Ap); % Step size
xk = xk + alpha * p; % Update solution

r_new = r - alpha * Ap; % New residual
z_new = L \ r_new ; % Apply preconditioner to new residual : solve L* z_new =
% r_new
q = L' \ z_new ; % Solve L '*q = z_new - q is the preconditioned new
% residual

beta = (r_new' * q) / (r' * p); % Calculate beta for direction update ( Fletcher - Reeves version )
p = q + beta * p; % Update search direction using preconditioned residuals

r = r_new; % Update residual
relres = norm(r) / normb ; % Update relative residual norm
it = it + 1; % Increment iteration counter
end

end