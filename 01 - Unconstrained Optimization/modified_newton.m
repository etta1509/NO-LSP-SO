function [xk, fk, gradfk_norm, k, xseq, btseq, rate_convergence] = modified_newton(x0, f, gradf, Hessf, kmax, tolgrad, c1, rho, btmax, varargin)

% MODIFIED_NEWTON - Modified Newton method with backtracking 
%
% Syntax :
% 1) [xk , fk , gradfk_norm , k, xseq , btseq , rate_convergence ] =
% modified_newton (x0 , f, gradf , Hessf , kmax , tolgrad , c1 , rho , btmax )

%
% 2) [xk , fk , gradfk_norm , k, xseq , btseq , rate_convergence ] =
% modified_newton (x0 , f, [], [], kmax , tolgrad , c1 , rho , btmax , h, type )
% => If the handles are empty or invalid , finite difference functions are used
% to calculate the gradient and Hessian .
%
% In both cases , the operation is identical .


% Function handle for the Armijo condition
farmijo = @(fk , alpha , gradfk , pk) fk + c1 * alpha * ( gradfk' * pk);

% Initializations
x0 = x0(:);
n = length(x0);
xseq = zeros(n, kmax);
btseq = zeros(1, kmax);

% For estimating the convergence rate
e = zeros(1, kmax );

xk = x0;
fk = f(xk);
gradfk = gradf (xk);
hessfk = Hessf (xk);
gradfk_norm = norm ( gradfk );
Ek = speye ( size ( hessfk )); % Sparse identity matrix , for Hessian correction


k = 0;

while k < kmax && gradfk_norm >= tolgrad
    tau = 0;
    % Calculate beta as the minimum between the gradient norm and the Frobenius norm of
    % the Hessian
    beta = min(norm(gradfk), sqrt(sum(hessfk.^2 , 'all')));
    pd_success = false ;
    jmax = 100;

    % Search for a tau such that Bk = Hessf + tau *I is PD ( Positive Definite )
    for j = 1:jmax
        Bk = hessfk + tau * Ek;
        [L, cholFlag ] = chol (Bk , 'lower');
        if cholFlag == 0
            pd_success = true;
            break;
        else
            tau = max (2 * tau , beta / 2);
        end
    end


    if ~ pd_success
        error ('Modified Newton : failed to obtain a PD Hessian after %d attempts .', jmax);
    end


    % Calculation of the modified Newton direction via Cholesky factorization
    y = L \ (- gradfk);
    pk = L' \ y;

    % Backtracking line search to satisfy the Armijo condition
    alpha = 1;
    xnew = xk + alpha * pk;
    fnew = f(xnew);
    bt = 0;
    while bt < btmax && fnew > farmijo(fk, alpha, gradfk, pk)
        alpha = rho * alpha ;
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        bt = bt + 1;
    end

    % Update iterate
    xk = xnew(:);
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    hessfk = Hessf(xk);

    k = k + 1;
    xseq(:, k) = xk;
    btseq(k) = bt;
end


% Resize sequences based on the actual number of iterations performed
xseq = xseq(:, 1:k);
btseq = btseq(1: k);

% Calculation of errors for estimating the convergence order
x_star = xseq(:, end); % The last iterate is considered the " true " solution
e = vecnorm (xseq - x_star, 2, 1);

% Vectorized estimation of the convergence order (if at least 3 iterations )
eps_est = 1e-12; % threshold to avoid division by zero
if k >= 3
    valid = (e(1: end -2) > eps_est) & (e(2: end -1) > eps_est) & (e(3: end ) > eps_est);
    p_est = log(e(3: end )./e(2: end -1)) ./ log(e(2: end -1) ./e (1: end -2));
    if any ( valid )
        rate_convergence = mean(p_est(valid));
    else
        rate_convergence = NaN;
    end
else
    rate_convergence = NaN;
end
end