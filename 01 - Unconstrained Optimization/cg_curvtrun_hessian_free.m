function xk = cg_curvtrun_hessian_free(Hv , b, x0 , tol )
% CG_CURVTRUN_HESSIAN_FREE Solves the system Hv(x) = b using the Hessian - Free truncated CG
% method ( without preconditioning ) handling potential negative curvature cases .
% Hessian - Free Version : Hv is a function handle that computes the
% Hessian - vector product , NOT the explicit Hessian matrix .
%
% xk = cg_curvtrun_hessian_free (Hv , b, x0 , tol )
%
% INPUTS :
% Hv - Function handle for the Hessian - vector product , Hv(p) = Hessian (f(xk)) * p
% b - Right - hand side vector ( typically -gradf )
% x0 - Initial vector
% tol - Tolerance on the relative residual
%
% OUTPUT :
% xk - Approximate solution ( computed direction )


% Initializations
xk = x0;
rk = b - Hv(xk); % Use Hv to calculate the Hessian - vector product
pk = rk;
norm_b = norm (b);
if norm_b == 0
    norm_b = 1; % Avoid division by zero if b is zero vector
end
relres = norm (rk) / norm_b ;
kmax = 100;
k = 0;


while ( relres > tol ) && (k < kmax )
    Apk = Hv(pk); % Use Hv to calculate the Hessian - vector product

    % Negative curvature check
    if pk' * Apk <= 0
        if k == 0
            % If negative curvature is detected at the first step ,
            % proceed with a step calculated along pk
            alphak = (rk' * pk) / max (pk' * Apk , eps );
            xk = xk + alphak * pk;
        end
        break ;
    end

        % Calculate step size using the classical CG method formula
        alphak = (rk' * rk) / (pk' * Apk);
        xk = xk + alphak * pk;
        
        rk_new = rk - alphak * Apk;
        beta = ( rk_new' * rk_new ) / (rk' * rk);
        
        % Update the search direction
        pk = rk_new + beta * pk;
        
        rk = rk_new ;
        relres = norm (rk) / norm_b ;
        k = k + 1;
end