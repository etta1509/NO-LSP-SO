function xk = cg_curvtrun_newt(A, b, x0, tol)
% CG_CURVTRUN_NEWT Solves the system A*x = b using the truncated CG method
% ( without preconditioning ) handling potential negative curvature cases .
%
% xk = cg_curvtrun_newt (A, b, x0 , tol )
%
% INPUTS :
% A - Matrix ( Hessian ) of the system
% b - Right - hand side vector ( typically -gradf )
% x0 - Initial vector
% tol - Tolerance on the relative residual
%
% OUTPUT :
% xk - Approximate solution ( computed direction )


% Initializations
xk = x0;
rk = b - A * xk;
pk = rk;
norm_b = norm (b);
if norm_b == 0
    norm_b = 1; % Avoid division by zero if b is zero vector
end
relres = norm (rk) / norm_b ;
kmax = 100;
k = 0;


while ( relres > tol ) && (k < kmax )
    Apk = A * pk;

    % If negative or near - zero curvature is detected , terminate the loop
    if pk' * Apk <= 0
        if k == 0
            % If negative curvature is detected at the first step ,
            % proceed with a step calculated along pk
            alphak = (rk' * pk) / max (pk' * Apk , eps);
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
    relres = norm(rk) / norm_b ;
    k = k + 1;
end


end