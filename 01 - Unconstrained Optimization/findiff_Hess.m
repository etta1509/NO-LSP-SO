function [ Hessfx ] = findiff_Hess(f, x, h)
% FINDIFF_HESS Approximates the Hessian of a function using finite differences .
%
% [ Hessfx ] = findiff_Hess (f, x, h)
%
% Optimized function to approximate the Hessian matrix of the function f at the point x (column vector )
% using finite difference methods . This version supports both scalar step size h and a vector step size h,
% where h(i) is used as the step size for the i-th coordinate .
%
% This implementation assumes the Hessian matrix is sparse with nonzero entries primarily
% on the main diagonal , the first sub - and super - diagonals , and the two corner elements
% at positions (1,n) and (n ,1) . This specific sparsity pattern is exploited for efficiency ,
% calculating only these potentially nonzero entries .
%
% INPUTS :
% f - Function handle describing a function f: R^n -> R;
% x - n- dimensional column vector representing the point at which to approximate the
% Hessian .
% h - Step size for the finite difference approximation . This can be:
% - A scalar : a uniform step size applied to all components .
% - An n- dimensional vector : component - specific step sizes , where h(i) is used for
% the i-th coordinate .
%
% OUTPUT :
% Hessfx - n-by -n sparse matrix representing the approximate Hessian of f at x,
% calculated using finite differences and exploiting the assumed sparsity
% pattern .


n = length(x); % Get the dimension of the input vector x
f0 = f(x); % Evaluate the function f at the point x ( base function value )


% Preallocate arrays to store function evaluations at perturbed points .
f_plus = zeros(n ,1) ; % Array to store f(x + h_i* e_i ) for each coordinate i
f_minus = zeros(n ,1) ; % Array to store f(x - h_i * e_i ) for each coordinate i


% Compute f(x + h_i* e_i ) and f(x - h_i *e_i) for each coordinate direction e_i.
for i = 1:n
    % Determine step size hi for the i-th coordinate
    if numel(h) > 1
        hi = h(i); % Use component - specific step size if h is a vector
    else
        hi = h; % Use uniform step size if h is a scalar
    end
    x_temp = x; % Create a temporary copy of x to avoid modifying the original x
    x_temp (i) = x_temp(i) + hi; % Perturb the i-th component by +hi
    f_plus (i) = f( x_temp ); % Evaluate f at the positively perturbed point
    
    x_temp = x; % Reset x_temp back to x for the negative perturbation
    x_temp (i) = x_temp(i) - hi; % Perturb the i-th component by -hi
    f_minus (i) = f( x_temp ); % Evaluate f at the negatively perturbed point
end

% Preallocate the Hessian matrix as a sparse matrix to efficiently store and compute only
% nonzero elements .
Hessfx = spalloc(n, n, 4*n); % Allocate space for a sparse n x n matrix , expecting
% approximately 4*n non - zero elements ( based on sparsity assumption )


% Compute diagonal entries of the Hessian using second - order central finite differences .
for i = 1:n
    % Determine step size hi for the i-th diagonal entry
    if numel(h) > 1
        hi = h(i); % Use component - specific step size if h is a vector
    else
        hi = h; % Use uniform step size if h is a scalar
    end
    % Second - order central difference formula for diagonal elements :
    % H(i,i) (f(x + h_i* e_i ) - 2*f(x) + f(x - h_i *e_i )) / ( h_i ^2)
    Hessfx(i,i) = ( f_plus (i) - 2*f0 + f_minus(i)) / (hi ^2) ;
end


% Compute off - diagonal entries for adjacent variables ( first sub - and super - diagonals ).
for i = 1:n -1
    % Determine step sizes hi and hj for the off - diagonal entry (i, i +1)
    if numel (h) > 1
        hi = h(i); % Step size for the i-th component
        hj = h(i +1) ; % Step size for the (i +1) -th component
    else
        hi = h; % Use uniform step size if h is a scalar
        hj = h; % Use uniform step size if h is a scalar
    end
    x_temp = x; % Create a temporary copy of x
    x_temp(i) = x_temp(i) + hi; % Perturb the i-th component by +hi
    x_temp(i+1) = x_temp(i +1) + hj; % Perturb the (i+1) -th component by +hj
    f_pair = f( x_temp ); % Evaluate f at the point perturbed in both i-th and (i +1) -th directions

    % Finite difference formula for off - diagonal elements (i, i +1) and (i+1, i):
    % H(i, i +1) = H(i+1, i) (f(x + h_i *e_i + h_j* e_j) - f(x + h_i* e_i ) - f(x + h_j * e_j )
    % + f(x)) / ( h_i * h_j )
    value = ( f_pair - f_plus(i) - f_plus(i +1) + f0) / (hi * hj);
    Hessfx(i, i+1) = value ; % Assign the computed value to the (i, i +1) entry
    Hessfx(i+1, i) = value ; % Assign the same value to the symmetric (i+1, i) entry ( Hessian is symmetric )
end


% Compute the corner elements : (1,n) and (n ,1) , assuming wrap - around connection .
if numel(h) > 1
    h1 = h(1); % Step size for the first component
    hn = h(n); % Step size for the n-th ( last ) component
else
    h1 = h; % Use uniform step size if h is a scalar
    hn = h; % Use uniform step size if h is a scalar
end
x_temp = x; % Create a temporary copy of x
x_temp(1) = x_temp(1) + h1; % Perturb the first component by +h1
x_temp(n) = x_temp(n) + hn; % Perturb the n-th component by +hn
f_pair = f( x_temp ); % Evaluate f at the point perturbed in both first and n-th directions


% Finite difference formula for corner elements (1, n) and (n, 1):
% H(1, n) = H(n, 1) (f(x + h_1 * e_1 + h_n *e_n ) - f(x + h_1 *e_1) - f(x + h_n* e_n ) + f(x))
% / (h_1 * h_n)
value = ( f_pair - f_plus(1) - f_plus(n) + f0) / (h1 * hn);
Hessfx(1, n) = value; % Assign the computed value to the (1, n) entry
Hessfx(n, 1) = value; % Assign the same value to the symmetric (n, 1) entry


end