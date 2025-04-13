function grad = findiff_grad(F, x, h, scheme)
% FINDIFF_GRAD Approximates the gradient of a function using finite differences .
%
% grad = findiff_grad (F, x, h, scheme )
%
% This function calculates an approximation of the gradient of a scalar function F
% at a point x using finite difference methods . It supports central and forward difference schemes .
%
% INPUTS :
% F - Function handle of the objective function , F: R^n -> R.
% x - Point ( vector ) at which to compute the gradient approximation .
% h - Step size for the finite difference approximation .
% - Can be a scalar ( uniform step size for all components )
% - or a vector of length n ( step size for each component ).
% scheme - String specifying the finite difference scheme to use:
% - 'c' for central difference (second - order accurate )
% - 'fw ' for forward difference (first - order accurate )
%
% OUTPUT :
% grad - Approximated gradient of F at x ( column vector ).

n = length(x); % Determine the dimension of the input vector x
grad = zeros(n ,1); % Initialize the gradient vector with zeros

% Loop through each component to compute the partial derivatives
for i = 1:n
    x_plus = x; % Create a copy of x for the positive perturbation
    x_minus = x; % Create a copy of x for the negative perturbation

    % Determine the step size hi for the current component i
    % If h is a vector , use the i-th component h(i) as step size for the i-th partial
    % derivative
    if numel(h) > 1
        hi = h(i);
    else
        hi = h; % If h is a scalar , use the same step size for all components
    end
    x_plus (i) = x_plus(i) + hi; % Perturb the i-th component of x by +hi
    x_minus (i) = x_minus(i) - hi; % Perturb the i-th component of x by -hi
    
    % Calculate the finite difference approximation based on the chosen scheme
    if strcmp (scheme ,'c')
        % Central difference scheme : (F(x + hi* e_i ) - F(x - hi*e_i)) / (2* hi)
        grad(i) = (F( x_plus ) - F( x_minus )) /(2* hi);
    elseif strcmp ( scheme ,'fw')
        % Forward difference scheme : (F(x + hi* e_i ) - F(x)) / hi
        grad(i) = (F( x_plus ) - F(x))/(hi);
    else
        error (' Unrecognized finite difference scheme . Please use ''c'' for central or ''fw '' for forward .');
    end
end
end