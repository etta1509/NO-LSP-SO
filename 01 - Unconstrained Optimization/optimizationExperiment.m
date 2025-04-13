function optimizationExperiment(f, gradf, hessf, x_base, kmax, tolgrad, c1, rho, btmax, dims, num_starting_points)
% OPTIMIZATIONEXPERIMENT Executes and compares different optimization methods for a given
% problem .
% This function performs a comparative analysis of optimization methods :
% Modified Newton , Truncated Newton without preconditioning , and Truncated
% Newton with preconditioning . It evaluates these methods across different
% problem dimensions and multiple initial starting points .
%
% INPUTS :
% f - Function handle for the objective function f(x).
% gradf - Function handle for the gradient of f, gradf (x).
% hessf - Function handle for the Hessian of f, hessf (x).
% x_base - Base vector used to generate initial conditions . It determines
% the dimension of the problem and serves as the first initial
% point .
% kmax - Maximum number of iterations for each optimization method .
% tolgrad - Tolerance for the gradient norm as the stopping criterion .
% c1 - Parameter for the Armijo condition in line search .
% rho - Reduction factor for backtracking line search .
% btmax - Maximum iterations for backtracking line search .
% dims - Vector of exponents for problem dimensions (n = 10^d, where d is
% from dims ).
% num_starting_points - Number of initial starting points to use for each dimension .
%
% OUTPUTS :
% This function does not explicitly return values , but it prints out the
% performance metrics ( iterations , convergence rate , execution time , gradient norm )
% for each optimization method , dimension , and initial starting point to the console .
% It 's designed to provide a comparative report of the methods ' effectiveness .

% --- Input Validation ---
if ~ isa (f, 'function_handle') || ~isa(gradf , 'function_handle') || ~isa (hessf , 'function_handle')
    error ('f, gradf , and hessf must be function handles.');
end
if ~ isvector ( x_base )
    error ('x_base must be a vector.');
end
if ~ isscalar (kmax) || ~ isscalar (tolgrad) || ~ isscalar (c1) || ~ isscalar (rho) || ~ isscalar (btmax)
    error ('kmax , tolgrad , c1 , rho , and btmax must be scalar values.');
end
if ~ isvector (dims) || ~ isnumeric (dims)
    error ('dims must be a numeric vector.');
end
if ~ isscalar (num_starting_points) || ~ isnumeric (num_starting_points) || num_starting_points < 1
    error ('num_starting_points must be a scalar numeric value greater than or equal to 1.');
end

% Loop through different problem dimensions defined by 'dims '
for d = dims
    n = 10^d; % Set problem dimension n based on the current exponent d
    fprintf ('\n ============================== \n');
    fprintf ('Dimension : n = 10^% d = %d\n', d, n);
    fprintf (' ============================== \n');

    % --- Creation of Initial Conditions ---
    initial_conditions = cell (1, num_starting_points ); % Cell array to store initial conditions
    names = cell (1, num_starting_points ); % Cell array to store names for initial conditions

    % First initial point is the provided base point 'x_base ' ( resized to dimension n)
    initial_conditions{1} = x_base(1: n); % Use the first n elements of x_base , or pad if x_base is shorter .
    names {1} = 'x_0';

    % Generate additional random initial points around x_base
    for i = 1:( num_starting_points - 1)
        x_rand = initial_conditions{1} + (2 * rand (1, n) - 1)'; % Random points within
                                                                  % hypercube around x_base
        initial_conditions {i+1} = x_rand ;
        names{i +1} = ['x_', num2str(i)];
    end

    % --- Loop over Initial Conditions for the current dimension ---
    for i = 1: length( initial_conditions )
        x0 = initial_conditions{i}; % Current initial condition
        fprintf ('\n-------------------------------------------\n');
        fprintf ('d = 10^%d, Initial Condition : %s\n', d, names{i});
        fprintf (' -------------------------------------------\n');

        % --- Modified Newton Method ---
        tic ; % Start timer for Modified Newton
        [xk_mn , fk_mn , grad_norm_mn , iter_mn , x_seq_mn , bt_seq_mn , rate_conv_mn ] = modified_newton (x0 , f, gradf , hessf , kmax , tolgrad , c1 , rho , btmax ); % Run
        % Modified Newton method
        time_mn = toc ; % Stop timer
        fprintf ('Modified Newton : iter = %d, rate = %f, time = %f sec , grad_norm = %e\n', iter_mn, rate_conv_mn, time_mn, grad_norm_mn); % Display results

        % --- Truncated Newton Method WITHOUT Preconditioning ---
        tic ; % Start timer for Truncated Newton without preconditioning
        [xk_tn , fk_tn , grad_norm_tn , iter_tn , x_seq_tn , bt_seq_tn , rate_conv_tn ] = truncated_newton (x0 , f, gradf , hessf , kmax , tolgrad , c1 , rho , btmax ); % Run
        % Truncated Newton method
        time_tn = toc ; % Stop timer
        fprintf ('Truncated Newton Without Preconditioning : iter = %d, rate = %f, time = %f, sec , grad_norm = %e\n', iter_tn, rate_conv_tn, time_tn, grad_norm_tn); % Display results

        % --- Truncated Newton Method WITH Preconditioning ---
        tic ; % Start timer for Truncated Newton with preconditioning
        [ xk_tnp , fk_tnp , grad_norm_tnp , iter_tnp , x_seq_tnp , bt_seq_tnp , rate_conv_tnp ] = truncated_newton_pre(x0 , f, gradf, hessf, kmax, tolgrad, c1, rho, btmax); % Run
        % Preconditioned Truncated Newton method
        time_tnp = toc; % Stop timer
        fprintf ('Truncated Newton Preconditioning : iter = %d, rate = %f, time = %f sec, grad_norm = %e\n', iter_tnp, rate_conv_tnp, time_tnp, grad_norm_tnp); % Display results

    end
end

end