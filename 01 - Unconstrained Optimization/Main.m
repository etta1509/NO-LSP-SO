%% MAIN 
clear
close all
clc

%% FUNCTIONS' DEFINITION

d = 3; % 4, 5
n = 10^d;

% ROSENBROCK
f_rosenbrock = @(x) 100*(x(2,:) - x(1,:).^2).^2 + (1 - x(1,:)).^2;
gradf_rose = @(x) [400*x(1,:).^3 - 400*x(1,:).*x(2,:) + 2*x(1,:) - 2;
                   200*(x(2,:) - x(1,:).^2) ];
Hessf_rose = @(x) [1200*x(1)^2 - 400*x(2) + 2, -400*x(1);
                   -400*x(1), 200];
% Starting Point
x1 = [1.2; 1.2];
x2 = [-1.2; 1];
x3 = [0; 0];
x4 = [-1; 1];
x5 = [-2; 1.5];


% PROBLEM 16 - BANDED TRIGONOMETRIC FUNCTION
f_bounded = @(x) F16(x);
gradf_bounded = @(x) gradF16(x);
Hessf_bounded = @(x) hessF16(x);
% Starting Point
x0_bounded = ones(n,1);

% PROBLEM 27 - PENALTY FUNCTION
f_penalty = @(x) 0.5*(sum(((1 / sqrt (100000) * (x-1)).^2)) + (sum(x.^2) - 0.25).^2 );
grad_penalty = @(x) (x-1) / 100000 + 2*x*(sum(x.^2) - 0.25);
Hess_diag = @(x) spdiags((1/100000 + 2*(sum(x.^2) - 0.25)) + 4*(x.^2), 0, length(x), length (x));
Hess_nondiag = @(x) 4*(x*x') - spdiags(4 * x.^2, 0, length(x), length (x));
Hess_penalty = @(x) sparse(Hess_diag(x) + Hess_nondiag(x));
% Starting Point
x0_penalty = 1:n;

% PROBLEM 76 - FUNCTION 76
f_problem76 = @(x) F76(x);
gradf_problem76 = @(x) gradF76(x);
Hessf_problem76 = @(x) hessF76(x); 
% Starting Point
x0_problem76 = 2*ones(n,1);

%% PROBLEM 16 - BOUNDED TRIGONOMETRIC FUNCTION

function Fval = F16(x)
    x = x(:);  
    n = length(x);

    if n == 1
        Fval = 1 - cos(x(1));
        return;
    end
    val1 = (1 - cos(x(1))) - sin(x(2));

    if n > 2
        ii = (2:n-1)';
        val2 = sum( ii .* (1 - cos(x(ii))) + sin(x(ii-1)) - sin(x(ii+1)) );
    else
        val2 = 0;
    end

    val3 = n*(1 - cos(x(n))) + n*sin(x(n-1));

    Fval = val1 + val2 + val3;
end

function g = gradF16(x)
    x = x(:);
    n = length(x);
    g = zeros(n,1);

    for k = 1:n
        switch k
            case 1
                g(1) = sin(x(1)) + cos(x(1));
            case 2
                g(2) = 2*sin(x(2));
            case {3 : (n-2)}
                g(k) = k * sin(x(k));
            case (n-1)
                g(n-1) = (n-1)*( sin(x(n-1)) + cos(x(n-1)) );
            case n
                g(n) = n*sin(x(n)) - cos(x(n));
        end
    end
end

function H = hessF16(x)
    x = x(:);
    n = length(x);

    d = zeros(n,1);
    d(1) = cos(x(1)) - sin(x(1));
    for k = 2 : (n-2)
        d(k) = k * cos(x(k));
    end

    if n >= 2
        d(n-1) = (n-1)*( cos(x(n-1)) - sin(x(n-1)) );
    end

    d(n) = n*cos(x(n)) + sin(x(n));
    H = sparse(1:n, 1:n, d, n, n);
end

%% PROBLEM 76 - FUNCTION

function y = F76(x)
    n = length(x);
    y = 0.5*sum( (x(1:n) - 0.1*[x(2:n); x(1)].^2).^2);
end

function g =gradF76(x)
    n = length(x);
    g = x(1:n) - [x(2:n); x(1)].^2*0.1 - 0.2*x(1:n).*([x(n); x(1:n-1)] - x(1:n).^2*0.1);
end

function H = hessF76(x)
    n = length(x);
    v1 = 1 - 0.2*[x(n); x(1:n-1)] + 3/50 * x(1:n).^2;
    v2 = -0.2*x(2:n);
    v3 = -0.2*x(1)*ones(2,1);
    H = sparse(1:n, 1:n, v1, n, n); % Main diagonal
    H = H + sparse(2:n, 1:n-1, v2, n, n); % Lower diagonal
    H = H + sparse(1:n-1, 2:n, v2, n, n); % Upper diagonal
    H = H + sparse([1,n], [n,1], v3, n, n);
end

%% INITIAL CONDITIONS

% For Rosenbrock
num_start_points = 5;
initial_conditions = cell(1, num_start_points);
point_names = cell(1, num_start_points);

for i = 1:num_start_points
    initial_conditions{i} = eval(['x', num2str(i)]); % Access variables x1, x2, ..., x5 dynamically
    point_names{i} = ['x', num2str(i)];
end

% For all the other functions
num_start_points = 11;
initial_conditions = cell(1, num_start_points);
point_names = cell(1, num_start_points);

% First starting point (as per original code)
initial_conditions{1} = x0_problem76;
point_names{1} = 'x_0';

% Generate the other 10 starting points: randomly in the hypercube [-1,1]^n
for i = 1:10 % Loop for the remaining 10 starting points
    % Sample a random point in the hypercube [xbar-1, xbar+1] for each coordinate
    x_rand = x0_problem76 + (2 * rand(1, n) - 1); % change for each of the function
    initial_conditions{i+1} = x_rand;
    point_names{i+1} = ['x_', num2str(i)];
end


%% HYPERPARAMETR GRID SEARCH
hyperparameter_grid_search(f, grad, hess, initial_conditions);

% hyperparameter_grid_search(f_rosenbrock, gradf_rose, Hessf_rose, initial_conditions);
% hyperparameter_grid_search(f_bounded, gradf_bounded, Hessf_bounded, initial_conditions);
% hyperparameter_grid_search(f_penalty, grad_penalty, Hess_penalty, initial_conditions);
% hyperparameter_grid_search(f_problem76, gradf_problem76, Hessf_problem76, initial_conditions);

%% OPTIMIZATION EXPERIMENT
% Change these value in relation to the function that you have to use, that
% are just an example
kmax = 1000;
c1 = 0.0001;
rho = 0.5;
btmax = 100;
tolgrad = 1e-6;
dims = [3, 4, 5];
num_starting_points = 11;

optimizationExperiment(f, gradf, hessf, x_base, kmax, tolgrad, c1, rho, btmax, dims, num_starting_points);

% optimizationExperiment(f_rosenbrock, gradf_rose, Hessf_rose, x_base, kmax, tolgrad, c1, rho, btmax, dims, num_starting_points)
% optimizationExperiment(f_bounded, gradf_bounded, Hessf_bounded, x0_bounded, kmax, tolgrad, c1, rho, btmax, dims, num_starting_points);
% optimizationExperiment(f_penalty, grad_penalty, Hess_penalty, x0_penalty, kmax, tolgrad, c1, rho, btmax, dims, num_starting_points);
% optimizationExperiment(f_problem76, gradf_problem76, Hessf_problem76, x0_problem76, kmax, tolgrad, c1, rho, btmax, dims, num_starting_points);

%% MODIFIED NEWTON METHOD
modified_newton(x0, f, gradf, Hessf, kmax, tolgrad, c1, rho, btmax, varargin);
modified_newton_matrixfree(x0, f, gradf, kmax, tolgrad, c1, rho, btmax);

%% TRUNCATED NEWTON METHOD (WITHOUT PRECONDITIONING)
truncated_newton(x0, f, gradf, Hessf, kmax, tolgrad, c1, rho, btmax);
truncated_newton_hessian_free(x0, f, gradf, kmax, tolgrad, c1, rho, btmax);

%% TRUNCATED NEWTON METHOD (WITH PRECONDITIONING)
truncated_newton_pre(x0, f, gradf, Hessf, kmax, tolgrad, c1, rho, btmax, varargin);
truncated_newton_pre_matrixfree(x0, f, gradf, kmax, tolgrad, c1, rho, btmax);

%% FINITE DIFFERENCE APPROACH
gridSearchFDParameters(initial_conditions, F, gradF, hessF);

% gridSearchFDParameters(initial_conditions, f_rosenbrock, gradf_rose, Hessf_rose);
% gridSearchFDParameters(initial_conditions, f_bounded, gradf_bounded, Hessf_bounded);
% gridSearchFDParameters(initial_conditions, f_penalty, grad_penalty, Hess_penalty);
% gridSearchFDParameters(initial_conditions, f_problem76, gradf_problem76, Hessf_problem76);