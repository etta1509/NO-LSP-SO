function [ output_str ] = gridSearchFDParameters( initial_conditions, F, gradF, hessF )
% GRIDSEARCHFDPARAMETERS Executes a grid search for finite difference parameters.
% This function performs a grid search to evaluate the performance of
% optimization algorithms (Modified Newton and Truncated Newton) using
% finite difference (FD) approximations for gradients and Hessians.
% It explores different step sizes for finite differences and compares
% central and forward difference schemes.
%
% INPUTS:
% initial_conditions : Cell array of initial points for optimization algorithms.
% F : Function handle for the objective function F(x).
% gradF : Function handle for the analytical gradient of F, gradF(x).
% hessF : Function handle for the analytical Hessian of F, hessF(x).
%
% OUTPUTS:
% output_str : String containing formatted results of the grid search,
% summarizing performance metrics for different FD parameters.
% Results : Structure (currently not returned, but could be implemented)
% containing detailed numerical results, including
% iterations, function values, and gradient norms for different
% parameter settings and methods.

n = 50;                 % Dimension of the problem (vector x is of size n)
kmax = 1000;            % Maximum number of iterations for optimization algorithms
tolgrad = 1e-6;         % Tolerance for the gradient norm (stopping criterion)
c1 = 1e-4;              % Armijo condition parameter for line search
rho = 0.5;              % Backtracking factor for line search
btmax = 100;            % Maximum number of backtracking iterations in line search

%% 3. DEFINITION OF THE PARAMETER "GRID"
num_starting_points = length(initial_conditions);  % Number of initial starting points to test
exponent_list = [2, 4, 6, 8, 10, 12];                % Exponents for step size h = 10^(-k), where k is from this list
n_params = length(exponent_list);                    % Number of different finite difference parameters to test

%% 4. PREALLOCATION OF RESULTS MATRICES
% Matrices to store iteration counts and final gradient norms for each configuration.
% The results are categorized by:
% - Finite Difference Type: Uniform (h is constant for all components) and Variable (h varies with x components)
% - Optimization Method: Modified Newton and Truncated Newton
% - Finite Difference Scheme: Central ('c') and Forward ('fw')

% --- Uniform FD (h = 10^(-k)) ---
% [MODIFIED NEWTON METHOD]
iter_uniform_mod_c = zeros(num_starting_points, n_params);  % Iterations for Modified Newton, Uniform FD, Central scheme
grad_uniform_mod_c = zeros(num_starting_points, n_params);  % Gradient norm for Modified Newton, Uniform FD, Central scheme
iter_uniform_mod_fw = zeros(num_starting_points, n_params);   % Iterations for Modified Newton, Uniform FD, Forward scheme
grad_uniform_mod_fw = zeros(num_starting_points, n_params);   % Gradient norm for Modified Newton, Uniform FD, Forward scheme

% [TRUNCATED NEWTON METHOD]
iter_uniform_trunc_c = zeros(num_starting_points, n_params);  % Iterations for Truncated Newton, Uniform FD, Central scheme
grad_uniform_trunc_c = zeros(num_starting_points, n_params);  % Gradient norm for Truncated Newton, Uniform FD, Central scheme
iter_uniform_trunc_fw = zeros(num_starting_points, n_params);   % Iterations for Truncated Newton, Uniform FD, Forward scheme
grad_uniform_trunc_fw = zeros(num_starting_points, n_params);   % Gradient norm for Truncated Newton, Uniform FD, Forward scheme

% --- FD with Variable h (h is proportional to |x|)
% [MODIFIED NEWTON METHOD]
iter_var_mod_c = zeros(num_starting_points, n_params);      % Iterations for Modified Newton, Variable FD, Central scheme
grad_var_mod_c = zeros(num_starting_points, n_params);      % Gradient norm for Modified Newton, Variable FD, Central scheme
iter_var_mod_fw = zeros(num_starting_points, n_params);       % Iterations for Modified Newton, Variable FD, Forward scheme
grad_var_mod_fw = zeros(num_starting_points, n_params);       % Gradient norm for Modified Newton, Variable FD, Forward scheme

% [TRUNCATED NEWTON METHOD]
iter_var_trunc_c = zeros(num_starting_points, n_params);      % Iterations for Truncated Newton, Variable FD, Central scheme
grad_var_trunc_c = zeros(num_starting_points, n_params);      % Gradient norm for Truncated Newton, Variable FD, Central scheme
iter_var_trunc_fw = zeros(num_starting_points, n_params);       % Iterations for Truncated Newton, Variable FD, Forward scheme
grad_var_trunc_fw = zeros(num_starting_points, n_params);       % Gradient norm for Truncated Newton, Variable FD, Forward scheme

%% 5. GRID SEARCH: LOOP OVER STARTING POINTS AND PARAMETERS
% This section performs the core grid search. It iterates through:
% - Initial starting points (defined in 'initial_conditions')
% - Exponents for step size 'h' (defined in 'exponent_list')
% For each combination, it tests four scenarios:
% 1. Modified Newton with Uniform FD (Central Difference)
% 2. Modified Newton with Uniform FD (Forward Difference)
% 3. Truncated Newton with Uniform FD (Central Difference)
% 4. Truncated Newton with Uniform FD (Forward Difference)
% 5. Modified Newton with Variable FD (Central Difference)
% 6. Modified Newton with Variable FD (Forward Difference)
% 7. Truncated Newton with Variable FD (Central Difference)
% 8. Truncated Newton with Variable FD (Forward Difference)

for i = 1:num_starting_points % Loop through each initial starting point
    x0 = initial_conditions{i}; % Get the current initial starting point
    for j = 1:n_params         % Loop through each finite difference parameter (exponent k)
        kexp = exponent_list(j); % Get the current exponent k for h = 10^(-k)
        
        %% 5.1 Uniform FD: h = 10^(-kexp)
        h_uniform = 10^(-kexp);  % Calculate the uniform step size h
        % Define function handles for finite difference gradient approximations using central and forward schemes
        grad_fdiff_c = @(x) findiff_grad(F, x, h_uniform, 'c');  % Central difference gradient approximation
        grad_fdiff_fw = @(x) findiff_grad(F, x, h_uniform, 'fw');  % Forward difference gradient approximation
        Hess_fdiff = @(x) findiff_Hess(F, x, h_uniform);           % Hessian approximation (using the same uniform h)
        
        % --- Modified Newton Method with Uniform FD ---
        % Modified Newton, Uniform FD, Central Difference Scheme
        try
            [~, ~, grad_norm_mod, iter_mod, ~, ~, ~] = ...
                modified_newton(x0, F, grad_fdiff_c, Hess_fdiff, kmax, tolgrad, c1, rho, btmax); % Run Modified Newton
            iter_uniform_mod_c(i,j) = iter_mod;  % Store iterations
            grad_uniform_mod_c(i,j) = grad_norm_mod;  % Store final gradient norm
        catch ME
            disp(['Error in modified_newton (uniform, central) for starting point ' num2str(i) ', k = ' num2str(kexp) ': ' ME.message]);
            iter_uniform_mod_c(i,j) = NaN;  % Store NaN for iterations in case of error
            grad_uniform_mod_c(i,j) = NaN;  % Store NaN for gradient norm in case of error
        end
        
        % Modified Newton, Uniform FD, Forward Difference Scheme
        try
            [~, ~, grad_norm_mod, iter_mod, ~, ~, ~] = ...
                modified_newton(x0, F, grad_fdiff_fw, Hess_fdiff, kmax, tolgrad, c1, rho, btmax); % Run Modified Newton
            iter_uniform_mod_fw(i,j) = iter_mod;  % Store iterations
            grad_uniform_mod_fw(i,j) = grad_norm_mod;  % Store final gradient norm
        catch ME
            disp(['Error in modified_newton (uniform, forward) for starting point ' num2str(i) ', k = ' num2str(kexp) ': ' ME.message]);
            iter_uniform_mod_fw(i,j) = NaN;  % Store NaN for iterations in case of error
            grad_uniform_mod_fw(i,j) = NaN;  % Store NaN for gradient norm in case of error
        end
        
        % --- Truncated Newton Method with Uniform FD ---
        % Truncated Newton, Uniform FD, Central Difference Scheme
        try
            [~, ~, grad_norm_trunc, iter_trunc, ~, ~, ~] = ...
                truncated_newton_pre(x0, F, grad_fdiff_c, Hess_fdiff, kmax, tolgrad, c1, rho, btmax); % Run Truncated Newton
            iter_uniform_trunc_c(i,j) = iter_trunc;  % Store iterations
            grad_uniform_trunc_c(i,j) = grad_norm_trunc;  % Store final gradient norm
        catch ME
            disp(['Error in truncated_newton_pre (uniform, central) for starting point ' num2str(i) ', k = ' num2str(kexp) ': ' ME.message]);
            iter_uniform_trunc_c(i,j) = NaN;  % Store NaN for iterations in case of error
            grad_uniform_trunc_c(i,j) = NaN;  % Store NaN for gradient norm in case of error
        end
        
        % Truncated Newton, Uniform FD, Forward Difference Scheme
        try
            [~, ~, grad_norm_trunc, iter_trunc, ~, ~, ~] = ...
                truncated_newton_pre(x0, F, grad_fdiff_fw, Hess_fdiff, kmax, tolgrad, c1, rho, btmax); % Run Truncated Newton
            iter_uniform_trunc_fw(i,j) = iter_trunc;  % Store iterations
            grad_uniform_trunc_fw(i,j) = grad_norm_trunc;  % Store final gradient norm
        catch ME
            disp(['Error in truncated_newton_pre (uniform, forward) for starting point ' num2str(i) ', k = ' num2str(kexp) ': ' ME.message]);
            iter_uniform_trunc_fw(i,j) = NaN;  % Store NaN for iterations in case of error
            grad_uniform_trunc_fw(i,j) = NaN;  % Store NaN for gradient norm in case of error
        end
        
        %% 5.2 Variable FD: h_var = h_uniform * abs(x0)
        h_var = h_uniform * abs(x0);  % Calculate variable step size h based on initial point x0
        grad_fdiff_var_c = @(x) findiff_grad(F, x, h_var, 'c');  % Central difference gradient with variable h
        grad_fdiff_var_fw = @(x) findiff_grad(F, x, h_var, 'fw'); % Forward difference gradient with variable h
        Hess_fdiff_var = @(x) findiff_Hess(F, x, h_var);          % Hessian approximation with variable h
        
        % --- Modified Newton Method with Variable FD ---
        % Modified Newton, Variable FD, Central Difference Scheme
        try
            [~, ~, grad_norm_mod, iter_mod, ~, ~, ~] = ...
                modified_newton(x0, F, grad_fdiff_var_c, Hess_fdiff_var, kmax, tolgrad, c1, rho, btmax); % Run Modified Newton
            iter_var_mod_c(i,j) = iter_mod;  % Store iterations
            grad_var_mod_c(i,j) = grad_norm_mod;  % Store final gradient norm
        catch ME
            disp(['Error in modified_newton (variable, central) for starting point ' num2str(i) ', kesp = ' num2str(kexp) ': ' ME.message]);
            iter_var_mod_c(i,j) = NaN;  % Store NaN for iterations in case of error
            grad_var_mod_c(i,j) = NaN;  % Store NaN for gradient norm in case of error
        end
        
        % Modified Newton, Variable FD, Forward Difference Scheme
        try
            [~, ~, grad_norm_mod, iter_mod, ~, ~, ~] = ...
                modified_newton(x0, F, grad_fdiff_var_fw, Hess_fdiff_var, kmax, tolgrad, c1, rho, btmax); % Run Modified Newton
            iter_var_mod_fw(i,j) = iter_mod;  % Store iterations
            grad_var_mod_fw(i,j) = grad_norm_mod;  % Store final gradient norm
        catch ME
            disp(['Error in modified_newton (variable, forward) for starting point ' num2str(i) ', kesp = ' num2str(kexp) ': ' ME.message]);
            iter_var_mod_fw(i,j) = NaN;  % Store NaN for iterations in case of error
            grad_var_mod_fw(i,j) = NaN;  % Store NaN for gradient norm in case of error
        end
        
        % --- Truncated Newton Method with Variable FD ---
        % Truncated Newton, Variable FD, Central Difference Scheme
        try
            [~, ~, grad_norm_trunc, iter_trunc, ~, ~, ~] = ...
                truncated_newton_pre(x0, F, grad_fdiff_var_c, Hess_fdiff_var, kmax, tolgrad, c1, rho, btmax); % Run Truncated Newton
            iter_var_trunc_c(i,j) = iter_trunc;  % Store iterations
            grad_var_trunc_c(i,j) = grad_norm_trunc;  % Store final gradient norm
        catch ME
            disp(['Error in truncated_newton_pre (variable, central) for starting point ' num2str(i) ', kesp = ' num2str(kexp) ': ' ME.message]);
            iter_var_trunc_c(i,j) = NaN;  % Store NaN for iterations in case of error
            grad_var_trunc_c(i,j) = NaN;  % Store NaN for gradient norm in case of error
        end
        
        % Truncated Newton, Variable FD, Forward Difference Scheme
        try
            [~, ~, grad_norm_trunc, iter_trunc, ~, ~, ~] = ...
                truncated_newton_pre(x0, F, grad_fdiff_var_fw, Hess_fdiff_var, kmax, tolgrad, c1, rho, btmax); % Run Truncated Newton
            iter_var_trunc_fw(i,j) = iter_trunc;  % Store iterations
            grad_var_trunc_fw(i,j) = grad_norm_trunc;  % Store final gradient norm
        catch ME
            disp(['Error in truncated_newton_pre (variable, forward) for starting point ' num2str(i) ', kesp = ' num2str(kexp) ': ' ME.message]);
            iter_var_trunc_fw(i,j) = NaN;  % Store NaN for iterations in case of error
            grad_var_trunc_fw(i,j) = NaN;  % Store NaN for gradient norm in case of error
        end
    end
end

%% 6. CALCULATION OF AVERAGES (OVER STARTING POINTS)
% Average the iteration counts and gradient norms over all starting points for each parameter setting.
% Uniform FD Averages
avg_iter_uniform_mod_c = mean(iter_uniform_mod_c, 1);  % Average iterations for Modified Newton, Uniform FD, Central
avg_grad_uniform_mod_c = mean(grad_uniform_mod_c, 1);  % Average gradient norm for Modified Newton, Uniform FD, Central
avg_iter_uniform_mod_fw = mean(iter_uniform_mod_fw, 1);  % Average iterations for Modified Newton, Uniform FD, Forward
avg_grad_uniform_mod_fw = mean(grad_uniform_mod_fw, 1);  % Average gradient norm for Modified Newton, Uniform FD, Forward

avg_iter_uniform_trunc_c = mean(iter_uniform_trunc_c, 1);  % Average iterations for Truncated Newton, Uniform FD, Central
avg_grad_uniform_trunc_c = mean(grad_uniform_trunc_c, 1);  % Average gradient norm for Truncated Newton, Uniform FD, Central
avg_iter_uniform_trunc_fw = mean(iter_uniform_trunc_fw, 1);  % Average iterations for Truncated Newton, Uniform FD, Forward
avg_grad_uniform_trunc_fw = mean(grad_uniform_trunc_fw, 1);  % Average gradient norm for Truncated Newton, Uniform FD, Forward

% Variable FD Averages
avg_iter_var_mod_c = mean(iter_var_mod_c, 1);  % Average iterations for Modified Newton, Variable FD, Central
avg_grad_var_mod_c = mean(grad_var_mod_c, 1);  % Average gradient norm for Modified Newton, Variable FD, Central
avg_iter_var_mod_fw = mean(iter_var_mod_fw, 1);  % Average iterations for Modified Newton, Variable FD, Forward
avg_grad_var_mod_fw = mean(grad_var_mod_fw, 1);  % Average gradient norm for Modified Newton, Variable FD, Forward

avg_iter_var_trunc_c = mean(iter_var_trunc_c, 1);  % Average iterations for Truncated Newton, Variable FD, Central
avg_grad_var_trunc_c = mean(grad_var_trunc_c, 1);  % Average gradient norm for Truncated Newton, Variable FD, Central
avg_iter_var_trunc_fw = mean(iter_var_trunc_fw, 1);  % Average iterations for Truncated Newton, Variable FD, Forward
avg_grad_var_trunc_fw = mean(grad_var_trunc_fw, 1);  % Average gradient norm for Truncated Newton, Variable FD, Forward

%% 7. SELECTION OF THE BEST PARAMETERS (for Modified Newton, Uniform FD)
% For demonstration, select the "best" parameter based on Modified Newton with Uniform FD.
% "Best" is defined as the parameter that achieves the minimum gradient norm,
% and among those, the minimum number of iterations.

% Central Difference - find best k based on minimum gradient norm, then minimum iterations among candidates
[ min_grad_uniform_mod_c, best_index_uniform_mod_c ] = min(avg_grad_uniform_mod_c);  % Find minimum average gradient norm and its index
candidates = find( abs(avg_grad_uniform_mod_c - min_grad_uniform_mod_c) < 1e-12 );       % Find indices with gradient norm close to minimum
if length(candidates) > 1  % If multiple candidates with similar gradient norms, choose based on minimum iterations
    [~, idx] = min( avg_iter_uniform_mod_c(candidates) );  % Find index of minimum iterations among candidates
    best_index_uniform_mod_c = candidates(idx);            % Update best index to the one with minimum iterations
end
best_k_uniform_mod_c = exponent_list(best_index_uniform_mod_c);  % Get the best k exponent for Uniform FD, Central scheme

% Forward Difference - find best k based on minimum gradient norm, then minimum iterations among candidates
[ min_grad_uniform_mod_fw, best_index_uniform_mod_fw ] = min(avg_grad_uniform_mod_fw);  % Find minimum average gradient norm and its index
candidates = find( abs(avg_grad_uniform_mod_fw - min_grad_uniform_mod_fw) < 1e-12 );       % Find indices with gradient norm close to minimum
if length(candidates) > 1  % If multiple candidates with similar gradient norms, choose based on minimum iterations
    [~, idx] = min( avg_iter_uniform_mod_fw(candidates) );  % Find index of minimum iterations among candidates
    best_index_uniform_mod_fw = candidates(idx);            % Update best index to the one with minimum iterations
end
best_k_uniform_mod_fw = exponent_list(best_index_uniform_mod_fw);  % Get the best k exponent for Uniform FD, Forward scheme

%% 8. CONSTRUCTION OF THE OUTPUT STRING (output_str)
% Format the results into a readable string for display.
output_str = "";  % Initialize output string
output_str = output_str + newline + "%% VISUALIZATION OF RESULTS " + newline;  % Add section header for visualization
output_str = output_str + "*** Grid Search Results ***" + newline + newline;      % Add title for grid search results

% --- Results for Uniform FD ---
output_str = output_str + "--- Uniform Finite Differences ---" + newline;  % Sub-section header for Uniform FD
output_str = output_str + " Modified Newton Method :" + newline;           % Method sub-header
output_str = output_str + " Central Difference Scheme :" + newline;        % Scheme sub-header
output_str = output_str + " Tested Parameters (k): " + string(mat2str(exponent_list)) + newline;  % List of tested k parameters
output_str = output_str + " Average Iterations : " + string(mat2str(avg_iter_uniform_mod_c, 4)) + newline;  % Average iterations array as string
output_str = output_str + " Average Gradient Norm : " + string(mat2str(avg_grad_uniform_mod_c, 4)) + newline;  % Average gradient norm array as string
output_str = output_str + "=> Best k = " + string(num2str(best_k_uniform_mod_c)) + " (Average Iterations = " + ...
    string(num2str(avg_iter_uniform_mod_c(best_index_uniform_mod_c), '%.2f')) + ", Average Gradient Norm = " + ...
    string(num2str(avg_grad_uniform_mod_c(best_index_uniform_mod_c), '%.3e')) + ")" + newline + newline;  % Best k and its performance metrics

output_str = output_str + " Forward Difference Scheme :" + newline;  % Scheme sub-header
output_str = output_str + " Tested Parameters (k): " + string(mat2str(exponent_list)) + newline;  % List of tested k parameters
output_str = output_str + " Average Iterations : " + string(mat2str(avg_iter_uniform_mod_fw, 4)) + newline;  % Average iterations array as string
output_str = output_str + " Average Gradient Norm : " + string(mat2str(avg_grad_uniform_mod_fw, 4)) + newline;  % Average gradient norm array as string
output_str = output_str + "=> Best k = " + string(num2str(best_k_uniform_mod_fw)) + " (Average Iterations = " + ...
    string(num2str(avg_iter_uniform_mod_fw(best_index_uniform_mod_fw), '%.2f')) + ", Average Gradient Norm = " + ...
    string(num2str(avg_grad_uniform_mod_fw(best_index_uniform_mod_fw), '%.3e')) + ")" + newline + newline;  % Best k and its performance metrics

output_str = output_str + " Truncated Newton Method :" + newline;  % Method sub-header
output_str = output_str + " Central Difference Scheme :" + newline;  % Scheme sub-header
output_str = output_str + " Tested Parameters (k): " + string(mat2str(exponent_list)) + newline;  % List of tested k parameters
output_str = output_str + " Average Iterations : " + string(mat2str(avg_iter_uniform_trunc_c, 4)) + newline;  % Average iterations array as string
output_str = output_str + " Average Gradient Norm : " + string(mat2str(avg_grad_uniform_trunc_c, 4)) + newline;  % Average gradient norm array as string
output_str = output_str + "=> Best k = " + string(num2str(best_k_uniform_mod_c)) + " (Average Iterations = " + ...
    string(num2str(avg_iter_uniform_trunc_c(best_index_uniform_mod_c), '%.2f')) + ", Average Gradient Norm = " + ...
    string(num2str(avg_grad_uniform_trunc_c(best_index_uniform_mod_c), '%.3e')) + ")" + newline + newline;  % Best k and its performance metrics

output_str = output_str + " Forward Difference Scheme :" + newline;  % Scheme sub-header
output_str = output_str + " Tested Parameters (k): " + string(mat2str(exponent_list)) + newline;  % List of tested k parameters
output_str = output_str + " Average Iterations : " + string(mat2str(avg_iter_uniform_trunc_fw, 4)) + newline;  % Average iterations array as string
output_str = output_str + " Average Gradient Norm : " + string(mat2str(avg_grad_uniform_mod_fw, 4)) + newline;  % Average gradient norm array as string
output_str = output_str + "=> Best k = " + string(num2str(best_k_uniform_mod_fw)) + " (Average Iterations = " + ...
    string(num2str(avg_iter_uniform_trunc_fw(best_index_uniform_mod_fw), '%.2f')) + ", Average Gradient Norm = " + ...
    string(num2str(avg_grad_uniform_trunc_fw(best_index_uniform_mod_fw), '%.3e')) + ")" + newline + newline;  % Best k and its performance metrics

% --- Results for Variable FD ---
output_str = output_str + newline + "--- Finite Differences with Variable h ---" + newline;  % Sub-section header for Variable FD
output_str = output_str + " Modified Newton Method :" + newline;  % Method sub-header
output_str = output_str + " Central Difference Scheme :" + newline;  % Scheme sub-header
output_str = output_str + " Tested Parameters (k): " + string(mat2str(exponent_list)) + newline;  % List of tested k parameters
output_str = output_str + " Average Iterations : " + string(mat2str(avg_iter_var_mod_c, 4)) + newline;  % Average iterations array as string
output_str = output_str + " Average Gradient Norm : " + string(mat2str(avg_grad_var_mod_c, 4)) + newline;  % Average gradient norm array as string
output_str = output_str + "=> Best k = " + string(num2str(exponent_list(best_index_uniform_mod_c))) + " (Average Iterations = " + ...
    string(num2str(avg_iter_var_mod_c(best_index_uniform_mod_c), '%.2f')) + ", Average Gradient Norm = " + ...
    string(num2str(avg_grad_var_mod_c(best_index_uniform_mod_c), '%.3e')) + ")" + newline + newline;  % Best k and its performance metrics

output_str = output_str + " Forward Difference Scheme :" + newline;  % Scheme sub-header
output_str = output_str + " Tested Parameters (k): " + string(mat2str(exponent_list)) + newline;  % List of tested k parameters
output_str = output_str + " Average Iterations : " + string(mat2str(avg_iter_var_mod_fw, 4)) + newline;  % Average iterations array as string
output_str = output_str + " Average Gradient Norm : " + string(mat2str(avg_grad_var_mod_fw, 4)) + newline;  % Average gradient norm array as string
output_str = output_str + "=> Best k = " + string(num2str(exponent_list(best_index_uniform_mod_fw))) + " (Average Iterations = " + ...
    string(num2str(avg_iter_var_mod_fw(best_index_uniform_mod_fw), '%.2f')) + ", Average Gradient Norm = " + ...
    string(num2str(avg_grad_var_mod_fw(best_index_uniform_mod_fw), '%.3e')) + ")" + newline + newline;  % Best k and its performance metrics

output_str = output_str + " Truncated Newton Method :" + newline;  % Method sub-header
output_str = output_str + " Central Difference Scheme :" + newline;  % Scheme sub-header
output_str = output_str + " Tested Parameters (k): " + string(mat2str(exponent_list)) + newline;  % List of tested k parameters
output_str = output_str + " Average Iterations : " + string(mat2str(avg_iter_var_trunc_c, 4)) + newline;  % Average iterations array as string
output_str = output_str + " Average Gradient Norm : " + string(mat2str(avg_grad_var_trunc_c, 4)) + newline;  % Average gradient norm array as string
output_str = output_str + "=> Best k = " + string(num2str(exponent_list(best_index_uniform_mod_c))) + " (Average Iterations = " + ...
    string(num2str(avg_iter_var_trunc_c(best_index_uniform_mod_c), '%.2f')) + ", Average Gradient Norm = " + ...
    string(num2str(avg_grad_var_trunc_c(best_index_uniform_mod_c), '%.3e')) + ")" + newline + newline;  % Best k and its performance metrics

output_str = output_str + " Forward Difference Scheme :" + newline;  % Scheme sub-header
output_str = output_str + " Tested Parameters (k): " + string(mat2str(exponent_list)) + newline;  % List of tested k parameters
output_str = output_str + " Average Iterations : " + string(mat2str(avg_iter_var_trunc_fw, 4)) + newline;  % Average iterations array as string
output_str = output_str + " Average Gradient Norm : " + string(mat2str(avg_grad_var_fw, 4)) + newline;  % Average gradient norm array as string
output_str = output_str + "=> Best k = " + string(num2str(exponent_list(best_index_uniform_mod_fw))) + " (Average Iterations = " + ...
    string(num2str(avg_iter_var_trunc_fw(best_index_uniform_mod_fw), '%.2f')) + ", Average Gradient Norm = " + ...
    string(num2str(avg_grad_var_trunc_fw(best_index_uniform_mod_fw), '%.3e')) + ")" + newline + newline;  % Best k and its performance metrics

output_str = output_str + "*** End of Grid Search Analysis ***" + newline;  % Footer for output string
disp(output_str);  % Display the formatted output string in the command window

end
