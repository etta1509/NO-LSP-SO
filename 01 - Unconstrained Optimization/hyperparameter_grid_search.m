function [ best_params , best_avg_iter ] = hyperparameter_grid_search(f, grad, hess, initial_conditions)
% HYPERPARAMETER_GRID_SEARCH Executes a grid search over hyperparameters for optimization
% methods .
%
% [ best_params , best_avg_iter ] = hyperparameter_grid_search (f, grad , hess ,
% initial_conditions )
%
% This function performs a grid search across combinations of hyperparameters
% to evaluate and optimize the performance of numerical optimization methods .
% It varies the parameters c1 , rho , and btmax for the ’Modified Newton ’ and
% ’Truncated Newton Pre ’ methods , assessing the average performance across
% several provided initial conditions .
%
% INPUTS :
% f - Function handle for the objective function to be minimized .
% grad - Function handle for the gradient of the objective function .
% hess - Function handle for the Hessian of the objective function .
% initial_conditions - Cell array of column vectors , representing different
% starting points for the optimization .
%
% OUTPUTS :
% best_params - Struct containing the best hyperparameter combination found
% during the grid search . The fields of the struct are:
% .c1 , .rho , . btmax .
% best_avg_iter - Scalar representing the average number of iterations obtained
% with the best hyperparameter combination , averaged
% over all optimization methods and initial conditions tested .
%
%
% HYPERPARAMETERS DEFINED FOR GRID SEARCH :
% Hyperparameters being varied :
% c1: from 1e -4 to 1e -2 ( Armijo condition parameter )
% rho: from 0.5 to 0.9 ( Backtracking line search reduction factor )
% btmax : from 10 to 50 ( Maximum backtracking iterations )

%% HYPERPARAMETER GRID DEFINITION
% Defines the ranges of values to explore for each hyperparameter
c1_vals = [1e-4, 1e-3, 1e-2]; % Range of values for c1 ( Armijo condition parameter )
rho_vals = [0.5 , 0.6 , 0.7 , 0.8 , 0.9]; % Range of values for rho ( backtracking
                                        % reduction factor ); note : 0.9 might not always satisfy the Armijo condition
btmax_vals = [30 , 40, 50]; % Range of values for btmax (max backtracking iterations );
                             % note : 10, 20, 30 might not always satisfy the Armijo condition

kmax = 1000;
% Calculate the number of values for each hyperparameter
num_c1 = numel ( c1_vals ); % Number of c1 values to test
num_rho = numel ( rho_vals ); % Number of rho values to test
num_btmax = numel ( btmax_vals ); % Number of btmax values to test

%% VARIABLES TO STORE THE BEST RESULT
% Initialize with a large value ( infinity ) so any computed average is better
best_avg_iter = Inf ;
% Structure to store the best hyperparameter combination found so far
best_params = struct ('c1', NaN , 'rho', NaN , 'btmax', NaN ); % Initialize best parameters to NaN

%% GRID SEARCH OVER PARAMETERS
% Cell array to hold the names of the optimization methods to be tested
methods = {'Modified Newton', 'Truncated Newton Pre'}; % List of optimization methods
num_methods = numel ( methods ); % Number of optimization methods

% Nested loops to iterate through all combinations of hyperparameters
for ic1 = 1: num_c1 % Iterate over each value of c1
    for irho = 1: num_rho % Iterate over each value of rho
        for ibt = 1: num_btmax % Iterate over each value of btmax
            % Get the current hyperparameter values for this iteration
            c1 = c1_vals ( ic1 ); % Current c1 value
            rho = rho_vals ( irho ); % Current rho value
            btmax = btmax_vals ( ibt ); % Current btmax value

            % Variable to accumulate the total number of iterations across methods and
            % initial conditions
            iter_total = 0;
            % Total number of runs ( number of methods multiplied by number of initial
            % conditions )
            run_count = 0;

            % Loop over the different initial conditions to test robustness
            for idx = 1: length ( initial_conditions )
                x0 = initial_conditions { idx }; % Get the current initial condition

                % --- Modified Newton Method ---
                try
                    % Call the modified_newton function with the current hyperparameters
                    % and initial condition
                    [~, ~, ~, k, ~, ~, ~] = modified_newton (x0, f, grad, hess, kmax, tolgrad , c1 , rho , btmax);
                    iter_total = iter_total + k; % Add the number of iterations ’k’ if
                    % the method converges successfully
                catch ME
                    % If the method fails ( throws an error ), assume it took the maximum
                    % number of iterations
                    iter_total = iter_total + kmax ; % Penalize failed runs by adding the
                    % maximum iteration count
                end
                run_count = run_count + 1; % Increment the run count for Modified Newton


                % --- Preconditioned Truncated Newton Method ---
                try
                    % Call the truncated_newton_pre function with the current
                    % hyperparameters and initial condition
                    [~, ~, ~, k, ~, ~, ~] = truncated_newton_pre (x0, f, grad, hess, kmax, tolgrad, c1, rho, btmax);
                    iter_total = iter_total + k; % Add the number of iterations ’k’ if
                                                 % the method converges successfully
                catch ME
                    iter_total = iter_total + kmax ; % Penalize failed runs by adding the maximum iteration count
                end
                run_count = run_count + 1; % Increment the run count for Truncated
                                           % Newton Preconditioned
            end

            % Calculate the average number of iterations for the current hyperparameter
            % combination
            avg_iter = iter_total / run_count ;

            % Update the best combination if the current average number of iterations is
            % lower ( better performance )
            if avg_iter < best_avg_iter
                best_avg_iter = avg_iter ; % Update the best average iteration count
                best_params .c1 = c1; % Update the best c1 parameter
                best_params . rho = rho ; % Update the best rho parameter
                best_params . btmax = btmax ; % Update the best btmax parameter
            end
        end
    end
end


%% GRID SEARCH RESULT
% Display the best hyperparameter combination found and the corresponding average
% iterations
disp ([ 'Grid search result - Best combination : c1 = ', num2str(best_params.c1), ', rho = ', num2str(best_params.rho ), ', btmax = ', num2str(best_params.btmax)]);
disp ([ 'Overall average iterations ( over all methods and initial conditions ): ', num2str(best_avg_iter)]);
end