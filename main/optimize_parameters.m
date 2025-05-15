function optimal_params = optimize_parameters()
    % OPTIMIZE_PARAMETERS - Automatically optimize algorithm parameters
    % This script uses MATLAB's optimization tools to find optimal parameter values
    % for the enhanced auction algorithm.
    %
    % The parameters optimized are:
    %   - epsilon: minimum bid increment
    %   - alpha: bid calculation weights
    %   - gamma: consensus weight factor
    %   - lambda: information decay rate
    %   - beta: recovery auction weights
    
    % Add required paths
    addpath('../common');
    addpath('../utils');
    
    % Load utility functions
    enhanced_auction_utils = enhanced_auction_utils();
    robot_utils = robot_utils();
    task_utils = task_utils();
    env_utils = environment_utils();
    
    % Create optimization options
    options = optimoptions('particleswarm', ...
                           'Display', 'iter', ...
                           'SwarmSize', 50, ...
                           'MaxIterations', 100, ...
                           'FunctionTolerance', 1e-3, ...
                           'PlotFcn', @pswplotbestf);
    
    % Define parameter bounds
    % [epsilon, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, gamma, lambda, beta_1, beta_2, beta_3]
    lb = [0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.5, 0.5, 0.5];   % Lower bounds
    ub = [0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 0.9, 0.5, 3.0, 3.0, 3.0];     % Upper bounds
    
    % Start with a reasonable initial guess (based on prior experiments)
    initial_params = [0.05, 0.8, 0.3, 1.0, 1.2, 0.2, 0.5, 0.1, 2.0, 1.5, 1.0];
    
    % Create evaluation function
    eval_func = @(params) evaluate_parameters(params);
    
    % Run optimization
    fprintf('Starting parameter optimization...\n');
    [optimal_param_vector, final_objective] = particleswarm(eval_func, length(lb), lb, ub, options);
    
    % Convert to structure
    optimal_params = struct();
    optimal_params.epsilon = optimal_param_vector(1);
    optimal_params.alpha = optimal_param_vector(2:6);
    optimal_params.gamma = optimal_param_vector(7);
    optimal_params.lambda = optimal_param_vector(8);
    optimal_params.beta = optimal_param_vector(9:11);
    optimal_params.final_objective = final_objective;
    
    % Run a final evaluation with the optimal parameters
    fprintf('Evaluating optimal parameters...\n');
    [final_objective, metrics] = evaluate_parameters(optimal_param_vector, true);
    
    % Add metrics to optimal params
    optimal_params.metrics = metrics;
    
    % Save results
    save_path = '../results/parameter_optimization/optimal_params.mat';
    mkdir_if_not_exists(fileparts(save_path));
    save(save_path, 'optimal_params');
    
    % Generate plots
    plot_optimization_results(optimal_params);
    
    fprintf('Parameter optimization completed.\n');
    fprintf('Optimal parameters:\n');
    fprintf('  Epsilon: %.4f\n', optimal_params.epsilon);
    fprintf('  Alpha: [%.2f, %.2f, %.2f, %.2f, %.2f]\n', optimal_params.alpha);
    fprintf('  Gamma: %.2f\n', optimal_params.gamma);
    fprintf('  Lambda: %.4f\n', optimal_params.lambda);
    fprintf('  Beta: [%.2f, %.2f, %.2f]\n', optimal_params.beta);
    fprintf('  Final objective value: %.4f\n', optimal_params.final_objective);
    
    return;
end

function [objective, metrics] = evaluate_parameters(param_vector, verbose)
    % Evaluate a parameter set by running multiple simulations
    
    if nargin < 2
        verbose = false;
    end
    
    % Extract parameters
    epsilon = param_vector(1);
    alpha = param_vector(2:6);
    gamma = param_vector(7);
    lambda = param_vector(8);
    beta = param_vector(9:11);
    
    % Create parameter structure
    params = struct();
    params.epsilon = epsilon;
    params.alpha = alpha;
    params.gamma = gamma;
    params.lambda = lambda;
    params.beta = beta;
    params.comm_delay = 50;         % Fixed value for optimization
    params.packet_loss_prob = 0.1;  % Fixed value for optimization
    params.heartbeat_interval = 3;  % Heartbeat signal interval
    params.missed_heartbeat_threshold = 3;  % Number of missed heartbeats for failure detection
    params.min_progress_rate = 0.05;  % Minimum progress rate
    params.sync_timeout = 5;        % Timeout for synchronization
    params.max_sync_distance = 0.5;  % Maximum distance for synchronization
    
    % Set up simulations
    enhanced_auction_utils = enhanced_auction_utils();
    robot_utils = robot_utils();
    task_utils = task_utils();
    env_utils = environment_utils();
    
    env = env_utils.createEnvironment(4, 4);
    
    % Run multiple simulations with different task counts
    task_counts = [8, 16];  % Use moderate task counts for optimization
    num_runs = 5;           % Repeat each case multiple times
    
    % Initialize metrics storage
    all_makespans = zeros(length(task_counts), num_runs);
    all_iterations = zeros(length(task_counts), num_runs);
    all_optimality_gaps = zeros(length(task_counts), num_runs);
    all_messages = zeros(length(task_counts), num_runs);
    all_recovery_times = zeros(length(task_counts), num_runs);
    all_converged = false(length(task_counts), num_runs);
    
    % Run simulations
    for t_idx = 1:length(task_counts)
        K = task_counts(t_idx);
        
        for run = 1:num_runs
            % Set random seed for reproducibility
            rng(1000 + run);
            
            % Create robots
            robots = robot_utils.createRobots(2, env);
            
            % Create tasks
            tasks = task_utils.createTasks(K, env);
            tasks = task_utils.addTaskDependencies(tasks, 0.3);
            
            % Add collaborative tasks (approximately 20% of tasks)
            num_collaborative = max(1, round(K * 0.2));
            collab_indices = randperm(K, min(num_collaborative, K));
            for i = collab_indices
                tasks(i).collaborative = true;
            end
            
            % Add failure scenario in some runs
            if mod(run, 2) == 0
                params.failure_time = 20;
                params.failed_robot = 1;
            else
                params.failure_time = Inf;
                params.failed_robot = [];
            end
            
            % Run simulation
            [metrics, converged] = enhanced_auction_utils.runAuctionSimulation(params, env, robots, tasks, false);
            
            % Store metrics
            all_makespans(t_idx, run) = metrics.makespan;
            all_iterations(t_idx, run) = metrics.iterations;
            all_optimality_gaps(t_idx, run) = metrics.optimality_gap;
            all_messages(t_idx, run) = metrics.messages;
            all_converged(t_idx, run) = converged;
            
            if params.failure_time < Inf && isfield(metrics, 'recovery_time') && metrics.recovery_time > 0
                all_recovery_times(t_idx, run) = metrics.recovery_time;
            end
        end
    end
    
    % Calculate mean metrics
    mean_makespan = mean(all_makespans(:));
    mean_iterations = mean(all_iterations(:));
    mean_gap = mean(all_optimality_gaps(:));
    mean_messages = mean(all_messages(:));
    convergence_rate = mean(all_converged(:));
    mean_recovery = mean(all_recovery_times(all_recovery_times > 0));
    
    % Normalize metrics to [0,1] scale
    norm_makespan = mean_makespan / 100;              % Assuming max makespan around 100
    norm_iterations = mean_iterations / 200;          % Assuming max iterations around 200
    norm_gap = mean_gap / (2 * epsilon);             % Normalizing by theoretical bound
    norm_messages = mean_messages / 1000;             % Assuming max messages around 1000
    norm_convergence = 1 - convergence_rate;          % Perfect convergence = 0
    if isempty(mean_recovery)
        norm_recovery = 0;
    else
        norm_recovery = mean_recovery / 50;           % Assuming max recovery time around 50
    end
    
    % Weights for multi-objective function
    weights = [0.25, 0.15, 0.25, 0.15, 0.1, 0.1];
    
    % Calculate weighted objective (lower is better)
    objective = weights(1) * norm_makespan + ...
                weights(2) * norm_iterations + ...
                weights(3) * norm_gap + ...
                weights(4) * norm_messages + ...
                weights(5) * norm_convergence + ...
                weights(6) * norm_recovery;
    
    % Store metrics for verbose output
    if verbose
        metrics = struct();
        metrics.makespan = all_makespans;
        metrics.iterations = all_iterations;
        metrics.optimality_gap = all_optimality_gaps;
        metrics.messages = all_messages;
        metrics.recovery_time = all_recovery_times;
        metrics.converged = all_converged;
        
        metrics.mean_makespan = mean_makespan;
        metrics.mean_iterations = mean_iterations;
        metrics.mean_gap = mean_gap;
        metrics.mean_messages = mean_messages;
        metrics.convergence_rate = convergence_rate;
        metrics.mean_recovery = mean_recovery;
        
        fprintf('Evaluation results:\n');
        fprintf('  Mean makespan: %.2f\n', mean_makespan);
        fprintf('  Mean iterations: %.2f\n', mean_iterations);
        fprintf('  Mean optimality gap: %.4f (%.2f%% of bound)\n', mean_gap, 100*norm_gap);
        fprintf('  Mean messages: %.2f\n', mean_messages);
        fprintf('  Convergence rate: %.2f%%\n', 100*convergence_rate);
        if ~isempty(mean_recovery)
            fprintf('  Mean recovery time: %.2f\n', mean_recovery);
        else
            fprintf('  Mean recovery time: N/A\n');
        end
        fprintf('  Objective value: %.4f\n', objective);
    else
        metrics = [];
    end
end

function plot_optimization_results(optimal_params)
    % Generate plots for optimization results
    
    % Create output directory
    plot_dir = '../figures/parameter_optimization';
    mkdir_if_not_exists(fileparts(plot_dir));
    
    % Plot 1: Parameter values
    figure('Name', 'Optimal Parameter Values', 'Position', [100, 100, 1000, 600]);
    
    % Plot epsilon
    subplot(1, 3, 1);
    bar(optimal_params.epsilon);
    ylabel('Value');
    title('Optimal Epsilon');
    grid on;
    
    % Plot alpha values
    subplot(1, 3, 2);
    bar(optimal_params.alpha);
    xlabel('Parameter Index');
    ylabel('Value');
    title('Optimal Alpha Values');
    xticklabels({'Distance', 'Config', 'Capability', 'Workload', 'Energy'});
    grid on;
    
    % Plot other parameters
    subplot(1, 3, 3);
    other_params = [optimal_params.gamma, optimal_params.lambda, optimal_params.beta];
    bar(other_params);
    xlabel('Parameter');
    ylabel('Value');
    title('Other Parameters');
    xticklabels({'Gamma', 'Lambda', 'Beta 1', 'Beta 2', 'Beta 3'});
    grid on;
    
    saveas(gcf, fullfile(plot_dir, 'optimal_parameters.png'));
    
    % Plot 2: Performance metrics
    if isfield(optimal_params, 'metrics')
        metrics = optimal_params.metrics;
        
        figure('Name', 'Performance with Optimal Parameters', 'Position', [100, 100, 1000, 600]);
        
        % Plot makespan
        subplot(2, 3, 1);
        bar(mean(metrics.makespan));
        title('Makespan');
        grid on;
        
        % Plot iterations
        subplot(2, 3, 2);
        bar(mean(metrics.iterations));
        title('Iterations to Converge');
        grid on;
        
        % Plot optimality gap
        subplot(2, 3, 3);
        bar(mean(metrics.optimality_gap));
        hold on;
        plot([0, 3], [2*optimal_params.epsilon, 2*optimal_params.epsilon], '--r', 'LineWidth', 2);
        title('Optimality Gap');
        legend('Actual', 'Theoretical Bound');
        grid on;
        
        % Plot messages
        subplot(2, 3, 4);
        bar(mean(metrics.messages));
        title('Message Count');
        grid on;
        
        % Plot convergence rate
        subplot(2, 3, 5);
        bar(100 * mean(metrics.converged));
        title('Convergence Rate (%)');
        ylim([0, 100]);
        grid on;
        
        % Plot recovery time
        subplot(2, 3, 6);
        recovery_times = metrics.recovery_time;
        recovery_times(recovery_times == 0) = NaN; % Ignore zeros
        bar(mean(recovery_times, 'omitnan'));
        title('Recovery Time');
        grid on;
        
        saveas(gcf, fullfile(plot_dir, 'optimal_performance.png'));
    end
end

function mkdir_if_not_exists(dir_path)
    % Create directory if it doesn't exist
    if ~exist(dir_path, 'dir')
        mkdir(dir_path);
    end
end