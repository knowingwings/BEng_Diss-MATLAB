% Parameter Sensitivity Analysis for Distributed Auction Algorithm
% This script runs multiple simulations with different parameter settings
% to analyze their impact on algorithm performance

%% Setup
clear all;
close all;
clc;

% Add utility functions to path
addpath('../common');

% Load utility functions
env_utils = environment_utils();
robot_utils = robot_utils();
task_utils = task_utils();
auction_utils = auction_utils();

% Create results structure
results = struct();

% IMPROVED: Better default parameter values
base_params.epsilon = 0.05;        % Minimum bid increment
base_params.alpha = [0.8, 0.3, 1.0, 1.2, 0.2];  % IMPROVED: Better bid calculation weights
base_params.gamma = 0.5;           % Consensus weight factor
base_params.lambda = 0.1;          % Information decay rate
base_params.beta = [2.0, 1.5];     % IMPROVED: Stronger recovery weights
base_params.comm_delay = 0;        % Communication delay (in iterations)
base_params.packet_loss_prob = 0;  % Probability of packet loss
base_params.failure_time = inf;    % Time of robot failure (inf = no failure)
base_params.failed_robot = [];     % Which robot fails

% IMPROVED: More comprehensive parameter ranges
% Better range for epsilon to clearly show theoretical impact
epsilon_range = [0.01, 0.025, 0.05, 0.1, 0.2, 0.5];  

% More gradual packet loss range
packet_loss_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5];

% IMPROVED: More points for task count to better see scaling behavior
task_count_range = [5, 10, 15, 20, 25, 30];

% IMPROVED: Increased trials for better statistical significance
num_trials = 15;  

%% 1. Epsilon (minimum bid increment) Sensitivity Analysis
fprintf('Running epsilon sensitivity analysis...\n');

epsilon_results = struct();
epsilon_results.epsilon_values = epsilon_range;
epsilon_results.iterations = zeros(length(epsilon_range), num_trials);
epsilon_results.optimality_gap = zeros(length(epsilon_range), num_trials);
epsilon_results.makespan = zeros(length(epsilon_range), num_trials);
epsilon_results.optimal_makespan = zeros(length(epsilon_range), num_trials);
epsilon_results.theoretical_gap = zeros(length(epsilon_range), 1);
epsilon_results.oscillation_count = zeros(length(epsilon_range), num_trials);

for e_idx = 1:length(epsilon_range)
    fprintf('Testing epsilon = %.3f\n', epsilon_range(e_idx));
    params = base_params;
    params.epsilon = epsilon_range(e_idx);
    
    % IMPROVED: Store theoretical bound
    epsilon_results.theoretical_gap(e_idx) = 2 * params.epsilon;
    
    for trial = 1:num_trials
        % Create environment and tasks
        env = env_utils.createEnvironment(4, 4);
        robots = robot_utils.createRobots(2, env);
        
        % IMPROVED: Ensure consistent number of tasks without too many dependencies
        tasks = task_utils.createTasks(10, env);
        tasks = task_utils.addTaskDependencies(tasks, 0.2);  % Lower probability for cleaner tests
        
        % Run simulation
        [metrics, ~] = auction_utils.runAuctionSimulation(params, env, robots, tasks, false);
        
        % Store results
        epsilon_results.iterations(e_idx, trial) = metrics.iterations;
        epsilon_results.optimality_gap(e_idx, trial) = metrics.optimality_gap;
        epsilon_results.makespan(e_idx, trial) = metrics.makespan;
        epsilon_results.optimal_makespan(e_idx, trial) = metrics.optimal_makespan;
        epsilon_results.oscillation_count(e_idx, trial) = metrics.oscillation_count;
    end
end

% Calculate means and standard deviations
epsilon_results.iterations_mean = mean(epsilon_results.iterations, 2);
epsilon_results.iterations_std = std(epsilon_results.iterations, 0, 2);
epsilon_results.optimality_gap_mean = mean(epsilon_results.optimality_gap, 2);
epsilon_results.optimality_gap_std = std(epsilon_results.optimality_gap, 0, 2);
epsilon_results.makespan_mean = mean(epsilon_results.makespan, 2);
epsilon_results.makespan_std = std(epsilon_results.makespan, 0, 2);
epsilon_results.oscillation_count_mean = mean(epsilon_results.oscillation_count, 2);
epsilon_results.gap_ratio = epsilon_results.optimality_gap_mean ./ epsilon_results.theoretical_gap;

% Store in results
results.epsilon = epsilon_results;

%% 2. Communication Reliability Analysis
fprintf('\nRunning communication reliability analysis...\n');

comm_results = struct();
comm_results.packet_loss_values = packet_loss_range;
comm_results.iterations = zeros(length(packet_loss_range), num_trials);
comm_results.optimality_gap = zeros(length(packet_loss_range), num_trials);
comm_results.convergence_success = zeros(length(packet_loss_range), num_trials);
comm_results.oscillation_count = zeros(length(packet_loss_range), num_trials);

for pl_idx = 1:length(packet_loss_range)
    fprintf('Testing packet loss probability = %.2f\n', packet_loss_range(pl_idx));
    params = base_params;
    params.packet_loss_prob = packet_loss_range(pl_idx);
    
    for trial = 1:num_trials
        % Create environment and tasks
        env = env_utils.createEnvironment(4, 4);
        robots = robot_utils.createRobots(2, env);
        tasks = task_utils.createTasks(10, env);
        tasks = task_utils.addTaskDependencies(tasks, 0.2);
        
        % Run simulation
        [metrics, converged] = auction_utils.runAuctionSimulation(params, env, robots, tasks, false);
        
        % Store results
        comm_results.iterations(pl_idx, trial) = metrics.iterations;
        comm_results.optimality_gap(pl_idx, trial) = metrics.optimality_gap;
        comm_results.convergence_success(pl_idx, trial) = converged;
        comm_results.oscillation_count(pl_idx, trial) = metrics.oscillation_count;
    end
end

% Calculate means and standard deviations
comm_results.iterations_mean = mean(comm_results.iterations, 2);
comm_results.iterations_std = std(comm_results.iterations, 0, 2);
comm_results.optimality_gap_mean = mean(comm_results.optimality_gap, 2);
comm_results.optimality_gap_std = std(comm_results.optimality_gap, 0, 2);
comm_results.convergence_rate = mean(comm_results.convergence_success, 2);
comm_results.oscillation_count_mean = mean(comm_results.oscillation_count, 2);

% Store in results
results.communication = comm_results;

%% 3. Scalability Analysis (Task Count)
fprintf('\nRunning scalability analysis...\n');

scalability_results = struct();
scalability_results.task_count = task_count_range;
scalability_results.iterations = zeros(length(task_count_range), num_trials);
scalability_results.theoretical_bound = zeros(length(task_count_range), 1);
scalability_results.messages = zeros(length(task_count_range), num_trials);
scalability_results.optimality_gap = zeros(length(task_count_range), num_trials);
scalability_results.oscillation_count = zeros(length(task_count_range), num_trials);

for t_idx = 1:length(task_count_range)
    fprintf('Testing task count = %d\n', task_count_range(t_idx));
    params = base_params;
    
    % Calculate theoretical bound
    K = task_count_range(t_idx);
    b_max = max(params.alpha);
    epsilon = params.epsilon;
    scalability_results.theoretical_bound(t_idx) = K^2 * b_max / epsilon;
    
    for trial = 1:num_trials
        % Create environment and tasks
        env = env_utils.createEnvironment(4, 4);
        robots = robot_utils.createRobots(2, env);
        tasks = task_utils.createTasks(task_count_range(t_idx), env);
        
        % IMPROVED: Limit task dependencies for large task counts to avoid gridlock
        depend_prob = min(0.3, 3.0 / task_count_range(t_idx));
        tasks = task_utils.addTaskDependencies(tasks, depend_prob);
        
        % Run simulation
        [metrics, ~] = auction_utils.runAuctionSimulation(params, env, robots, tasks, false);
        
        % Store results
        scalability_results.iterations(t_idx, trial) = metrics.iterations;
        scalability_results.messages(t_idx, trial) = metrics.messages;
        scalability_results.optimality_gap(t_idx, trial) = metrics.optimality_gap;
        scalability_results.oscillation_count(t_idx, trial) = metrics.oscillation_count;
    end
end

% Calculate means and standard deviations
scalability_results.iterations_mean = mean(scalability_results.iterations, 2);
scalability_results.iterations_std = std(scalability_results.iterations, 0, 2);
scalability_results.messages_mean = mean(scalability_results.messages, 2);
scalability_results.messages_std = std(scalability_results.messages, 0, 2);
scalability_results.optimality_gap_mean = mean(scalability_results.optimality_gap, 2);
scalability_results.bound_ratio = scalability_results.iterations_mean ./ scalability_results.theoretical_bound;
scalability_results.oscillation_count_mean = mean(scalability_results.oscillation_count, 2);

% Store in results
results.scalability = scalability_results;

%% 4. Failure Recovery Analysis
fprintf('\nRunning failure recovery analysis...\n');

recovery_results = struct();
recovery_results.task_count = task_count_range;
recovery_results.recovery_time = zeros(length(task_count_range), num_trials);
recovery_results.theoretical_bound = zeros(length(task_count_range), num_trials);
recovery_results.failed_task_count = zeros(length(task_count_range), num_trials);
recovery_results.makespan_degradation = zeros(length(task_count_range), num_trials);

for t_idx = 1:length(task_count_range)
    fprintf('Testing recovery with task count = %d\n', task_count_range(t_idx));
    params = base_params;
    
    % IMPROVED: Adjust failure time based on task count
    params.failure_time = 10 + ceil(task_count_range(t_idx) / 5);  
    params.failed_robot = 1;   % Robot 1 fails
    
    for trial = 1:num_trials
        % Create environment and tasks
        env = env_utils.createEnvironment(4, 4);
        robots = robot_utils.createRobots(2, env);
        tasks = task_utils.createTasks(task_count_range(t_idx), env);
        
        % IMPROVED: Limit task dependencies for large task counts
        depend_prob = min(0.3, 3.0 / task_count_range(t_idx));
        tasks = task_utils.addTaskDependencies(tasks, depend_prob);
        
        % Run simulation
        [metrics, ~] = auction_utils.runAuctionSimulation(params, env, robots, tasks, false);
        
        % Store results
        recovery_results.recovery_time(t_idx, trial) = metrics.recovery_time;
        recovery_results.failed_task_count(t_idx, trial) = metrics.failed_task_count;
        
        % IMPROVED: Track makespan degradation
        if isfield(metrics, 'makespan_before_failure') && isfield(metrics, 'makespan')
            recovery_results.makespan_degradation(t_idx, trial) = metrics.makespan - metrics.makespan_before_failure;
        end
        
        % Calculate theoretical bound for this trial
        T_f = metrics.failed_task_count;  % Number of tasks assigned to failed robot
        b_max = max(params.alpha);
        epsilon = params.epsilon;
        recovery_results.theoretical_bound(t_idx, trial) = T_f + b_max / epsilon;
    end
end

% Calculate means and standard deviations
recovery_results.recovery_time_mean = mean(recovery_results.recovery_time, 2);
recovery_results.recovery_time_std = std(recovery_results.recovery_time, 0, 2);
recovery_results.theoretical_bound_mean = mean(recovery_results.theoretical_bound, 2);
recovery_results.failed_task_count_mean = mean(recovery_results.failed_task_count, 2);
recovery_results.makespan_degradation_mean = mean(recovery_results.makespan_degradation, 2);
recovery_results.bound_ratio = recovery_results.recovery_time_mean ./ recovery_results.theoretical_bound_mean;

% Store in results
results.recovery = recovery_results;

%% Plot results with improved visualizations
% 1. Epsilon Sensitivity
figure('Name', 'Epsilon Sensitivity Analysis', 'Position', [100, 100, 1200, 600]);

subplot(2, 3, 1);
errorbar(epsilon_range, epsilon_results.iterations_mean, epsilon_results.iterations_std, '-o', 'LineWidth', 1.5);
xlabel('Epsilon (ε)');
ylabel('Iterations to Converge');
title('Convergence Speed vs. Epsilon');
grid on;

subplot(2, 3, 2);
errorbar(epsilon_range, epsilon_results.optimality_gap_mean, epsilon_results.optimality_gap_std, '-o', 'LineWidth', 1.5);
hold on;
plot(epsilon_range, 2*epsilon_range, '--r', 'LineWidth', 1.5);
legend('Measured Gap', 'Theoretical Bound (2ε)');
xlabel('Epsilon (ε)');
ylabel('Optimality Gap');
title('Solution Quality vs. Epsilon');
grid on;

subplot(2, 3, 3);
bar(epsilon_range, epsilon_results.gap_ratio);
xlabel('Epsilon (ε)');
ylabel('Gap Ratio (Actual/Theoretical)');
title('Gap Ratio vs. Epsilon');
grid on;
ylim([0, max(epsilon_results.gap_ratio)*1.1]);

subplot(2, 3, 4);
bar(epsilon_range, epsilon_results.makespan_mean);
hold on;
errorbar(epsilon_range, epsilon_results.makespan_mean, epsilon_results.makespan_std, 'k.', 'LineWidth', 1.5);
xlabel('Epsilon (ε)');
ylabel('Makespan');
title('Final Makespan vs. Epsilon');
grid on;

subplot(2, 3, 5);
plot(epsilon_range, epsilon_results.oscillation_count_mean, '-o', 'LineWidth', 1.5);
xlabel('Epsilon (ε)');
ylabel('Task Oscillations');
title('Task Oscillations vs. Epsilon');
grid on;

% 2. Communication Reliability
figure('Name', 'Communication Reliability Analysis', 'Position', [100, 550, 1200, 400]);

subplot(1, 3, 1);
errorbar(packet_loss_range, comm_results.iterations_mean, comm_results.iterations_std, '-o', 'LineWidth', 1.5);
xlabel('Packet Loss Probability');
ylabel('Iterations to Converge');
title('Convergence Speed vs. Packet Loss');
grid on;

subplot(1, 3, 2);
errorbar(packet_loss_range, comm_results.optimality_gap_mean, comm_results.optimality_gap_std, '-o', 'LineWidth', 1.5);
xlabel('Packet Loss Probability');
ylabel('Optimality Gap');
title('Solution Quality vs. Packet Loss');
grid on;

subplot(1, 3, 3);
bar(packet_loss_range, comm_results.convergence_rate);
xlabel('Packet Loss Probability');
ylabel('Convergence Rate');
title('Convergence Success vs. Packet Loss');
grid on;
ylim([0, 1.1]);

% 3. Scalability Analysis
figure('Name', 'Scalability Analysis', 'Position', [600, 100, 1200, 400]);

subplot(1, 3, 1);
errorbar(task_count_range, scalability_results.iterations_mean, scalability_results.iterations_std, '-o', 'LineWidth', 1.5);
hold on;
plot(task_count_range, scalability_results.theoretical_bound, '--r', 'LineWidth', 1.5);
legend('Actual Iterations', 'Theoretical Bound');
xlabel('Number of Tasks (K)');
ylabel('Iterations to Converge');
title('Convergence Speed vs. Task Count');
grid on;

subplot(1, 3, 2);
errorbar(task_count_range, scalability_results.messages_mean, scalability_results.messages_std, '-o', 'LineWidth', 1.5);
xlabel('Number of Tasks (K)');
ylabel('Number of Messages');
title('Communication Overhead vs. Task Count');
grid on;

subplot(1, 3, 3);
plot(task_count_range, scalability_results.optimality_gap_mean, '-o', 'LineWidth', 1.5);
xlabel('Number of Tasks (K)');
ylabel('Optimality Gap');
title('Solution Quality vs. Task Count');
grid on;

% 4. Failure Recovery Analysis
figure('Name', 'Failure Recovery Analysis', 'Position', [600, 550, 1200, 400]);

subplot(1, 3, 1);
errorbar(task_count_range, recovery_results.recovery_time_mean, recovery_results.recovery_time_std, '-o', 'LineWidth', 1.5);
hold on;
errorbar(task_count_range, recovery_results.theoretical_bound_mean, recovery_results.recovery_time_std, '--r', 'LineWidth', 1.5);
legend('Actual Recovery Time', 'Theoretical Bound');
xlabel('Number of Tasks (K)');
ylabel('Recovery Time (iterations)');
title('Recovery Time vs. Task Count');
grid on;

subplot(1, 3, 2);
bar(task_count_range, recovery_results.bound_ratio);
xlabel('Number of Tasks (K)');
ylabel('Ratio (Actual/Theoretical)');
title('Recovery Time Bound Ratio');
grid on;
ylim([0, max(recovery_results.bound_ratio)*1.1]);

subplot(1, 3, 3);
bar(task_count_range, recovery_results.failed_task_count_mean);
xlabel('Number of Tasks (K)');
ylabel('Average Tasks to Reassign');
title('Failed Task Count');
grid on;

% Save results
if ~exist('../results', 'dir')
    mkdir('../results');
end
save('../results/auction_algorithm_sensitivity_results.mat', 'results');

% Save figures
if ~exist('../figures/parameter_sensitivity_plots', 'dir')
    mkdir('../figures/parameter_sensitivity_plots');
end

saveas(figure(1), '../figures/parameter_sensitivity_plots/epsilon_sensitivity.fig');
saveas(figure(1), '../figures/parameter_sensitivity_plots/epsilon_sensitivity.png');
saveas(figure(2), '../figures/parameter_sensitivity_plots/communication_reliability.fig');
saveas(figure(2), '../figures/parameter_sensitivity_plots/communication_reliability.png');
saveas(figure(3), '../figures/parameter_sensitivity_plots/scalability_analysis.fig');
saveas(figure(3), '../figures/parameter_sensitivity_plots/scalability_analysis.png');
saveas(figure(4), '../figures/parameter_sensitivity_plots/failure_recovery.fig');
saveas(figure(4), '../figures/parameter_sensitivity_plots/failure_recovery.png');

fprintf('\nSensitivity analysis complete. Results saved to auction_algorithm_sensitivity_results.mat\n');