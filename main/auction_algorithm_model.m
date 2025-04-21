% Distributed Auction Algorithm Verification Model
% For Decentralized Control of Dual Mobile Manipulators
% Based on Zavlanos et al. (2008) with extensions for task dependencies and failure recovery

%% Setup and Configuration
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

% Set random seed for reproducibility
rng(42);

% Algorithm parameters
params.epsilon = 0.05;        % Minimum bid increment
params.alpha = [1.0, 0.5, 2.0, 0.8, 0.3];  % Bid calculation weights
params.gamma = 0.5;           % Consensus weight factor
params.lambda = 0.1;          % Information decay rate
params.beta = [1.5, 1.0];     % Recovery auction weights
params.comm_delay = 0;        % Communication delay (in iterations)
params.packet_loss_prob = 0;  % Probability of packet loss
params.failure_time = inf;    % Time of robot failure (inf = no failure)
params.failed_robot = [];     % Which robot fails

% Create a simulation environment
env = env_utils.createEnvironment(4, 4);  % 4m x 4m workspace

% Create robots
robots = robot_utils.createRobots(2, env);

% Create tasks with dependencies
num_tasks = 10;
tasks = task_utils.createTasks(num_tasks, env);

% Add task dependencies (create a DAG)
tasks = task_utils.addTaskDependencies(tasks);

% Create data structures for the auction algorithm
auction_data = auction_utils.initializeAuctionData(tasks, robots);

% Visualization setup
figure('Name', 'Distributed Auction Algorithm Simulation', 'Position', [100, 100, 1200, 800]);
subplot(2, 3, [1, 4]);
env_utils.visualizeEnvironment(env, robots, tasks, auction_data);
title('Environment');

% Performance metrics
metrics = struct();
metrics.iterations = 0;
metrics.messages = 0;
metrics.convergence_history = [];
metrics.price_history = zeros(length(tasks), 1000);  % Preallocate
metrics.assignment_history = zeros(length(tasks), 1000);
metrics.completion_time = 0;
metrics.optimality_gap = 0;
metrics.recovery_time = 0;
metrics.failure_time = 0;  % Initialize failure_time field

%% Main simulation loop
max_iterations = 1000;
converged = false;

% Available tasks (initially only those with no prerequisites)
available_tasks = task_utils.findAvailableTasks(tasks, []);

fprintf('Starting auction algorithm simulation...\n');
fprintf('Number of robots: %d, Number of tasks: %d\n', length(robots), length(tasks));

% Print task details for inspection
fprintf('\nTask details:\n');
for i = 1:length(tasks)
    prereqs = tasks(i).prerequisites;
    if isempty(prereqs)
        prereq_str = 'none';
    else
        prereq_str = sprintf('%d ', prereqs);
    end
    fprintf('Task %d: position=[%.1f, %.1f], time=%.1f, prereqs=[%s]\n', ...
        i, tasks(i).position(1), tasks(i).position(2), tasks(i).execution_time, prereq_str);
end

% Track iterations without changes for convergence detection
unchanged_iterations = 0;

for iter = 1:max_iterations
    metrics.iterations = iter;
    
    % Check for robot failure
    if iter == params.failure_time && ~isempty(params.failed_robot)
        fprintf('Robot %d has failed at iteration %d\n', params.failed_robot, iter);
        robots(params.failed_robot).failed = true;
        metrics.failure_time = iter;  % Store failure time in metrics
        
        % Initiate recovery process
        auction_data = auction_utils.initiateRecovery(auction_data, robots, tasks, params.failed_robot);
    end
    
    % Simulate one step of the distributed auction algorithm
    [auction_data, new_assignments, messages] = auction_utils.distributedAuctionStep(auction_data, robots, tasks, available_tasks, params);
    metrics.messages = metrics.messages + messages;
    
    % Update visualization
    subplot(2, 3, [1, 4]);
    env_utils.visualizeEnvironment(env, robots, tasks, auction_data);
    title(sprintf('Environment (Iteration %d)', iter));
    
    % Update metrics
    metrics.price_history(:, iter) = auction_data.prices;
    metrics.assignment_history(:, iter) = auction_data.assignment;
    
    % Calculate convergence metric (change in assignments)
    if iter > 1
        conv_metric = sum(metrics.assignment_history(:, iter) ~= metrics.assignment_history(:, iter-1));
        metrics.convergence_history(iter) = conv_metric;
        
        % Update unchanged iterations counter
        if conv_metric == 0
            unchanged_iterations = unchanged_iterations + 1;
        else
            unchanged_iterations = 0;
        end
    else
        metrics.convergence_history(iter) = NaN;
        unchanged_iterations = 0;
    end
    
    % Check if any new tasks become available due to completed prerequisites
    completed_tasks = find(auction_data.completion_status == 1);
    available_tasks = task_utils.findAvailableTasks(tasks, completed_tasks);
    
    % Check for convergence - more stringent criteria
    if iter > 20 && unchanged_iterations >= 15 && all(available_tasks <= length(tasks))
        converged = true;
        fprintf('Auction algorithm converged after %d iterations (stable for %d iterations)\n', ...
            iter, unchanged_iterations);
        break;
    end
    
    % Additional convergence check - all tasks assigned
    if all(auction_data.assignment > 0) && unchanged_iterations >= 5
        converged = true;
        fprintf('Auction algorithm converged after %d iterations (all tasks assigned)\n', iter);
        break;
    end
    
    % Update recovery time if in recovery mode
    if ~isempty(params.failed_robot) && metrics.failure_time > 0 && iter > metrics.failure_time && metrics.recovery_time == 0
        % Check if all tasks have been reassigned
        if all(auction_data.assignment(auction_data.assignment == params.failed_robot) == 0)
            metrics.recovery_time = iter - metrics.failure_time;
            fprintf('Recovery completed after %d iterations\n', metrics.recovery_time);
        end
    end
    
    % Diagnostics every 10 iterations
    if mod(iter, 10) == 0
        auction_utils.analyzeTaskAllocation(auction_data, tasks);
    end
    
    % Pause for visualization
    pause(0.01);
end

% Trim history matrices to actual size
metrics.price_history = metrics.price_history(:, 1:iter);
metrics.assignment_history = metrics.assignment_history(:, 1:iter);

% Plot results
subplot(2, 3, 2);
env_utils.plotTaskPrices(metrics.price_history);
title('Task Prices Over Time');

subplot(2, 3, 3);
env_utils.plotAssignments(metrics.assignment_history, length(robots));
title('Task Assignments Over Time');

subplot(2, 3, 5);
env_utils.plotConvergence(metrics.convergence_history);
title('Convergence Metric');

subplot(2, 3, 6);
env_utils.plotWorkload(metrics.assignment_history(:, end), tasks, robots);
title('Final Workload Distribution');

% Calculate optimality gap - fixed to use absolute value
optimal_makespan = robot_utils.calculateOptimalMakespan(tasks, robots);
achieved_makespan = robot_utils.calculateMakespan(auction_data.assignment, tasks, robots);
metrics.optimality_gap = abs(achieved_makespan - optimal_makespan);
metrics.theoretical_gap_bound = 2 * params.epsilon;

% Display final metrics
fprintf('\n--- Final Performance Metrics ---\n');
fprintf('Iterations to converge: %d\n', metrics.iterations);
fprintf('Theoretical bound: O(K² · bₘₐₓ/ε) = O(%d)\n', numel(tasks)^2 * max(params.alpha) / params.epsilon);
fprintf('Messages exchanged: %d\n', metrics.messages);
fprintf('Achieved makespan: %.2f\n', achieved_makespan);
fprintf('Optimal makespan: %.2f\n', optimal_makespan);
fprintf('Optimality gap: %.2f\n', metrics.optimality_gap);
fprintf('Theoretical gap bound (2ε): %.2f\n', metrics.theoretical_gap_bound);
if ~isinf(params.failure_time)
    fprintf('Recovery time: %d iterations\n', metrics.recovery_time);
    fprintf('Theoretical recovery bound: O(|Tᶠ|) + O(bₘₐₓ/ε) ≈ %d\n', ...
        numel(find(auction_data.initial_assignment == params.failed_robot)) + ...
        round(max(params.alpha) / params.epsilon));
end

% Detailed final analysis
fprintf('\n--- Final Task Allocation Analysis ---\n');
auction_utils.analyzeTaskAllocation(auction_data, tasks);
auction_utils.analyzeBidDistribution(auction_data, robots, tasks);

% Save results
if ~exist('../results', 'dir')
    mkdir('../results');
end
save('../results/auction_algorithm_results.mat', 'metrics', 'params', 'tasks', 'robots');

% Save figure
if ~exist('../figures/auction_convergence_plots', 'dir')
    mkdir('../figures/auction_convergence_plots');
end
saveas(gcf, '../figures/auction_convergence_plots/auction_algorithm_results.fig');
saveas(gcf, '../figures/auction_convergence_plots/auction_algorithm_results.png');