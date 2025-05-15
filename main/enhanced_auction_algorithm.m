% Enhanced Distributed Auction Algorithm Implementation
% Based on mathematical foundations for decentralized control of dual mobile manipulators

%% Setup and Configuration
clear all;
close all;
clc;

% Add utility functions to path
addpath('../common');

% Load utility functions
enhanced_auction_utils = enhanced_auction_utils();
env_utils = environment_utils();
robot_utils = robot_utils();
task_utils = task_utils();
scheduler_utils = scheduler_utils();

% Set random seed for reproducibility
rng(42);

% Algorithm parameters
params.epsilon = 0.05;        % Minimum bid increment
params.alpha = [0.8, 0.3, 1.0, 1.2, 0.2];  % Bid calculation weights [distance, config, capability, workload, energy]
params.gamma = 0.5;           % Consensus weight factor
params.lambda = 0.1;          % Information decay rate
params.beta = [2.0, 1.5, 1.0];  % Recovery auction weights
params.comm_delay = 0;        % Communication delay (in iterations)
params.packet_loss_prob = 0;  % Probability of packet loss
params.failure_time = 50;     % Time of robot failure (inf = no failure)
params.failed_robot = 1;      % Which robot fails
params.heartbeat_interval = 3;  % Heartbeat signal interval
params.missed_heartbeat_threshold = 3;  % Number of missed heartbeats for failure detection
params.min_progress_rate = 0.05;  % Minimum progress rate
params.sync_timeout = 5;      % Timeout for synchronization
params.max_sync_distance = 0.5;  % Maximum distance for synchronization

% Create a simulation environment
env = env_utils.createEnvironment(4, 4);  % 4m x 4m workspace

% Create robots
robots = robot_utils.createRobots(2, env);

% Create tasks with dependencies
num_tasks = 15;
tasks = task_utils.createTasks(num_tasks, env);
tasks = task_utils.addTaskDependencies(tasks, 0.3);  % Add dependencies with 30% probability

% Add collaborative tasks (randomly select a few tasks as collaborative)
collab_task_indices = [3, 8, 12];  % Example collaborative tasks
for i = collab_task_indices
    if i <= num_tasks
        tasks(i).collaborative = true;
    end
end

% Create data structures for the auction algorithm
auction_data = enhanced_auction_utils.initializeAuctionData(tasks, robots);
auction_data.collaborative_tasks(collab_task_indices) = true;

% Configure communication model
auction_data = setupCommunicationModel(auction_data, robots, params);

% Visualization setup
figure('Name', 'Enhanced Distributed Auction Algorithm Simulation', 'Position', [100, 100, 1200, 800]);
subplot(2, 3, [1, 4]);
env_utils.visualizeEnvironment(env, robots, tasks, auction_data);
title('Initial Environment');

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
metrics.failure_time = 0;
metrics.makespan = 0;
metrics.optimal_makespan = 0;
metrics.makespan_before_failure = 0;
metrics.theoretical_recovery_bound = 0;
metrics.oscillation_count = 0;
metrics.converged = false;

%% Main simulation loop
max_iterations = 200;
converged = false;

% Available tasks (initially only those with no prerequisites)
available_tasks = task_utils.findAvailableTasks(tasks, []);

fprintf('Starting enhanced auction algorithm simulation...\n');
fprintf('Number of robots: %d, Number of tasks: %d\n', length(robots), length(tasks));
fprintf('Number of collaborative tasks: %d\n', sum([tasks.collaborative]));

% Print task details for inspection
fprintf('\nTask details:\n');
for i = 1:length(tasks)
    prereqs = tasks(i).prerequisites;
    if isempty(prereqs)
        prereq_str = 'none';
    else
        prereq_str = sprintf('%d ', prereqs);
    end
    
    collab_str = '';
    if isfield(tasks, 'collaborative') && tasks(i).collaborative
        collab_str = ' (Collaborative)';
    end
    
    fprintf('Task %d: position=[%.1f, %.1f], time=%.1f, prereqs=[%s]%s\n', ...
        i, tasks(i).position(1), tasks(i).position(2), tasks(i).execution_time, prereq_str, collab_str);
end

for iter = 1:max_iterations
    % Update current time and metrics
    params.current_time = iter;
    metrics.iterations = iter;
    
    % 1. Check for robot failures
    [failures, auction_data] = enhanced_auction_utils.detectFailures(auction_data, robots, params);
    
    % 2. Handle any new failures
    if any(failures)
        for i = find(failures)
            fprintf('Robot %d has failed at iteration %d\n', i, iter);
            robots(i).failed = true;
            
            % Record metrics before failure
            metrics.failed_task_count = sum(auction_data.assignment == i);
            metrics.makespan_before_failure = robot_utils.calculateMakespan(auction_data.assignment, tasks, robots);
            metrics.failed_robot = i;
            metrics.failure_time = iter;
            
            % Initiate recovery process
            [auction_data, recovery_initiated] = enhanced_auction_utils.enhancedRecovery(auction_data, robots, tasks, i, params);
            
            if recovery_initiated
                fprintf('Recovery initiated for robot %d failures\n', i);
            end
        end
    end
    
    % 3. Run the distributed auction algorithm step
    [auction_data, new_assignments, messages] = enhanced_auction_utils.enhancedDistributedAuctionStep(auction_data, robots, tasks, available_tasks, params);
    metrics.messages = metrics.messages + messages;
    
    % 4. Update visualization
    subplot(2, 3, [1, 4]);
    env_utils.visualizeEnvironment(env, robots, tasks, auction_data);
    title(sprintf('Environment (Iteration %d)', iter));
    
    % 5. Update metrics
    metrics.price_history(:, iter) = auction_data.prices;
    metrics.assignment_history(:, iter) = auction_data.assignment;
    
    % Calculate convergence metric (change in assignments)
    if iter > 1
        conv_metric = sum(metrics.assignment_history(:, iter) ~= metrics.assignment_history(:, iter-1));
        metrics.convergence_history(iter) = conv_metric;
    else
        metrics.convergence_history(iter) = NaN;
    end
    
    % 6. Check if any new tasks become available due to completed prerequisites
    completed_tasks = find(auction_data.completion_status == 1);
    [auction_data, available_tasks] = enhanced_auction_utils.manageTaskDependencies(auction_data, tasks, completed_tasks);
    
    % 7. Check for recovery completion
    if metrics.failure_time > 0 && metrics.recovery_time == 0
        % Check if all tasks previously assigned to the failed robot have been reassigned
        failed_robot_tasks = find(auction_data.failure_assignment == metrics.failed_robot);
        if all(auction_data.assignment(failed_robot_tasks) ~= metrics.failed_robot) && ...
           all(auction_data.assignment(failed_robot_tasks) ~= 0) && ...
           all(auction_data.assignment(failed_robot_tasks) ~= -1)
            metrics.recovery_time = iter - metrics.failure_time;
            fprintf('Recovery completed after %d iterations\n', metrics.recovery_time);
        end
    end
    
    % 8. Check for convergence
    [auction_data, converged, convergence_metrics] = enhanced_auction_utils.analyzeBidConvergence(auction_data, params);
    
    if converged
        fprintf('Auction algorithm converged after %d iterations\n', iter);
        metrics.converged = true;
        break;
    end
    
    % 9. Diagnostics every 10 iterations
    if mod(iter, 10) == 0
        fprintf('Iteration %d: ', iter);
        auction_utils.analyzeTaskAllocation(auction_data, tasks);
    end
    
    % 10. Pause for visualization
    pause(0.01);
    
    % Stop if we reach max iterations
    if iter == max_iterations
        fprintf('Maximum iterations (%d) reached without convergence\n', max_iterations);
        break;
    end
end

% Trim history matrices to actual size
metrics.price_history = metrics.price_history(:, 1:iter);
metrics.assignment_history = metrics.assignment_history(:, 1:iter);

% Calculate final metrics
metrics.makespan = robot_utils.calculateMakespan(auction_data.assignment, tasks, robots);
metrics.optimal_makespan = robot_utils.calculateOptimalMakespan(tasks, robots);
metrics.optimality_gap = abs(metrics.makespan - metrics.optimal_makespan);

% Calculate theoretical recovery bound
if metrics.failure_time > 0
    T_f = metrics.failed_task_count;
    b_max = max(params.alpha);
    epsilon = params.epsilon;
    metrics.theoretical_recovery_bound = T_f + round(b_max/epsilon);
end

% Count task oscillations
if isfield(auction_data, 'task_oscillation_count')
    metrics.oscillation_count = sum(auction_data.task_oscillation_count);
end

% Plot results
subplot(2, 3, 2);
plot(metrics.price_history', 'LineWidth', 1.5);
title('Task Prices Over Time');
xlabel('Iteration');
ylabel('Price');
grid on;

subplot(2, 3, 3);
imagesc(metrics.assignment_history);
colormap(jet);
colorbar;
title('Task Assignments Over Time');
xlabel('Iteration');
ylabel('Task');
grid on;

subplot(2, 3, 5);
plot(metrics.convergence_history, 'LineWidth', 1.5);
title('Convergence Metric');
xlabel('Iteration');
ylabel('Number of Changes');
grid on;

% Add vertical line for failure time if applicable
if metrics.failure_time > 0
    hold on;
    xline(metrics.failure_time, 'r--', 'LineWidth', 2);
    if metrics.recovery_time > 0
        xline(metrics.failure_time + metrics.recovery_time, 'g--', 'LineWidth', 2);
        legend('Changes', 'Failure', 'Recovery Complete');
    else
        legend('Changes', 'Failure');
    end
    hold off;
end

subplot(2, 3, 6);
robot_workloads = zeros(length(robots), 1);
for j = 1:length(tasks)
    r = auction_data.assignment(j);
    if r > 0
        if isfield(tasks, 'execution_time')
            robot_workloads(r) = robot_workloads(r) + tasks(j).execution_time;
        else
            robot_workloads(r) = robot_workloads(r) + 1;
        end
    end
end
bar(robot_workloads);
title('Final Workload Distribution');
xlabel('Robot');
ylabel('Total Execution Time');
grid on;

% Generate schedule based on final assignment
schedule = scheduler_utils.generateSchedule(auction_data.assignment, tasks, robots);

% Display final metrics
fprintf('\n--- Final Performance Metrics ---\n');
fprintf('Iterations to converge: %d\n', metrics.iterations);
fprintf('Theoretical bound: O(K² · bₘₐₓ/ε) = O(%d)\n', length(tasks)^2 * max(params.alpha) / params.epsilon);
fprintf('Messages exchanged: %d\n', metrics.messages);
fprintf('Achieved makespan: %.2f\n', metrics.makespan);
fprintf('Optimal makespan: %.2f\n', metrics.optimal_makespan);
fprintf('Optimality gap: %.2f\n', metrics.optimality_gap);
fprintf('Theoretical gap bound (2ε): %.2f\n', 2 * params.epsilon);
if metrics.failure_time > 0
    fprintf('Recovery time: %d iterations\n', metrics.recovery_time);
    fprintf('Theoretical recovery bound: O(|Tᶠ|) + O(bₘₐₓ/ε) ≈ %d\n', metrics.theoretical_recovery_bound);
end
fprintf('Task oscillations: %d\n', metrics.oscillation_count);

% Detailed final analysis
fprintf('\n--- Final Task Allocation Analysis ---\n');
auction_utils.analyzeTaskAllocation(auction_data, tasks);

% Create schedule visualization in a new figure
figure('Name', 'Task Schedule', 'Position', [100, 100, 1200, 600]);
scheduler_utils.visualizeSchedule(schedule, tasks, robots);

% Save results
if ~exist('../results', 'dir')
    mkdir('../results');
end
save('../results/enhanced_auction_results.mat', 'metrics', 'params', 'tasks', 'robots', 'auction_data', 'schedule');

% Helper function to setup communication model
function auction_data = setupCommunicationModel(auction_data, robots, params)
    num_robots = length(robots);
    
    % Set up communication delays
    if isfield(params, 'comm_delay')
        if isscalar(params.comm_delay)
            % Uniform delay
            auction_data.comm_delay = params.comm_delay * ones(num_robots, num_robots);
            % Set diagonal to zero (no delay to self)
            for i = 1:num_robots
                auction_data.comm_delay(i, i) = 0;
            end
        else
            % Custom delay matrix
            auction_data.comm_delay = params.comm_delay;
        end
    else
        % Default: no delay
        auction_data.comm_delay = zeros(num_robots, num_robots);
    end
    
    % Set up packet loss probabilities
    if isfield(params, 'packet_loss_prob')
        if isscalar(params.packet_loss_prob)
            % Uniform probability
            auction_data.packet_loss_prob = params.packet_loss_prob * ones(num_robots, num_robots);
            % Set diagonal to zero (no loss to self)
            for i = 1:num_robots
                auction_data.packet_loss_prob(i, i) = 0;
            end
        else
            % Custom probability matrix
            auction_data.packet_loss_prob = params.packet_loss_prob;
        end
    else
        % Default: no packet loss
        auction_data.packet_loss_prob = zeros(num_robots, num_robots);
    end
    
    % Set up communication ranges
    if isfield(params, 'comm_range')
        if isscalar(params.comm_range)
            % Uniform range
            auction_data.comm_range = params.comm_range * ones(num_robots, 1);
        else
            % Custom range vector
            auction_data.comm_range = params.comm_range;
        end
    else
        % Default: infinite range
        auction_data.comm_range = Inf * ones(num_robots, 1);
    end
    
    return;
end