% Consensus Protocol and Failure Recovery Testing
% This script specifically focuses on validating the time-weighted consensus protocol
% and the failure recovery mechanisms for dual mobile manipulators

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

% Set random seed for reproducibility
rng(42);

%% Part 1: Time-Weighted Consensus Protocol Validation

% System parameters
num_robots = 2;
state_dim = 5;  % Dimensions of the state vector
num_iterations = 50;
gamma = 0.5;    % Base weight factor
lambda = 0.1;   % Information decay rate

% Initialize state vectors for both robots
% Each robot has its own estimate of the global state
X = zeros(state_dim, num_robots, num_iterations);
X(:,:,1) = 10 * rand(state_dim, num_robots);  % Random initial states

% Last communication time matrix
last_comm_time = zeros(num_robots, num_robots);

% Communication delay profile (iterations)
comm_delay = zeros(num_robots, num_robots);
comm_delay(1,2) = 2;  % Robot 1 receives information from Robot 2 with delay
comm_delay(2,1) = 1;  % Robot 2 receives information from Robot 1 with delay

% Initialize true state value (for reference)
true_state = mean(X(:,:,1), 2);

% Run consensus algorithm
fprintf('Running time-weighted consensus protocol simulation...\n');
for k = 2:num_iterations
    % Update last communication time
    last_comm_time = last_comm_time + 1;
    
    % For each robot
    for i = 1:num_robots
        % Start with current state
        X(:,i,k) = X(:,i,k-1);
        
        % For each neighbor
        for j = 1:num_robots
            if i ~= j
                % Check if communication happens (could add packet loss here)
                if rand() < 0.9  % 90% communication success rate
                    % Calculate time since last update from this robot
                    time_diff = last_comm_time(i,j);
                    
                    % Calculate weight based on time decay
                    weight = gamma * exp(-lambda * time_diff);
                    
                    % Get delayed state from other robot
                    delay = comm_delay(i,j);
                    if k - delay > 0
                        other_state = X(:,j,k-delay);
                    else
                        other_state = X(:,j,1);  % Use initial state if not enough history
                    end
                    
                    % Update state with weighted information
                    X(:,i,k) = X(:,i,k) + weight * (other_state - X(:,i,k-1));
                    
                    % Reset last communication time
                    last_comm_time(i,j) = 0;
                end
            end
        end
    end
end

% Calculate convergence metrics
convergence_error = zeros(num_iterations, num_robots);
for i = 1:num_robots
    for k = 1:num_iterations
        convergence_error(k,i) = norm(X(:,i,k) - true_state);
    end
end

% Calculate average consensus error
avg_consensus_error = mean(convergence_error, 2);

% Fit exponential decay model to verify exponential convergence
% Use data from iteration 5 onwards to avoid initial transients
start_fit = 5;
[fit_model, gof] = fit((start_fit:num_iterations)', avg_consensus_error(start_fit:end), 'a*exp(-b*x)+c', 'StartPoint', [avg_consensus_error(start_fit), 0.1, 0]);
convergence_rate = fit_model.b;

% Plot results
figure('Name', 'Time-Weighted Consensus Protocol Validation', 'Position', [100, 100, 1200, 800]);

% Plot state evolution for each dimension and robot
subplot(2, 2, 1);
for i = 1:num_robots
    for d = 1:state_dim
        plot(1:num_iterations, squeeze(X(d,i,:)), 'LineWidth', 1.5);
        hold on;
    end
end
for d = 1:state_dim
    plot(1:num_iterations, true_state(d) * ones(1, num_iterations), 'k--', 'LineWidth', 1);
end
xlabel('Iteration');
ylabel('State Value');
title('State Evolution');
grid on;

% Plot convergence error for each robot
subplot(2, 2, 2);
for i = 1:num_robots
    semilogy(1:num_iterations, convergence_error(:,i), 'LineWidth', 1.5);
    hold on;
end
semilogy(1:num_iterations, avg_consensus_error, 'k-', 'LineWidth', 2);
semilogy(start_fit:num_iterations, fit_model(start_fit:num_iterations), 'r--', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Error (log scale)');
title('Convergence Error');
legend([arrayfun(@(i) sprintf('Robot %d', i), 1:num_robots, 'UniformOutput', false), 'Average', 'Exponential Fit']);
grid on;

% Display convergence rate
text_str = sprintf('Fitted convergence rate: %.4f\nR-squared: %.4f\nTheoretical rate: %.4f', ...
                  convergence_rate, gof.rsquare, -log(1-2*gamma));
annotation('textbox', [0.58, 0.8, 0.3, 0.1], 'String', text_str, 'FitBoxToText', 'on', 'BackgroundColor', 'white');

%% Part 2: Failure Recovery Mechanism Testing

fprintf('\nRunning failure recovery mechanism testing...\n');

% Set random seed for reproducibility
rng(42);

% Create simulation environment
env = env_utils.createEnvironment(4, 4);
robots = robot_utils.createRobots(2, env);
num_tasks = 15;
tasks = task_utils.createTasks(num_tasks, env);
tasks = task_utils.addTaskDependencies(tasks);

% Parameters
params.epsilon = 0.05;        % Minimum bid increment
params.alpha = [1.0, 0.5, 2.0, 0.8, 0.3];  % Bid calculation weights
params.gamma = 0.5;           % Consensus weight factor
params.lambda = 0.1;          % Information decay rate
params.beta = [1.5, 1.0];     % Recovery auction weights
params.comm_delay = 0;        % Communication delay (in iterations)
params.packet_loss_prob = 0;  % Probability of packet loss
params.failure_time = 20;     % Time of robot failure 
params.failed_robot = 1;      % Robot 1 fails

% Initialize auction data
auction_data = auction_utils.initializeAuctionData(tasks, robots);

% Performance metrics
metrics = struct();
metrics.iterations = 0;
metrics.messages = 0;
metrics.convergence_history = [];
metrics.price_history = zeros(length(tasks), 200);  % Preallocate
metrics.assignment_history = zeros(length(tasks), 200);
metrics.completion_time = 0;
metrics.optimality_gap = 0;
metrics.recovery_time = 0;
metrics.failure_time = params.failure_time;

% Create figure for visualization
figure('Name', 'Failure Recovery Simulation', 'Position', [100, 100, 1200, 800]);

subplot(2, 3, [1, 4]);
env_utils.visualizeEnvironment(env, robots, tasks, auction_data);
title('Initial Environment');

% Main simulation loop
max_iterations = 100;
converged = false;

% Available tasks (initially only those with no prerequisites)
available_tasks = task_utils.findAvailableTasks(tasks, []);

% Record initial state - before failure
pre_failure_state = struct();

for iter = 1:max_iterations
    metrics.iterations = iter;
    
    % Record pre-failure state at the iteration before failure
    if iter == params.failure_time - 1
        pre_failure_state.assignment = auction_data.assignment;
        pre_failure_state.prices = auction_data.prices;
        pre_failure_state.available_tasks = available_tasks;
        pre_failure_state.robot_workloads = [robots.workload];
        pre_failure_state.iteration = iter;
        
        % Count tasks assigned to the robot that will fail
        pre_failure_state.tasks_to_reassign = sum(auction_data.assignment == params.failed_robot);
        fprintf('Robot %d will fail with %d tasks assigned to it\n', ...
                params.failed_robot, pre_failure_state.tasks_to_reassign);
    end
    
    % Check for robot failure
    if iter == params.failure_time
        fprintf('Robot %d has failed at iteration %d\n', params.failed_robot, iter);
        robots(params.failed_robot).failed = true;
        
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
    else
        metrics.convergence_history(iter) = NaN;
    end
    
    % Check if any new tasks become available due to completed prerequisites
    completed_tasks = find(auction_data.completion_status == 1);
    available_tasks = task_utils.findAvailableTasks(tasks, completed_tasks);
    
    % Check for recovery completion
    if iter > metrics.failure_time && metrics.recovery_time == 0
        % Check if all tasks previously assigned to the failed robot have been reassigned
        failed_robot_tasks = find(metrics.assignment_history(:, metrics.failure_time) == params.failed_robot);
        if all(metrics.assignment_history(failed_robot_tasks, iter) ~= params.failed_robot)
            metrics.recovery_time = iter - metrics.failure_time;
            fprintf('Recovery completed after %d iterations\n', metrics.recovery_time);
        end
    end
    
    % Check for convergence (no change in assignments for several iterations)
    if iter > 10 && all(metrics.convergence_history(iter-5:iter) == 0) && metrics.recovery_time > 0
        converged = true;
        fprintf('Auction algorithm converged after %d iterations\n', iter);
        break;
    end
    
    % Pause for visualization
    pause(0.05);
end

% Trim history matrices to actual size
metrics.price_history = metrics.price_history(:, 1:iter);
metrics.assignment_history = metrics.assignment_history(:, 1:iter);

% Calculate theoretical recovery time bound
tasks_reassigned = sum(pre_failure_state.assignment == params.failed_robot);
b_max = max(params.alpha);
epsilon = params.epsilon;
theoretical_recovery_bound = tasks_reassigned + round(b_max/epsilon);

% Plot recovery-specific results
subplot(2, 3, 2);
plot(metrics.convergence_history, 'LineWidth', 1.5);
hold on;
xline(params.failure_time, 'r--', 'LineWidth', 2);
if metrics.recovery_time > 0
    xline(params.failure_time + metrics.recovery_time, 'g--', 'LineWidth', 2);
end
xlabel('Iteration');
ylabel('Number of Assignment Changes');
title('Convergence Metric');
grid on;
legend('Changes', 'Failure', 'Recovery Complete', 'Location', 'best');

subplot(2, 3, 3);
bar([tasks_reassigned, metrics.recovery_time, theoretical_recovery_bound]);
set(gca, 'XTickLabel', {'Tasks to Reassign', 'Actual Recovery Time', 'Theoretical Bound'});
title('Recovery Performance');
grid on;

% Calculate workload before and after failure
workload_before = zeros(1, length(robots));
workload_after = zeros(1, length(robots));

for i = 1:length(robots)
    for j = 1:length(tasks)
        if metrics.assignment_history(j, metrics.failure_time-1) == i
            workload_before(i) = workload_before(i) + tasks(j).execution_time;
        end
        if metrics.assignment_history(j, end) == i
            workload_after(i) = workload_after(i) + tasks(j).execution_time;
        end
    end
end

subplot(2, 3, 5);
bar([workload_before; workload_after]');
set(gca, 'XTickLabel', arrayfun(@(i) sprintf('Robot %d', i), 1:length(robots), 'UniformOutput', false));
legend('Before Failure', 'After Recovery');
title('Workload Redistribution');
ylabel('Total Execution Time');
grid on;

% Calculate makespan before and after failure
makespan_before = max(workload_before);
makespan_after = max(workload_after(~[robots.failed]));
makespan_degradation = makespan_after - makespan_before;

subplot(2, 3, 6);
bar([makespan_before, makespan_after, makespan_degradation]);
set(gca, 'XTickLabel', {'Before Failure', 'After Recovery', 'Degradation'});
title('Makespan Impact');
ylabel('Makespan');
grid on;

% Display summary
fprintf('\n--- Recovery Performance Summary ---\n');
fprintf('Tasks assigned to failed robot: %d\n', tasks_reassigned);
fprintf('Recovery time: %d iterations\n', metrics.recovery_time);
fprintf('Theoretical recovery bound: O(|Tᶠ|) + O(bₘₐₓ/ε) ≈ %d\n', theoretical_recovery_bound);
fprintf('Recovery time / bound ratio: %.2f\n', metrics.recovery_time / theoretical_recovery_bound);
fprintf('Makespan before failure: %.2f\n', makespan_before);
fprintf('Makespan after recovery: %.2f\n', makespan_after);
fprintf('Makespan degradation: %.2f (%.2f%%)\n', makespan_degradation, makespan_degradation/makespan_before*100);

% Save results
if ~exist('../results', 'dir')
    mkdir('../results');
end
save('../results/consensus_recovery_results.mat', 'X', 'convergence_error', 'convergence_rate', 'metrics', 'pre_failure_state');

% Save figures
if ~exist('../figures/failure_recovery_plots', 'dir')
    mkdir('../figures/failure_recovery_plots');
end
saveas(figure(1), '../figures/failure_recovery_plots/consensus_validation.fig');
saveas(figure(1), '../figures/failure_recovery_plots/consensus_validation.png');
saveas(figure(2), '../figures/failure_recovery_plots/failure_recovery.fig');
saveas(figure(2), '../figures/failure_recovery_plots/failure_recovery.png');

fprintf('\nTesting complete. Results saved to consensus_recovery_results.mat\n');