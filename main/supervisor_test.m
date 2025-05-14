% test_fixed_utils.m - Test the fixed utility functions
clear all;
close all;
clc;

% Add path properly
current_dir = pwd;
common_dir = fullfile(current_dir, '..', 'common');
addpath(common_dir);

% Get utility functions
fprintf('Loading utility functions...\n');
env_utils = environment_utils();
robot_utils = robot_utils();
task_utils = task_utils();
auction_utils = auction_utils();
sim_utils = simulation_supervisor();

% Try creating an environment
fprintf('Creating environment...\n');
env = env_utils.createEnvironment(4, 4);
fprintf('Environment created successfully!\n');

% Try creating robots
fprintf('Creating robots...\n');
robots = robot_utils.createRobots(2, env);
fprintf('Robots created successfully!\n');

% Try creating tasks
fprintf('Creating tasks...\n');
tasks = task_utils.createTasks(5, env);
fprintf('Tasks created successfully!\n');

% Try initializing auction data
fprintf('Initializing auction data...\n');
auction_data = auction_utils.initializeAuctionData(tasks, robots);
fprintf('Auction data initialized successfully!\n');

% Try visualizing the environment
fprintf('Visualizing environment...\n');
figure;
env_utils.visualizeEnvironment(env, robots, tasks, auction_data);
title('Test Visualization');
fprintf('Visualization created successfully!\n');

% Try initializing simulation
fprintf('Initializing simulation...\n');
params = struct(...
    'env_width', 4, ...
    'env_height', 4, ...
    'num_robots', 2, ...
    'num_tasks', 5, ...
    'add_dependencies', false, ...
    'use_detailed_robots', false, ...
    'time_step', 0.1, ...
    'simulation_duration', 10, ...
    'epsilon', 0.05, ...
    'alpha', [1.0, 0.5, 2.0, 0.8, 0.3], ...
    'comm_delay', 0, ...
    'packet_loss_prob', 0 ...
);

sim_data = sim_utils.initializeSimulation(params);
fprintf('Simulation initialized successfully!\n');

% Run a few simulation steps
fprintf('Running simulation steps...\n');
for i = 1:5
    fprintf('Step %d...\n', i);
    [sim_data, status] = sim_utils.stepSimulation(sim_data);
    fprintf('Status: %s\n', status);
end

% Visualize simulation
fprintf('Visualizing simulation...\n');
sim_utils.visualizeSimulation(sim_data);
fprintf('Visualization successful!\n');

% Record metrics
fprintf('Recording metrics...\n');
metrics = sim_utils.recordMetrics(sim_data);
fprintf('Metrics recorded successfully!\n');

% Print metrics
fprintf('\nSimulation metrics:\n');
fprintf('Iterations: %d\n', metrics.iterations);
fprintf('Messages: %d\n', metrics.messages);
fprintf('Messages per task: %.2f\n', metrics.messages_per_task);

% Export results to a file
fprintf('Exporting results...\n');
sim_utils.exportResults(sim_data, 'test_simulation_results.mat');

fprintf('\nAll tests passed!\n');