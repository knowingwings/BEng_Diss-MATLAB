% Demo script for enhanced MATLAB implementation
clear all;
close all;
clc;

% Add all directories to MATLAB path
current_dir = pwd;
common_dir = fullfile(current_dir, '..', 'common');
addpath(common_dir);

fprintf('Loading utilities...\n');

% Use the utils manager to avoid naming conflicts
utils = utils_manager();

% Set simulation parameters
params = struct();
params.env_width = 4;
params.env_height = 4;
params.num_robots = 2;
params.num_tasks = 10;
params.add_dependencies = true;
params.dependency_probability = 0.3;
params.use_detailed_robots = true;
params.time_step = 0.1;
params.simulation_duration = 100;

% Auction algorithm parameters
params.epsilon = 0.05;
params.alpha = [1.0, 0.5, 2.0, 0.8, 0.3];
params.gamma = 0.5;
params.lambda = 0.1;
params.beta = [1.5, 1.0];
params.comm_delay = 0;
params.packet_loss_prob = 0;

% Failure scenario
params.failure_time = 30;
params.failed_robot = 1;

fprintf('Initializing simulation...\n');

% Initialize simulation
sim_data = utils.sim.initializeSimulation(params);

% Run simulation
fprintf('Starting enhanced simulation...\n');

% Create a real-time visualization
figure_handle = figure('Name', 'Enhanced Distributed Auction Simulation', 'Position', [100, 100, 1200, 800]);
sim_data.figure_handle = figure_handle;

% Simulation loop
while strcmp(sim_data.state, 'running')
    % Step simulation
    [sim_data, status] = utils.sim.stepSimulation(sim_data);
    
    % Visualize every 5 steps to improve performance
    if mod(sim_data.iteration, 5) == 0
        utils.sim.visualizeSimulation(sim_data, true);  % Use 3D visualization
        pause(0.01);  % Small pause for visualization
    end
end

% Final visualization
utils.sim.visualizeSimulation(sim_data, true);

% Record final metrics
metrics = utils.sim.recordMetrics(sim_data);

% Display summary
fprintf('\n--- Simulation Results ---\n');
fprintf('Simulation time: %.2f seconds\n', sim_data.clock.time);
fprintf('Iterations: %d\n', metrics.iterations);
fprintf('Messages: %d (%.2f per task)\n', metrics.messages, metrics.messages_per_task);
fprintf('Makespan: %.2f (Optimal: %.2f, Gap: %.2f)\n', metrics.makespan, metrics.optimal_makespan, metrics.optimality_gap);

if metrics.failure_time > 0
    fprintf('Failure time: %.2f seconds\n', metrics.failure_time);
    fprintf('Failed robot: %d with %d assigned tasks\n', params.failed_robot, metrics.failed_task_count);
    
    if metrics.recovery_time > 0
        fprintf('Recovery time: %.2f seconds\n', metrics.recovery_time);
        fprintf('Theoretical recovery bound: %.2f seconds\n', metrics.theoretical_recovery_bound * params.time_step);
        fprintf('Recovery efficiency: %.2f%%\n', 100 * metrics.theoretical_recovery_bound * params.time_step / metrics.recovery_time);
    else
        fprintf('Recovery not completed\n');
    end
end

% Export results
utils.sim.exportResults(sim_data, 'enhanced_simulation_results.mat');