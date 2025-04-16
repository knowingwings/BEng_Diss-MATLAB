%% Parameter Sensitivity Analysis Script
% This script performs sensitivity analysis on key parameters of the
% distributed auction algorithm to identify optimal settings.
%
% Tom Le Huray - Mechatronics Engineering Final Year Project
% University of Gloucestershire
% March 2025

clear all;
close all;
clc;

% Add all subdirectories to path
addpath(genpath('../src'));
addpath(genpath('../data'));
addpath(genpath('../models'));

% Load scenario
if exist('../data/scenarios/assembly_scenario.mat', 'file')
    load('../data/scenarios/assembly_scenario.mat');
    disp('Loaded assembly scenario from file.');
else
    disp('Error: No scenario file found. Please run run_verification.m first.');
    return;
end

%% Define parameter ranges for sensitivity analysis
parameter_ranges = struct();

% Auction algorithm parameters
parameter_ranges.epsilon = [0.01, 0.05, 0.1, 0.2, 0.5];  % Minimum bid increment

% Bid calculation weights
parameter_ranges.alpha1 = [5, 7.5, 10, 12.5, 15];        % Distance weight
parameter_ranges.alpha3 = [4, 6, 8, 10, 12];             % Capability match weight

% Consensus parameters
parameter_ranges.gamma = [0.3, 0.4, 0.5, 0.6, 0.7];      % Consensus weight
parameter_ranges.lambda = [0.05, 0.1, 0.2, 0.3, 0.5];    % Time decay parameter

% Communication parameters
parameter_ranges.comm_delay = [0.05, 0.1, 0.2, 0.3, 0.5]; % Communication delay
parameter_ranges.packet_loss_prob = [0, 0.1, 0.2, 0.3, 0.5]; % Packet loss probability

% Define base parameters (default values)
base_parameters = struct();
base_parameters.epsilon = 0.1;
base_parameters.alpha1 = 10;
base_parameters.alpha3 = 8;
base_parameters.gamma = 0.5;
base_parameters.lambda = 0.1;
base_parameters.comm_delay = 0.2;
base_parameters.packet_loss_prob = 0.1;
base_parameters.execution_speed = 0.1;
base_parameters.movement_speed = 0.5;

% Function to run simulation with given parameters
simulation_func = @run_simulation_with_parameters;

% Number of samples for each parameter combination
num_samples = 5;  % For example, adjust based on your time constraints

% Run parameter sensitivity analysis
disp('Starting parameter sensitivity analysis...');
disp(['This will test ', num2str(length(fieldnames(parameter_ranges))), ' parameters with ', ...
      num2str(num_samples), ' samples each...']);
disp('This may take a while. Progress will be displayed...');

sensitivity_results = parameter_sensitivity_analysis(parameter_ranges, base_parameters, simulation_func, num_samples);

% Save results
if ~exist('../data/results', 'dir')
    mkdir('../data/results');
end
save('../data/results/sensitivity_analysis_results.mat', 'sensitivity_results', 'parameter_ranges', 'base_parameters');

% Display summary
disp('Parameter Sensitivity Analysis Complete');
disp('Summary of Results:');

% Summarise parameter impacts
disp('Parameter impact ranking (most to least influential):');
param_names = fieldnames(sensitivity_results.sensitivity);
impact_values = zeros(length(param_names), 1);

for i = 1:length(param_names)
    impact_values(i) = sensitivity_results.sensitivity.(param_names{i});
end

[sorted_values, idx] = sort(impact_values, 'descend');
sorted_names = param_names(idx);

for i = 1:length(sorted_names)
    disp(['  ', num2str(i), '. ', sorted_names{i}, ' - Sensitivity: ', num2str(sorted_values(i), '%.3f')]);
end

% Identify optimal parameter values
disp('Recommended parameter values based on analysis:');
for p = 1:length(param_names)
    param = param_names{p};
    values = parameter_ranges.(param);
    
    % Get makespan metric for this parameter
    makespan_data = sensitivity_results.metrics.makespan{p};
    mean_makespan = mean(makespan_data, 2);
    
    % Find best value (minimum makespan)
    [~, best_idx] = min(mean_makespan);
    best_value = values(best_idx);
    
    disp(['  ', param, ' = ', num2str(best_value), ' (Base value: ', ...
          num2str(base_parameters.(param)), ')']);
end

% Display path to results
disp(['Detailed results saved to: ../data/results/sensitivity_analysis_results.mat']);
disp('Visualisations have been generated.');

%% Helper function to run simulation with parameters
function metrics = run_simulation_with_parameters(parameters)
    % Set simulation parameters
    sim_parameters = parameters;
    
    % Set random seed for reproducibility
    rng(42);
    
    % Load the scenario (it should be in the workspace)
    scenario = evalin('base', 'scenario');
    
    % Create temporary simulation for this parameter set
    model_name = 'temp_sensitivity_model';
    copyfile('../models/DistributedAuctionVerification.slx', [model_name, '.slx']);
    
    % Set model parameters
    assignin('base', 'sim_parameters', sim_parameters);
    assignin('base', 'robot_states', scenario.robot_states);
    assignin('base', 'tasks', scenario.tasks);
    assignin('base', 'execution_times', scenario.execution_times);
    
    % Initialise other variables
    num_tasks = size(scenario.tasks, 1);
    task_assignments = zeros(num_tasks, 1);
    task_prices = zeros(num_tasks, 1);
    task_progress = zeros(num_tasks, 1);
    task_start_times = zeros(num_tasks, 1);
    task_finish_times = zeros(num_tasks, 1);
    
    assignin('base', 'task_assignments', task_assignments);
    assignin('base', 'task_prices', task_prices);
    assignin('base', 'task_progress', task_progress);
    assignin('base', 'task_start_times', task_start_times);
    assignin('base', 'task_finish_times', task_finish_times);
    
    % Run simulation
    try
        sim_out = sim(model_name, 'StopTime', '60');  % Shorter time for sensitivity analysis
        
        % Extract metrics
        if isfield(sim_out, 'metrics')
            metrics = sim_out.metrics.Data(end);
        else
            % Create default metrics if simulation didn't produce valid metrics
            metrics = create_default_metrics();
        end
    catch ME
        warning(['Simulation failed: ', ME.message]);
        metrics = create_default_metrics();
    end
    
    % Cleanup
    delete([model_name, '.slx']);
end

function metrics = create_default_metrics()
    % Create default metrics structure for failed simulations
    metrics = struct();
    metrics.makespan = Inf;
    metrics.robot_utilistion = [0; 0];
    metrics.load_balance = 0;
    metrics.total_messages = 0;
    metrics.messages_per_task = 0;
    metrics.critical_path_length = 0;
    metrics.critical_path_utilistion = 0;
    metrics.avg_waiting_time = Inf;
end