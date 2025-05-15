% Initialization script for the distributed auction algorithm
% This script adds all the necessary paths

% Add common utilities
addpath('./common');

% Add main scripts
addpath('./main');

% Add experiments
addpath('./experiments');

% Add tests
addpath('./tests');

% Add optimization tools
addpath('./optimization');

% Add utility scripts
addpath('./utils');

% Create results and figures directories if they don't exist
if ~exist('./results', 'dir')
    mkdir('./results');
    mkdir('./results/basic_convergence');
    mkdir('./results/communication_constraints');
    mkdir('./results/failure_recovery');
    mkdir('./results/task_dependencies');
    mkdir('./results/parameter_optimization');
    mkdir('./results/full_experiment');
end

if ~exist('./figures', 'dir')
    mkdir('./figures');
    mkdir('./figures/basic_convergence');
    mkdir('./figures/communication_constraints');
    mkdir('./figures/failure_recovery');
    mkdir('./figures/task_dependencies');
    mkdir('./figures/parameter_optimization');
    mkdir('./figures/full_experiment');
end

% Display welcome message
fprintf('Distributed Auction Algorithm Initialization Complete\n');
fprintf('To run the enhanced auction algorithm: enhanced_auction_algorithm\n');
fprintf('To run all experiments: run_all_experiments\n');
fprintf('To run the full factorial experiment: run_full_experiment\n');
fprintf('To optimize parameters: optimize_parameters\n');