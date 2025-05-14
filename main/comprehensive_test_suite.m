% fixed_comprehensive_test_suite.m
% A comprehensive test suite for validating the distributed auction algorithm
% implementation against theoretical properties and guarantees - WITH PATH FIXES

clear all;
close all;
clc;

% First run the path setup script to ensure all directories are properly added
run('setup_paths.m');

% Verify the path and utilities are loaded properly
fprintf('==============================================\n');
fprintf('COMPREHENSIVE VALIDATION TEST SUITE\n');
fprintf('==============================================\n\n');

% Load utility functions
try
    utils = utils_manager();
    fprintf('Successfully loaded utility functions.\n\n');
catch ME
    fprintf('ERROR: Failed to load utility functions: %s\n', ME.message);
    fprintf('Please run setup_paths.m manually and ensure all utility files exist.\n');
    return;
end

% Initialize test results tracking
test_results = struct();
test_results.total_tests = 0;
test_results.passed_tests = 0;
test_results.failed_tests = 0;
test_results.test_details = {};

%% 1. Convergence Time Complexity Tests
fprintf('1. TESTING CONVERGENCE TIME COMPLEXITY...\n');

% Theoretical bound: O(K² · bₘₐₓ/ε)
task_counts = [5, 10, 15, 20];
epsilons = [0.05, 0.1, 0.2];

% Results storage
convergence_iterations = zeros(length(task_counts), length(epsilons));
theoretical_bounds = zeros(length(task_counts), length(epsilons));
bound_ratios = zeros(length(task_counts), length(epsilons));

% Base parameters
base_params = struct();
base_params.env_width = 4;
base_params.env_height = 4;
base_params.num_robots = 2;
base_params.add_dependencies = false;
base_params.dependency_probability = 0.2;
base_params.use_detailed_robots = false;
base_params.time_step = 0.1;
base_params.simulation_duration = 100;
base_params.alpha = [1.0, 0.5, 2.0, 0.8, 0.3];
base_params.gamma = 0.5;
base_params.lambda = 0.1;
base_params.beta = [1.5, 1.0];
base_params.comm_delay = 0;
base_params.packet_loss_prob = 0;
base_params.failure_time = inf;
base_params.failed_robot = [];

try
    % Run tests for each task count and epsilon
    % Note: For brevity and quick testing, we'll only run one test combination
    i = 1;  % Just test with task_count = 5
    j = 1;  % Just test with epsilon = 0.05
    
    test_results.total_tests = test_results.total_tests + 1;
    
    fprintf('  - Testing K=%d, ε=%.2f: ', task_counts(i), epsilons(j));
    
    % Set test parameters
    params = base_params;
    params.num_tasks = task_counts(i);
    params.epsilon = epsilons(j);
    
    % Create environment for this test
    env = utils.env.createEnvironment(params.env_width, params.env_height);
    robots = utils.robot.createRobots(params.num_robots, env);
    tasks = utils.task.createTasks(params.num_tasks, env);
    
    % Set random seed for reproducibility
    rng(42);
    
    % Run algorithm
    [metrics, converged] = utils.auction.runAuctionSimulation(params, env, robots, tasks, false);
    
    % Calculate theoretical bound
    b_max = max(params.alpha);
    K = params.num_tasks;
    epsilon = params.epsilon;
    theoretical_bound = K^2 * b_max / epsilon;
    
    % Store results
    convergence_iterations(i, j) = metrics.iterations;
    theoretical_bounds(i, j) = theoretical_bound;
    bound_ratios(i, j) = metrics.iterations / theoretical_bound;
    
    % Check if bound is respected (with some margin for randomness)
    margin_factor = 1.2; % Allow 20% margin
    if converged && metrics.iterations <= margin_factor * theoretical_bound
        fprintf('PASSED - Iterations: %d, Bound: %.2f, Ratio: %.2f\n', ...
                metrics.iterations, theoretical_bound, bound_ratios(i, j));
        test_results.passed_tests = test_results.passed_tests + 1;
        test_results.test_details{end+1} = sprintf('Convergence Test (K=%d, ε=%.2f): PASSED', task_counts(i), epsilons(j));
    else
        fprintf('FAILED - Iterations: %d, Bound: %.2f, Ratio: %.2f\n', ...
                metrics.iterations, theoretical_bound, bound_ratios(i, j));
        test_results.failed_tests = test_results.failed_tests + 1;
        test_results.test_details{end+1} = sprintf('Convergence Test (K=%d, ε=%.2f): FAILED', task_counts(i), epsilons(j));
    end
    
    % Plot results
    figure('Name', 'Convergence Time Complexity Validation');
    subplot(1, 2, 1);
    bar(convergence_iterations(i, j));
    hold on;
    bar(theoretical_bounds(i, j), 'FaceColor', [0.8, 0.8, 0.8]);
    legend('Actual', 'Theoretical Bound');
    xlabel('Test Case');
    ylabel('Iterations to Converge');
    title('Convergence Time vs. Bound');
    grid on;
    
    subplot(1, 2, 2);
    bar(bound_ratios(i, j));
    xlabel('Test Case');
    ylabel('Ratio (Actual/Theoretical)');
    title('Bound Tightness');
    grid on;
    
    fprintf('Completed basic convergence test. Full test suite in comprehensive_test_suite.m\n');
    
catch ME
    fprintf('ERROR during convergence testing: %s\n', ME.message);
    fprintf('Line %d: %s\n', ME.stack(1).line, ME.stack(1).name);
    fprintf('Try running setup_paths.m first to ensure all paths are correct.\n');
end

%% Display short summary
fprintf('\n==============================================\n');
fprintf('TEST SUMMARY\n');
fprintf('==============================================\n');
fprintf('Tests Run: %d\n', test_results.total_tests);
fprintf('Tests Passed: %d\n', test_results.passed_tests);
fprintf('Tests Failed: %d\n', test_results.failed_tests);

fprintf('\nNote: This is a simplified version of the test suite to verify your setup.\n');
fprintf('The full test suite is available in comprehensive_test_suite.m\n');
fprintf('First run setup_paths.m to configure your environment properly.\n');