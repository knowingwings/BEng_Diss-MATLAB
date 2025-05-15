% Run all experiments for the enhanced auction algorithm
% This script runs all experiments and generates comprehensive results

% Add required paths
addpath('../common');
addpath('../utils');

% Load utility functions
enhanced_auction_utils = enhanced_auction_utils();
scheduler_utils = scheduler_utils();
consensus_utils = consensus_utils();

% Create results and figures directories
if ~exist('../results', 'dir')
    mkdir('../results');
end
if ~exist('../figures', 'dir')
    mkdir('../figures');
end

% Set random seed for reproducibility
rng(42);

fprintf('Starting all experiments for enhanced auction algorithm...\n\n');

%% Part 1: Basic convergence test
fprintf('Running basic convergence test...\n');
run_basic_convergence_test();

%% Part 2: Communication constraints test
fprintf('Running communication constraints test...\n');
run_communication_constraints_test();

%% Part 3: Failure recovery test
fprintf('Running failure recovery test...\n');
run_failure_recovery_test();

%% Part 4: Collaborative tasks test
fprintf('Running collaborative tasks test...\n');
run_collaborative_tasks_test();

%% Part 5: Parameter optimization
fprintf('Running parameter optimization...\n');
optimal_params = optimize_parameters();

%% Part 6: Full factorial experiment
fprintf('Running full factorial experiment...\n');
results = run_full_experiment();

fprintf('\nAll experiments completed!\n');

function run_basic_convergence_test()
    % Run basic convergence test with different epsilon values
    
    % Load utility functions
    enhanced_auction_utils = enhanced_auction_utils();
    robot_utils = robot_utils();
    task_utils = task_utils();
    env_utils = environment_utils();
    
    % Create environment and robots
    env = env_utils.createEnvironment(4, 4);
    robots = robot_utils.createRobots(2, env);
    
    % Create tasks
    num_tasks = 10;
    tasks = task_utils.createTasks(num_tasks, env);
    
    % Test different epsilon values
    epsilon_values = [0.01, 0.05, 0.1, 0.2];
    
    % Initialize results
    results = struct();
    results.epsilon = epsilon_values;
    results.iterations = zeros(length(epsilon_values), 1);
    results.optimality_gap = zeros(length(epsilon_values), 1);
    results.makespan = zeros(length(epsilon_values), 1);
    results.optimal_makespan = zeros(length(epsilon_values), 1);
    
    for i = 1:length(epsilon_values)
        % Set parameters
        params = struct();
        params.epsilon = epsilon_values(i);
        params.alpha = [0.8, 0.3, 1.0, 1.2, 0.2];
        params.gamma = 0.5;
        params.lambda = 0.1;
        params.beta = [2.0, 1.5, 1.0];
        params.failure_time = inf;
        params.failed_robot = [];
        
        % Run simulation
        [metrics, converged] = enhanced_auction_utils.runAuctionSimulation(params, env, robots, tasks, false);
        
        % Store results
        results.iterations(i) = metrics.iterations;
        results.optimality_gap(i) = metrics.optimality_gap;
        results.makespan(i) = metrics.makespan;
        results.optimal_makespan(i) = metrics.optimal_makespan;
        
        fprintf('  Epsilon = %.2f: Iterations = %d, Optimality Gap = %.4f, Makespan = %.2f\n', ...
                epsilon_values(i), metrics.iterations, metrics.optimality_gap, metrics.makespan);
    end
    
    % Create plots
    figure('Name', 'Basic Convergence Test', 'Position', [100, 100, 1000, 400]);
    
    subplot(1, 2, 1);
    plot(epsilon_values, results.iterations, 'o-', 'LineWidth', 2);
    title('Convergence Speed vs. Epsilon');
    xlabel('Epsilon (ε)');
    ylabel('Iterations to Converge');
    grid on;
    
    subplot(1, 2, 2);
    plot(epsilon_values, results.optimality_gap, 'o-', 'LineWidth', 2);
    hold on;
    plot(epsilon_values, 2*epsilon_values, '--r', 'LineWidth', 2);
    title('Optimality Gap vs. Epsilon');
    xlabel('Epsilon (ε)');
    ylabel('Optimality Gap');
    legend('Actual Gap', 'Theoretical Bound (2ε)');
    grid on;
    
    % Save figure
    saveas(gcf, '../figures/basic_convergence/convergence_test.png');
    
    % Save results
    save('../results/basic_convergence/convergence_results.mat', 'results');
end

function run_communication_constraints_test()
    % Run tests with different communication constraints
    
    % Load utility functions
    enhanced_auction_utils = enhanced_auction_utils();
    robot_utils = robot_utils();
    task_utils = task_utils();
    env_utils = environment_utils();
    
    % Create environment and robots
    env = env_utils.createEnvironment(4, 4);
    robots = robot_utils.createRobots(2, env);
    
    % Create tasks
    num_tasks = 10;
    tasks = task_utils.createTasks(num_tasks, env);
    tasks = task_utils.addTaskDependencies(tasks, 0.3);
    
    % Test different communication delays
    delays = [0, 50, 200, 500];
    delay_results = struct();
    delay_results.delays = delays;
    delay_results.iterations = zeros(length(delays), 1);
    delay_results.messages = zeros(length(delays), 1);
    delay_results.optimality_gap = zeros(length(delays), 1);
    
    fprintf('  Testing communication delays...\n');
    for i = 1:length(delays)
        % Set parameters
        params = struct();
        params.epsilon = 0.05;
        params.alpha = [0.8, 0.3, 1.0, 1.2, 0.2];
        params.gamma = 0.5;
        params.lambda = 0.1;
        params.beta = [2.0, 1.5, 1.0];
        params.comm_delay = delays(i);
        params.packet_loss_prob = 0;
        params.failure_time = inf;
        params.failed_robot = [];
        
        % Run simulation
        [metrics, converged] = enhanced_auction_utils.runAuctionSimulation(params, env, robots, tasks, false);
        
        % Store results
        delay_results.iterations(i) = metrics.iterations;
        delay_results.messages(i) = metrics.messages;
        delay_results.optimality_gap(i) = metrics.optimality_gap;
        
        fprintf('    Delay = %d ms: Iterations = %d, Messages = %d, Optimality Gap = %.4f\n', ...
                delays(i), metrics.iterations, metrics.messages, metrics.optimality_gap);
    end
    
    % Test different packet loss probabilities
    loss_probs = [0, 0.1, 0.3, 0.5];
    loss_results = struct();
    loss_results.probs = loss_probs;
    loss_results.iterations = zeros(length(loss_probs), 1);
    loss_results.messages = zeros(length(loss_probs), 1);
    loss_results.optimality_gap = zeros(length(loss_probs), 1);
    
    fprintf('  Testing packet loss probabilities...\n');
    for i = 1:length(loss_probs)
        % Set parameters
        params = struct();
        params.epsilon = 0.05;
        params.alpha = [0.8, 0.3, 1.0, 1.2, 0.2];
        params.gamma = 0.5;
        params.lambda = 0.1;
        params.beta = [2.0, 1.5, 1.0];
        params.comm_delay = 0;
        params.packet_loss_prob = loss_probs(i);
        params.failure_time = inf;
        params.failed_robot = [];
        
        % Run simulation
        [metrics, converged] = enhanced_auction_utils.runAuctionSimulation(params, env, robots, tasks, false);
        
        % Store results
        loss_results.iterations(i) = metrics.iterations;
        loss_results.messages(i) = metrics.messages;
        loss_results.optimality_gap(i) = metrics.optimality_gap;
        
        fprintf('    Packet Loss = %.1f: Iterations = %d, Messages = %d, Optimality Gap = %.4f\n', ...
                loss_probs(i), metrics.iterations, metrics.messages, metrics.optimality_gap);
    end
    
    % Create plots
    figure('Name', 'Communication Delay Test', 'Position', [100, 100, 800, 600]);
    
    subplot(2, 1, 1);
    plot(delays, delay_results.iterations, 'o-', 'LineWidth', 2);
    title('Iterations vs. Communication Delay');
    xlabel('Delay (ms)');
    ylabel('Iterations to Converge');
    grid on;
    
    subplot(2, 1, 2);
    plot(delays, delay_results.messages, 'o-', 'LineWidth', 2);
    title('Messages vs. Communication Delay');
    xlabel('Delay (ms)');
    ylabel('Number of Messages');
    grid on;
    
    saveas(gcf, '../figures/communication_constraints/delay_test.png');
    
    figure('Name', 'Packet Loss Test', 'Position', [100, 100, 800, 600]);
    
    subplot(2, 1, 1);
    plot(loss_probs, loss_results.iterations, 'o-', 'LineWidth', 2);
    title('Iterations vs. Packet Loss Probability');
    xlabel('Packet Loss Probability');
    ylabel('Iterations to Converge');
    grid on;
    
    subplot(2, 1, 2);
    plot(loss_probs, loss_results.messages, 'o-', 'LineWidth', 2);
    title('Messages vs. Packet Loss Probability');
    xlabel('Packet Loss Probability');
    ylabel('Number of Messages');
    grid on;
    
    saveas(gcf, '../figures/communication_constraints/packet_loss_test.png');
    
    % Save results
    save('../results/communication_constraints/delay_results.mat', 'delay_results');
    save('../results/communication_constraints/loss_results.mat', 'loss_results');
end

function run_failure_recovery_test()
    % Run failure recovery tests
    
    % Load utility functions
    enhanced_auction_utils = enhanced_auction_utils();
    robot_utils = robot_utils();
    task_utils = task_utils();
    env_utils = environment_utils();
    
    % Create environment and robots
    env = env_utils.createEnvironment(4, 4);
    robots = robot_utils.createRobots(2, env);
    
    % Test different task counts
    task_counts = [5, 10, 15, 20];
    
    % Initialize results
    results = struct();
    results.task_counts = task_counts;
    results.recovery_time = zeros(length(task_counts), 1);
    results.failed_task_count = zeros(length(task_counts), 1);
    results.makespan_before = zeros(length(task_counts), 1);
    results.makespan_after = zeros(length(task_counts), 1);
    results.theoretical_bound = zeros(length(task_counts), 1);
    
    for i = 1:length(task_counts)
        % Create tasks
        num_tasks = task_counts(i);
        tasks = task_utils.createTasks(num_tasks, env);
        tasks = task_utils.addTaskDependencies(tasks, min(0.3, 3/num_tasks));
        
        % Set parameters
        params = struct();
        params.epsilon = 0.05;
        params.alpha = [0.8, 0.3, 1.0, 1.2, 0.2];
        params.gamma = 0.5;
        params.lambda = 0.1;
        params.beta = [2.0, 1.5, 1.0];
        params.comm_delay = 0;
        params.packet_loss_prob = 0;
        params.failure_time = 20;  % Fixed failure time
        params.failed_robot = 1;   % Robot 1 fails
        
        % Run simulation
        [metrics, converged] = enhanced_auction_utils.runAuctionSimulation(params, env, robots, tasks, false);
        
        % Store results
        results.recovery_time(i) = metrics.recovery_time;
        results.failed_task_count(i) = metrics.failed_task_count;
        results.makespan_before(i) = metrics.makespan_before_failure;
        results.makespan_after(i) = metrics.makespan;
        
        % Calculate theoretical bound
        b_max = max(params.alpha);
        epsilon = params.epsilon;
        results.theoretical_bound(i) = metrics.failed_task_count + round(b_max/epsilon);
        
        fprintf('  Tasks = %d: Failed Tasks = %d, Recovery Time = %d, Theoretical Bound = %d\n', ...
                task_counts(i), metrics.failed_task_count, metrics.recovery_time, results.theoretical_bound(i));
    end
    
    % Create plots
    figure('Name', 'Failure Recovery Test', 'Position', [100, 100, 1000, 800]);
    
    subplot(2, 2, 1);
    plot(task_counts, results.failed_task_count, 'o-', 'LineWidth', 2);
    title('Failed Tasks vs. Task Count');
    xlabel('Number of Tasks');
    ylabel('Number of Failed Tasks');
    grid on;
    
    subplot(2, 2, 2);
    plot(task_counts, results.recovery_time, 'o-', 'LineWidth', 2);
    hold on;
    plot(task_counts, results.theoretical_bound, '--r', 'LineWidth', 2);
    title('Recovery Time vs. Task Count');
    xlabel('Number of Tasks');
    ylabel('Recovery Time (iterations)');
    legend('Actual Recovery Time', 'Theoretical Bound');
    grid on;
    
    subplot(2, 2, 3);
    plot(task_counts, results.makespan_before, 'o-', 'LineWidth', 2);
    hold on;
    plot(task_counts, results.makespan_after, 'o-', 'LineWidth', 2);
    title('Makespan Before and After Failure');
    xlabel('Number of Tasks');
    ylabel('Makespan');
    legend('Before Failure', 'After Recovery');
    grid on;
    
    subplot(2, 2, 4);
    degradation = results.makespan_after - results.makespan_before;
    degradation_pct = 100 * degradation ./ results.makespan_before;
    plot(task_counts, degradation_pct, 'o-', 'LineWidth', 2);
    title('Makespan Degradation (%)');
    xlabel('Number of Tasks');
    ylabel('Degradation (%)');
    grid on;
    
    saveas(gcf, '../figures/failure_recovery/recovery_test.png');
    
    % Save results
    save('../results/failure_recovery/recovery_results.mat', 'results');
end

function run_collaborative_tasks_test()
    % Run tests with collaborative tasks
    
    % Load utility functions
    enhanced_auction_utils = enhanced_auction_utils();
    robot_utils = robot_utils();
    task_utils = task_utils();
    env_utils = environment_utils();
    scheduler_utils = scheduler_utils();
    
    % Create environment and robots
    env = env_utils.createEnvironment(4, 4);
    robots = robot_utils.createRobots(2, env);
    
    % Create tasks
    num_tasks = 12;
    tasks = task_utils.createTasks(num_tasks, env);
    tasks = task_utils.addTaskDependencies(tasks, 0.3);
    
    % Test different collaborative task ratios
    collab_ratios = [0, 0.2, 0.4, 0.6];
    
    % Initialize results
    results = struct();
    results.collab_ratios = collab_ratios;
    results.iterations = zeros(length(collab_ratios), 1);
    results.makespan = zeros(length(collab_ratios), 1);
    results.optimality_gap = zeros(length(collab_ratios), 1);
    results.messages = zeros(length(collab_ratios), 1);
    
    for i = 1:length(collab_ratios)
        % Reset collaborative flags
        for j = 1:num_tasks
            tasks(j).collaborative = false;
        end
        
        % Set collaborative tasks
        num_collaborative = round(num_tasks * collab_ratios(i));
        collab_indices = randperm(num_tasks, num_collaborative);
        for j = collab_indices
            tasks(j).collaborative = true;
        end
        
        % Set parameters
        params = struct();
        params.epsilon = 0.05;
        params.alpha = [0.8, 0.3, 1.0, 1.2, 0.2];
        params.gamma = 0.5;
        params.lambda = 0.1;
        params.beta = [2.0, 1.5, 1.0];
        params.comm_delay = 0;
        params.packet_loss_prob = 0;
        params.failure_time = inf;
        params.failed_robot = [];
        params.sync_timeout = 5;
        params.max_sync_distance = 0.5;
        
        % Run simulation
        [metrics, converged] = enhanced_auction_utils.runAuctionSimulation(params, env, robots, tasks, false);
        
        % Store results
        results.iterations(i) = metrics.iterations;
        results.makespan(i) = metrics.makespan;
        results.optimality_gap(i) = metrics.optimality_gap;
        results.messages(i) = metrics.messages;
        
        fprintf('  Collaborative Ratio = %.1f: Iterations = %d, Makespan = %.2f, Optimality Gap = %.4f\n', ...
                collab_ratios(i), metrics.iterations, metrics.makespan, metrics.optimality_gap);
    end
    
    % Create plots
    figure('Name', 'Collaborative Tasks Test', 'Position', [100, 100, 1000, 400]);
    
    subplot(1, 3, 1);
    plot(collab_ratios, results.iterations, 'o-', 'LineWidth', 2);
    title('Iterations vs. Collaborative Ratio');
    xlabel('Collaborative Task Ratio');
    ylabel('Iterations to Converge');
    grid on;
    
    subplot(1, 3, 2);
    plot(collab_ratios, results.makespan, 'o-', 'LineWidth', 2);
    title('Makespan vs. Collaborative Ratio');
    xlabel('Collaborative Task Ratio');
    ylabel('Makespan');
    grid on;
    
    subplot(1, 3, 3);
    plot(collab_ratios, results.messages, 'o-', 'LineWidth', 2);
    title('Messages vs. Collaborative Ratio');
    xlabel('Collaborative Task Ratio');
    ylabel('Number of Messages');
    grid on;
    
    saveas(gcf, '../figures/collaborative_tasks/collaborative_test.png');
    
    % Create visualization of a schedule with collaborative tasks
    % Run simulation with 40% collaborative tasks
    num_collaborative = round(num_tasks * 0.4);
    collab_indices = randperm(num_tasks, num_collaborative);
    for j = 1:num_tasks
        tasks(j).collaborative = false;
    end
    for j = collab_indices
        tasks(j).collaborative = true;
    end
    
    params.epsilon = 0.05;
    [metrics, converged] = enhanced_auction_utils.runAuctionSimulation(params, env, robots, tasks, false);
    
    % Generate schedule
    schedule = scheduler_utils.generateSchedule(metrics.assignment_history(:, end), tasks, robots);
    
    % Visualize schedule
    figure('Name', 'Schedule with Collaborative Tasks', 'Position', [100, 100, 1200, 600]);
    scheduler_utils.visualizeSchedule(schedule, tasks, robots);
    
    saveas(gcf, '../figures/collaborative_tasks/collaborative_schedule.png');
    
    % Save results
    save('../results/collaborative_tasks/collaborative_results.mat', 'results');
end