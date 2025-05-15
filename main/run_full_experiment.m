function results = run_full_experiment()
    % RUN_FULL_EXPERIMENT - Implements the full experimental design from methodology
    % This script runs a full factorial design of experiments with the following
    % control variables:
    %   - K: Number of tasks [4, 8, 16, 32]
    %   - τᵢⱼ: Communication delay [0, 50, 200, 500] ms
    %   - pₗᵒₛₛ: Packet loss probability [0, 0.1, 0.3, 0.5]
    %   - ε: Minimum bid increment [0.01, 0.05, 0.2, 0.5]
    %
    % The primary response variables include:
    %   - Makespan (completion time)
    %   - Communication overhead (message count) 
    %   - Optimality gap compared to centralised solutions
    %   - Recovery time after failure
    
    % Add required paths
    addpath('../common');
    addpath('../utils');
    
    % Load utility functions
    enhanced_auction_utils = enhanced_auction_utils();
    robot_utils = robot_utils();
    task_utils = task_utils();
    env_utils = environment_utils();
    
    % Define experimental factors
    task_counts = [4, 8, 16, 32];
    comm_delays = [0, 50, 200, 500];
    packet_loss_probs = [0, 0.1, 0.3, 0.5];
    epsilon_values = [0.01, 0.05, 0.2, 0.5];
    
    % Number of repetitions for statistical significance
    num_repetitions = 15;
    
    % Create results structure
    results = struct();
    results.task_counts = task_counts;
    results.comm_delays = comm_delays;
    results.packet_loss_probs = packet_loss_probs;
    results.epsilon_values = epsilon_values;
    results.num_repetitions = num_repetitions;
    
    % Initialize results arrays
    num_task_levels = length(task_counts);
    num_delay_levels = length(comm_delays);
    num_loss_levels = length(packet_loss_probs);
    num_epsilon_levels = length(epsilon_values);
    
    % Results arrays for each response variable
    results.makespan = zeros(num_task_levels, num_delay_levels, num_loss_levels, num_epsilon_levels, num_repetitions);
    results.messages = zeros(num_task_levels, num_delay_levels, num_loss_levels, num_epsilon_levels, num_repetitions);
    results.optimality_gap = zeros(num_task_levels, num_delay_levels, num_loss_levels, num_epsilon_levels, num_repetitions);
    results.recovery_time = zeros(num_task_levels, num_delay_levels, num_loss_levels, num_epsilon_levels, num_repetitions);
    results.iterations = zeros(num_task_levels, num_delay_levels, num_loss_levels, num_epsilon_levels, num_repetitions);
    results.converged = false(num_task_levels, num_delay_levels, num_loss_levels, num_epsilon_levels, num_repetitions);
    
    % Create environment
    env = env_utils.createEnvironment(4, 4);
    
    % Counter for progress tracking
    total_experiments = num_task_levels * num_delay_levels * num_loss_levels * num_epsilon_levels * num_repetitions;
    current_experiment = 0;
    
    % Timer for estimating completion time
    start_time = tic;
    
    % Run full factorial experiment
    fprintf('Starting full factorial experiment with %d total runs...\n', total_experiments);
    
    for t_idx = 1:num_task_levels
        K = task_counts(t_idx);
        
        for d_idx = 1:num_delay_levels
            delay = comm_delays(d_idx);
            
            for p_idx = 1:num_loss_levels
                p_loss = packet_loss_probs(p_idx);
                
                for e_idx = 1:num_epsilon_levels
                    epsilon = epsilon_values(e_idx);
                    
                    % Set base parameters
                    params = struct();
                    params.epsilon = epsilon;
                    params.alpha = [0.8, 0.3, 1.0, 1.2, 0.2];  % Bid calculation weights
                    params.gamma = 0.5;                         % Consensus weight factor
                    params.lambda = 0.1;                        % Information decay rate
                    params.beta = [2.0, 1.5, 1.0];              % Recovery auction weights
                    params.comm_delay = delay;                  % Communication delay
                    params.packet_loss_prob = p_loss;           % Packet loss probability
                    params.heartbeat_interval = 3;              % Heartbeat signal interval
                    params.missed_heartbeat_threshold = 3;      % Number of missed heartbeats for failure detection
                    params.min_progress_rate = 0.05;            % Minimum progress rate
                    params.sync_timeout = 5;                    % Timeout for synchronization
                    params.max_sync_distance = 0.5;             % Maximum distance for synchronization
                    
                    % For each repetition
                    for rep = 1:num_repetitions
                        % Update progress
                        current_experiment = current_experiment + 1;
                        elapsed_time = toc(start_time);
                        estimated_total_time = elapsed_time * (total_experiments / current_experiment);
                        estimated_remaining_time = estimated_total_time - elapsed_time;
                        
                        fprintf('Running experiment %d/%d (%.1f%%) - ETA: %.1f minutes\n', ...
                                current_experiment, total_experiments, ...
                                100 * current_experiment / total_experiments, ...
                                estimated_remaining_time / 60);
                        
                        % Set random seed for reproducibility
                        rng(1000 + rep);  % Different seed for each repetition
                        
                        % Create robots
                        robots = robot_utils.createRobots(2, env);
                        
                        % Create tasks
                        tasks = task_utils.createTasks(K, env);
                        
                        % For larger task sets, reduce dependency density to avoid deadlock
                        if K <= 10
                            dependency_prob = 0.3;
                        else
                            dependency_prob = min(0.3, 3.0 / K);
                        end
                        
                        tasks = task_utils.addTaskDependencies(tasks, dependency_prob);
                        
                        % Add collaborative tasks (approximately 20% of tasks)
                        num_collaborative = max(1, round(K * 0.2));
                        collab_indices = randperm(K, min(num_collaborative, K));
                        for i = collab_indices
                            tasks(i).collaborative = true;
                        end
                        
                        % Add failure scenario for recovery tests
                        % Only test recovery on a subset of runs to save time
                        if mod(rep, 3) == 0
                            params.failure_time = 20;
                            params.failed_robot = 1;
                        else
                            params.failure_time = Inf;
                            params.failed_robot = [];
                        end
                        
                        % Run simulation
                        [metrics, converged] = enhanced_auction_utils.runAuctionSimulation(params, env, robots, tasks, false);
                        
                        % Store results
                        results.makespan(t_idx, d_idx, p_idx, e_idx, rep) = metrics.makespan;
                        results.messages(t_idx, d_idx, p_idx, e_idx, rep) = metrics.messages;
                        results.optimality_gap(t_idx, d_idx, p_idx, e_idx, rep) = metrics.optimality_gap;
                        results.iterations(t_idx, d_idx, p_idx, e_idx, rep) = metrics.iterations;
                        results.converged(t_idx, d_idx, p_idx, e_idx, rep) = converged;
                        
                        % Store recovery time if applicable
                        if params.failure_time < Inf && metrics.recovery_time > 0
                            results.recovery_time(t_idx, d_idx, p_idx, e_idx, rep) = metrics.recovery_time;
                        end
                    end
                end
            end
        end
    end
    
    % Calculate summary statistics
    results = calculate_statistics(results);
    
    % Save results
    save_path = '../results/full_experiment/experiment_results.mat';
    mkdir_if_not_exists(fileparts(save_path));
    save(save_path, 'results');
    
    % Generate plots
    generate_experiment_plots(results);
    
    fprintf('Full experiment completed and results saved.\n');
    
    return;
end

function results = calculate_statistics(results)
    % Calculate mean, std, and confidence intervals for each metric
    
    % For each response variable
    variables = {'makespan', 'messages', 'optimality_gap', 'iterations', 'recovery_time'};
    
    for v = 1:length(variables)
        var_name = variables{v};
        if ~isfield(results, var_name)
            continue;
        end
        
        % Calculate mean across repetitions
        mean_var = mean(results.(var_name), 5, 'omitnan');
        results.([var_name '_mean']) = mean_var;
        
        % Calculate standard deviation
        std_var = std(results.(var_name), 0, 5, 'omitnan');
        results.([var_name '_std']) = std_var;
        
        % Calculate 95% confidence intervals
        [~, ~, ci_var] = ttest(results.(var_name), [], 'Alpha', 0.05, 'Dim', 5);
        results.([var_name '_ci']) = ci_var;
    end
    
    % Calculate convergence rate
    results.convergence_rate = mean(results.converged, 5);
    
    return;
end

function generate_experiment_plots(results)
    % Generate plots for the experiment results
    
    % Create output directory
    plot_dir = '../figures/full_experiment';
    mkdir_if_not_exists(plot_dir);
    
    % Plot 1: Effect of task count on makespan
    figure('Name', 'Effect of Task Count on Makespan', 'Position', [100, 100, 1000, 600]);
    hold on;
    
    % Average across other factors
    makespan_vs_tasks = squeeze(mean(mean(mean(results.makespan_mean, 4), 3), 2));
    makespan_std_vs_tasks = squeeze(mean(mean(mean(results.makespan_std, 4), 3), 2));
    
    errorbar(results.task_counts, makespan_vs_tasks, makespan_std_vs_tasks, 'o-', 'LineWidth', 2);
    
    xlabel('Number of Tasks (K)');
    ylabel('Makespan');
    title('Effect of Task Count on Makespan');
    grid on;
    
    saveas(gcf, fullfile(plot_dir, 'task_count_vs_makespan.png'));
    
    % Plot 2: Effect of communication delay on convergence
    figure('Name', 'Effect of Communication Delay on Convergence', 'Position', [100, 100, 1000, 600]);
    hold on;
    
    % Average across other factors
    iterations_vs_delay = squeeze(mean(mean(mean(results.iterations_mean, 4), 3), 1));
    iterations_std_vs_delay = squeeze(mean(mean(mean(results.iterations_std, 4), 3), 1));
    
    errorbar(results.comm_delays, iterations_vs_delay, iterations_std_vs_delay, 'o-', 'LineWidth', 2);
    
    xlabel('Communication Delay (ms)');
    ylabel('Iterations to Converge');
    title('Effect of Communication Delay on Convergence');
    grid on;
    
    saveas(gcf, fullfile(plot_dir, 'delay_vs_iterations.png'));
    
    % Plot 3: Effect of packet loss on message count
    figure('Name', 'Effect of Packet Loss on Message Count', 'Position', [100, 100, 1000, 600]);
    hold on;
    
    % Average across other factors
    messages_vs_loss = squeeze(mean(mean(mean(results.messages_mean, 4), 2), 1));
    messages_std_vs_loss = squeeze(mean(mean(mean(results.messages_std, 4), 2), 1));
    
    errorbar(results.packet_loss_probs, messages_vs_loss, messages_std_vs_loss, 'o-', 'LineWidth', 2);
    
    xlabel('Packet Loss Probability');
    ylabel('Number of Messages');
    title('Effect of Packet Loss on Message Count');
    grid on;
    
    saveas(gcf, fullfile(plot_dir, 'packet_loss_vs_messages.png'));
    
    % Plot 4: Effect of epsilon on optimality gap
    figure('Name', 'Effect of Epsilon on Optimality Gap', 'Position', [100, 100, 1000, 600]);
    hold on;
    
    % Average across other factors
    gap_vs_epsilon = squeeze(mean(mean(mean(results.optimality_gap_mean, 3), 2), 1));
    gap_std_vs_epsilon = squeeze(mean(mean(mean(results.optimality_gap_std, 3), 2), 1));
    
    errorbar(results.epsilon_values, gap_vs_epsilon, gap_std_vs_epsilon, 'o-', 'LineWidth', 2);
    plot(results.epsilon_values, 2 * results.epsilon_values, '--r', 'LineWidth', 2);
    
    xlabel('Epsilon (ε)');
    ylabel('Optimality Gap');
    title('Effect of Epsilon on Optimality Gap');
    legend('Actual Gap', 'Theoretical Bound (2ε)');
    grid on;
    
    saveas(gcf, fullfile(plot_dir, 'epsilon_vs_gap.png'));
    
    % Plot 5: Recovery time vs task count
    figure('Name', 'Recovery Time vs Task Count', 'Position', [100, 100, 1000, 600]);
    hold on;
    
    % Average across other factors
    recovery_vs_tasks = squeeze(mean(mean(mean(results.recovery_time_mean, 4), 3), 2));
    recovery_std_vs_tasks = squeeze(mean(mean(mean(results.recovery_time_std, 4), 3), 2));
    
    % Calculate theoretical bounds (approximation)
    theoretical_bound = results.task_counts / 2 + max(results.epsilon_values) / min(results.epsilon_values);
    
    errorbar(results.task_counts, recovery_vs_tasks, recovery_std_vs_tasks, 'o-', 'LineWidth', 2);
    plot(results.task_counts, theoretical_bound, '--r', 'LineWidth', 2);
    
    xlabel('Number of Tasks (K)');
    ylabel('Recovery Time (iterations)');
    title('Recovery Time vs Task Count');
    legend('Actual Recovery Time', 'Theoretical Bound');
    grid on;
    
    saveas(gcf, fullfile(plot_dir, 'task_count_vs_recovery.png'));
    
    % Plot 6: ANOVA results heatmap
    figure('Name', 'Factors Impact Analysis', 'Position', [100, 100, 1000, 800]);
    
    % Perform simplified ANOVA for each response variable
    variables = {'makespan', 'messages', 'optimality_gap', 'iterations'};
    factors = {'Task Count', 'Comm Delay', 'Packet Loss', 'Epsilon'};
    
    % Create a matrix to hold p-values
    p_values = zeros(length(variables), length(factors));
    
    % For each response variable
    for v = 1:length(variables)
        var_name = variables{v};
        data = results.(var_name);
        
        % For each factor
        for f = 1:length(factors)
            % Reshape data to analyze this factor
            data_reshaped = reshape(data, [], size(data, f));
            
            % Perform one-way ANOVA
            [~, tbl] = anova1(data_reshaped, [], 'off');
            
            % Extract p-value
            p_values(v, f) = tbl{2, 6};
        end
    end
    
    % Create heatmap
    subplot(2, 1, 1);
    h = heatmap(factors, variables, -log10(p_values), 'ColorbarVisible', 'on');
    title('Factor Significance (-log10(p-value))');
    colormap(hot);
    
    % Create effect size heatmap
    effect_size = zeros(length(variables), length(factors));
    
    % For each response variable
    for v = 1:length(variables)
        var_name = variables{v};
        var_mean = results.([var_name '_mean']);
        
        % For each factor
        for f = 1:length(factors)
            % Calculate min and max across this factor
            min_val = min(var_mean, [], f);
            max_val = max(var_mean, [], f);
            
            % Calculate effect size as (max-min)/mean
            effect_size(v, f) = mean((max_val(:) - min_val(:)) ./ mean(var_mean(:)));
        end
    end
    
    subplot(2, 1, 2);
    h = heatmap(factors, variables, effect_size, 'ColorbarVisible', 'on');
    title('Factor Effect Size (normalized)');
    colormap(parula);
    
    saveas(gcf, fullfile(plot_dir, 'factors_impact_analysis.png'));
end

function mkdir_if_not_exists(dir_path)
    % Create directory if it doesn't exist
    if ~exist(dir_path, 'dir')
        mkdir(dir_path);
    end
end