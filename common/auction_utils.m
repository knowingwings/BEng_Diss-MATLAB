function utils = auction_utils()
    % AUCTION_UTILS - Returns function handles for auction-related functions
    utils = struct(...
        'initializeAuctionData', @local_initializeAuctionData, ...
        'distributedAuctionStep', @local_distributedAuctionStep, ...
        'calculateBid', @local_calculateBid, ...
        'initiateRecovery', @local_initiateRecovery, ...
        'consensusUpdate', @local_consensusUpdate, ...
        'runAuctionSimulation', @local_runAuctionSimulation, ...
        'analyzeBidDistribution', @local_analyzeBidDistribution, ...
        'analyzeTaskAllocation', @local_analyzeTaskAllocation ...
    );
end

function auction_data = local_initializeAuctionData(tasks, robots)
    num_tasks = length(tasks);
    num_robots = length(robots);
    
    auction_data.prices = zeros(num_tasks, 1);  % Initial prices are zero
    auction_data.assignment = zeros(num_tasks, 1);  % 0 means unassigned
    auction_data.initial_assignment = zeros(num_tasks, 1);  % For recovery analysis
    auction_data.completion_status = zeros(num_tasks, 1);  % 0 means incomplete
    auction_data.bids = zeros(num_robots, num_tasks);  % Bid matrix
    auction_data.utilities = zeros(num_robots, num_tasks);  % Utility matrix
    auction_data.last_update_time = zeros(num_robots, 1);  % For time-weighted consensus
    auction_data.recovery_mode = false;  % Whether in recovery mode
    auction_data.all_utilities = zeros(num_robots, num_tasks, 1000);  % Historical utilities for diagnosis
    auction_data.utility_iter = 1;  % Current iteration for utility history
    auction_data.failure_assignment = [];  % Store assignment at time of failure
    auction_data.task_oscillation_count = zeros(num_tasks, 1);  % Count of assignment changes for each task
    auction_data.task_last_robot = zeros(num_tasks, 1);  % Last robot assigned to each task
    auction_data.unassigned_iterations = zeros(num_tasks, 1);  % Track how long tasks remain unassigned
end

function [auction_data, new_assignments, messages] = local_distributedAuctionStep(auction_data, robots, tasks, available_tasks, params, varargin)
    % Handle optional iter parameter for backward compatibility
    if length(varargin) >= 1
        iter = varargin{1};
    else
        iter = auction_data.utility_iter; % Use utility_iter as a fallback for iteration count
    end

    num_robots = length(robots);
    new_assignments = false;
    messages = 0;
    
    % Convert "in recovery" tasks to unassigned before bidding
    recovery_tasks = find(auction_data.assignment == -1);
    if ~isempty(recovery_tasks)
        auction_data.assignment(recovery_tasks) = 0;
        available_tasks = union(available_tasks, recovery_tasks);
    end
    
    % Calculate current workload for each robot
    robot_workloads = zeros(1, num_robots);
    for i = 1:num_robots
        if isfield(robots, 'failed') && all(robots(i).failed)
            continue;
        end
        for j = 1:length(tasks)
            if all(auction_data.assignment(j) == i)
                if isfield(tasks, 'execution_time')
                    try
                        robot_workloads(i) = robot_workloads(i) + double(tasks(j).execution_time);
                    catch
                        robot_workloads(i) = robot_workloads(i) + 1;
                    end
                else
                    robot_workloads(i) = robot_workloads(i) + 1;
                end
            end
        end
    end
    
    % Calculate min and max workloads and the difference - used for adaptive pricing
    active_robots = find(~[robots.failed]);
    if ~isempty(active_robots)
        active_workloads = robot_workloads(active_robots);
        min_workload = min(active_workloads);
        max_workload = max(active_workloads);
        workload_diff = max_workload - min_workload;
        
        % Calculate workload imbalance
        if max_workload > 0
            workload_imbalance = workload_diff / max_workload;
        else
            workload_imbalance = 0;
        end
    else
        min_workload = 0;
        max_workload = 0;
        workload_diff = 0;
        workload_imbalance = 0;
    end
    
    % UPDATE: Track unassigned iterations counter for all tasks
    for j = 1:length(tasks)
        if all(auction_data.assignment(j) == 0)
            auction_data.unassigned_iterations(j) = auction_data.unassigned_iterations(j) + 1;
        else
            auction_data.unassigned_iterations(j) = 0;
        end
    end
    
    % Adaptive epsilon based on iteration
    if auction_data.utility_iter < 10
        base_epsilon = params.epsilon .* 1.5; % Higher initial epsilon to prevent oscillations
    else
        base_epsilon = params.epsilon;
    end
    
    % IMPROVED: Adaptive batch sizing based on multiple factors
    if auction_data.utility_iter < 10
        base_batch_size = ceil(length(available_tasks)/3); % More aggressive initially
    else
        base_batch_size = ceil(length(available_tasks)/5); % Base batch size
    end
    
    % Adjust batch size based on workload imbalance
    if workload_imbalance > 0.3
        imbalance_factor = 1.5; % Larger batches when imbalance is high
    elseif workload_imbalance > 0.15
        imbalance_factor = 1.25;
    else
        imbalance_factor = 1.0;
    end
    
    % Adjust batch size based on unassigned tasks
    unassigned_count = sum(auction_data.assignment == 0);
    if unassigned_count > 0
        unassigned_factor = 1 + (unassigned_count / length(tasks)) .* 0.5;
    else
        unassigned_factor = 1.0;
    end
    
    % Calculate final batch size
    max_bids_per_iteration = max(2, ceil(base_batch_size .* imbalance_factor .* unassigned_factor));
    
    % Prevent excessive batching
    max_bids_per_iteration = min(max_bids_per_iteration, length(available_tasks));
    
    % For each robot that is not failed
    for i = 1:num_robots
        if isfield(robots, 'failed') && all(robots(i).failed)
            continue;
        end
        
        % Update the robot's workload field
        if isfield(robots, 'workload')
            robots(i).workload = robot_workloads(i);
        end
        
        % Adjust bid calculation based on workload relative to others
        workload_ratio = 1.0;
        if max_workload > 0
            workload_ratio = robot_workloads(i) / max_workload;
        end
        
        % Calculate bids for all available tasks
        for j = available_tasks
            % Calculate bid with enhanced global objective consideration
            % Pass iteration number to the bid calculation function
            single_bid = local_calculateImprovedBid(i, j, robot_workloads(i), workload_ratio, workload_imbalance, auction_data, params, iter);
            
            % Ensure bid is a scalar
            auction_data.bids(i, j) = single_bid;
            auction_data.utilities(i, j) = single_bid - auction_data.prices(j);
            
            % Apply a penalty to the utility if this task has recently oscillated between robots
            if all(auction_data.task_oscillation_count(j) > 3) && all(auction_data.task_last_robot(j) ~= i)
                auction_data.utilities(i, j) = auction_data.utilities(i, j) .* 0.9;
            end
            
            % ADD: Special handling for long-unassigned tasks with escalating incentives
            if all(auction_data.assignment(j) == 0)
                unassigned_iter = auction_data.unassigned_iterations(j);
                if unassigned_iter > 20
                    bonus_factor = min(3.0, 1.0 + (unassigned_iter - 20) .* 0.1);
                    auction_data.utilities(i, j) = auction_data.utilities(i, j) .* bonus_factor;
                end
            end
        end
        
        % Store utilities for diagnosis
        if auction_data.utility_iter < size(auction_data.all_utilities, 3)
            auction_data.all_utilities(i, :, auction_data.utility_iter) = auction_data.utilities(i, :);
        end
        
        % Sort tasks by utility
        [sorted_utilities, sorted_indices] = sort(auction_data.utilities(i, available_tasks), 'descend');
        sorted_tasks = available_tasks(sorted_indices);
        
        % Select best tasks with positive utility - limited by batch size
        bid_count = 0;
        for j = sorted_tasks
            % Force scalar logical comparisons with all()
            if all(auction_data.utilities(i, j) > 0) && all(auction_data.assignment(j) ~= i)
                % Message prioritization and persistence
                % Implement resend attempts based on packet loss probability
                resend_attempts = min(2, ceil(3 .* params.packet_loss_prob));
                sent_successfully = false;
                
                for attempt = 1:resend_attempts+1
                    if rand() > params.packet_loss_prob  % No packet loss
                        sent_successfully = true;
                        break;
                    end
                    % In a real system, a small delay would occur between retries
                end
                
                if sent_successfully
                    % Broadcast bid
                    old_assignment = auction_data.assignment(j);
                    
                    % Track oscillation (robot changes) for this task
                    if all(old_assignment > 0) && all(old_assignment ~= i)
                        auction_data.task_oscillation_count(j) = auction_data.task_oscillation_count(j) + 1;
                    end
                    auction_data.task_last_robot(j) = i;
                    
                    auction_data.assignment(j) = i;
                    
                    % Dynamic price increment with multiple factors
                    effective_epsilon = base_epsilon;
                    
                    % Factor 1: Task oscillation history
                    if all(auction_data.task_oscillation_count(j) > 2)
                        effective_epsilon = effective_epsilon .* (1 + 0.15 .* auction_data.task_oscillation_count(j));
                    end
                    
                    % Factor 2: Workload balancing
                    if workload_diff > 0
                        if workload_ratio > 1.2  % Robot has >20% more workload than minimum
                            effective_epsilon = effective_epsilon .* 1.5;  % Increase price faster
                        elseif workload_ratio < 0.8  % Robot has <80% of maximum workload
                            effective_epsilon = effective_epsilon .* 0.7;  % Increase price slower
                        end
                    end
                    
                    % Cap prices to prevent them from getting too high
                    max_price = 3.0 .* max(params.alpha); % Maximum reasonable price
                    if auction_data.prices(j) + effective_epsilon > max_price
                        effective_epsilon = max(0, max_price - auction_data.prices(j));
                    end
                    
                    % Use the adjusted epsilon for price increment
                    auction_data.prices(j) = auction_data.prices(j) + effective_epsilon;
                    
                    new_assignments = true;
                    messages = messages + 1;
                    
                    % If this is the first assignment, record it for recovery analysis
                    if all(auction_data.initial_assignment(j) == 0)
                        auction_data.initial_assignment(j) = i;
                    end
                    
                    % Update workload in our records (and in robot structure if possible)
                    if all(old_assignment > 0)
                        if isfield(robots, 'workload')
                            if isfield(tasks, 'execution_time')
                                try
                                    current_workload = double(robots(old_assignment).workload);
                                    task_exec_time = double(tasks(j).execution_time);
                                    robots(old_assignment).workload = current_workload - task_exec_time;
                                catch
                                    % Fallback if conversion fails
                                    robots(old_assignment).workload = 0;
                                end
                            else
                                try
                                    robots(old_assignment).workload = double(robots(old_assignment).workload) - 1;
                                catch
                                    robots(old_assignment).workload = 0;
                                end
                            end
                        end
                    end
                    
                    if isfield(robots, 'workload')
                        if isfield(tasks, 'execution_time')
                            try
                                current_workload = double(robots(i).workload);
                                task_exec_time = double(tasks(j).execution_time);
                                robots(i).workload = current_workload + task_exec_time;
                            catch
                                % Fallback if conversion fails
                                robots(i).workload = 1;
                            end
                        else
                            try
                                robots(i).workload = double(robots(i).workload) + 1;
                            catch
                                robots(i).workload = 1;
                            end
                        end
                    end
                    
                    bid_count = bid_count + 1;
                    if bid_count >= max_bids_per_iteration
                        break;  % Limit bids per iteration for more uniform communication
                    end
                end
            end
        end
    end
    
    % ENHANCED: Progressive price reduction for unassigned tasks
    if iter > 10
        unassigned_tasks = find(auction_data.assignment == 0);
        
        if ~isempty(unassigned_tasks)
            % Sort by how long they've been unassigned
            [sorted_unassigned_times, sort_idx] = sort(auction_data.unassigned_iterations(unassigned_tasks), 'descend');
            sorted_unassigned_tasks = unassigned_tasks(sort_idx);
            
            for task_idx = 1:length(sorted_unassigned_tasks)
                j = sorted_unassigned_tasks(task_idx);
                unassigned_iter = auction_data.unassigned_iterations(j);
                
                % Progressive price reduction - more aggressive for longer unassigned tasks
                if unassigned_iter > 30
                    reduction_factor = 0.3; % Very aggressive reduction
                elseif unassigned_iter > 20
                    reduction_factor = 0.5; % Strong reduction
                elseif unassigned_iter > 10
                    reduction_factor = 0.7; % Moderate reduction
                else
                    reduction_factor = 0.9; % Mild reduction
                end
                
                auction_data.prices(j) = auction_data.prices(j) .* reduction_factor;
                
                % Ensure price doesn't go below zero
                auction_data.prices(j) = max(0, auction_data.prices(j));
                
                % Clear oscillation history to allow fresh bidding
                if unassigned_iter > 15
                    auction_data.task_oscillation_count(j) = 0;
                end
                
                % Add status updates for difficult tasks to track progress
                if unassigned_iter > 25 && mod(iter, 5) == 0
                    fprintf('Task %d remains unassigned for %d iterations - price reduced to %.2f\n', ...
                            j, unassigned_iter, auction_data.prices(j));
                end
            end
        end
    end
    
    % Update utility iteration counter
    auction_data.utility_iter = auction_data.utility_iter + 1;
    
    % Update last update time for all robots
    auction_data.last_update_time = auction_data.last_update_time + 1;
end

function bid = local_calculateBid(robot_id, task_id, robot_workload, auction_data, params)
    % This is the original function but isn't used anymore
    bid = 0;
end

function bid = local_calculateImprovedBid(robot_id, task_id, robot_workload, workload_ratio, workload_imbalance, auction_data, params, varargin)
    % Handle optional iter parameter for backward compatibility
    if length(varargin) >= 1
        iter = varargin{1};
    else
        iter = auction_data.utility_iter; % Use utility_iter as a fallback for iteration count
    end

    % Make sure we're working with scalar values
    robot_id = double(robot_id(1));
    task_id = double(task_id(1));
    robot_workload = double(robot_workload(1));
    
    % Try to extract position data for real distance calculation
    try
        distance = 0;
        if isfield(robots, 'position') && isfield(tasks, 'position')
            distance = norm(robots(robot_id).position - tasks(task_id).position);
        else
            % Fallback to ID-based distance approximation
            distance = abs(robot_id - task_id);
        end
    catch
        % If that fails, use a fallback approach
        distance = abs(robot_id - task_id);
    end
    
    % Better distance factor normalization
    d_factor = 1 / (1 + distance);
    
    % Configuration cost - keep simple
    c_factor = 1;
    
    % Try to calculate real capability matching if possible
    try
        capability_match = 0.8;  % Default value
        if isfield(robots, 'capabilities') && isfield(tasks, 'capabilities_required')
            robot_cap = robots(robot_id).capabilities;
            task_cap = tasks(task_id).capabilities_required;
            
            % Normalize vectors to unit length
            robot_cap_norm = robot_cap / norm(robot_cap);
            task_cap_norm = task_cap / norm(task_cap);
            
            % Compute cosine similarity (normalized dot product)
            capability_match = dot(robot_cap_norm, task_cap_norm);
        end
    catch
        % If calculation fails, use a default value
        capability_match = 0.8;
    end
    
    % Progressive workload factor with stronger imbalance penalty
    % This creates a non-linear penalty that increases more rapidly with workload
    % The exponent (1.8) makes the penalty grow faster than before
    workload_factor = robot_workload / 10 .* (workload_ratio.^1.8);
    
    % Add global balance consideration
    global_balance_factor = workload_imbalance .* 0.5;
    
    % Simple energy factor
    energy_factor = distance .* 0.1;
    
    % Standard weights with better balance
    alpha = [0.8, 0.3, 1.0, 1.2, 0.2];  % Default values
    
    % Override with provided parameters if available
    if isfield(params, 'alpha') && length(params.alpha) >= 5
        for i = 1:5
            alpha(i) = params.alpha(i);
        end
    end
    
    % Adjust workload penalty when imbalance is high
    adjusted_workload_alpha = alpha(4);
    if workload_ratio > 1.2 || workload_ratio < 0.8
        adjusted_workload_alpha = adjusted_workload_alpha .* 1.5;  % 50% increase in penalty
    end
    
    % IMPROVED: Enhanced bonus for unassigned tasks with progressive scaling
    task_unassigned_bonus = 0;
    if all(auction_data.assignment(task_id) == 0)
        unassigned_iter = auction_data.unassigned_iterations(task_id);
        
        % Exponential bonus scaling for persistent unassigned tasks
        if unassigned_iter > 25
            task_unassigned_bonus = min(6.0, 0.5 .* exp(unassigned_iter./10));
        elseif unassigned_iter > 15
            task_unassigned_bonus = min(4.0, 0.4 .* exp(unassigned_iter./12));
        elseif unassigned_iter > 5
            task_unassigned_bonus = min(3.0, 0.3 .* unassigned_iter);
        end
    end
    
    % Add scaling factor for tasks that need to be assigned quickly
    iteration_factor = 0;
    if iter > 20 && all(auction_data.assignment(task_id) == 0)
        iteration_factor = min(2.0, 0.05 .* iter);
    end
    
    % Calculate bid
    if auction_data.recovery_mode
        % Add recovery-specific terms when in recovery mode
        % Extract beta parameters safely
        beta = [2.0, 1.5, 1.0];  % Added third parameter for balance
        if isfield(params, 'beta') && length(params.beta) >= 2
            beta(1) = params.beta(1);
            beta(2) = params.beta(2);
            if length(params.beta) >= 3
                beta(3) = params.beta(3);
            end
        end
        
        % Calculate bid components with improved recovery bias
        progress_term = beta(1) .* (1 - 0);  % Assuming no partial progress
        criticality_term = beta(2) .* 0.5;   % Default criticality
        
        % Recovery bids favor robots with lower workload more strongly
        recovery_workload_factor = (1 - workload_ratio) .* beta(1) .* 0.8;
        
        % Add stronger global balance factor during recovery
        global_recovery_factor = workload_imbalance .* beta(3) .* 0.7;
        
        bid = alpha(1) .* d_factor + ...
              alpha(2) .* c_factor + ...
              alpha(3) .* capability_match - ...
              adjusted_workload_alpha .* workload_factor - ...
              alpha(5) .* energy_factor + ...
              progress_term + ...
              criticality_term + ...
              recovery_workload_factor + ...
              global_recovery_factor + ...
              global_balance_factor + ...
              task_unassigned_bonus + ...
              iteration_factor;
    else
        bid = alpha(1) .* d_factor + ...
              alpha(2) .* c_factor + ...
              alpha(3) .* capability_match - ...
              adjusted_workload_alpha .* workload_factor - ...
              alpha(5) .* energy_factor + ...
              global_balance_factor + ...
              task_unassigned_bonus + ...
              iteration_factor;
    end
    
    % Add small random noise to break ties (slightly increased to help prevent cycling)
    bid = double(bid) + 0.001 .* rand();
end

function auction_data = local_initiateRecovery(auction_data, robots, tasks, failed_robot_id)
    % INITIATERECOVERY Initiate recovery process after a robot failure
    auction_data.recovery_mode = true;
    
    % Store current assignment state for recovery analysis
    auction_data.failure_assignment = auction_data.assignment;
    
    % Find tasks assigned to the failed robot
    failed_tasks = find(auction_data.assignment == failed_robot_id);
    
    % Prioritize critical tasks during recovery
    if ~isempty(failed_tasks)
        % Calculate criticality scores for failed tasks
        criticality_scores = zeros(size(failed_tasks));
        
        for i = 1:length(failed_tasks)
            task_idx = failed_tasks(i);
            
            % 1. Consider execution time
            if isfield(tasks, 'execution_time')
                criticality_scores(i) = tasks(task_idx).execution_time;
            else
                criticality_scores(i) = 1;
            end
            
            % 2. Add bonus for tasks with dependencies
            if isfield(tasks, 'prerequisites')
                % Count how many other tasks depend on this one
                dependent_count = 0;
                for j = 1:length(tasks)
                    if ismember(task_idx, tasks(j).prerequisites)
                        dependent_count = dependent_count + 1;
                    end
                end
                criticality_scores(i) = criticality_scores(i) + dependent_count .* 2;
            end
        end
        
        % Sort failed tasks by criticality (highest first)
        [~, critical_order] = sort(criticality_scores, 'descend');
        prioritized_failed_tasks = failed_tasks(critical_order);
        
        % Process in order of criticality
        auction_data.task_oscillation_count(failed_tasks) = 0;
        
        for i = 1:length(prioritized_failed_tasks)
            task_idx = prioritized_failed_tasks(i);
            
            % Mark as in recovery with priority level reflected in the price reset
            % Most critical tasks get bigger price reductions
            priority_factor = 1 - (i-1)/length(prioritized_failed_tasks);
            auction_data.assignment(task_idx) = -1;  % -1 indicates "in recovery"
            
            % More aggressive price reset for high-priority tasks
            reset_factor = 0.3 .* (1 + priority_factor);
            auction_data.prices(task_idx) = auction_data.prices(task_idx) .* reset_factor;
        end
    end
    
    fprintf('Recovery initiated for robot %d. %d tasks need reassignment.\n', ...
            failed_robot_id, length(failed_tasks));
end

function x_consensus = local_consensusUpdate(x_i, x_others, last_update_times, gamma, lambda)
    % CONSENSUSUPDATE Perform a time-weighted consensus update
    x_consensus = x_i;
    
    % Update state based on information from other robots
    for j = 1:size(x_others, 2)
        % Calculate time-weighted factor
        time_diff = last_update_times(j);
        weight = gamma .* exp(-lambda .* time_diff);
        
        % Update state
        x_consensus = x_consensus + weight .* (x_others(:, j) - x_i);
    end
end

function [metrics, converged] = local_runAuctionSimulation(params, env, robots, tasks, visualize)
    % RUNAUCTIONSIMULATION Run a complete auction algorithm simulation
    
    % Ensure epsilon is sufficient but not excessive
    if params.epsilon < 0.01
        params.epsilon = 0.05;
        if visualize
            fprintf('Warning: Epsilon was too small. Adjusted to 0.05\n');
        end
    elseif params.epsilon > 1.0
        params.epsilon = 1.0;
        if visualize
            fprintf('Warning: Epsilon was too large. Adjusted to 1.0\n');
        end
    end
    
    % Get utility functions directly for avoiding scope issues
    env_utils = environment_utils();
    robot_utils = robot_utils();
    task_utils = task_utils();
    auction_utils = auction_utils();
    
    auction_data = local_initializeAuctionData(tasks, robots);
    
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
    metrics.failed_task_count = 0;
    metrics.failure_time = 0;  % Initialize failure_time field
    metrics.makespan = 0;
    metrics.optimal_makespan = 0;
    metrics.theoretical_recovery_bound = 0;
    metrics.oscillation_count = 0;
    
    % Track workload before failure for better analysis
    metrics.makespan_before_failure = 0;
    
    % Main simulation loop
    max_iterations = 1000;
    converged = false;
    
    % More robust convergence tracking
    unchanged_iterations = 0;
    stable_workload_iterations = 0;
    minor_change_threshold = 0.05;  % 5% change is considered minor
    
    % Minimum iterations before considering convergence to prevent premature termination
    min_iterations_before_convergence = 15;
    
    % Available tasks (initially only those with no prerequisites)
    try
        available_tasks = task_utils.findAvailableTasks(tasks, []);
    catch
        % If function fails, assume all tasks are available at the start
        available_tasks = 1:length(tasks);
    end
    
    % Check if we have an unusually small number of available tasks
    if length(available_tasks) < 2 && length(tasks) > 3
        if visualize
            fprintf('Warning: Only %d/%d tasks available initially. Check task dependencies.\n', ...
                length(available_tasks), length(tasks));
        end
    end
    
    % Track workload history for stability detection
    prev_workloads = zeros(length(robots), 5);  % Store last 5 iterations
    workload_history_idx = 1;
    
    for iter = 1:max_iterations
        metrics.iterations = iter;
        
        % Check for robot failure
        if iter == params.failure_time && ~isempty(params.failed_robot)
            if visualize
                fprintf('Robot %d has failed at iteration %d\n', params.failed_robot, iter);
            end
            if isfield(robots, 'failed')
                robots(params.failed_robot).failed = true;
            end
            metrics.failure_time = iter;  % Record the failure time
            
            % Record workload and makespan before failure
            robot_loads = zeros(length(robots), 1);
            for j = 1:length(tasks)
                r = auction_data.assignment(j);
                if r > 0 && r <= length(robot_loads)
                    if isfield(tasks, 'execution_time')
                        robot_loads(r) = robot_loads(r) + tasks(j).execution_time;
                    else
                        robot_loads(r) = robot_loads(r) + 1;
                    end
                end
            end
            metrics.makespan_before_failure = max(robot_loads);
            
            % Record number of tasks assigned to failed robot
            metrics.failed_task_count = sum(auction_data.assignment == params.failed_robot);
            
            % Initiate recovery process
            auction_data = local_initiateRecovery(auction_data, robots, tasks, params.failed_robot);
            
            % Reset convergence tracking
            unchanged_iterations = 0;
            stable_workload_iterations = 0;
        end
        
        % Simulate one step of the distributed auction algorithm
        % Pass the current iteration number to the step function
        [auction_data, new_assignments, messages] = local_distributedAuctionStep(auction_data, robots, tasks, available_tasks, params, iter);
        metrics.messages = metrics.messages + messages;
        
        % Diagnostic output every 10 iterations
        if visualize && mod(iter, 10) == 0
            local_analyzeTaskAllocation(auction_data, tasks);
        end
        
        % Update visualization if enabled
        if visualize
            try
                subplot(2, 3, [1, 4]);
                env_utils.visualizeEnvironment(env, robots, tasks, auction_data);
                title(sprintf('Environment (Iteration %d)', iter));
                pause(0.01);
            catch
                % Visualization failed, skip it
                fprintf('Warning: Visualization failed, continuing without it\n');
            end
        end
        
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
        
        % Track workload stability
        current_workloads = zeros(length(robots), 1);
        for r = 1:length(robots)
            if ~(isfield(robots, 'failed') && all(robots(r).failed))
                % Calculate workload for this robot
                workload = 0;
                for j = 1:length(tasks)
                    if all(auction_data.assignment(j) == r)
                        if isfield(tasks, 'execution_time')
                            workload = workload + tasks(j).execution_time;
                        else
                            workload = workload + 1;
                        end
                    end
                end
                current_workloads(r) = workload;
            end
        end
        
        % Store current workloads in history
        prev_workloads(:, workload_history_idx) = current_workloads;
        workload_history_idx = mod(workload_history_idx, 5) + 1;
        
        % Check workload stability (after collecting enough history)
        if iter > 5
            % Calculate average workload over last 5 iterations
            avg_workloads = mean(prev_workloads, 2);
            
            % Check if current workloads are within minor_change_threshold of average
            max_deviation = 0;
            for r = 1:length(robots)
                if avg_workloads(r) > 0
                    deviation = abs(current_workloads(r) - avg_workloads(r)) ./ avg_workloads(r);
                    max_deviation = max(max_deviation, deviation);
                end
            end
            
            if max_deviation < minor_change_threshold
                stable_workload_iterations = stable_workload_iterations + 1;
            else
                stable_workload_iterations = 0;
            end
        end
        
        % Check if any new tasks become available due to completed prerequisites
        try
            completed_tasks = find(auction_data.completion_status == 1);
            available_tasks = task_utils.findAvailableTasks(tasks, completed_tasks);
        catch
            % If function fails, assume all tasks are available
            available_tasks = 1:length(tasks);
        end
        
        % NEW: Implement a reset mechanism for stalled situations
        if mod(iter, 50) == 0 && iter > 100 && any(auction_data.assignment == 0)
            unassigned_count = sum(auction_data.assignment == 0);
            fprintf('Reset triggered at iteration %d with %d unassigned tasks\n', iter, unassigned_count);
            
            % Reset prices for unassigned tasks to zero to restart bidding
            unassigned_tasks = find(auction_data.assignment == 0);
            auction_data.prices(unassigned_tasks) = 0;
            
            % Reset oscillation counts to allow fresh bidding
            auction_data.task_oscillation_count(unassigned_tasks) = 0;
            
            % Reset convergence tracking
            unchanged_iterations = 0;
            stable_workload_iterations = 0;
        end
        
        % IMPROVED: Scale required stable iterations with problem complexity
        min_required_stable = max(10, ceil(log(length(tasks)) .* 5));
        required_workload_stability = max(5, ceil(log(length(tasks)) .* 3));
        
        % Diagnostic for hard-to-assign tasks
        if unchanged_iterations > 0 && mod(iter, 10) == 0
            unassigned_count = sum(auction_data.assignment == 0);
            if unassigned_count > 0 && visualize
                unassigned_tasks = find(auction_data.assignment == 0);
                fprintf('WARNING: %d tasks remain unassigned after %d iterations.\n', ...
                        unassigned_count, iter);
                for j = unassigned_tasks
                    fprintf('  Task %d: unassigned for %d iterations, price=%.2f\n', ...
                            j, auction_data.unassigned_iterations(j), auction_data.prices(j));
                end
            end
        end
        
        % Primary convergence criteria - stable assignments for sufficient iterations
        if iter > min_iterations_before_convergence && unchanged_iterations >= min_required_stable && stable_workload_iterations >= required_workload_stability
            unassigned_count = sum(auction_data.assignment == 0);
            
            % Allow convergence even with unassigned tasks if we've been stable for much longer
            if unassigned_count == 0 || (unchanged_iterations >= 2*min_required_stable)
                converged = true;
                if visualize
                    if unassigned_count == 0
                        fprintf('Auction algorithm converged after %d iterations (all tasks assigned and stable)\n', iter);
                    else
                        fprintf('Auction algorithm converged after %d iterations with %d unassigned tasks\n', iter, unassigned_count);
                    end
                end
                break;
            end
        end
        
        % Update recovery time if in recovery mode
        if ~isempty(params.failed_robot) && params.failure_time < inf && metrics.recovery_time == 0
            % Check if all tasks from the failed robot have been reassigned
            if isfield(auction_data, 'failure_assignment') && ~isempty(auction_data.failure_assignment)
                failed_tasks = find(auction_data.failure_assignment == params.failed_robot);
                if all(auction_data.assignment(failed_tasks) ~= params.failed_robot) && ...
                   all(auction_data.assignment(failed_tasks) ~= 0) && ...
                   all(auction_data.assignment(failed_tasks) ~= -1)
                    metrics.recovery_time = iter - metrics.failure_time;
                    if visualize
                        fprintf('Recovery completed after %d iterations\n', metrics.recovery_time);
                    end
                end
            else
                if ~any(auction_data.assignment == params.failed_robot) && ...
                   ~any(auction_data.assignment == -1)
                    metrics.recovery_time = iter - metrics.failure_time;
                    if visualize
                        fprintf('Recovery completed after %d iterations (using fallback)\n', metrics.recovery_time);
                    end
                end
            end
        end
        
        % Stop if we reach max iterations without convergence
        if iter == max_iterations
            fprintf('WARNING: Maximum iterations (%d) reached without convergence!\n', max_iterations);
            % Evaluate final performance even without convergence
            converged = false;
            break;
        end
    end
    
    % Trim history matrices to actual size
    metrics.price_history = metrics.price_history(:, 1:iter);
    metrics.assignment_history = metrics.assignment_history(:, 1:iter);
    
    % NEW: Track unassigned tasks explicitly
    metrics.unassigned_tasks = sum(auction_data.assignment == 0);
    
    % Calculate makespan
    try
        metrics.makespan = robot_utils.calculateMakespan(auction_data.assignment, tasks, robots);
        metrics.optimal_makespan = robot_utils.calculateOptimalMakespan(tasks, robots);
    catch
        % If makespan calculation fails, use a simplified approach
        robot_loads = zeros(length(robots), 1);
        for j = 1:length(tasks)
            r = auction_data.assignment(j);
            if r > 0 && r <= length(robot_loads)
                if isfield(tasks, 'execution_time')
                    robot_loads(r) = robot_loads(r) + tasks(j).execution_time;
                else
                    robot_loads(r) = robot_loads(r) + 1;
                end
            end
        end
        metrics.makespan = max(robot_loads);
        
        % Better optimal makespan approximation
        total_load = sum(robot_loads);
        active_robots = sum(~[robots.failed]);
        if active_robots > 0
            balanced_load = total_load ./ active_robots;
        else
            balanced_load = total_load;
        end
        
        % Optimal makespan is at least the balanced load or the max task time
        max_task_time = 0;
        for j = 1:length(tasks)
            if isfield(tasks, 'execution_time')
                max_task_time = max(max_task_time, tasks(j).execution_time);
            else
                max_task_time = 1;
            end
        end
        
        metrics.optimal_makespan = max(balanced_load, max_task_time);
    end
    
    % Count task oscillations
    metrics.oscillation_count = sum(auction_data.task_oscillation_count);
    
    % Ensure optimality gap is correctly calculated
    metrics.optimality_gap = abs(metrics.makespan - metrics.optimal_makespan);
    
    % Theoretical recovery bound
    if ~isempty(params.failed_robot) && params.failure_time < inf
        T_f = metrics.failed_task_count;
        b_max = max(params.alpha);
        epsilon = params.epsilon;
        metrics.theoretical_recovery_bound = T_f + round(b_max./epsilon);
    end
    
    % Final diagnostic if requested
    if visualize
        fprintf('\n--- Final Task Allocation ---\n');
        local_analyzeTaskAllocation(auction_data, tasks);
        fprintf('Makespan: %.2f (Optimal: %.2f, Gap: %.2f)\n', ...
            metrics.makespan, metrics.optimal_makespan, metrics.optimality_gap);
        fprintf('Total task oscillations: %d\n', metrics.oscillation_count);
        
        if metrics.unassigned_tasks > 0
            fprintf('Final result has %d unassigned tasks.\n', metrics.unassigned_tasks);
        end
    end
end

% Diagnostic functions
function local_analyzeBidDistribution(auction_data, robots, tasks)
    fprintf('\n--- Bid Analysis ---\n');
    for i = 1:length(robots)
        if isfield(robots, 'failed') && all(robots(i).failed)
            fprintf('Robot %d: FAILED\n', i);
            continue;
        end
        fprintf('Robot %d bids:\n', i);
        for j = 1:length(tasks)
            fprintf('  Task %d: Bid=%.3f, Utility=%.3f, Price=%.3f\n', ...
                j, auction_data.bids(i,j), auction_data.utilities(i,j), auction_data.prices(j));
        end
    end
end

function local_analyzeTaskAllocation(auction_data, tasks)
    robot_tasks = cell(2,1);
    for j = 1:length(tasks)
        r = auction_data.assignment(j);
        if r > 0
            robot_tasks{r} = [robot_tasks{r}, j];
        end
    end
    
    fprintf('Task Allocation: ');
    for i = 1:2
        fprintf('R%d: [', i);
        if ~isempty(robot_tasks{i})
            fprintf('%d ', robot_tasks{i});
        end
        fprintf('] ');
    end
    fprintf('\n');
    
    % Calculate workload balance
    workload = zeros(2,1);
    for j = 1:length(tasks)
        r = auction_data.assignment(j);
        if r > 0
            if isfield(tasks, 'execution_time')
                workload(r) = workload(r) + tasks(j).execution_time;
            else
                workload(r) = workload(r) + 1;  % Default execution time
            end
        end
    end
    
    if workload(1) == 0 && workload(2) == 0
        workload_ratio = 0;
    elseif workload(1) == 0
        workload_ratio = Inf;
    elseif workload(2) == 0
        workload_ratio = Inf;
    else
        workload_ratio = max(workload)/min(workload);
    end
    
    fprintf('Workload: R1=%.2f, R2=%.2f, Ratio=%.2f\n', ...
        workload(1), workload(2), workload_ratio);
    
    % Report unassigned tasks if any
    unassigned = sum(auction_data.assignment == 0);
    if unassigned > 0
        fprintf('WARNING: %d tasks remain unassigned!\n', unassigned);
        
        % Show which tasks are unassigned and how long they've been unassigned
        unassigned_tasks = find(auction_data.assignment == 0);
        fprintf('Unassigned tasks: ');
        for j = unassigned_tasks
            fprintf('T%d (unassigned for %d iterations) ', j, auction_data.unassigned_iterations(j));
        end
        fprintf('\n');
    end
    
    % Report oscillations
    fprintf('Task oscillations: ');
    for j = 1:length(tasks)
        if auction_data.task_oscillation_count(j) > 0
            fprintf('T%d:%d ', j, auction_data.task_oscillation_count(j));
        end
    end
    fprintf('\n');
end