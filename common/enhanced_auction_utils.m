function utils = enhanced_auction_utils()
    % ENHANCED_AUCTION_UTILS - Returns function handles for enhanced auction algorithm
    utils = struct(...
        'initializeAuctionData', @initializeAuctionData, ...
        'enhancedDistributedAuctionStep', @enhancedDistributedAuctionStep, ...
        'detectFailures', @detectFailures, ...
        'initiateRecovery', @initiateRecovery, ...
        'calculateBid', @calculateBid, ...
        'handleCollaborativeTasks', @handleCollaborativeTasks, ...
        'analyzeTaskAllocation', @analyzeTaskAllocation, ...
        'analyzeBidDistribution', @analyzeBidDistribution, ...
        'runEnhancedAuctionSimulation', @runEnhancedAuctionSimulation ...  % Add this line
    );
    
    % =========================================================================
    function auction_data = initializeAuctionData(tasks, robots)
        % INITIALIZEAUCTIONDATA Initializes the auction data structure
        %
        % Parameters:
        %   tasks  - Array of task structures
        %   robots - Array of robot structures
        %
        % Returns:
        %   auction_data - Initialized auction data structure
        
        num_tasks = length(tasks);
        num_robots = length(robots);
        
        % Standard auction data initialization
        auction_data.prices = zeros(num_tasks, 1);  % Initial prices are zero
        auction_data.assignment = zeros(num_tasks, 1);  % 0 means unassigned
        auction_data.initial_assignment = zeros(num_tasks, 1);  % For recovery analysis
        auction_data.completion_status = zeros(num_tasks, 1);  % 0 means incomplete
        auction_data.bids = zeros(num_robots, num_tasks);  % Bid matrix
        auction_data.utilities = zeros(num_robots, num_tasks);  % Utility matrix
        auction_data.utility_iter = 1;  % Current iteration for utility history
        
        % Task tracking
        auction_data.task_oscillation_count = zeros(num_tasks, 1);  % Count of assignment changes for each task
        auction_data.task_last_robot = zeros(num_tasks, 1);  % Last robot assigned to each task
        auction_data.unassigned_iterations = zeros(num_tasks, 1);  % Track how long tasks remain unassigned
        
        % Recovery mode tracking
        auction_data.recovery_mode = false;  % Whether in recovery mode
        auction_data.failure_assignment = [];  % Store assignment at time of failure
        
        % Collaborative task data
        auction_data.collaborative_tasks = zeros(num_tasks, 1);
        for i = 1:num_tasks
            if isfield(tasks, 'collaborative') && any(tasks(i).collaborative)
                auction_data.collaborative_tasks(i) = 1;
            end
        end
        
        % Initialize heartbeat data
        auction_data.last_heartbeat = zeros(num_robots, 1);
        auction_data.current_time = 1;  % Start at iteration 1
        
        % Set initial heartbeats for all robots (they're all alive at start)
        for i = 1:num_robots
            auction_data.last_heartbeat(i) = auction_data.current_time;
        end
        
        % For tracking failure history
        auction_data.failed_robots = zeros(num_robots, 1);
        auction_data.failure_time = zeros(num_robots, 1);
        auction_data.recovery_time = zeros(num_robots, 1);
        
        % For progress tracking (used in failure detection)
        auction_data.task_progress = zeros(num_tasks, 1);
        auction_data.last_progress_check = zeros(num_tasks, 1);
        auction_data.last_progress_time = zeros(num_tasks, 1);
        
        % Historical data for diagnosis
        auction_data.all_utilities = zeros(num_robots, num_tasks, 1000);  % Historical utilities
        auction_data.last_update_time = zeros(num_robots, 1);  % For time-weighted consensus
    end
    
    % =========================================================================
    function [auction_data, new_assignments, messages] = enhancedDistributedAuctionStep(auction_data, robots, tasks, available_tasks, params, varargin)
        % ENHANCEDDISTRIBUTEDAUCTIONSTEP Performs one step of the distributed auction algorithm
        %
        % Parameters:
        %   auction_data    - Auction data structure
        %   robots          - Array of robot structures
        %   tasks           - Array of task structures
        %   available_tasks - List of available task IDs
        %   params          - Algorithm parameters
        %   varargin        - Optional iteration number
        %
        % Returns:
        %   auction_data    - Updated auction data structure
        %   new_assignments - Whether any new assignments were made
        %   messages        - Number of messages exchanged
        
        % Handle optional iter parameter for backward compatibility
        if length(varargin) >= 1
            iter = varargin{1};
        else
            iter = auction_data.utility_iter; % Use utility_iter as fallback
        end

        num_robots = length(robots);
        num_tasks = length(tasks);
        new_assignments = false;
        messages = 0;
        
        % Update simulation time
        auction_data.current_time = iter;
        
        % ---- HEARTBEAT MECHANISM ----
        % Each active robot sends a heartbeat
        for i = 1:num_robots
            % Only send heartbeats from non-failed robots
            if ~isfield(robots, 'failed') || ~robots(i).failed
                % Simulate heartbeat message transmission
                transmission_success = true;
                
                % Apply packet loss probability if specified
                if isfield(params, 'packet_loss_prob') && params.packet_loss_prob > 0
                    if rand() < params.packet_loss_prob
                        transmission_success = false;
                    end
                end
                
                % If heartbeat was successfully transmitted, update timestamp
                if transmission_success
                    auction_data.last_heartbeat(i) = auction_data.current_time;
                    messages = messages + 1; % Count heartbeat as a message
                end
            end
        end
        
        % Convert "in recovery" tasks to unassigned before bidding
        recovery_tasks = find(auction_data.assignment == -1);
        if ~isempty(recovery_tasks)
            auction_data.assignment(recovery_tasks) = 0;
            available_tasks = union(available_tasks, recovery_tasks);
        end
        
        % Calculate current workload for each robot
        robot_workloads = zeros(1, num_robots);
        for i = 1:num_robots
            if isfield(robots, 'failed') && robots(i).failed
                continue;
            end
            for j = 1:length(tasks)
                if auction_data.assignment(j) == i
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
        active_robots = [];
        for i = 1:num_robots
            if ~isfield(robots, 'failed') || ~robots(i).failed
                active_robots = [active_robots, i];
            end
        end
        
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
            if auction_data.assignment(j) == 0
                auction_data.unassigned_iterations(j) = auction_data.unassigned_iterations(j) + 1;
            else
                auction_data.unassigned_iterations(j) = 0;
            end
        end
        
        % Adaptive epsilon based on iteration
        if auction_data.utility_iter < 10
            base_epsilon = params.epsilon * 1.5; % Higher initial epsilon to prevent oscillations
        else
            base_epsilon = params.epsilon;
        end
        
        % Adaptive batch sizing based on multiple factors
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
            unassigned_factor = 1 + (unassigned_count / length(tasks)) * 0.5;
        else
            unassigned_factor = 1.0;
        end
        
        % Calculate final batch size
        max_bids_per_iteration = max(2, ceil(base_batch_size * imbalance_factor * unassigned_factor));
        
        % Prevent excessive batching
        max_bids_per_iteration = min(max_bids_per_iteration, length(available_tasks));
        
        % For each robot that is not failed
        for i = 1:num_robots
            if isfield(robots, 'failed') && robots(i).failed
                continue;
            end
            
            % Adjust bid calculation based on workload relative to others
            workload_ratio = 1.0;
            if max_workload > 0
                workload_ratio = robot_workloads(i) / max_workload;
            end
            
            % Calculate bids for all available tasks
            for j = available_tasks
                % Check if this is a collaborative task
                if auction_data.collaborative_tasks(j) == 1
                    % Handle collaborative tasks differently
                    [auction_data, sync_successful] = handleCollaborativeTasks(auction_data, robots, tasks, j, params);
                    if sync_successful
                        continue; % Skip normal bidding process for this task
                    end
                end
                
                % Calculate bid with enhanced global objective consideration
                single_bid = calculateBid(i, j, robot_workloads(i), workload_ratio, workload_imbalance, auction_data, params, iter);
                
                % Ensure bid is a scalar
                auction_data.bids(i, j) = single_bid;
                auction_data.utilities(i, j) = single_bid - auction_data.prices(j);
                
                % Apply a penalty to the utility if this task has recently oscillated between robots
                if any(auction_data.task_oscillation_count(j) > 3) && any(auction_data.task_last_robot(j) ~= i)
                    auction_data.utilities(i, j) = auction_data.utilities(i, j) * 0.9;
                end
                
                % ADD: Special handling for long-unassigned tasks with escalating incentives
                if auction_data.assignment(j) == 0
                    unassigned_iter = auction_data.unassigned_iterations(j);
                    if unassigned_iter > 20
                        bonus_factor = min(3.0, 1.0 + (unassigned_iter - 20) * 0.1);
                        auction_data.utilities(i, j) = auction_data.utilities(i, j) * bonus_factor;
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
                if any(auction_data.utilities(i, j)) > 0 && any(auction_data.assignment(j) ~= i)
                    % Message prioritization and persistence
                    % Implement resend attempts based on packet loss probability
                    resend_attempts = 0;
                    if isfield(params, 'packet_loss_prob') && params.packet_loss_prob > 0
                        resend_attempts = min(2, ceil(3 * params.packet_loss_prob));
                    end
                    sent_successfully = false;
                    
                    for attempt = 1:(resend_attempts+1)
                        if ~isfield(params, 'packet_loss_prob') || params.packet_loss_prob == 0 || rand() > params.packet_loss_prob
                            sent_successfully = true;
                            break;
                        end
                        % In a real system, a small delay would occur between retries
                    end
                    
                    if sent_successfully
                        % Broadcast bid
                        old_assignment = auction_data.assignment(j);
                        
                        % Track oscillation (robot changes) for this task
                        if any(old_assignment > 0) && any(old_assignment ~= i)
                            auction_data.task_oscillation_count(j) = auction_data.task_oscillation_count(j) + 1;
                        end
                        auction_data.task_last_robot(j) = i;
                        
                        auction_data.assignment(j) = i;
                        
                        % Dynamic price increment with multiple factors
                        effective_epsilon = base_epsilon;
                        
                        % Factor 1: Task oscillation history
                        if auction_data.task_oscillation_count(j) > 2
                            effective_epsilon = effective_epsilon * (1 + 0.15 * auction_data.task_oscillation_count(j));
                        end
                        
                        % Factor 2: Workload balancing
                        if workload_diff > 0
                            if workload_ratio > 1.2  % Robot has >20% more workload than minimum
                                effective_epsilon = effective_epsilon * 1.5;  % Increase price faster
                            elseif workload_ratio < 0.8  % Robot has <80% of maximum workload
                                effective_epsilon = effective_epsilon * 0.7;  % Increase price slower
                            end
                        end
                        
                        % Cap prices to prevent them from getting too high
                        max_price = 3.0 * max(params.alpha); % Maximum reasonable price
                        if auction_data.prices(j) + effective_epsilon > max_price
                            effective_epsilon = max(0, max_price - auction_data.prices(j));
                        end
                        
                        % Use the adjusted epsilon for price increment
                        auction_data.prices(j) = auction_data.prices(j) + effective_epsilon;
                        
                        new_assignments = true;
                        messages = messages + 1;
                        
                        % If this is the first assignment, record it for recovery analysis
                        if auction_data.initial_assignment(j) == 0
                            auction_data.initial_assignment(j) = i;
                        end
                        
                        bid_count = bid_count + 1;
                        if bid_count >= max_bids_per_iteration
                            break;  % Limit bids per iteration for more uniform communication
                        end
                    end
                end
            end
        end
        
        % Progressive price reduction for unassigned tasks
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
                    
                    auction_data.prices(j) = auction_data.prices(j) * reduction_factor;
                    
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
    
    % =========================================================================
    function [failures, auction_data] = detectFailures(auction_data, robots, params)
        % DETECTFAILURES Detects robot failures using multiple methods
        %
        % Parameters:
        %   auction_data - Auction data structure
        %   robots       - Array of robot structures
        %   params       - Algorithm parameters
        %
        % Returns:
        %   failures     - List of newly detected failed robot IDs
        %   auction_data - Updated auction data structure
        
        failures = [];
        
        % Don't detect failures during warm-up period to avoid false positives
        if ~isfield(params, 'warmup_iterations')
            params.warmup_iterations = 10; % Default warm-up period
        end
        
        if auction_data.current_time < params.warmup_iterations
            return;
        end
        
        % Set default heartbeat timeout if not specified
        if ~isfield(params, 'heartbeat_timeout')
            params.heartbeat_timeout = 5; % Default timeout in iterations
        end
        
        % Check heartbeats for each robot
        for i = 1:length(robots)
            % Skip already failed robots
            if isfield(robots, 'failed') && robots(i).failed
                continue;
            end
            
            % Calculate time since last heartbeat
            heartbeat_age = auction_data.current_time - auction_data.last_heartbeat(i);
            
            % For debugging - uncomment if needed
            % fprintf('Iteration %d: Robot %d heartbeat age: %.2f (threshold: %.2f)\n', ...
            %        auction_data.current_time, i, heartbeat_age, params.heartbeat_timeout);
            
            % Check if heartbeat timeout has occurred
            if heartbeat_age > params.heartbeat_timeout
                fprintf('Robot %d has failed at iteration %d (heartbeat timeout)\n', ...
                       i, auction_data.current_time);
                
                % Record the failure
                failures = [failures, i];
                
                % Mark the robot as failed if it has a 'failed' field
                if isfield(robots, 'failed')
                    robots(i).failed = true;
                end
                
                % Record failure time for recovery tracking
                auction_data.failed_robots(i) = 1;
                auction_data.failure_time(i) = auction_data.current_time;
            end
        end
        
        % Secondary detection: Check task progress (only if enabled)
        if isfield(params, 'enable_progress_detection') && params.enable_progress_detection
            for j = 1:length(auction_data.task_progress)
                % Only check assigned tasks that are in progress
                robot_id = auction_data.assignment(j);
                if robot_id > 0 && auction_data.completion_status(j) == 0
                    
                    % Calculate progress rate (assuming you have a progress field)
                    if isfield(auction_data, 'task_progress') && isfield(auction_data, 'last_progress_check')
                        current_progress = auction_data.task_progress(j);
                        last_progress = auction_data.last_progress_check(j);
                        time_delta = auction_data.current_time - auction_data.last_progress_time(j);
                        
                        if time_delta > 0
                            recent_progress = (current_progress - last_progress) / time_delta;
                            
                            % Check if progress is below minimum acceptable rate
                            if isfield(params, 'min_progress_rate') && any(recent_progress < params.min_progress_rate)
                                % Task is stalled, might indicate robot failure
                                if ~ismember(robot_id, failures) && ~robots(robot_id).failed
                                    fprintf('Robot %d failure detected from lack of progress on task %d\n', ...
                                           robot_id, j);
                                    failures = [failures, robot_id];
                                    
                                    % Mark the robot as failed
                                    if isfield(robots, 'failed')
                                        robots(robot_id).failed = true;
                                    end
                                    
                                    % Record failure time
                                    auction_data.failed_robots(robot_id) = 1;
                                    auction_data.failure_time(robot_id) = auction_data.current_time;
                                end
                            end
                        end
                        
                        % Update progress tracking variables
                        auction_data.last_progress_check(j) = current_progress;
                        auction_data.last_progress_time(j) = auction_data.current_time;
                    end
                end
            end
        end
        
        % Only return failures that haven't been detected already
        failures = setdiff(failures, find(auction_data.failed_robots));
    end
    
    % =========================================================================
    function [auction_data, recovery_status] = initiateRecovery(auction_data, robots, tasks, failed_robot_id)
        % INITIATERECOVERY Initiate recovery process after a robot failure
        %
        % Parameters:
        %   auction_data    - Auction data structure
        %   robots          - Array of robot structures
        %   tasks           - Array of task structures
        %   failed_robot_id - ID of the failed robot
        %
        % Returns:
        %   auction_data    - Updated auction data structure
        %   recovery_status - Status code (1=success, 0=partial, -1=failure)
        
        % Initialize return status
        recovery_status = 1; % 1 = success, 0 = partial success, -1 = cannot recover
        
        % Store current assignment state for recovery analysis
        auction_data.recovery_mode = true;
        auction_data.failure_assignment = auction_data.assignment;
        
        % Check if all robots are failed - critical edge case!
        all_failed = true;
        for i = 1:length(robots)
            if ~isfield(robots, 'failed') || ~robots(i).failed
                all_failed = false;
                break;
            end
        end
        
        if all_failed
            fprintf('WARNING: All robots have failed. Recovery not possible without intervention.\n');
            recovery_status = -1;
            
            % Option 1: Emergency restoration of one robot (uncomment if desired)
            % if length(robots) > 0
            %     fprintf('Emergency restoration of Robot 1 to enable recovery\n');
            %     robots(1).failed = false;
            %     auction_data.failed_robots(1) = 0;
            % end
            
            % Option 2: Just report the issue and return
            return;
        end
        
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
                    criticality_scores(i) = criticality_scores(i) + dependent_count * 2;
                end
                
                % 3. Add bonus for collaborative tasks
                if auction_data.collaborative_tasks(task_idx) == 1
                    criticality_scores(i) = criticality_scores(i) * 1.5;
                end
            end
            
            % Sort failed tasks by criticality (highest first)
            [~, critical_order] = sort(criticality_scores, 'descend');
            prioritized_failed_tasks = failed_tasks(critical_order);
            
            % Reset oscillation counters for these tasks to avoid bias
            auction_data.task_oscillation_count(failed_tasks) = 0;
            
            for i = 1:length(prioritized_failed_tasks)
                task_idx = prioritized_failed_tasks(i);
                
                % Mark as in recovery with priority level reflected in the price reset
                % Most critical tasks get bigger price reductions for faster reassignment
                priority_factor = 1 - (i-1)/length(prioritized_failed_tasks);
                auction_data.assignment(task_idx) = -1;  % -1 indicates "in recovery"
                
                % More aggressive price reset for high-priority tasks
                reset_factor = 0.3 * (1 + priority_factor);
                auction_data.prices(task_idx) = auction_data.prices(task_idx) * reset_factor;
            end
        end
        
        fprintf('Recovery initiated for robot %d. %d tasks need reassignment.\n', ...
                failed_robot_id, length(failed_tasks));
        
        return;
    end
    
    % =========================================================================
    function bid = calculateBid(robot_id, task_id, robot_workload, workload_ratio, workload_imbalance, auction_data, params, iter)
        % CALCULATEBID Calculate a robot's bid for a specific task
        %
        % Parameters:
        %   robot_id          - Robot ID
        %   task_id           - Task ID
        %   robot_workload    - Current workload of the robot
        %   workload_ratio    - Ratio of robot's workload to maximum workload
        %   workload_imbalance - Global workload imbalance measure
        %   auction_data      - Auction data structure
        %   params            - Algorithm parameters
        %   iter              - Current iteration number
        %
        % Returns:
        %   bid               - Calculated bid value
        
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
        workload_factor = robot_workload / 10 * (workload_ratio^1.8);
        
        % Add global balance consideration
        global_balance_factor = workload_imbalance * 0.5;
        
        % Simple energy factor
        energy_factor = distance * 0.1;
        
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
            adjusted_workload_alpha = adjusted_workload_alpha * 1.5;  % 50% increase in penalty
        end
        
        % Enhanced bonus for unassigned tasks with progressive scaling
        task_unassigned_bonus = 0;
        if auction_data.assignment(task_id) == 0
            unassigned_iter = auction_data.unassigned_iterations(task_id);
            
            % Exponential bonus scaling for persistent unassigned tasks
            if unassigned_iter > 25
                task_unassigned_bonus = min(6.0, 0.5 * exp(unassigned_iter/10));
            elseif unassigned_iter > 15
                task_unassigned_bonus = min(4.0, 0.4 * exp(unassigned_iter/12));
            elseif unassigned_iter > 5
                task_unassigned_bonus = min(3.0, 0.3 * unassigned_iter);
            end
        end
        
        % Add scaling factor for tasks that need to be assigned quickly
        iteration_factor = 0;
        if iter > 20 && auction_data.assignment(task_id) == 0
            iteration_factor = min(2.0, 0.05 * iter);
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
            progress_term = beta(1) * (1 - 0);  % Assuming no partial progress
            criticality_term = beta(2) * 0.5;   % Default criticality
            
            % Recovery bids favor robots with lower workload more strongly
            recovery_workload_factor = (1 - workload_ratio) * beta(1) * 0.8;
            
            % Add stronger global balance factor during recovery
            global_recovery_factor = workload_imbalance * beta(3) * 0.7;
            
            bid = alpha(1) * d_factor + ...
                  alpha(2) * c_factor + ...
                  alpha(3) * capability_match - ...
                  adjusted_workload_alpha * workload_factor - ...
                  alpha(5) * energy_factor + ...
                  progress_term + ...
                  criticality_term + ...
                  recovery_workload_factor + ...
                  global_recovery_factor + ...
                  global_balance_factor + ...
                  task_unassigned_bonus + ...
                  iteration_factor;
        else
            bid = alpha(1) * d_factor + ...
                  alpha(2) * c_factor + ...
                  alpha(3) * capability_match - ...
                  adjusted_workload_alpha * workload_factor - ...
                  alpha(5) * energy_factor + ...
                  global_balance_factor + ...
                  task_unassigned_bonus + ...
                  iteration_factor;
        end
        
        % Add small random noise to break ties (slightly increased to help prevent cycling)
        bid = double(bid) + 0.001 * rand();
    end
    
    % =========================================================================
    function [auction_data, sync_successful] = handleCollaborativeTasks(auction_data, robots, tasks, task_id, params)
        % HANDLECOLLABORATIVETASKS Special handling for collaborative tasks requiring multiple robots
        %
        % Parameters:
        %   auction_data - Auction data structure
        %   robots       - Array of robot structures
        %   tasks        - Array of task structures
        %   task_id      - ID of the collaborative task
        %   params       - Algorithm parameters
        %
        % Returns:
        %   auction_data    - Updated auction data structure
        %   sync_successful - Whether synchronization was successful
        
        % Default return value
        sync_successful = false;
        
        % Verify this is actually a collaborative task
        if ~auction_data.collaborative_tasks(task_id)
            return;
        end
        
        % For a two-robot system, we need both robots to be available
        if length(robots) < 2
            return;
        end
        
        % Skip if either robot has failed
        if (isfield(robots, 'failed') && (robots(1).failed || robots(2).failed))
            return;
        end
        
        % For simplicity, we refer to them as robot1 and robot2
        robot1 = 1;
        robot2 = 2;
        
        % Calculate workloads for both robots (previously missing)
        robot_workloads = zeros(1, length(robots));
        for i = 1:length(robots)
            if ~isfield(robots, 'failed') || ~robots(i).failed
                for j = 1:length(tasks)
                    if auction_data.assignment(j) == i
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
        end
        
        % Calculate workload ratio
        max_workload = max(robot_workloads);
        workload_ratios = ones(1, length(robots));
        if max_workload > 0
            workload_ratios = robot_workloads / max_workload;
        end
        
        % Calculate workload imbalance
        active_robots = [];
        for i = 1:length(robots)
            if ~isfield(robots, 'failed') || ~robots(i).failed
                active_robots = [active_robots, i];
            end
        end
        
        workload_imbalance = 0;
        if length(active_robots) >= 2
            active_workloads = robot_workloads(active_robots);
            min_workload = min(active_workloads);
            max_workload = max(active_workloads);
            workload_diff = max_workload - min_workload;
            
            if max_workload > 0
                workload_imbalance = workload_diff / max_workload;
            end
        end
        
        % Determine leader/follower roles
        % Calculate leader cost for each robot
        cost1 = calculateLeaderCost(robot1, task_id, robots, tasks, auction_data);
        cost2 = calculateLeaderCost(robot2, task_id, robots, tasks, auction_data);
        
        % Lower cost means better suited as leader
        if cost1 <= cost2
            leader = robot1;
            follower = robot2;
        else
            leader = robot2;
            follower = robot1;
        end
        
        % Calculate joint bid - using local variables now instead of undefined ones
        leader_bid = calculateBid(leader, task_id, robot_workloads(leader), workload_ratios(leader), workload_imbalance, auction_data, params, auction_data.utility_iter);
        follower_bid = calculateBid(follower, task_id, robot_workloads(follower), workload_ratios(follower), workload_imbalance, auction_data, params, auction_data.utility_iter);
        
        % Joint bid is the minimum of the two (limiting factor)
        joint_bid = min(leader_bid, follower_bid);
        
        % Utility is bid minus price
        joint_utility = joint_bid - auction_data.prices(task_id);
        
        % Only proceed if utility is positive
        if joint_utility <= 0
            return;
        end
        
        % Both robots must be unassigned or already assigned to this task
        if (auction_data.assignment(task_id) ~= leader && auction_data.assignment(task_id) ~= follower && auction_data.assignment(task_id) ~= 0)
            return;
        end
        
        % If we made it here, we can assign the task collaboratively
        auction_data.assignment(task_id) = -2;  % Special marker for collaborative tasks
        
        % We need to store which robot is leader and which is follower
        if ~isfield(auction_data, 'collaborative_leaders')
            auction_data.collaborative_leaders = zeros(length(tasks), 1);
            auction_data.collaborative_followers = zeros(length(tasks), 1);
        end
        
        auction_data.collaborative_leaders(task_id) = leader;
        auction_data.collaborative_followers(task_id) = follower;
        
        % Update price
        auction_data.prices(task_id) = auction_data.prices(task_id) + params.epsilon;
        
        sync_successful = true;
    end
    
    % =========================================================================
    function cost = calculateLeaderCost(robot_id, task_id, robots, tasks, auction_data)
        % CALCULATELEADERCOST Calculate the cost for a robot to be the leader in a collaborative task
        %
        % Parameters:
        %   robot_id     - Robot ID
        %   task_id      - Task ID
        %   robots       - Array of robot structures
        %   tasks        - Array of task structures
        %   auction_data - Auction data structure
        %
        % Returns:
        %   cost         - Calculated cost value (lower is better for leader)
        
        % Default cost - higher values are worse
        cost = 1000;
        
        try
            % Distance cost
            if isfield(robots, 'position') && isfield(tasks, 'position')
                distance = norm(robots(robot_id).position - tasks(task_id).position);
                distance_cost = distance;
            else
                distance_cost = abs(robot_id - task_id);
            end
            
            % Capability matching
            if isfield(robots, 'capabilities') && isfield(tasks, 'capabilities_required')
                robot_cap = robots(robot_id).capabilities;
                task_cap = tasks(task_id).capabilities_required;
                
                % Normalize vectors to unit length
                robot_cap_norm = robot_cap / norm(robot_cap);
                task_cap_norm = task_cap / norm(task_cap);
                
                % Compute cosine similarity (normalized dot product)
                capability_match = dot(robot_cap_norm, task_cap_norm);
                capability_cost = 1 - capability_match;
            else
                capability_cost = 0.2;
            end
            
            % Workload cost
            workload = 0;
            for j = 1:length(tasks)
                if auction_data.assignment(j) == robot_id
                    if isfield(tasks, 'execution_time')
                        workload = workload + tasks(j).execution_time;
                    else
                        workload = workload + 1;
                    end
                end
            end
            workload_cost = workload / 10;
            
            % Combine costs with weights
            cost = 0.4 * distance_cost + 0.4 * capability_cost + 0.2 * workload_cost;
        catch
            % If there's an error, return the default high cost
            cost = 1000;
        end
    end
    
    % =========================================================================
    function analyzeTaskAllocation(auction_data, tasks)
        % ANALYZETASKALLOCATION Analyzes the current task allocation
        %
        % Parameters:
        %   auction_data - Auction data structure
        %   tasks        - Array of task structures
        
        % Group tasks by assigned robot
        num_robots = 10;  % Assume up to 10 robots for simplicity
        robot_tasks = cell(num_robots, 1);
        
        for j = 1:length(tasks)
            r = auction_data.assignment(j);
            if r > 0 && r <= num_robots
                if isempty(robot_tasks{r})
                    robot_tasks{r} = j;
                else
                    robot_tasks{r} = [robot_tasks{r}, j];
                end
            end
        end
        
        % Print task allocation
        fprintf('Task Allocation: ');
        for i = 1:num_robots
            if ~isempty(robot_tasks{i})
                fprintf('R%d: [', i);
                fprintf('%d ', robot_tasks{i});
                fprintf('] ');
            end
        end
        fprintf('\n');
        
        % Calculate workload balance
        workload = zeros(num_robots, 1);
        for j = 1:length(tasks)
            r = auction_data.assignment(j);
            if r > 0 && r <= num_robots
                if isfield(tasks, 'execution_time')
                    workload(r) = workload(r) + tasks(j).execution_time;
                else
                    workload(r) = workload(r) + 1;  % Default execution time
                end
            end
        end
        
        % Calculate workload ratio (max to min)
        active_robots = find(workload > 0);
        if length(active_robots) >= 2
            workload_ratio = max(workload(active_robots)) / min(workload(active_robots));
        else
            workload_ratio = 0;
        end
        
        fprintf('Workload: ');
        for i = 1:min(2, num_robots)  % Show workload for at least first two robots
            fprintf('R%d=%.2f, ', i, workload(i));
        end
        fprintf('Ratio=%.2f\n', workload_ratio);
        
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
        has_oscillations = false;
        for j = 1:length(tasks)
            if auction_data.task_oscillation_count(j) > 0
                fprintf('T%d:%d ', j, auction_data.task_oscillation_count(j));
                has_oscillations = true;
            end
        end
        if ~has_oscillations
            fprintf('None');
        end
        fprintf('\n');
    end
    
    % =========================================================================
    function analyzeBidDistribution(auction_data, robots, tasks)
        % ANALYZEBIDDISTRIBUTION Analyzes the current bid distribution
        %
        % Parameters:
        %   auction_data - Auction data structure
        %   robots       - Array of robot structures
        %   tasks        - Array of task structures
        
        fprintf('\n--- Bid Analysis ---\n');
        for i = 1:length(robots)
            if isfield(robots, 'failed') && robots(i).failed
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
end

% =========================================================================
function [metrics, converged] = runEnhancedAuctionSimulation(params, env, robots, tasks, visualize)
    % RUNENHANCEDAUCTIONSIMULATION Run a complete enhanced auction algorithm simulation
    %
    % Parameters:
    %   params    - Algorithm parameters structure
    %   env       - Environment structure
    %   robots    - Array of robot structures
    %   tasks     - Array of task structures
    %   visualize - Boolean flag for visualization
    %
    % Returns:
    %   metrics   - Performance metrics structure
    %   converged - Boolean indicating convergence

    % Other utility functions needed
    robot_util = robot_utils();
    task_util = task_utils();
    env_utils = environment_utils();
    
    % Display robot capabilities if visualization is enabled
    if visualize
        fprintf('Robot capabilities:\n');
        for i = 1:length(robots)
            fprintf('Robot %d: [%.2f %.2f %.2f %.2f %.2f]\n', i, robots(i).capabilities);
        end
        
        % Count collaborative tasks
        num_collaborative = 0;
        if isfield(tasks, 'collaborative')
            for i = 1:length(tasks)
                if any(tasks(i).collaborative == true) || any(tasks(i).collaborative == 1)
                        num_collaborative = num_collaborative + 1;
                end
            end
        end
        
        fprintf('Starting enhanced auction algorithm simulation...\n');
        fprintf('Number of robots: %d, Number of tasks: %d\n', length(robots), length(tasks));
        fprintf('Number of collaborative tasks: %d\n', num_collaborative);
        
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
            if isfield(tasks, 'collaborative') && (any(tasks(i).collaborative == true) || any(tasks(i).collaborative == 1))
                collab_str = ' (Collaborative)';
            end
            
            fprintf('Task %d: position=[%.1f, %.1f], time=%.1f, prereqs=[%s]%s\n', ...
                i, tasks(i).position(1), tasks(i).position(2), tasks(i).execution_time, prereq_str, collab_str);
        end
    end
    
    % Initialize heartbeat parameters if not provided
    if ~isfield(params, 'warmup_iterations')
        params.warmup_iterations = 10;
    end
    if ~isfield(params, 'heartbeat_timeout')
        params.heartbeat_timeout = 5;
    end
    if ~isfield(params, 'enable_progress_detection')
        params.enable_progress_detection = false;
    end
    
    % Create data structures for the auction algorithm
    utils = enhanced_auction_utils();
    auction_data = utils.initializeAuctionData(tasks, robots);
    
    % Visualization setup if enabled
    if visualize
        figure('Name', 'Enhanced Distributed Auction Algorithm Simulation', 'Position', [100, 100, 1200, 800]);
        subplot(2, 3, [1, 4]);
        env_utils.visualizeEnvironment(env, robots, tasks, auction_data);
        title('Environment');
    end
    
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
    metrics.failure_time = 0;  % Initialize failure_time field
    metrics.makespan = 0;
    metrics.makespan_before_failure = 0;
    
    % Main simulation loop
    max_iterations = 150;
    converged = false;
    
    % Track iterations without changes for convergence detection
    unchanged_iterations = 0;
    stable_workload_iterations = 0;
    
    % Find initially available tasks (only those with no prerequisites)
    available_tasks = task_util.findAvailableTasks(tasks, []);
    
    % Main loop
    for iter = 1:max_iterations
        metrics.iterations = iter;
        
        % Check for programmed robot failure (from parameters)
        if iter == params.failure_time && ~isempty(params.failed_robot)
            if visualize
                fprintf('Robot %d has failed at iteration %d (programmed)\n', params.failed_robot, iter);
            end
            if isfield(robots, 'failed')
                robots(params.failed_robot).failed = true;
            end
            
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
            
            % Initiate recovery process for programmed failure
            [auction_data, recovery_status] = utils.initiateRecovery(auction_data, robots, tasks, params.failed_robot);
            metrics.failure_time = iter;
        end
        
        % Check for heartbeat failures
        [failures, auction_data] = utils.detectFailures(auction_data, robots, params);
        
        % Handle any detected failures
        for i = 1:length(failures)
            [auction_data, recovery_status] = utils.initiateRecovery(auction_data, robots, tasks, failures(i));
            if visualize
                fprintf('Recovery initiated for robot %d failures\n', failures(i));
                
                % Handle failed recovery
                if recovery_status == -1
                    fprintf('WARNING: Recovery failed due to all robots being unavailable.\n');
                end
            end
        end
        
        % Simulate one step of the distributed auction algorithm
        [auction_data, new_assignments, messages] = utils.enhancedDistributedAuctionStep(auction_data, robots, tasks, available_tasks, params, iter);
        metrics.messages = metrics.messages + messages;
        
        % Calculate makespan for current assignment
        try
            current_makespan = robot_util.calculateMakespan(auction_data.assignment, tasks, robots);
            metrics.makespan = current_makespan;
        catch e
            if visualize
                fprintf('Warning: %s\n', e.message);
            end
        end
        
        % Update visualization if enabled
        if visualize
            subplot(2, 3, [1, 4]);
            env_utils.visualizeEnvironment(env, robots, tasks, auction_data);
            title(sprintf('Environment (Iteration %d)', iter));
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
        
        % Check if any new tasks become available due to completed prerequisites
        completed_tasks = find(auction_data.completion_status == 1);
        available_tasks = task_util.findAvailableTasks(tasks, completed_tasks);
        
        % Track workload stability for convergence detection
        current_workloads = zeros(length(robots), 1);
        for r = 1:length(robots)
            if ~(isfield(robots, 'failed') && robots(r).failed)
                % Calculate workload for this robot
                workload = 0;
                for j = 1:length(tasks)
                    if auction_data.assignment(j) == r
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
        
        % Check for workload stability
        if iter > 5
            % Calculate workload deviation from moving average
            % This could be enhanced in a full implementation
            workload_stable = true;
            
            if workload_stable
                stable_workload_iterations = stable_workload_iterations + 1;
            else
                stable_workload_iterations = 0;
            end
        end
        
        % Reset mechanism for stalled situations
        if mod(iter, 50) == 0 && iter > 100 && any(auction_data.assignment == 0)
            unassigned_count = sum(auction_data.assignment == 0);
            if visualize
                fprintf('Reset triggered at iteration %d with %d unassigned tasks\n', iter, unassigned_count);
            end
            
            % Reset prices for unassigned tasks to zero to restart bidding
            unassigned_tasks = find(auction_data.assignment == 0);
            auction_data.prices(unassigned_tasks) = 0;
            
            % Reset oscillation counts to allow fresh bidding
            auction_data.task_oscillation_count(unassigned_tasks) = 0;
            
            % Reset convergence tracking
            unchanged_iterations = 0;
            stable_workload_iterations = 0;
        end
        
        % Scale required stable iterations with problem complexity
        min_required_stable = max(10, ceil(log(length(tasks)) * 5));
        required_workload_stability = max(5, ceil(log(length(tasks)) * 3));
        
        % Diagnostics every 10 iterations
        if visualize && mod(iter, 10) == 0
            fprintf('Iteration %d: ', iter);
            utils.analyzeTaskAllocation(auction_data, tasks);
        end
        
        % Update recovery time if in recovery mode
        if metrics.failure_time > 0 && metrics.recovery_time == 0
            % Find tasks that were assigned to failed robots
            failed_robot_tasks = [];
            for r = 1:length(robots)
                if isfield(robots, 'failed') && robots(r).failed
                    % Find tasks that were assigned to this robot at failure time
                    if isfield(auction_data, 'failure_assignment') && ~isempty(auction_data.failure_assignment)
                        robot_tasks = find(auction_data.failure_assignment == r);
                        failed_robot_tasks = [failed_robot_tasks, robot_tasks];
                    end
                end
            end
            
            % Check if all failed robot tasks have been reassigned
            all_reassigned = true;
            for j = 1:length(failed_robot_tasks)
                task_id = failed_robot_tasks(j);
                % Task should be assigned to a non-failed robot or completed
                if auction_data.assignment(task_id) <= 0 || ...
                   (auction_data.assignment(task_id) > 0 && ...
                    isfield(robots, 'failed') && ...
                    robots(auction_data.assignment(task_id)).failed)
                    all_reassigned = false;
                    break;
                end
            end
            
            if all_reassigned
                metrics.recovery_time = iter - metrics.failure_time;
                if visualize
                    fprintf('Recovery completed after %d iterations\n', metrics.recovery_time);
                end
            end
        end
        
        % Check for convergence - stable assignments for sufficient iterations
        if iter > 20 && unchanged_iterations >= min_required_stable && stable_workload_iterations >= required_workload_stability
            unassigned_count = sum(auction_data.assignment == 0);
            
            % Allow convergence even with unassigned tasks if stable for longer
            if unassigned_count == 0 || (unchanged_iterations >= 2*min_required_stable)
                converged = true;
                if visualize
                    fprintf('Auction algorithm converged after %d iterations\n', iter);
                end
                break;
            end
        end
        
        % Pause for visualization
        if visualize
            pause(0.01);
        end
    end
    
    % Trim history matrices to actual size
    metrics.price_history = metrics.price_history(:, 1:iter);
    metrics.assignment_history = metrics.assignment_history(:, 1:iter);
    
    % Plot results if visualization is enabled
    if visualize
        subplot(2, 3, 2);
        env_utils.plotTaskPrices(metrics.price_history);
        title('Task Prices Over Time');
        
        subplot(2, 3, 3);
        env_utils.plotAssignments(metrics.assignment_history, length(robots));
        title('Task Assignments Over Time');
        
        subplot(2, 3, 5);
        env_utils.plotConvergence(metrics.convergence_history);
        title('Convergence Metric');
        
        subplot(2, 3, 6);
        env_utils.plotWorkload(metrics.assignment_history(:, end), tasks, robots);
        title('Final Workload Distribution');
    end
    
    % Calculate optimality gap
    try
        optimal_makespan = robot_util.calculateOptimalMakespan(tasks, robots);
        metrics.optimality_gap = abs(metrics.makespan - optimal_makespan);
        metrics.theoretical_gap_bound = 2 * params.epsilon;
    catch e
        if visualize
            fprintf('Warning: Error calculating optimality gap: %s\n', e.message);
        end
        metrics.optimality_gap = NaN;
        metrics.theoretical_gap_bound = NaN;
    end
    
    % Display final metrics if visualization is enabled
    if visualize
        fprintf('\n--- Final Performance Metrics ---\n');
        fprintf('Iterations to converge: %d\n', metrics.iterations);
        fprintf('Theoretical bound: O(K² · bₘₐₓ/ε) = O(%d)\n', numel(tasks)^2 * max(params.alpha) / params.epsilon);
        fprintf('Messages exchanged: %d\n', metrics.messages);
        fprintf('Achieved makespan: %.2f\n', metrics.makespan);
        
        if ~isnan(metrics.optimality_gap)
            fprintf('Optimal makespan: %.2f\n', optimal_makespan);
            fprintf('Optimality gap: %.2f\n', metrics.optimality_gap);
            fprintf('Theoretical gap bound (2ε): %.2f\n', metrics.theoretical_gap_bound);
        end
        
        if metrics.failure_time > 0
            fprintf('Recovery time: %d iterations\n', metrics.recovery_time);
            
            % Count tasks assigned to failed robots
            failed_task_count = 0;
            for r = 1:length(robots)
                if isfield(robots, 'failed') && robots(r).failed
                    if isfield(auction_data, 'failure_assignment') && ~isempty(auction_data.failure_assignment)
                        failed_task_count = failed_task_count + sum(auction_data.failure_assignment == r);
                    end
                end
            end
            
            fprintf('Tasks reassigned after failure: %d\n', failed_task_count);
            fprintf('Theoretical recovery bound: O(|Tᶠ|) + O(bₘₐₓ/ε) ≈ %d\n', ...
                failed_task_count + round(max(params.alpha) / params.epsilon));
            
            if metrics.makespan_before_failure > 0
                makespan_degradation = metrics.makespan - metrics.makespan_before_failure;
                fprintf('Makespan before failure: %.2f\n', metrics.makespan_before_failure);
                fprintf('Makespan degradation: %.2f (%.2f%%)\n', makespan_degradation, ...
                        makespan_degradation / metrics.makespan_before_failure * 100);
            end
        end
        
        % Detailed final analysis
        fprintf('\n--- Final Task Allocation Analysis ---\n');
        utils.analyzeTaskAllocation(auction_data, tasks);
        utils.analyzeBidDistribution(auction_data, robots, tasks);
        
        % Analyze collaborative tasks
        if isfield(tasks, 'collaborative')
            collaborative_tasks = [];
            for i = 1:length(tasks)
                if any(tasks(i).collaborative == true) || any(tasks(i).collaborative == 1)
                    collaborative_tasks = [collaborative_tasks, i];
                end
            end
            
            if ~isempty(collaborative_tasks)
                fprintf('\n--- Collaborative Task Analysis ---\n');
                fprintf('Collaborative tasks: ');
                for i = collaborative_tasks
                    status = 'Unassigned';
                    if auction_data.assignment(i) == -2
                        status = 'Assigned collaboratively';
                    elseif auction_data.assignment(i) > 0
                        status = sprintf('Assigned to Robot %d only', auction_data.assignment(i));
                    end
                    fprintf('Task %d: %s\n', i, status);
                end
            end
        end
    end
end