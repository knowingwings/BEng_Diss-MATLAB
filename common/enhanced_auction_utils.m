function utils = enhanced_auction_utils()
    % ENHANCED_AUCTION_UTILS - Returns function handles for enhanced auction algorithm
    % This file extends the original auction_utils with additional functionality
    % based on the mathematical foundations document
    
    % Get base utilities
    base_utils = auction_utils();
    
    % Create enhanced utilities structure
    utils = base_utils;
    
    % Override/add enhanced functions
    utils.initializeAuctionData = @enhancedInitializeAuctionData;
    utils.distributedAuctionStep = @enhancedDistributedAuctionStep;
    utils.calculateEnhancedBid = @calculateEnhancedBid;
    utils.detectFailures = @detectFailures;
    utils.initiateRecovery = @enhancedRecovery;
    utils.manageTaskDependencies = @manageTaskDependencies;
    utils.handleCollaborativeTasks = @handleCollaborativeTasks;
    utils.analyzeBidConvergence = @analyzeBidConvergence;
    utils.runTimeWeightedConsensus = @runTimeWeightedConsensus;
    utils.runAuctionSimulation = @runEnhancedAuctionSimulation;
    
    % Enhanced auction data initialization
    function auction_data = enhancedInitializeAuctionData(tasks, robots)
        % Start with base initialization
        auction_data = base_utils.initializeAuctionData(tasks, robots);
        
        % Add additional fields for enhanced algorithms
        num_robots = length(robots);
        num_tasks = length(tasks);
        
        % Consensus protocol related fields
        auction_data.state_vectors = cell(num_robots, 1);
        for i = 1:num_robots
            % Format: [robot positions, task prices, task assignments, task completion]
            auction_data.state_vectors{i} = zeros(2*num_robots + 2*num_tasks, 1);
            
            % Initialize with known information
            % Robot positions
            for r = 1:num_robots
                auction_data.state_vectors{i}(2*(r-1)+1:2*r) = robots(r).position;
            end
            
            % Task prices
            auction_data.state_vectors{i}(2*num_robots+1:2*num_robots+num_tasks) = zeros(num_tasks, 1);
            
            % Task assignments
            auction_data.state_vectors{i}(2*num_robots+num_tasks+1:2*num_robots+2*num_tasks) = zeros(num_tasks, 1);
        end
        
        auction_data.consensus_errors = [];
        auction_data.last_update_time = zeros(num_robots, num_robots);
        
        % Communication model related fields
        auction_data.comm_delay = zeros(num_robots, num_robots);
        auction_data.packet_loss_prob = zeros(num_robots, num_robots);
        auction_data.comm_range = Inf * ones(num_robots, 1);
        
        % Failure detection related fields
        auction_data.heartbeat_timestamps = zeros(num_robots, 1);
        auction_data.progress_history = zeros(num_tasks, 10);  % Store last 10 progress values
        auction_data.progress_rate = zeros(num_tasks, 1);
        
        % Task dependency related fields
        auction_data.dependency_matrix = zeros(num_tasks, num_tasks);
        for i = 1:num_tasks
            if isfield(tasks, 'prerequisites') && ~isempty(tasks(i).prerequisites)
                auction_data.dependency_matrix(tasks(i).prerequisites, i) = 1;
            end
        end
        auction_data.critical_path = [];
        
        % Collaborative task related fields
        auction_data.collaborative_tasks = false(num_tasks, 1);
        auction_data.leader_assignment = zeros(num_tasks, 1);
        auction_data.sync_status = zeros(num_tasks, 1);
        auction_data.sync_start_time = zeros(num_tasks, 1);
        auction_data.execution_start_time = zeros(num_tasks, 1);
        
        % Theoretical performance analysis fields
        auction_data.utility_iter = 1;
        
        return;
    end
    
    % Enhanced distributed auction step with consensus and collaboration
    function [auction_data, new_assignments, messages] = enhancedDistributedAuctionStep(auction_data, robots, tasks, available_tasks, params)
        % Run time-weighted consensus to maintain consistent global view
        auction_data = runTimeWeightedConsensus(auction_data, robots, params);
        
        % Run the base auction step with enhanced bid calculation
        [auction_data, new_assignments, messages] = base_utils.distributedAuctionStep(auction_data, robots, tasks, available_tasks, params);
        
        % Handle collaborative tasks if any
        if any(auction_data.collaborative_tasks)
            for j = find(auction_data.collaborative_tasks')
                if auction_data.assignment(j) > 0 
                    [auction_data, sync_successful] = handleCollaborativeTasks(auction_data, robots, j, params);
                end
            end
        end
        
        % Update task dependencies
        completed_tasks = find(auction_data.completion_status == 1);
        [auction_data, available_tasks] = manageTaskDependencies(auction_data, tasks, completed_tasks);
        
        return;
    end
    
    % Enhanced bid calculation based on mathematical foundations
    function bid = calculateEnhancedBid(robot_id, task_id, robot_workload, workload_ratio, auction_data, params, tasks, robots)
        % Extract base factors
        distance_factor = 0;
        config_factor = 0;
        capability_match = 0;
        
        % Try to extract position data for real distance calculation
        try
            distance = norm(robots(robot_id).position - tasks(task_id).position);
            distance_factor = 1 / (1 + distance);  % Normalize with soft max
        catch
            % If positional data unavailable
            distance_factor = 0.5;  % Default value
        end
        
        % Try to calculate configuration transition cost
        try
            if isfield(robots, 'configuration')
                current_config = robots(robot_id).configuration;
                target_config = getTargetConfiguration(robots(robot_id), tasks(task_id));
                config_factor = 1 / (1 + norm(current_config - target_config));
            else
                config_factor = 0.5; % Default
            end
        catch
            % If configuration data unavailable
            config_factor = 0.5;  % Default value
        end
        
        % Calculate capability match score
        try
            if isfield(robots, 'capabilities') && isfield(tasks, 'capabilities_required')
                robot_capabilities = robots(robot_id).capabilities;
                task_requirements = tasks(task_id).capabilities_required;
                
                % Normalize vectors
                robot_cap_norm = robot_capabilities / norm(robot_capabilities);
                task_cap_norm = task_requirements / norm(task_requirements);
                
                % Compute cosine similarity (normalized dot product)
                capability_match = dot(robot_cap_norm, task_cap_norm);
            else
                capability_match = 0.5; % Default
            end
        catch
            % If capability data unavailable
            capability_match = 0.5;  % Default value
        end
        
        % Calculate workload factor (penalize overloaded robots)
        workload_factor = robot_workload * (workload_ratio^1.8);
        
        % Calculate energy consumption estimate
        try
            energy_factor = distance * 0.1;
        catch
            energy_factor = 0.3;  % Default value
        end
        
        % Get weights
        alpha = params.alpha;
        
        % Calculate bid
        bid = alpha(1) * distance_factor + ...
              alpha(2) * config_factor + ...
              alpha(3) * capability_match - ...
              alpha(4) * workload_factor - ...
              alpha(5) * energy_factor;
        
        % Special handling for collaborative tasks
        if isfield(auction_data, 'collaborative_tasks') && ...
           length(auction_data.collaborative_tasks) >= task_id && ...
           auction_data.collaborative_tasks(task_id)
            % Apply collaborative task penalties/bonuses
            % Add synchronization cost factor
            sync_factor = 0.2;  % Default value
            bid = bid - alpha(4) * sync_factor;
        end
        
        % Handle tasks on critical path - boost priority
        if isfield(auction_data, 'critical_path') && ...
           ismember(task_id, auction_data.critical_path)
            critical_path_bonus = 0.5;
            bid = bid + critical_path_bonus;
        end
        
        % Handle task dependencies - prioritize tasks that enable many others
        if isfield(auction_data, 'dependency_matrix') && ...
           size(auction_data.dependency_matrix, 1) >= task_id
            dependency_count = sum(auction_data.dependency_matrix(task_id, :));
            if dependency_count > 0
                dependency_bonus = 0.1 * dependency_count;
                bid = bid + dependency_bonus;
            end
        end
        
        % Add small random noise to break ties
        bid = bid + 0.001 * rand();
        
        return;
    end
    
    % Failure detection based on mathematical foundations
    function [failures, auction_data] = detectFailures(auction_data, robots, params)
        failures = false(size(robots));
        num_robots = length(robots);
        
        % Get current time from params
        if isfield(params, 'current_time')
            current_time = params.current_time;
        else
            current_time = auction_data.utility_iter;
        end
        
        % 1. Explicit detection via heartbeats
        for i = 1:num_robots
            if ~robots(i).failed
                % Update heartbeat for active robots
                auction_data.heartbeat_timestamps(i) = current_time;
            end
        end
        
        % Check for missing heartbeats
        if isfield(params, 'heartbeat_interval') && isfield(params, 'missed_heartbeat_threshold')
            heartbeat_threshold = params.heartbeat_interval * params.missed_heartbeat_threshold;
            for i = 1:num_robots
                if ~robots(i).failed && (current_time - auction_data.heartbeat_timestamps(i) > heartbeat_threshold)
                    % No heartbeat received for too long
                    failures(i) = true;
                end
            end
        end
        
        % 2. Implicit detection via task progress
        num_tasks = length(auction_data.assignment);
        for i = 1:num_robots
            if robots(i).failed || failures(i)
                continue;  % Skip already failed robots
            end
            
            % Check tasks assigned to this robot
            robot_tasks = find(auction_data.assignment == i);
            if isempty(robot_tasks)
                continue;  % No tasks assigned
            end
            
            % Get progress history for these tasks
            progress_stalled = false;
            for j = robot_tasks
                if j > size(auction_data.progress_history, 1)
                    continue;
                end
                
                % Shift history and add new progress
                auction_data.progress_history(j, 1:end-1) = auction_data.progress_history(j, 2:end);
                auction_data.progress_history(j, end) = auction_data.completion_status(j);
                
                % Calculate progress rate
                if size(auction_data.progress_history, 2) >= 3
                    recent_progress = auction_data.progress_history(j, end) - auction_data.progress_history(j, end-3);
                    auction_data.progress_rate(j) = recent_progress;
                    
                    % Check if progress has stalled
                    if isfield(params, 'min_progress_rate') && recent_progress < params.min_progress_rate
                        progress_stalled = true;
                    end
                end
            end
            
            % If progress has stalled on all tasks, suspect failure
            if progress_stalled
                failures(i) = true;
            end
        end
        
        % 3. Scheduled failures for simulation
        if isfield(params, 'failure_time') && isfield(params, 'failed_robot')
            if params.failure_time == current_time && ~isempty(params.failed_robot)
                failures(params.failed_robot) = true;
            end
        end
        
        return;
    end
    
    % Enhanced recovery mechanism based on mathematical foundations
    function [auction_data, recovered] = enhancedRecovery(auction_data, robots, tasks, failed_robot_id, params)
        % Start with the base recovery
        auction_data = base_utils.initiateRecovery(auction_data, robots, tasks, failed_robot_id);
        
        % Get tasks assigned to the failed robot
        failed_tasks = find(auction_data.failure_assignment == failed_robot_id);
        num_failed_tasks = length(failed_tasks);
        
        % Calculate theoretical recovery bound
        b_max = max(params.alpha);
        epsilon = params.epsilon;
        auction_data.recovery_bound = num_failed_tasks + round(b_max / epsilon);
        
        % Enhanced recovery prioritization
        if ~isempty(failed_tasks)
            % Calculate criticality for each task
            criticality_scores = zeros(num_failed_tasks, 1);
            
            for i = 1:num_failed_tasks
                task_id = failed_tasks(i);
                
                % 1. Factor: Execution time
                if isfield(tasks, 'execution_time')
                    criticality_scores(i) = tasks(task_id).execution_time;
                else
                    criticality_scores(i) = 1;
                end
                
                % 2. Factor: Dependency count (how many tasks depend on this)
                if isfield(auction_data, 'dependency_matrix') && ...
                   size(auction_data.dependency_matrix, 1) >= task_id
                    dependency_count = sum(auction_data.dependency_matrix(task_id, :));
                    criticality_scores(i) = criticality_scores(i) + dependency_count * 2;
                end
                
                % 3. Factor: Critical path membership
                if isfield(auction_data, 'critical_path') && ...
                   ismember(task_id, auction_data.critical_path)
                    criticality_scores(i) = criticality_scores(i) * 1.5;
                end
                
                % 4. Factor: Completion status
                completion = auction_data.completion_status(task_id);
                criticality_scores(i) = criticality_scores(i) * (1 - completion) * 0.5;
            end
            
            % Sort by criticality (highest first)
            [~, critical_order] = sort(criticality_scores, 'descend');
            prioritized_tasks = failed_tasks(critical_order);
            
            % Mark them for enhanced recovery auction
            auction_data.recovery_tasks = prioritized_tasks;
            auction_data.recovery_criticality = criticality_scores(critical_order);
            
            % Reset oscillation counts and prices based on priority
            for i = 1:length(prioritized_tasks)
                task_id = prioritized_tasks(i);
                priority_factor = 1 - (i-1)/length(prioritized_tasks);
                
                % Aggressive price reset for high-priority tasks
                reset_factor = 0.3 * (1 + priority_factor);
                auction_data.prices(task_id) = auction_data.prices(task_id) * reset_factor;
                
                % Clear oscillation history
                auction_data.task_oscillation_count(task_id) = 0;
            end
        end
        
        recovered = (num_failed_tasks > 0);
        return;
    end
    
    % Task dependency management based on mathematical foundations
    function [auction_data, available_tasks] = manageTaskDependencies(auction_data, tasks, completed_tasks)
        num_tasks = length(tasks);
        
        % Update dependency matrix if needed
        if ~isfield(auction_data, 'dependency_matrix') || ...
           any(size(auction_data.dependency_matrix) ~= [num_tasks, num_tasks])
            auction_data.dependency_matrix = zeros(num_tasks, num_tasks);
            for i = 1:num_tasks
                if isfield(tasks, 'prerequisites') && ~isempty(tasks(i).prerequisites)
                    auction_data.dependency_matrix(tasks(i).prerequisites, i) = 1;
                end
            end
        end
        
        % Calculate earliest start times (forward pass)
        earliest_start = zeros(num_tasks, 1);
        earliest_finish = zeros(num_tasks, 1);
        
        % Topological sort to respect dependencies
        visited = false(num_tasks, 1);
        topo_order = [];
        
        for i = 1:num_tasks
            if ~visited(i)
                [visited, topo_order] = topologicalSort(i, visited, topo_order, auction_data.dependency_matrix);
            end
        end
        
        % Calculate earliest times based on topological order
        for i = topo_order
            % Find prerequisites
            prereqs = find(auction_data.dependency_matrix(:, i));
            
            if ~isempty(prereqs)
                earliest_start(i) = max(earliest_finish(prereqs));
            end
            
            % Calculate finish time
            if isfield(tasks, 'execution_time')
                earliest_finish(i) = earliest_start(i) + tasks(i).execution_time;
            else
                earliest_finish(i) = earliest_start(i) + 1;  % Default
            end
        end
        
        % Calculate latest start times (backward pass)
        latest_finish = max(earliest_finish) * ones(num_tasks, 1);
        latest_start = zeros(num_tasks, 1);
        
        % Reverse topological order for backward pass
        for i = fliplr(topo_order)
            % Find tasks that depend on this one
            dependents = find(auction_data.dependency_matrix(i, :));
            
            if ~isempty(dependents)
                latest_finish(i) = min(latest_start(dependents));
            end
            
            % Calculate latest start
            if isfield(tasks, 'execution_time')
                latest_start(i) = latest_finish(i) - tasks(i).execution_time;
            else
                latest_start(i) = latest_finish(i) - 1;  % Default
            end
        end
        
        % Calculate slack and identify critical path
        slack = latest_start - earliest_start;
        auction_data.critical_path = find(slack < 0.001);  % Near-zero slack
        
        % Find available tasks
        available_tasks = [];
        for i = 1:num_tasks
            if auction_data.completion_status(i) == 0  % Not completed
                % Check if all prerequisites are completed
                prereqs = find(auction_data.dependency_matrix(:, i));
                if all(ismember(prereqs, completed_tasks))
                    available_tasks = [available_tasks, i];
                end
            end
        end
        
        return;
    end
    
    % Helper function for topological sort
    function [visited, topo_order] = topologicalSort(node, visited, topo_order, dep_matrix)
        visited(node) = true;
        
        % Visit all dependencies first
        for i = find(dep_matrix(node, :))
            if ~visited(i)
                [visited, topo_order] = topologicalSort(i, visited, topo_order, dep_matrix);
            end
        end
        
        % Add current node after all its dependencies
        topo_order = [topo_order, node];
    end
    
    % Collaborative task handling based on mathematical foundations
    function [auction_data, sync_successful] = handleCollaborativeTasks(auction_data, robots, task_id, params)
        sync_successful = false;
        
        if ~auction_data.collaborative_tasks(task_id)
            % Not a collaborative task
            sync_successful = true;
            return;
        end
        
        % Get the two robots
        robot1 = 1;
        robot2 = 2;
        
        % Skip if either robot has failed
        if robots(robot1).failed || robots(robot2).failed
            return;
        end
        
        % Get current time
        if isfield(params, 'current_time')
            current_time = params.current_time;
        else
            current_time = auction_data.utility_iter;
        end
        
        % Determine leader based on bid values or capability
        if auction_data.leader_assignment(task_id) == 0
            % Calculate leader cost
            cost1 = calculateLeaderCost(robot1, task_id, robots, tasks, auction_data);
            cost2 = calculateLeaderCost(robot2, task_id, robots, tasks, auction_data);
            
            if cost1 <= cost2
                auction_data.leader_assignment(task_id) = robot1;
            else
                auction_data.leader_assignment(task_id) = robot2;
            end
        end
        
        leader = auction_data.leader_assignment(task_id);
        follower = 3 - leader;  % Other robot (either 1 or 2)
        
        % Implement the three-step synchronization protocol
        switch auction_data.sync_status(task_id)
            case 0  % Initial state
                % Leader sends notification
                auction_data.sync_status(task_id) = 1;
                auction_data.sync_start_time(task_id) = current_time;
                
            case 1  % Leader has sent notification
                % Check if follower can acknowledge
                follower_ready = isRobotReadyForTask(follower, task_id, robots, tasks, params);
                
                if follower_ready
                    auction_data.sync_status(task_id) = 2;
                end
                
            case 2  % Follower has acknowledged
                % Joint execution can proceed
                auction_data.sync_status(task_id) = 3;
                auction_data.execution_start_time(task_id) = current_time;
                
            case 3  % Execution in progress
                % Calculate execution progress
                if isfield(tasks, 'execution_time')
                    task_duration = tasks(task_id).execution_time;
                else
                    task_duration = 5; % Default duration
                end
                
                elapsed_time = current_time - auction_data.execution_start_time(task_id);
                progress = min(1.0, elapsed_time / task_duration);
                
                % Check if execution is complete
                if progress >= 1.0
                    auction_data.sync_status(task_id) = 4; % Completed
                    auction_data.completion_status(task_id) = 1;
                    sync_successful = true;
                end
        end
        
        % Check for timeout
        if auction_data.sync_status(task_id) > 0 && auction_data.sync_status(task_id) < 3
            if isfield(params, 'sync_timeout') && ...
               current_time - auction_data.sync_start_time(task_id) > params.sync_timeout
                % Synchronization timeout
                auction_data.sync_status(task_id) = 0; % Reset synchronization
                auction_data.leader_assignment(task_id) = 0; % Reassign leader
            end
        end
        
        return;
    end
    
    % Helper function to calculate leader cost
    function cost = calculateLeaderCost(robot_id, task_id, robots, tasks, auction_data)
        % Distance factor - leader should be closer to task
        distance = norm(robots(robot_id).position - tasks(task_id).position);
        distance_factor = distance;
        
        % Capability factor - leader should have better match with task requirements
        capability_match = 0.5; % Default
        if isfield(robots, 'capabilities') && isfield(tasks, 'capabilities_required')
            robot_capabilities = robots(robot_id).capabilities;
            task_requirements = tasks(task_id).capabilities_required;
            
            % Normalize vectors
            robot_cap_norm = robot_capabilities / norm(robot_capabilities);
            task_cap_norm = task_requirements / norm(task_requirements);
            
            % Compute similarity (0 to 1, higher is better)
            capability_match = dot(robot_cap_norm, task_cap_norm);
        end
        
        % Workload factor - leader should have less workload
        workload = 0;
        if isfield(robots(robot_id), 'workload')
            workload = robots(robot_id).workload;
        else
            % Calculate workload from assignments
            task_count = sum(auction_data.assignment == robot_id);
            workload = task_count;
        end
        
        % Weighted cost (lower is better)
        weights = [0.4, 0.4, 0.2]; % Distance, capability, workload
        cost = weights(1) * distance_factor + ...
               weights(2) * (1 - capability_match) + ...
               weights(3) * workload;
        
        return;
    end
    
    % Helper function to check if robot is ready for task
    function ready = isRobotReadyForTask(robot_id, task_id, robots, tasks, params)
        % Check if robot has failed
        if robots(robot_id).failed
            ready = false;
            return;
        end
        
        % Check distance to task
        distance = norm(robots(robot_id).position - tasks(task_id).position);
        max_sync_distance = 0.5; % Default
        if isfield(params, 'max_sync_distance')
            max_sync_distance = params.max_sync_distance;
        end
        position_ready = (distance <= max_sync_distance);
        
        % Check if robot has necessary capabilities
        capability_ready = true;
        if isfield(robots, 'capabilities') && isfield(tasks, 'capabilities_required')
            robot_capabilities = robots(robot_id).capabilities;
            task_requirements = tasks(task_id).capabilities_required;
            
            % Normalize vectors
            robot_cap_norm = robot_capabilities / norm(robot_capabilities);
            task_cap_norm = task_requirements / norm(task_requirements);
            
            capability_match = dot(robot_cap_norm, task_cap_norm);
            capability_ready = (capability_match >= 0.7);
        end
        
        % Check if robot is not busy with another task
        busy_tasks = find(auction_data.assignment == robot_id & auction_data.completion_status < 1);
        not_busy = isempty(busy_tasks) || ismember(task_id, busy_tasks);
        
        ready = position_ready && capability_ready && not_busy;
        return;
    end
    
    % Time-weighted consensus update based on mathematical foundations
    function auction_data = runTimeWeightedConsensus(auction_data, robots, params)
        % Get number of robots
        num_robots = length(robots);
        num_tasks = length(auction_data.assignment);
        
        % Create state vectors for each robot if they don't exist
        if ~isfield(auction_data, 'state_vectors')
            auction_data.state_vectors = cell(num_robots, 1);
            for i = 1:num_robots
                % Format: [robot positions, task prices, task assignments, task completion]
                auction_data.state_vectors{i} = zeros(2*num_robots + 2*num_tasks, 1);
                
                % Initialize with known information
                % Robot positions
                for r = 1:num_robots
                    auction_data.state_vectors{i}(2*(r-1)+1:2*r) = robots(r).position;
                end
                
                % Task prices
                auction_data.state_vectors{i}(2*num_robots+1:2*num_robots+num_tasks) = auction_data.prices;
                
                % Task assignments
                auction_data.state_vectors{i}(2*num_robots+num_tasks+1:2*num_robots+2*num_tasks) = auction_data.assignment;
            end
            
            % Initialize last update timestamp matrix
            auction_data.last_update_time = zeros(num_robots, num_robots);
        end
        
        % Update timestamps
        auction_data.last_update_time = auction_data.last_update_time + 1;
        
        % First, update local state vectors with latest information
        for i = 1:num_robots
            % Update own position
            auction_data.state_vectors{i}(2*(i-1)+1:2*i) = robots(i).position;
            
            % Update own knowledge of prices and assignments
            auction_data.state_vectors{i}(2*num_robots+1:2*num_robots+num_tasks) = auction_data.prices;
            auction_data.state_vectors{i}(2*num_robots+num_tasks+1:2*num_robots+2*num_tasks) = auction_data.assignment;
        end
        
        % For each pair of robots, attempt information exchange
        for i = 1:num_robots
            if robots(i).failed
                continue;
            end
            
            % Get robot i's state vector
            x_i = auction_data.state_vectors{i};
            
            % For each other robot j
            for j = setdiff(1:num_robots, i)
                if robots(j).failed
                    continue;
                end
                
                % Check communication conditions
                dist_ij = norm(robots(i).position - robots(j).position);
                
                % Check if within communication range
                if isfield(auction_data, 'comm_range') && ...
                   length(auction_data.comm_range) >= i && ...
                   dist_ij > auction_data.comm_range(i)
                    continue;
                end
                
                % Check packet loss probability
                if isfield(auction_data, 'packet_loss_prob') && ...
                   size(auction_data.packet_loss_prob, 1) >= i && ...
                   size(auction_data.packet_loss_prob, 2) >= j && ...
                   rand() < auction_data.packet_loss_prob(i, j)
                    continue;  % Packet lost
                end
                
                % Get robot j's state vector with delay
                delay_ij = 0;
                if isfield(auction_data, 'comm_delay') && ...
                   size(auction_data.comm_delay, 1) >= i && ...
                   size(auction_data.comm_delay, 2) >= j && ...
                   auction_data.comm_delay(i, j) > 0
                    delay_ij = auction_data.comm_delay(i, j);
                end
                
                % For simplicity in simulation, we just use the current state
                % In a real system, this would use a delayed state vector
                x_j = auction_data.state_vectors{j};
                
                % Calculate time-decaying weight
                time_diff = auction_data.last_update_time(i, j);
                gamma = 0.5; % Default
                lambda = 0.1; % Default
                if isfield(params, 'gamma')
                    gamma = params.gamma;
                end
                if isfield(params, 'lambda')
                    lambda = params.lambda;
                end
                weight = gamma * exp(-lambda * time_diff);
                
                % Update state with time-weighted consensus
                auction_data.state_vectors{i} = x_i + weight * (x_j - x_i);
                
                % Reset last update time
                auction_data.last_update_time(i, j) = 0;
            end
        end
        
        % Extract consistent global view from state vectors
        
        % Update prices with consensus average - only for robot's own assigned tasks
        for j = 1:num_tasks
            assigned_robot = auction_data.assignment(j);
            if assigned_robot > 0 && assigned_robot <= num_robots && ~robots(assigned_robot).failed
                % Get the assigned robot's view of the price
                auction_data.prices(j) = auction_data.state_vectors{assigned_robot}(2*num_robots + j);
            else
                % For unassigned tasks, use average price across active robots
                total_price = 0;
                active_count = 0;
                for i = 1:num_robots
                    if ~robots(i).failed
                        total_price = total_price + auction_data.state_vectors{i}(2*num_robots + j);
                        active_count = active_count + 1;
                    end
                end
                if active_count > 0
                    auction_data.prices(j) = total_price / active_count;
                end
            end
        end
        
        % Calculate consensus error for monitoring convergence
        if ~isfield(auction_data, 'consensus_errors')
            auction_data.consensus_errors = [];
        end
        
        error = 0;
        active_robots = find(~[robots.failed]);
        if length(active_robots) >= 2
            for i = active_robots
                for j = active_robots
                    if i < j
                        % Calculate error for price consensus
                        price_error = norm(auction_data.state_vectors{i}(2*num_robots+1:2*num_robots+num_tasks) - ...
                                          auction_data.state_vectors{j}(2*num_robots+1:2*num_robots+num_tasks));
                        
                        % Calculate error for assignment consensus
                        assign_error = sum(auction_data.state_vectors{i}(2*num_robots+num_tasks+1:2*num_robots+2*num_tasks) ~= ...
                                          auction_data.state_vectors{j}(2*num_robots+num_tasks+1:2*num_robots+2*num_tasks));
                        
                        error = error + price_error + assign_error;
                    end
                end
            end
        end
        
        auction_data.consensus_errors = [auction_data.consensus_errors; error];
        
        return;
    end
    
    % Analyze convergence of the bidding process
    function [auction_data, converged, metrics] = analyzeBidConvergence(auction_data, params)
        converged = false;
        
        % Calculate theoretical convergence time
        num_tasks = length(auction_data.assignment);
        b_max = max(params.alpha);
        epsilon = params.epsilon;
        theoretical_bound = num_tasks^2 * b_max / epsilon;
        
        % Check if we've converged based on iterations with no changes
        unchanged_iterations = 0;
        if isfield(auction_data, 'convergence_history') && length(auction_data.convergence_history) > 0
            for i = length(auction_data.convergence_history):-1:1
                if auction_data.convergence_history(i) == 0
                    unchanged_iterations = unchanged_iterations + 1;
                else
                    break;
                end
            end
        end
        
        % Require several unchanged iterations for convergence
        required_unchanged = max(5, ceil(log(num_tasks) * 3));
        
        if unchanged_iterations >= required_unchanged
            converged = true;
            
            % Calculate optimality gap compared to theoretical optimal
            robot_utils_obj = robot_utils();
            achieved_makespan = robot_utils_obj.calculateMakespan(auction_data.assignment, tasks, robots);
            optimal_makespan = robot_utils_obj.calculateOptimalMakespan(tasks, robots);
            auction_data.optimality_gap = abs(achieved_makespan - optimal_makespan);
            
            % Check against theoretical bound (2ε)
            theoretical_gap = 2 * epsilon;
            
            % Prepare metrics
            metrics.converged = true;
            metrics.iterations = auction_data.utility_iter;
            metrics.theoretical_bound = theoretical_bound;
            metrics.iterations_ratio = auction_data.utility_iter / theoretical_bound;
            metrics.achieved_makespan = achieved_makespan;
            metrics.optimal_makespan = optimal_makespan;
            metrics.optimality_gap = auction_data.optimality_gap;
            metrics.theoretical_gap = theoretical_gap;
            metrics.gap_ratio = auction_data.optimality_gap / theoretical_gap;
        else
            metrics.converged = false;
        end
        
        return;
    end
    
    % Run enhanced auction simulation
    function [metrics, converged] = runEnhancedAuctionSimulation(params, env, robots, tasks, visualize)
        % Initialize auction data
        auction_data = enhancedInitializeAuctionData(tasks, robots);
        
        % Set up communication model
        auction_data = setupCommunicationModel(auction_data, robots, params);
        
        % Initialize metrics
        metrics = initializeMetrics();
        
        % Main simulation loop
        max_iterations = 1000;
        converged = false;
        
        % Find initially available tasks
        task_utils_obj = task_utils();
        available_tasks = task_utils_obj.findAvailableTasks(tasks, []);
        
        for iter = 1:max_iterations
            % Set current time
            params.current_time = iter;
            metrics.iterations = iter;
            
            % Check for robot failures
            [failures, auction_data] = detectFailures(auction_data, robots, params);
            
            % Handle any new failures
            if any(failures)
                for i = find(failures)
                    if visualize
                        fprintf('Robot %d has failed at iteration %d\n', i, iter);
                    end
                    robots(i).failed = true;
                    
                    % Record metrics before failure
                    metrics = recordPreFailureMetrics(metrics, auction_data, robots, tasks, i);
                    
                    % Initiate recovery process
                    [auction_data, recovery_initiated] = enhancedRecovery(auction_data, robots, tasks, i, params);
                    
                    if recovery_initiated && visualize
                        fprintf('Recovery initiated for robot %d failures\n', i);
                    end
                    metrics.failure_time = iter;
                end
            end
            
            % Execute the distributed auction algorithm step
            [auction_data, new_assignments, messages] = enhancedDistributedAuctionStep(auction_data, robots, tasks, available_tasks, params);
            metrics.messages = metrics.messages + messages;
            
            % Update visualization
            if visualize
                try
                    subplot(2, 3, [1, 4]);
                    env_utils = environment_utils();
                    env_utils.visualizeEnvironment(env, robots, tasks, auction_data);
                    title(sprintf('Environment (Iteration %d)', iter));
                    pause(0.01);
                catch
                    % Visualization failed, skip it
                    if visualize
                        fprintf('Warning: Visualization failed, continuing without it\n');
                    end
                end
            end
            
            % Update metrics
            metrics = updateMetricsHistory(metrics, auction_data, iter);
            
            % Update available tasks based on completed prerequisites
            completed_tasks = find(auction_data.completion_status == 1);
            [auction_data, available_tasks] = manageTaskDependencies(auction_data, tasks, completed_tasks);
            
            % Check for recovery completion
            if metrics.failure_time > 0 && metrics.recovery_time == 0
                if isRecoveryComplete(auction_data, robots, metrics.failed_robot)
                    metrics.recovery_time = iter - metrics.failure_time;
                    if visualize
                        fprintf('Recovery completed after %d iterations\n', metrics.recovery_time);
                    end
                end
            end
            
            % Check for convergence
            [auction_data, converged, convergence_metrics] = analyzeBidConvergence(auction_data, params);
            
            if converged
                if visualize
                    fprintf('Auction algorithm converged after %d iterations\n', iter);
                end
                metrics.converged = true;
                break;
            end
            
            % Update utility iteration
            auction_data.utility_iter = auction_data.utility_iter + 1;
            
            % Stop if we reach max iterations
            if iter == max_iterations
                if visualize
                    fprintf('Maximum iterations (%d) reached without convergence\n', max_iterations);
                end
                break;
            end
        end
        
        % Calculate final metrics
        metrics = calculateFinalMetrics(metrics, auction_data, robots, tasks, params);
        
        return;
    end
    
    % Helper function to setup communication model
    function auction_data = setupCommunicationModel(auction_data, robots, params)
        num_robots = length(robots);
        
        % Set up communication delays
        if isfield(params, 'comm_delay')
            if isscalar(params.comm_delay)
                % Uniform delay
                auction_data.comm_delay = params.comm_delay * ones(num_robots, num_robots);
            else
                % Custom delay matrix
                auction_data.comm_delay = params.comm_delay;
            end
        else
            % Default: no delay
            auction_data.comm_delay = zeros(num_robots, num_robots);
        end
        
        % Set up packet loss probabilities
        if isfield(params, 'packet_loss_prob')
            if isscalar(params.packet_loss_prob)
                % Uniform probability
                auction_data.packet_loss_prob = params.packet_loss_prob * ones(num_robots, num_robots);
            else
                % Custom probability matrix
                auction_data.packet_loss_prob = params.packet_loss_prob;
            end
        else
            % Default: no packet loss
            auction_data.packet_loss_prob = zeros(num_robots, num_robots);
        end
        
        % Set up communication ranges
        if isfield(params, 'comm_range')
            if isscalar(params.comm_range)
                % Uniform range
                auction_data.comm_range = params.comm_range * ones(num_robots, 1);
            else
                % Custom range vector
                auction_data.comm_range = params.comm_range;
            end
        else
            % Default: infinite range
            auction_data.comm_range = Inf * ones(num_robots, 1);
        end
        
        return;
    end
    
    % Helper function to initialize metrics
    function metrics = initializeMetrics()
        metrics = struct();
        metrics.iterations = 0;
        metrics.messages = 0;
        metrics.convergence_history = [];
        metrics.price_history = [];
        metrics.assignment_history = [];
        metrics.completion_time = 0;
        metrics.optimality_gap = 0;
        metrics.recovery_time = 0;
        metrics.failed_task_count = 0;
        metrics.failure_time = 0;
        metrics.makespan = 0;
        metrics.optimal_makespan = 0;
        metrics.makespan_before_failure = 0;
        metrics.theoretical_recovery_bound = 0;
        metrics.oscillation_count = 0;
        metrics.converged = false;
    end
    
    % Helper function to record pre-failure metrics
    function metrics = recordPreFailureMetrics(metrics, auction_data, robots, tasks, failed_robot)
        % Count tasks assigned to the failed robot
        metrics.failed_task_count = sum(auction_data.assignment == failed_robot);
        
        % Record makespan before failure
        robot_utils_obj = robot_utils();
        metrics.makespan_before_failure = robot_utils_obj.calculateMakespan(auction_data.assignment, tasks, robots);
        
        % Record workload distribution before failure
        robot_workloads = zeros(length(robots), 1);
        for j = 1:length(tasks)
            r = auction_data.assignment(j);
            if r > 0
                if isfield(tasks, 'execution_time')
                    robot_workloads(r) = robot_workloads(r) + tasks(j).execution_time;
                else
                    robot_workloads(r) = robot_workloads(r) + 1;
                end
            end
        end
        metrics.workload_before_failure = robot_workloads;
        
        % Store tasks assigned to the failed robot
        metrics.failed_robot = failed_robot;
        metrics.failed_robot_tasks = find(auction_data.assignment == failed_robot);
        
        return;
    end
    
    % Helper function to update metrics history
    function metrics = updateMetricsHistory(metrics, auction_data, iter)
        % Update price history
        if isempty(metrics.price_history)
            metrics.price_history = zeros(length(auction_data.prices), 1000);  % Preallocate
        end
        if iter <= size(metrics.price_history, 2)
            metrics.price_history(:, iter) = auction_data.prices;
        end
        
        % Update assignment history
        if isempty(metrics.assignment_history)
            metrics.assignment_history = zeros(length(auction_data.assignment), 1000);  % Preallocate
        end
        if iter <= size(metrics.assignment_history, 2)
            metrics.assignment_history(:, iter) = auction_data.assignment;
        end
        
        % Calculate convergence metric (change in assignments)
        if iter > 1 && iter <= size(metrics.assignment_history, 2)
            conv_metric = sum(metrics.assignment_history(:, iter) ~= metrics.assignment_history(:, iter-1));
            metrics.convergence_history(iter) = conv_metric;
        else
            metrics.convergence_history(iter) = NaN;
        end
        
        return;
    end
    
    % Helper function to calculate final metrics
    function metrics = calculateFinalMetrics(metrics, auction_data, robots, tasks, params)
        % Trim history matrices to actual size
        iter = metrics.iterations;
        if ~isempty(metrics.price_history) && size(metrics.price_history, 2) >= iter
            metrics.price_history = metrics.price_history(:, 1:iter);
        end
        if ~isempty(metrics.assignment_history) && size(metrics.assignment_history, 2) >= iter
            metrics.assignment_history = metrics.assignment_history(:, 1:iter);
        end
        
        % Calculate makespan
        robot_utils_obj = robot_utils();
        try
            metrics.makespan = robot_utils_obj.calculateMakespan(auction_data.assignment, tasks, robots);
            metrics.optimal_makespan = robot_utils_obj.calculateOptimalMakespan(tasks, robots);
        catch
            % Fallback if direct calculation fails
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
                balanced_load = total_load / active_robots;
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
        if isfield(auction_data, 'task_oscillation_count')
            metrics.oscillation_count = sum(auction_data.task_oscillation_count);
        end
        
        % Ensure optimality gap is correctly calculated
        metrics.optimality_gap = abs(metrics.makespan - metrics.optimal_makespan);
        
        % Theoretical recovery bound
        if isfield(metrics, 'failed_robot') && ~isempty(metrics.failed_robot)
            T_f = metrics.failed_task_count;
            if isfield(params, 'alpha') && isfield(params, 'epsilon')
                b_max = max(params.alpha);
                epsilon = params.epsilon;
                metrics.theoretical_recovery_bound = T_f + round(b_max/epsilon);
            end
        end
        
        return;
    end
    
    % Helper function to check if recovery is complete
    function complete = isRecoveryComplete(auction_data, robots, failed_robot)
        complete = true;
        
        % Check if all tasks previously assigned to the failed robot
        % have been reassigned to active robots
        if isfield(auction_data, 'failure_assignment') && ~isempty(auction_data.failure_assignment)
            failed_tasks = find(auction_data.failure_assignment == failed_robot);
            
            for i = 1:length(failed_tasks)
                task_id = failed_tasks(i);
                new_assignment = auction_data.assignment(task_id);
                
                % Task is still unassigned or marked for recovery
                if new_assignment == 0 || new_assignment == -1 || new_assignment == failed_robot
                    complete = false;
                    break;
                end
                
                % Check if the new robot is active
                if new_assignment > 0 && new_assignment <= length(robots) && robots(new_assignment).failed
                    complete = false;
                    break;
                end
            end
        else
            % Fallback approach if failure_assignment is not available
            complete = ~any(auction_data.assignment == failed_robot);
        end
        
        return;
    end
    
    % Return the utils structure
    return;
end