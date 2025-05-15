function utils = auction_utils()
    % AUCTION_UTILS - Returns function handles for auction-related functions
    utils = struct(...
        'initializeAuctionData', @local_initializeAuctionData, ...
        'distributedAuctionStep', @local_distributedAuctionStep, ...
        'calculateBid', @local_calculateBid, ...
        'calculateFutureBid', @local_calculateFutureBid, ...
        'calculateBalancedBid', @local_calculateBalancedBid, ...
        'initiateRecovery', @local_initiateRecovery, ...
        'runAuctionSimulation', @local_runAuctionSimulation, ...
        'analyzeTaskAllocation', @local_analyzeTaskAllocation, ...
        'analyzeBidDistribution', @local_analyzeBidDistribution, ...
        'resetPricesForBlockedTasks', @local_resetPricesForBlockedTasks, ...
        'printTaskStatus', @local_printTaskStatus, ...
        'progressTaskExecution', @local_progressTaskExecution, ...
        'updateProjectedCompletionTimes', @local_updateProjectedCompletionTimes, ...
        'calculateEarliestStartTime', @local_calculateEarliestStartTime, ...
        'processDependencyBasedBatches', @local_processDependencyBasedBatches, ...
        'calculateDependencyLevels', @local_calculateDependencyLevels, ...
        'runBatchAuction', @local_runBatchAuction ...
    );
end

function auction_data = local_initializeAuctionData(tasks, robots)
    % INITIALIZEAUCTIONDATA Initialize auction data structure
    %
    % Parameters:
    %   tasks - Array of task structures
    %   robots - Array of robot structures
    %
    % Returns:
    %   auction_data - Structure containing auction state
    
    % Initialize auction data structure
    auction_data = struct();
    
    % Create arrays for task assignments and prices
    auction_data.assignment = zeros(length(tasks), 1);  % 0 = unassigned
    auction_data.prices = zeros(length(tasks), 1);
    auction_data.bids = zeros(length(robots), length(tasks));
    auction_data.utilities = zeros(length(robots), length(tasks));
    auction_data.completion_status = zeros(length(tasks), 1);  % 0 = not completed, 1 = completed
    
    % Task progress tracking
    auction_data.task_progress = zeros(length(tasks), 1);
    
    % ADDITION: Add tentative assignments for future planning
    auction_data.tentative_assignment = zeros(length(tasks), 1);
    auction_data.projected_completion_time = inf(length(tasks), 1);
    
    % ADDITION: Track expected available time for each robot
    auction_data.robot_available_time = zeros(length(robots), 1);
    
    % ADDITION: Keep track of initial assignment for recovery
    auction_data.initial_assignment = auction_data.assignment;
    
    % ADDITION: Add task assignment history to track oscillations
    auction_data.assignment_count = zeros(length(tasks), 1);
    auction_data.last_assigned_robot = zeros(length(tasks), 1);
    
    % ADDITION: Track task oscillations
    auction_data.task_oscillation_count = zeros(length(tasks), 1);
    
    % ADDITION: Recovery mode flag
    auction_data.recovery_mode = false;
    auction_data.failed_robot = [];
    
    % ADDITION: For storing assignment and price history
    auction_data.assignment_history = [];
    auction_data.price_history = [];
    
    % ADDITION: Track simulation time
    auction_data.simulation_time = 0;
    
    % ADDITION: Track task queue for dependency handling
    auction_data.task_queue = [];
    
    % ADDITION: Add expected completion time for each task
    auction_data.expected_completion_time = zeros(length(tasks), 1);
    
    % ADDITION: Add dependency status for better tracking
    auction_data.dependency_met = false(length(tasks), 1);
    % Initially mark tasks with no prerequisites as having dependencies met
    for i = 1:length(tasks)
        if isempty(tasks(i).prerequisites)
            auction_data.dependency_met(i) = true;
        end
    end
    
    % NEW ADDITION: Add task execution schedule for better temporal planning
    auction_data.task_start_time = inf(length(tasks), 1);
    auction_data.task_end_time = inf(length(tasks), 1);
    
    % NEW ADDITION: Track task dependency levels
    auction_data.dependency_levels = zeros(length(tasks), 1);
    
    % NEW ADDITION: Track robot schedules
    auction_data.robot_schedule = cell(length(robots), 1);
    for i = 1:length(robots)
        auction_data.robot_schedule{i} = [];
    end
    
    % NEW ADDITION: Track workload balance metrics
    auction_data.workload_balance = 1.0;  % 1.0 means perfectly balanced
    auction_data.min_max_workload_ratio = 1.0;
end

function [auction_data, new_assignments, messages] = local_distributedAuctionStep(auction_data, robots, tasks, available_tasks, params)
    % DISTRIBUTEDAUCTIONSTEP Perform one step of the distributed auction algorithm
    %
    % Parameters:
    %   auction_data - Auction data structure
    %   robots - Array of robot structures
    %   tasks - Array of task structures
    %   available_tasks - List of task IDs that are currently available for assignment
    %   params - Algorithm parameters
    %
    % Returns:
    %   auction_data - Updated auction data structure
    %   new_assignments - Number of new assignments made in this step
    %   messages - Number of messages exchanged in this step
    
    % Track new assignments and messages
    new_assignments = 0;
    messages = 0;
    
    % Get current assignments and prices
    assignment = auction_data.assignment;
    prices = auction_data.prices;
    
    % Store previous completion status to check for new completions
    previous_completion_status = auction_data.completion_status;
    
    % Reset bids and utilities for this step
    auction_data.bids = zeros(length(robots), length(tasks));
    auction_data.utilities = zeros(length(robots), length(tasks));
    
    % Apply communication constraints if specified
    comm_delay = 0;
    packet_loss_prob = 0;
    
    if isfield(params, 'comm_delay')
        comm_delay = params.comm_delay;
    end
    
    if isfield(params, 'packet_loss_prob')
        packet_loss_prob = params.packet_loss_prob;
    end
    
    % Calculate dependency levels for tasks if not already done
    if all(auction_data.dependency_levels == 0)
        auction_data.dependency_levels = local_calculateDependencyLevels(tasks);
    end
    
    % NEW: Identify future tasks that will be available soon
    future_available_tasks = local_identifyFutureTasks(tasks, auction_data, 0.7);
    
    % NEW: Combine currently available and future available tasks for bidding
    bidding_tasks = union(available_tasks, future_available_tasks);
    
    % FIX: Add logging for available tasks
    if ~isempty(bidding_tasks)
        fprintf('Current bidding tasks: ');
        fprintf('%d ', bidding_tasks);
        fprintf('\n');
    end
    
    % NEW: Update workload balance metrics
    auction_data = local_updateWorkloadBalanceMetrics(auction_data, robots);
    
    % Phase 1: Bidding
    % Each robot bids on available and soon-to-be-available tasks
    for r = 1:length(robots)
        % Skip failed robots
        if isfield(robots, 'failed') && robots(r).failed
            continue;
        end
        
        % For each available task
        for i = 1:length(bidding_tasks)
            task_idx = bidding_tasks(i);
            
            % Skip tasks that are already completed
            if auction_data.completion_status(task_idx) == 1
                continue;
            end
            
            % Calculate bid for this task
            if ismember(task_idx, available_tasks)
                % Regular bid for currently available tasks
                if auction_data.recovery_mode && isfield(params, 'beta')
                    % Adjust bid calculation during recovery
                    bid = local_calculateRecoveryBid(robots(r), tasks(task_idx), prices(task_idx), params);
                else
                    % NEW: Use balanced bid calculation that considers workload distribution
                    bid = local_calculateBalancedBid(robots(r), tasks(task_idx), prices(task_idx), params, auction_data, robots);
                end
            else
                % Future bid with discount for tasks that will be available soon
                bid = local_calculateFutureBid(robots(r), tasks(task_idx), prices(task_idx), params, auction_data);
            end
            
            % Store the bid
            auction_data.bids(r, task_idx) = bid;
            
            % Calculate utility
            utility = bid - prices(task_idx);
            auction_data.utilities(r, task_idx) = utility;
            
            % Simulate communication
            if comm_delay > 0 || packet_loss_prob > 0
                messages = messages + 1;
                
                % Simulate packet loss
                if rand() < packet_loss_prob
                    auction_data.bids(r, task_idx) = 0;
                    auction_data.utilities(r, task_idx) = 0;
                end
            end
        end
    end
    
    % Phase 2: Assignment
    % Process bids and update assignments
    for i = 1:length(bidding_tasks)
        task_idx = bidding_tasks(i);
        
        % Skip tasks that are already completed
        if auction_data.completion_status(task_idx) == 1
            continue;
        end
        
        % Find the highest bidder
        [max_bid, max_bidder] = max(auction_data.bids(:, task_idx));
        
        % If there is a valid bid
        if max_bid > 0
            % Check if this assignment is new or different
            if assignment(task_idx) ~= max_bidder
                % If previously assigned to another robot, count as reassignment
                if assignment(task_idx) > 0
                    auction_data.task_oscillation_count(task_idx) = auction_data.task_oscillation_count(task_idx) + 1;
                end
                
                % Update assignment
                old_assignment = assignment(task_idx);
                assignment(task_idx) = max_bidder;
                new_assignments = new_assignments + 1;
                
                % If this is a new assignment, increment counter
                if old_assignment == 0
                    auction_data.assignment_count(task_idx) = auction_data.assignment_count(task_idx) + 1;
                end
                
                % NEW: Calculate earliest possible start time based on prerequisites
                earliest_start_time = local_calculateEarliestStartTime(task_idx, tasks, auction_data);
                
                % Update robot workloads
                if isfield(tasks, 'execution_time')
                    task_time = tasks(task_idx).execution_time;
                else
                    task_time = 1;  % Default execution time
                end
                
                % NEW: Update projected timeline with temporal planning
                [auction_data.robot_available_time, auction_data.projected_completion_time] = ...
                    local_updateTaskTimeline(auction_data.robot_available_time, auction_data.projected_completion_time, ...
                                             max_bidder, task_idx, earliest_start_time, task_time, auction_data.simulation_time);
                
                % FIX: Log the assignment
                fprintf('Task %d assigned to Robot %d (bid: %.2f, price: %.2f, start: %.2f)\n', ...
                       task_idx, max_bidder, max_bid, prices(task_idx), earliest_start_time);
                
                % Update robot workload using utils_manager
                utils = utils_manager();
                robot_utils = utils.robot;
                robots = robot_utils.updateRobotWorkload(robots, old_assignment, max_bidder, task_idx, task_time);
                
                % NEW: Update robot schedule
                auction_data.robot_schedule{max_bidder} = [auction_data.robot_schedule{max_bidder}; ...
                                                           struct('task_id', task_idx, ...
                                                                 'start_time', earliest_start_time, ...
                                                                 'end_time', earliest_start_time + task_time)];
                
                % Sort schedule by start time
                [~, sort_idx] = sort([auction_data.robot_schedule{max_bidder}.start_time]);
                auction_data.robot_schedule{max_bidder} = auction_data.robot_schedule{max_bidder}(sort_idx);
            end
            
            % Update task price
            second_highest_bid = 0;
            bids = auction_data.bids(:, task_idx);
            if sum(bids > 0) > 1
                sorted_bids = sort(bids(bids > 0), 'descend');
                second_highest_bid = sorted_bids(2);
            end
            
            if second_highest_bid > 0
                prices(task_idx) = second_highest_bid + params.epsilon;
            else
                prices(task_idx) = prices(task_idx) + params.epsilon;
            end
        end
    end
    
    % Update auction data
    auction_data.assignment = assignment;
    auction_data.prices = prices;
    
    % ADDITION: Task progress update
    auction_data = local_progressTaskExecution(auction_data, tasks, robots, params);
    
    % ADDITION: Check for new completions and update available tasks
    if any(auction_data.completion_status ~= previous_completion_status)
        % Get newly completed tasks
        newly_completed = find(auction_data.completion_status == 1 & previous_completion_status == 0);
        
        if ~isempty(newly_completed)
            fprintf('Tasks completed: ');
            fprintf('%d ', newly_completed);
            fprintf('\n');
            
            % Update dependency status based on new completions
            for i = 1:length(tasks)
                if auction_data.completion_status(i) == 0 && ~auction_data.dependency_met(i)
                    % Check if all prerequisites are now completed
                    all_completed = true;
                    for j = 1:length(tasks(i).prerequisites)
                        prereq = tasks(i).prerequisites(j);
                        if auction_data.completion_status(prereq) == 0
                            all_completed = false;
                            break;
                        end
                    end
                    
                    if all_completed
                        auction_data.dependency_met(i) = true;
                        fprintf('Dependencies for Task %d are now met\n', i);
                    end
                end
            end
        end
    end
end

function auction_data = local_progressTaskExecution(auction_data, tasks, robots, params)
    % PROGRESSTASKEXECUTION - Progress the execution of assigned tasks
    %
    % Parameters:
    %   auction_data - Auction data structure
    %   tasks - Array of task structures
    %   robots - Array of robot structures
    %   params - Algorithm parameters
    %
    % Returns:
    %   auction_data - Updated auction data structure
    
    % Update simulation time
    auction_data.simulation_time = auction_data.simulation_time + params.time_step;
    
    % Process each task
    for i = 1:length(tasks)
        if auction_data.assignment(i) > 0 && auction_data.completion_status(i) == 0
            robot_id = auction_data.assignment(i);
            
            % If robot is not failed
            if robot_id <= length(robots) && ~(isfield(robots, 'failed') && robots(robot_id).failed)
                % Check if prerequisites are satisfied
                prerequisites_satisfied = true;
                if isfield(tasks(i), 'prerequisites') && ~isempty(tasks(i).prerequisites)
                    for prereq = tasks(i).prerequisites
                        if auction_data.completion_status(prereq) == 0
                            prerequisites_satisfied = false;
                            break;
                        end
                    end
                end
                
                % Only progress tasks when prerequisites are completed
                if prerequisites_satisfied
                    % Check if task has started based on projected start time
                    if auction_data.task_start_time(i) == inf
                        % Set start time if not set
                        auction_data.task_start_time(i) = auction_data.simulation_time;
                    end
                    
                    % Increment progress by time step
                    auction_data.task_progress(i) = auction_data.task_progress(i) + params.time_step;
                    
                    % If task has been executed for its required time, mark as completed
                    if auction_data.task_progress(i) >= tasks(i).execution_time
                        auction_data.completion_status(i) = 1;
                        
                        % Set end time
                        auction_data.task_end_time(i) = auction_data.simulation_time;
                        
                        % Log task completion
                        fprintf('Task %d completed by Robot %d (Duration: %.2f)\n', ...
                                i, robot_id, auction_data.simulation_time - auction_data.task_start_time(i));
                        
                        % Update robot's completed tasks list
                        utils = utils_manager();
                        robot_utils = utils.robot;
                        robots = robot_utils.updateRobotCompletedTasks(robots, i, robot_id);
                        
                        % Update robot available time
                        auction_data.robot_available_time(robot_id) = auction_data.simulation_time;
                        
                        % FIX: Check for tasks that become available due to this completion
                        for j = 1:length(tasks)
                            if auction_data.completion_status(j) == 0 && ~isempty(tasks(j).prerequisites)
                                % If this task depends on the just-completed task
                                if ismember(i, tasks(j).prerequisites)
                                    % Check if all prerequisites are now completed
                                    all_completed = true;
                                    for prereq = tasks(j).prerequisites
                                        if auction_data.completion_status(prereq) == 0
                                            all_completed = false;
                                            break;
                                        end
                                    end
                                    
                                    if all_completed
                                        auction_data.dependency_met(j) = true;
                                        fprintf('Task %d is now ready for assignment (dependencies met)\n', j);
                                    end
                                end
                            end
                        end
                    end
                else
                    % Log waiting for dependencies
                    if mod(auction_data.simulation_time, 1) < params.time_step
                        fprintf('Task %d assigned to Robot %d is waiting for prerequisites\n', i, robot_id);
                    end
                end
            end
        end
    end
end

function bid = local_calculateBid(robot, task, current_price, params)
    % CALCULATEBID Calculate bid value for a task
    
    % Calculate capability match with proper normalization
    capability_match = dot(robot.capabilities, task.capabilities_required) / ...
                      (norm(robot.capabilities) * norm(task.capabilities_required));
    
    % Calculate distance-based cost with exponential decay
    distance = norm(robot.position - task.position);
    distance_cost = params.alpha(3) * (1 - exp(-distance/2));
    
    % Improve workload factor calculation
    workload_factor = params.alpha(2) * (1 / (robot.workload + 1));
    
    % FIXED: Add value for tasks on critical path and for tasks with dependencies
    critical_path_bonus = 0;
    dependency_bonus = 0;
    
    % Use any() function to handle non-scalar logical values
    if isfield(task, 'on_critical_path') && (isscalar(task.on_critical_path) && task.on_critical_path)
        critical_path_bonus = params.alpha(4) * 3;  % Triple importance for critical path tasks
    end
    
    % Add bonus for tasks with no prerequisites (to get the process started)
    if ~isfield(task, 'prerequisites') || isempty(task.prerequisites)
        dependency_bonus = params.alpha(5) * 2;
    end
    
    % Add time efficiency factor - reward shorter tasks
    time_efficiency = 0;
    if isfield(task, 'execution_time') && task.execution_time > 0
        time_efficiency = params.alpha(4) * (5 / task.execution_time);
    end
    
    % Calculate bid value with proper weighting
    bid = params.alpha(1) * capability_match + ...
          workload_factor - ...
          distance_cost + ...
          time_efficiency + ...
          critical_path_bonus + ...
          dependency_bonus;
    
    % Ensure bid exceeds current price by at least epsilon
    if bid <= current_price
        bid = current_price + params.epsilon;
    end
end

function bid = local_calculateBalancedBid(robot, task, current_price, params, auction_data, all_robots)
    % CALCULATEBALANCEDBID Calculate a bid that accounts for workload balancing
    %
    % Parameters:
    %   robot - Robot structure
    %   task - Task structure
    %   current_price - Current price of the task
    %   params - Algorithm parameters
    %   auction_data - Current auction data for context
    %   all_robots - Array of all robot structures
    %
    % Returns:
    %   bid - Calculated balanced bid value
    
    % Get standard bid calculation
    base_bid = local_calculateBid(robot, task, current_price, params);
    
    % Calculate workload balance metrics
    robot_workloads = zeros(1, length(all_robots));
    for i = 1:length(all_robots)
        robot_workloads(i) = all_robots(i).workload;
    end
    
    % Skip failed robots when calculating workload metrics
    valid_robots = true(1, length(all_robots));
    for i = 1:length(all_robots)
        if isfield(all_robots(i), 'failed') && all_robots(i).failed
            valid_robots(i) = false;
        end
    end
    
    valid_workloads = robot_workloads(valid_robots);
    
    % Avoid division by zero
    if isempty(valid_workloads) || all(valid_workloads == 0)
        return;
    end
    
    % Calculate workload balance metrics
    min_workload = min(valid_workloads);
    max_workload = max(valid_workloads);
    avg_workload = mean(valid_workloads);
    workload_range = max_workload - min_workload;
    
    % Calculate current robot's workload relative to others
    relative_workload = robot.workload / avg_workload;
    
    % Apply balancing factor - higher bids for under-utilized robots
    balance_factor = 1.0;
    
    if workload_range > 0
        % Robots with less workload get a boost
        if robot.workload < avg_workload
            % The further below average, the bigger the boost
            % Maximum boost is 2.0 (doubling bid) when workload is 0 and max disparity
            imbalance_ratio = (avg_workload - robot.workload) / workload_range;
            balance_factor = 1.0 + imbalance_ratio;
        else
            % Robots with more workload get a penalty
            % Maximum penalty is 0.5 (halving bid) when workload is at maximum
            imbalance_ratio = (robot.workload - avg_workload) / workload_range;
            balance_factor = 1.0 / (1.0 + imbalance_ratio);
        end
    end
    
    % NEW: Temporal considerations - adjust bid based on how busy the robot is
    if isfield(auction_data, 'robot_available_time')
        earliest_start = local_calculateEarliestStartTime(task.id, [], auction_data);
        
        % Prefer robots that are available closer to when the task can start
        time_match = abs(auction_data.robot_available_time(robot.id) - earliest_start);
        time_factor = exp(-0.1 * time_match); % Exponential decay based on time mismatch
        
        % Adjust balance factor
        balance_factor = balance_factor * (0.7 + 0.3 * time_factor);
    end
    
    % Apply the balance factor to the base bid
    balanced_bid = base_bid * balance_factor;
    
    % Ensure balanced bid still exceeds current price 
    if balanced_bid <= current_price
        balanced_bid = current_price + params.epsilon;
    end
    
    bid = balanced_bid;
end

function bid = local_calculateFutureBid(robot, task, current_price, params, auction_data)
    % CALCULATEFUTUREBID Calculate bid value for a task that will be available in the future
    %
    % Parameters:
    %   robot - Robot structure
    %   task - Task structure
    %   current_price - Current price of the task
    %   params - Algorithm parameters
    %   auction_data - Current auction data for context
    %
    % Returns:
    %   bid - Calculated future bid value
    
    % Calculate base bid
    base_bid = local_calculateBid(robot, task, current_price, params);
    
    % Estimate when this task will be available based on prerequisites
    earliest_available_time = local_calculateEarliestStartTime(task.id, [], auction_data);
    
    % Calculate when the robot will be available
    robot_available_time = auction_data.robot_available_time(robot.id);
    
    % Calculate timing alignment factor
    timing_match = 1.0;
    time_difference = abs(earliest_available_time - robot_available_time);
    
    % Better match if robot becomes available close to when task is ready
    if time_difference < 1.0
        timing_match = 1.3;  % Boost for perfect timing
    elseif time_difference < 3.0
        timing_match = 1.1;  % Small boost for good timing
    else
        timing_match = 1.0 / (1.0 + 0.1 * time_difference); % Diminishing returns for worse timing
    end
    
    % Calculate time discount factor (value decreases with wait time)
    time_to_availability = max(0, earliest_available_time - auction_data.simulation_time);
    
    % Use gentler time discount to encourage future task bidding
    time_discount = exp(-0.03 * time_to_availability);  % Reduced from 0.05 to 0.03 for slower decay
    
    % Apply timing match and discount to bid
    discounted_bid = base_bid * time_discount * timing_match;
    
    % Ensure minimum bid
    if discounted_bid <= current_price
        discounted_bid = current_price + params.epsilon;
    end
    
    bid = discounted_bid;
end

function bid = local_calculateRecoveryBid(robot, task, current_price, params)
    % CALCULATERECOVERYBID Calculate bid value during recovery phase
    %
    % Parameters:
    %   robot - Robot structure
    %   task - Task structure
    %   current_price - Current price of the task
    %   params - Algorithm parameters
    %
    % Returns:
    %   bid - Calculated recovery bid value
    
    % Use standard bid as baseline
    base_bid = local_calculateBid(robot, task, current_price, params);
    
    % Add recovery bonus to prioritize task reassignment
    recovery_bonus = params.beta(1);
    
    % Scale bonus based on task criticality if the task is on critical path
    if isfield(task, 'on_critical_path') && task.on_critical_path
        recovery_bonus = recovery_bonus * params.beta(2);
    end
    
    % Calculate recovery bid
    bid = base_bid + recovery_bonus;
    
    % Ensure bid exceeds current price by at least epsilon
    if bid <= current_price
        bid = current_price + params.epsilon;
    end
    
    return;
end

function auction_data = local_initiateRecovery(auction_data, robots, tasks, failed_robot)
    % INITIATERECOVERY Initiate recovery process after robot failure
    %
    % Parameters:
    %   auction_data - Auction data structure
    %   robots - Array of robot structures
    %   tasks - Array of task structures
    %   failed_robot - ID of the failed robot
    %
    % Returns:
    %   auction_data - Updated auction data structure
    
    % Set recovery mode flag
    auction_data.recovery_mode = true;
    auction_data.failed_robot = failed_robot;
    
    % Store assignment at time of failure for analysis
    if ~isfield(auction_data, 'failure_assignment')
        auction_data.failure_assignment = auction_data.assignment;
    end
    
    % Get tasks assigned to failed robot
    failed_tasks = find(auction_data.assignment == failed_robot);
    
    % Log recovery initiation
    fprintf('Initiating recovery for Robot %d failure. %d tasks to reassign: ', ...
            failed_robot, length(failed_tasks));
    fprintf('%d ', failed_tasks);
    fprintf('\n');
    
    % Reset assignments for tasks assigned to failed robot
    for i = 1:length(failed_tasks)
        task_id = failed_tasks(i);
        
        % Reset assignment for this task
        auction_data.assignment(task_id) = 0;
        
        % Reset task progress for incomplete tasks
        if auction_data.completion_status(task_id) == 0
            % Reset progress if work had started
            if auction_data.task_progress(task_id) > 0
                fprintf('Task %d progress reset (was %.1f%%)\n', ...
                        task_id, auction_data.task_progress(task_id) / tasks(task_id).execution_time * 100);
                auction_data.task_progress(task_id) = 0;
            end
        end
        
        % Reset task timeline entries
        auction_data.task_start_time(task_id) = inf;
        auction_data.task_end_time(task_id) = inf;
        auction_data.projected_completion_time(task_id) = inf;
    end
    
    % Clear failed robot's schedule
    if failed_robot <= length(auction_data.robot_schedule)
        auction_data.robot_schedule{failed_robot} = [];
    end
    
    % Reset robot available time to current time
    auction_data.robot_available_time(failed_robot) = inf;
    
    return;
end

function [metrics, converged] = local_runAuctionSimulation(params, env, robots, tasks, visualize)
    % RUNAUCTIONSIMULATION Run the complete auction simulation
    
    % Load utilities
    utils = utils_manager();
    task_utils = utils.task;
    
    % IMPROVED: Pre-analyze task dependencies to identify critical path
    [critical_path, task_depths] = findCriticalPath(tasks);
    
    % Mark tasks on critical path
    for i = 1:length(critical_path)
        task_id = critical_path(i);
        if task_id <= length(tasks)
            tasks(task_id).on_critical_path = true;
        end
    end
    
    % Initialize auction data
    auction_data = local_initializeAuctionData(tasks, robots);
    
    % Performance metrics
    metrics = struct();
    metrics.iterations = 0;
    metrics.messages = 0;
    metrics.convergence_history = zeros(1, 1000);  % Pre-allocate with zeros
    metrics.price_history = zeros(length(tasks), 1000);  % Preallocate
    metrics.assignment_history = zeros(length(tasks), 1000);
    metrics.completion_time = 0;
    metrics.optimality_gap = 0;
    metrics.recovery_time = 0;
    metrics.failure_time = 0;
    metrics.makespan = 0;
    metrics.optimal_makespan = 0;
    metrics.task_oscillation_count = zeros(length(tasks), 1);
    metrics.failed_task_count = 0;
    metrics.critical_path = critical_path;
    metrics.workload_balance_history = zeros(1, 1000);
    
    % ADDITION: Add time step to parameters if not present
    if ~isfield(params, 'time_step')
        params.time_step = 0.1; % Default time step
    end
    
    % NEW: Process tasks in dependency-based batches
    dependency_levels = local_calculateDependencyLevels(tasks);
    auction_data.dependency_levels = dependency_levels;
    
    % NEW: Use batch execution option (if enabled)
    use_batch_auction = false;
    if isfield(params, 'use_batch_auction') && params.use_batch_auction
        use_batch_auction = true;
    end
    
    % FIX: Increase maximum iterations for complex scenarios
    max_iterations = 2000;
    converged = false;
    
    % Available tasks (initially only those with no prerequisites)
    completed_tasks = [];
    available_tasks = task_utils.findAvailableTasks(tasks, completed_tasks);
    fprintf('Initial available tasks: ');
    fprintf('%d ', available_tasks);
    fprintf('\n');
    
    % Store initial assignments for comparison
    prev_assignments = auction_data.assignment;
    unchanged_iterations = 0;
    
    % Visualization setup if requested
    if visualize
        figure('Name', 'Distributed Auction Algorithm Simulation', 'Position', [100, 100, 1200, 800]);
        subplot(2, 3, [1, 4]);
        env_utils = environment_utils();
        env_utils.visualizeEnvironment(env, robots, tasks, auction_data);
        title('Initial Environment');
    end
    
    % Main simulation loop
    for iter = 1:max_iterations
        metrics.iterations = iter;
        
        % Check for robot failure
        if isfield(params, 'failure_time') && iter == params.failure_time && ~isempty(params.failed_robot)
            fprintf('Robot %d has failed at iteration %d\n', params.failed_robot, iter);
            robots(params.failed_robot).failed = true;
            metrics.failure_time = iter;  % Store failure time in metrics
            
            % Count tasks assigned to failed robot
            metrics.failed_task_count = sum(auction_data.assignment == params.failed_robot);
            
            % Initiate recovery process
            auction_data = local_initiateRecovery(auction_data, robots, tasks, params.failed_robot);
        end
        
        % Reset prices for blocked tasks
        auction_data = local_resetPricesForBlockedTasks(auction_data, tasks, available_tasks);
        
        % Update available tasks based on newly completed tasks
        completed_tasks = find(auction_data.completion_status == 1);
        available_tasks = task_utils.findAvailableTasks(tasks, completed_tasks);
        
        % FIX: Add debugging for available tasks
        if mod(iter, 5) == 0
            fprintf('Available tasks at iteration %d: ', iter);
            fprintf('%d ', available_tasks);
            fprintf('\n');
        end
        
        % NEW: Use dependency-based batch processing if enabled
        if use_batch_auction
            [auction_data, new_assignments, messages] = local_processDependencyBasedBatches(auction_data, robots, tasks, available_tasks, params);
        else
            % Standard distributed auction algorithm step
            [auction_data, new_assignments, messages] = local_distributedAuctionStep(auction_data, robots, tasks, available_tasks, params);
        end
        
        metrics.messages = metrics.messages + messages;
        
        % Update performance metrics
        metrics.price_history(:, iter) = auction_data.prices;
        metrics.assignment_history(:, iter) = auction_data.assignment;
        
        % NEW: Record workload balance metric
        auction_data = local_updateWorkloadBalanceMetrics(auction_data, robots);
        metrics.workload_balance_history(iter) = auction_data.workload_balance;
        
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
            metrics.convergence_history(iter) = 0;  % Set first value to 0 instead of NaN
            unchanged_iterations = 0;
        end
        
        % Update visualization if requested
        if visualize && (mod(iter, 5) == 0 || iter < 10)
            subplot(2, 3, [1, 4]);
            env_utils.visualizeEnvironment(env, robots, tasks, auction_data);
            title(sprintf('Environment (Iteration %d, Time: %.1f s)', iter, auction_data.simulation_time));
            
            % Print detailed status
            local_printTaskStatus(auction_data, tasks, robots);
            
            % Update other plots
            if iter > 0
                % Price history
                subplot(2, 3, 2);
                env_utils.plotTaskPrices(metrics.price_history(:, 1:iter));
                title('Task Prices Over Time');
                
                % Assignment history
                subplot(2, 3, 3);
                env_utils.plotAssignments(metrics.assignment_history(:, 1:iter), length(robots));
                title('Task Assignments Over Time');
                
                % Convergence metric
                subplot(2, 3, 5);
                if iter >= 1
                    env_utils.plotConvergence(metrics.convergence_history(1:iter));
                else
                    % Create empty plot for first iteration
                    plot(0, 0);
                    xlim([0, 10]);
                    ylim([0, 1]);
                end
                title('Convergence Metric');
                
                % Workload distribution
                subplot(2, 3, 6);
                env_utils.plotWorkload(metrics.assignment_history(:, iter), tasks, robots);
                title('Current Workload Distribution');
            end
            
            drawnow;
            pause(0.01);
        end
        
        % IMPROVED: Better convergence criteria
        
        % Check if all tasks are completed
        all_tasks_completed = (sum(auction_data.completion_status) == length(tasks));
        
        % Check if all tasks are assigned or completed
        all_tasks_assigned_or_completed = all(auction_data.completion_status == 1 | auction_data.assignment > 0);
        
        % Check if we've made significant progress (at least half of tasks completed)
        significant_progress = (sum(auction_data.completion_status) >= length(tasks) / 2);
        
        % Check for simulation timeout
        simulation_timeout = auction_data.simulation_time >= 150; % Increased timeout
        
        % Check for stalled progress (no completions for a long time)
        stalled_progress = (unchanged_iterations >= 50) && (sum(auction_data.completion_status) > 0);
        
        % Determine if we've converged
        if all_tasks_completed || ...
           (all_tasks_assigned_or_completed && unchanged_iterations >= 20) || ...
           (significant_progress && unchanged_iterations >= 30) || ...
           simulation_timeout || ...
           stalled_progress
            if all_tasks_completed
                fprintf('Auction algorithm converged - all tasks completed\n');
            elseif all_tasks_assigned_or_completed
                fprintf('Auction algorithm converged - all tasks assigned or completed (stable for %d iterations)\n', unchanged_iterations);
            elseif significant_progress
                fprintf('Auction algorithm converged - significant progress made (%.1f%% tasks completed, stable for %d iterations)\n', ...
                        100 * sum(auction_data.completion_status) / length(tasks), unchanged_iterations);
            elseif simulation_timeout
                fprintf('Auction algorithm terminated - simulation timeout reached (%.1f seconds)\n', auction_data.simulation_time);
            else
                fprintf('Auction algorithm terminated - progress stalled for %d iterations\n', unchanged_iterations);
            end
            converged = true;
            break;
        end
        
        % Update recovery time if in recovery mode
        if isfield(auction_data, 'recovery_mode') && auction_data.recovery_mode && metrics.recovery_time == 0
            % Check if all tasks have been reassigned
            if isfield(auction_data, 'failed_robot') && ~isempty(auction_data.failed_robot)
                tasks_to_reassign = find(auction_data.initial_assignment == auction_data.failed_robot);
                if all(auction_data.assignment(tasks_to_reassign) ~= auction_data.failed_robot)
                    metrics.recovery_time = iter - metrics.failure_time;
                    fprintf('Recovery completed after %d iterations\n', metrics.recovery_time);
                end
            end
        end
        
        % Diagnostics every 10 iterations
        if mod(iter, 10) == 0
            local_analyzeTaskAllocation(auction_data, tasks);
            
            % Check progress of task completion
            num_completed = sum(auction_data.completion_status);
            num_assigned = sum(auction_data.assignment > 0 & auction_data.completion_status == 0);
            fprintf('Progress: %d/%d tasks completed, %d tasks in progress\n', ...
                    num_completed, length(tasks), num_assigned);
        end
    end
    
    % Trim history matrices to actual size
    metrics.price_history = metrics.price_history(:, 1:iter);
    metrics.assignment_history = metrics.assignment_history(:, 1:iter);
    metrics.convergence_history = metrics.convergence_history(1:iter);
    metrics.workload_balance_history = metrics.workload_balance_history(1:iter);
    
    % Final analysis
    fprintf('\n--- Final Task Allocation ---\n');
    local_analyzeTaskAllocation(auction_data, tasks);
    
    % Calculate makespan
    robot_utils = utils.robot;
    metrics.makespan = robot_utils.calculateMakespan(auction_data.assignment, tasks, robots);
    metrics.optimal_makespan = robot_utils.calculateOptimalMakespan(tasks, robots);
    metrics.optimality_gap = abs(metrics.makespan - metrics.optimal_makespan);
    
    % Store final state
    metrics.final_assignment = auction_data.assignment;
    metrics.final_prices = auction_data.prices;
    metrics.completion_status = auction_data.completion_status;
    metrics.task_oscillation_count = auction_data.task_oscillation_count;
    metrics.workload_balance = auction_data.workload_balance;
    
    % Print makespan information
    fprintf('Makespan: %.2f (Optimal: %.2f, Gap: %.2f)\n', ...
            metrics.makespan, metrics.optimal_makespan, metrics.optimality_gap);
    fprintf('Workload balance: %.2f (1.0 is perfect)\n', metrics.workload_balance);
    fprintf('Total task oscillations: %d\n', sum(auction_data.task_oscillation_count));
    
    % Check for unassigned tasks
    unassigned_count = sum(auction_data.assignment == 0);
    if unassigned_count > 0
        fprintf('Final result has %d unassigned tasks.\n', unassigned_count);
    end
end

function levels = local_calculateDependencyLevels(tasks)
    % CALCULATEDEPENDENCYLEVELS Calculate dependency level for each task
    %
    % Parameters:
    %   tasks - Array of task structures
    %
    % Returns:
    %   levels - Array of dependency levels (1 = no prerequisites, etc.)
    
    num_tasks = length(tasks);
    levels = zeros(num_tasks, 1);
    
    % First calculate indegree (number of prerequisites) for each task
    indegree = zeros(num_tasks, 1);
    for i = 1:num_tasks
        if isfield(tasks(i), 'prerequisites')
            indegree(i) = length(tasks(i).prerequisites);
        end
    end
    
    % Initialize levels - tasks with no prerequisites are at level 1
    levels(indegree == 0) = 1;
    
    % Calculate levels for remaining tasks using topological sort
    queue = find(indegree == 0);
    visited = false(num_tasks, 1);
    
    while ~isempty(queue)
        current = queue(1);
        queue(1) = [];
        
        if ~visited(current)
            visited(current) = true;
            
            % Find tasks that depend on this task
            for i = 1:num_tasks
                if isfield(tasks(i), 'prerequisites') && ismember(current, tasks(i).prerequisites)
                    % Update level if necessary
                    levels(i) = max(levels(i), levels(current) + 1);
                    
                    % Decrement indegree
                    indegree(i) = indegree(i) - 1;
                    
                    % Add to queue if all prerequisites have been processed
                    if indegree(i) == 0
                        queue = [queue, i];
                    end
                end
            end
        end
    end
    
    % Handle any tasks not visited (potential circular dependencies)
    if ~all(visited)
        warning('Potential circular dependencies detected.');
        unvisited = find(~visited);
        levels(unvisited) = max(levels) + 1;
    end
end

function [auction_data, new_assignments, messages] = local_processDependencyBasedBatches(auction_data, robots, tasks, available_tasks, params)
    % PROCESSDEPENDENCYBASEDBATCHES Process tasks in batches based on dependency levels
    %
    % Parameters:
    %   auction_data - Auction data structure
    %   robots - Array of robot structures
    %   tasks - Array of task structures
    %   available_tasks - List of currently available tasks
    %   params - Algorithm parameters
    %
    % Returns:
    %   auction_data - Updated auction data structure
    %   new_assignments - Number of new assignments
    %   messages - Number of messages exchanged
    
    % Initialize counters
    new_assignments = 0;
    messages = 0;
    
    % Calculate dependency levels if not already done
    if all(auction_data.dependency_levels == 0)
        auction_data.dependency_levels = local_calculateDependencyLevels(tasks);
    end
    
    % Create batches based on dependency levels
    max_level = max(auction_data.dependency_levels);
    batched_tasks = cell(max_level, 1);
    
    for level = 1:max_level
        batched_tasks{level} = find(auction_data.dependency_levels == level);
    end
    
    % Process each batch in order
    for level = 1:max_level
        batch = intersect(batched_tasks{level}, available_tasks);
        
        % Skip empty batches
        if isempty(batch)
            continue;
        end
        
        fprintf('Processing batch at dependency level %d: ', level);
        fprintf('%d ', batch);
        fprintf('\n');
        
        % Run auction for this batch
        [batch_auction_data, batch_new_assignments, batch_messages] = local_runBatchAuction(auction_data, robots, tasks, batch, params);
        
        % Update auction data
        auction_data = batch_auction_data;
        
        % Update counters
        new_assignments = new_assignments + batch_new_assignments;
        messages = messages + batch_messages;
    end
end

function [auction_data, new_assignments, messages] = local_runBatchAuction(auction_data, robots, tasks, batch, params)
    % RUNBATCHAUCTION Run auction algorithm for a specific batch of tasks
    %
    % Parameters:
    %   auction_data - Auction data structure
    %   robots - Array of robot structures
    %   tasks - Array of task structures
    %   batch - List of task IDs to include in this batch
    %   params - Algorithm parameters
    %
    % Returns:
    %   auction_data - Updated auction data structure
    %   new_assignments - Number of new assignments
    %   messages - Number of messages exchanged
    
    % Run simplified auction just for these tasks
    assignment = auction_data.assignment;
    prices = auction_data.prices;
    new_assignments = 0;
    messages = 0;
    
    % Reset bids for this batch
    bids = zeros(length(robots), length(tasks));
    
    % Bidding phase
    for r = 1:length(robots)
        % Skip failed robots
        if isfield(robots, 'failed') && robots(r).failed
            continue;
        end
        
        % For each task in the batch
        for i = 1:length(batch)
            task_idx = batch(i);
            
            % Skip completed tasks
            if auction_data.completion_status(task_idx) == 1
                continue;
            end
            
            % Calculate bid with balanced workload consideration
            bid = local_calculateBalancedBid(robots(r), tasks(task_idx), prices(task_idx), params, auction_data, robots);
            
            % Store bid
            bids(r, task_idx) = bid;
            
            % Count message
            messages = messages + 1;
        end
    end
    
    % Assignment phase - prioritize critical path tasks
    critical_tasks = [];
    normal_tasks = [];
    
    for i = 1:length(batch)
        task_idx = batch(i);
        if isfield(tasks(task_idx), 'on_critical_path') && tasks(task_idx).on_critical_path
            critical_tasks = [critical_tasks, task_idx];
        else
            normal_tasks = [normal_tasks, task_idx];
        end
    end
    
    % Process critical path tasks first
    process_order = [critical_tasks, normal_tasks];
    
    for i = 1:length(process_order)
        task_idx = process_order(i);
        
        % Skip completed tasks
        if auction_data.completion_status(task_idx) == 1
            continue;
        end
        
        % Find highest bidder
        [max_bid, max_bidder] = max(bids(:, task_idx));
        
        % If there is a valid bid
        if max_bid > 0
            % Check if assignment is new or different
            if assignment(task_idx) ~= max_bidder
                % If previously assigned, count as reassignment
                if assignment(task_idx) > 0
                    auction_data.task_oscillation_count(task_idx) = auction_data.task_oscillation_count(task_idx) + 1;
                end
                
                % Update assignment
                old_assignment = assignment(task_idx);
                assignment(task_idx) = max_bidder;
                new_assignments = new_assignments + 1;
                
                % Calculate earliest start time
                earliest_start_time = local_calculateEarliestStartTime(task_idx, tasks, auction_data);
                
                % Get task execution time
                if isfield(tasks, 'execution_time')
                    task_time = tasks(task_idx).execution_time;
                else
                    task_time = 1;
                end
                
                % Update projected timeline
                [auction_data.robot_available_time, auction_data.projected_completion_time] = ...
                    local_updateTaskTimeline(auction_data.robot_available_time, auction_data.projected_completion_time, ...
                                            max_bidder, task_idx, earliest_start_time, task_time, auction_data.simulation_time);
                
                % Log assignment
                fprintf('Batch: Task %d assigned to Robot %d (bid: %.2f, price: %.2f, start: %.2f)\n', ...
                        task_idx, max_bidder, max_bid, prices(task_idx), earliest_start_time);
                
                % Update robot workload
                utils = utils_manager();
                robot_utils = utils.robot;
                robots = robot_utils.updateRobotWorkload(robots, old_assignment, max_bidder, task_idx, task_time);
                
                % Update robot schedule
                auction_data.robot_schedule{max_bidder} = [auction_data.robot_schedule{max_bidder}; ...
                                                          struct('task_id', task_idx, ...
                                                                'start_time', earliest_start_time, ...
                                                                'end_time', earliest_start_time + task_time)];
                
                % Sort schedule by start time
                [~, sort_idx] = sort([auction_data.robot_schedule{max_bidder}.start_time]);
                auction_data.robot_schedule{max_bidder} = auction_data.robot_schedule{max_bidder}(sort_idx);
            end
            
            % Update price
            second_highest_bid = 0;
            current_bids = bids(:, task_idx);
            if sum(current_bids > 0) > 1
                sorted_bids = sort(current_bids(current_bids > 0), 'descend');
                second_highest_bid = sorted_bids(2);
            end
            
            if second_highest_bid > 0
                prices(task_idx) = second_highest_bid + params.epsilon;
            else
                prices(task_idx) = prices(task_idx) + params.epsilon;
            end
        end
    end
    
    % Update auction data
    auction_data.assignment = assignment;
    auction_data.prices = prices;
    auction_data.bids = bids;
    
    % Update workload balance metrics
    auction_data = local_updateWorkloadBalanceMetrics(auction_data, robots);
end

function earliest_start_time = local_calculateEarliestStartTime(task_id, tasks, auction_data)
    % CALCULATEEARLIESTSTARTIME Calculate the earliest possible start time for a task
    %
    % Parameters:
    %   task_id - ID of the task
    %   tasks - Array of task structures (optional if task info in auction_data)
    %   auction_data - Auction data structure
    %
    % Returns:
    %   earliest_start_time - Earliest possible start time for the task
    
    % Initialize with current time
    earliest_start_time = auction_data.simulation_time;
    
    % If tasks not provided, try to get prerequisites from auction_data
    if isempty(tasks) || nargin < 2
        % Try to extract prerequisites from auction_data
        % This requires tasks array to be stored in auction_data
        if isfield(auction_data, 'tasks')
            tasks = auction_data.tasks;
        else
            % No task data available, return current time
            return;
        end
    end
    
    % Check prerequisites
    if task_id <= length(tasks) && isfield(tasks(task_id), 'prerequisites') && ~isempty(tasks(task_id).prerequisites)
        for prereq = tasks(task_id).prerequisites
            if prereq <= length(auction_data.completion_status)
                % If prerequisite is completed, no delay
                if auction_data.completion_status(prereq) == 1
                    continue;
                end
                
                % If prerequisite is assigned but not completed, use projected completion time
                if auction_data.assignment(prereq) > 0
                    if auction_data.projected_completion_time(prereq) < inf
                        prereq_finish = auction_data.projected_completion_time(prereq);
                    else
                        % Fallback if no projection available
                        if isfield(tasks, 'execution_time')
                            prereq_finish = auction_data.simulation_time + tasks(prereq).execution_time;
                        else
                            prereq_finish = auction_data.simulation_time + 1;
                        end
                    end
                    
                    earliest_start_time = max(earliest_start_time, prereq_finish);
                else
                    % If prerequisite is not assigned, use a heuristic estimate
                    if isfield(tasks, 'execution_time')
                        prereq_estimate = auction_data.simulation_time + tasks(prereq).execution_time;
                    else
                        prereq_estimate = auction_data.simulation_time + 1;
                    end
                    
                    earliest_start_time = max(earliest_start_time, prereq_estimate);
                end
            end
        end
    end
    
    return;
end

function [robot_available_time, projected_completion_time] = local_updateTaskTimeline(robot_available_time, projected_completion_time, robot_id, task_id, earliest_start_time, task_time, current_time)
    % UPDATETASKTIMELINE Update the robot availability and task completion timeline
    %
    % Parameters:
    %   robot_available_time - Array of times when each robot will be available
    %   projected_completion_time - Array of projected completion times for tasks
    %   robot_id - ID of the robot being assigned
    %   task_id - ID of the task being assigned
    %   earliest_start_time - Earliest possible start time for the task
    %   task_time - Execution time for the task
    %   current_time - Current simulation time
    %
    % Returns:
    %   robot_available_time - Updated robot availability times
    %   projected_completion_time - Updated task completion projections
    
    % Calculate when robot will be available to start this task
    robot_start_time = max(robot_available_time(robot_id), earliest_start_time);
    
    % Calculate when this task will finish
    task_finish_time = robot_start_time + task_time;
    
    % Update robot's next available time
    robot_available_time(robot_id) = task_finish_time;
    
    % Update task's projected completion time
    projected_completion_time(task_id) = task_finish_time;
end

function auction_data = local_updateWorkloadBalanceMetrics(auction_data, robots)
    % UPDATEWORKLOADBALANCEMETRICS Update workload balance metrics
    %
    % Parameters:
    %   auction_data - Auction data structure
    %   robots - Array of robot structures
    %
    % Returns:
    %   auction_data - Updated auction data structure with balance metrics
    
    % Extract workloads
    robot_workloads = zeros(1, length(robots));
    for i = 1:length(robots)
        robot_workloads(i) = robots(i).workload;
    end
    
    % Skip failed robots
    valid_robots = true(1, length(robots));
    for i = 1:length(robots)
        if isfield(robots(i), 'failed') && robots(i).failed
            valid_robots(i) = false;
        end
    end
    
    valid_workloads = robot_workloads(valid_robots);
    
    % Avoid division by zero
    if isempty(valid_workloads) || all(valid_workloads == 0)
        auction_data.workload_balance = 1.0;
        auction_data.min_max_workload_ratio = 1.0;
        return;
    end
    
    % Calculate workload statistics
    min_workload = min(valid_workloads);
    max_workload = max(valid_workloads);
    avg_workload = mean(valid_workloads);
    std_workload = std(valid_workloads);
    
    % Calculate coefficient of variation as a balance metric
    % Lower CV means better balance
    if avg_workload > 0
        cv = std_workload / avg_workload;
        
        % Convert to a 0-1 scale where 1 is perfect balance
        balance = 1 / (1 + cv);
    else
        balance = 1.0;
    end
    
    % Calculate min/max workload ratio (another balance metric)
    if max_workload > 0
        min_max_ratio = min_workload / max_workload;
    else
        min_max_ratio = 1.0;
    end
    
    % Update auction data
    auction_data.workload_balance = balance;
    auction_data.min_max_workload_ratio = min_max_ratio;
end

function future_tasks = local_identifyFutureTasks(tasks, auction_data, progress_threshold)
    % IDENTIFYFUTURETASKS Identify tasks that will soon be available based on dependencies
    %
    % Parameters:
    %   tasks - Array of task structures
    %   auction_data - Auction data structure
    %   progress_threshold - Progress threshold to consider a task "almost complete" (0-1)
    %
    % Returns:
    %   future_tasks - Array of task IDs that will be available soon
    
    future_tasks = [];
    
    % For each task that is not completed and not available
    for i = 1:length(tasks)
        if auction_data.completion_status(i) == 0 && ~auction_data.dependency_met(i)
            % Check if all prerequisites are either completed or close to completion
            all_prerequisites_near_completion = true;
            
            if isfield(tasks(i), 'prerequisites') && ~isempty(tasks(i).prerequisites)
                for prereq = tasks(i).prerequisites
                    % Skip if prerequisite is already completed
                    if auction_data.completion_status(prereq) == 1
                        continue;
                    end
                    
                    % Check if prerequisite is in progress and near completion
                    if auction_data.assignment(prereq) > 0 && ...
                       (auction_data.task_progress(prereq) / tasks(prereq).execution_time) >= progress_threshold
                        continue;
                    end
                    
                    % If we get here, at least one prerequisite is not near completion
                    all_prerequisites_near_completion = false;
                    break;
                end
            end
            
            % If all prerequisites are near completion, add to future tasks
            if all_prerequisites_near_completion
                future_tasks = [future_tasks, i];
            end
        end
    end
end

function auction_data = local_resetPricesForBlockedTasks(auction_data, tasks, available_tasks)
    % RESETPRICESFORBLOCKEDTASKS Reset prices for tasks that are blocked by dependencies
    %
    % Parameters:
    %   auction_data - Auction data structure
    %   tasks - Array of task structures
    %   available_tasks - List of task IDs that are currently available
    %
    % Returns:
    %   auction_data - Updated auction data structure
    
    % Find tasks that are not available and not completed
    for i = 1:length(tasks)
        if auction_data.completion_status(i) == 0 && ~ismember(i, available_tasks)
            % Check if this task has any unfinished prerequisites
            has_unfinished_prereqs = false;
            
            if isfield(tasks(i), 'prerequisites') && ~isempty(tasks(i).prerequisites)
                for j = 1:length(tasks(i).prerequisites)
                    prereq = tasks(i).prerequisites(j);
                    if auction_data.completion_status(prereq) == 0
                        has_unfinished_prereqs = true;
                        break;
                    end
                end
            end
            
            % If task has unfinished prerequisites, reset its price to prevent inflation
            if has_unfinished_prereqs
                auction_data.prices(i) = 0;
            end
        end
    end
end

function local_printTaskStatus(auction_data, tasks, robots)
    % PRINTTASKSTATUS Print detailed status of tasks
    %
    % Parameters:
    %   auction_data - Auction data structure
    %   tasks - Array of task structures
    %   robots - Array of robot structures
    
    fprintf('\n--- Current Task Status ---\n');
    fprintf('ID\tAssigned\tProgress\tPrice\t\tPrereqs\t\tProjected Completion\n');
    fprintf('----------------------------------------------------------------------------------\n');
    
    for i = 1:length(tasks)
        % Get assignment status
        if auction_data.completion_status(i) == 1
            status = 'Completed';
        elseif auction_data.assignment(i) == 0
            status = 'Unassigned';
        else
            robot_id = auction_data.assignment(i);
            if robot_id <= length(robots) && isfield(robots, 'failed') && robots(robot_id).failed
                status = sprintf('R%d (Failed)', robot_id);
            else
                status = sprintf('Robot %d', robot_id);
            end
        end
        
        % Get progress
        if auction_data.completion_status(i) == 1
            progress = '100%';
        else
            if isfield(tasks, 'execution_time') && tasks(i).execution_time > 0
                progress_pct = (auction_data.task_progress(i) / tasks(i).execution_time) * 100;
                progress = sprintf('%.1f%%', progress_pct);
            else
                progress = '0%';
            end
        end
        
        % Get prerequisites
        if isfield(tasks(i), 'prerequisites') && ~isempty(tasks(i).prerequisites)
            prereq_str = sprintf('%d ', tasks(i).prerequisites);
        else
            prereq_str = 'None';
        end
        
        % Get projected completion
        if auction_data.completion_status(i) == 1
            proj_completion = 'Done';
        elseif auction_data.projected_completion_time(i) < inf
            proj_completion = sprintf('%.1f', auction_data.projected_completion_time(i));
        else
            proj_completion = 'N/A';
        end
        
        % Print task information
        fprintf('%d\t%s\t\t%s\t\t%.2f\t\t%s\t\t%s\n', ...
                i, status, progress, auction_data.prices(i), prereq_str, proj_completion);
    end
    
    fprintf('\n');
end

function local_analyzeTaskAllocation(auction_data, tasks)
    % ANALYZETASKALLOCATION Analyze the current task allocation status
    %
    % Parameters:
    %   auction_data - Auction data structure
    %   tasks - Array of task structures
    
    % Count assignments by robot
    assignments = auction_data.assignment;
    unique_robots = unique(assignments(assignments > 0));
    
    fprintf('\n--- Task Allocation Analysis ---\n');
    
    % Check for unassigned tasks
    unassigned_count = sum(assignments == 0);
    fprintf('Unassigned tasks: %d (%.1f%%)\n', ...
            unassigned_count, unassigned_count/length(tasks)*100);
    
    % Count completed tasks
    completed_count = sum(auction_data.completion_status == 1);
    fprintf('Completed tasks: %d (%.1f%%)\n', ...
            completed_count, completed_count/length(tasks)*100);
    
    % Calculate workload by robot
    fprintf('\nWorkload Distribution:\n');
    fprintf('Robot\tTask Count\tExecution Time\tUtilization\tAvailable at\n');
    fprintf('----------------------------------------------------------------------\n');
    
    % Calculate total execution time
    total_execution_time = 0;
    for i = 1:length(tasks)
        if isfield(tasks, 'execution_time')
            total_execution_time = total_execution_time + tasks(i).execution_time;
        else
            total_execution_time = total_execution_time + 1;  % Default execution time
        end
    end
    
    % Analyze robot workloads
    max_workload = 0;
    min_workload = inf;
    
    for robot_id = unique_robots'
        robot_tasks = find(assignments == robot_id);
        task_count = length(robot_tasks);
        
        % Calculate execution time for this robot
        robot_execution_time = 0;
        for i = 1:task_count
            task_id = robot_tasks(i);
            if isfield(tasks, 'execution_time')
                robot_execution_time = robot_execution_time + tasks(task_id).execution_time;
            else
                robot_execution_time = robot_execution_time + 1;  % Default execution time
            end
        end
        
        % Update min/max workload
        if robot_execution_time > max_workload
            max_workload = robot_execution_time;
        end
        if robot_execution_time < min_workload
            min_workload = robot_execution_time;
        end
        
        % Calculate utilization percentage
        if total_execution_time > 0
            utilization = robot_execution_time / total_execution_time * 100;
        else
            utilization = 0;
        end
        
        % Get robot available time
        if isfield(auction_data, 'robot_available_time') && robot_id <= length(auction_data.robot_available_time)
            available_at = auction_data.robot_available_time(robot_id);
            if available_at == inf
                available_str = 'N/A';
            else
                available_str = sprintf('%.1f', available_at);
            end
        else
            available_str = 'Unknown';
        end
        
        fprintf('%d\t%d\t\t%.2f\t\t%.1f%%\t\t%s\n', ...
                robot_id, task_count, robot_execution_time, utilization, available_str);
    end
    
    % Calculate workload imbalance
    if length(unique_robots) > 1 && min_workload < inf
        imbalance = (max_workload - min_workload) / max_workload * 100;
        fprintf('\nWorkload imbalance: %.1f%%\n', imbalance);
    end
    
    % Check for oscillating tasks
    if isfield(auction_data, 'task_oscillation_count')
        oscillating_tasks = find(auction_data.task_oscillation_count > 2);
        if ~isempty(oscillating_tasks)
            fprintf('\nTasks with high oscillation (>2 reassignments):\n');
            for i = 1:length(oscillating_tasks)
                task_id = oscillating_tasks(i);
                fprintf('Task %d: %d reassignments\n', task_id, auction_data.task_oscillation_count(task_id));
            end
        end
    end
    
    fprintf('\n');
end

function local_analyzeBidDistribution(auction_data, robots, tasks)
    % ANALYZEENDDISTRIBUTION Analyze the distribution of bids
    %
    % Parameters:
    %   auction_data - Auction data structure
    %   robots - Array of robot structures
    %   tasks - Array of task structures
    
    % Get bid data
    bids = auction_data.bids;
    
    fprintf('\n--- Bid Distribution Analysis ---\n');
    
    % Calculate bid statistics
    non_zero_bids = bids(bids > 0);
    
    if ~isempty(non_zero_bids)
        mean_bid = mean(non_zero_bids);
        min_bid = min(non_zero_bids);
        max_bid = max(non_zero_bids);
        std_bid = std(non_zero_bids);
        
        fprintf('Bid statistics:\n');
        fprintf('  Mean: %.2f\n', mean_bid);
        fprintf('  Min: %.2f\n', min_bid);
        fprintf('  Max: %.2f\n', max_bid);
        fprintf('  Std Dev: %.2f\n', std_bid);
        
        % Analyze bid spread for each task
        fprintf('\nTasks with highest bid spread:\n');
        fprintf('Task\tMin Bid\tMax Bid\tSpread\tAssigned\n');
        fprintf('----------------------------------------------\n');
        
        task_spreads = zeros(length(tasks), 1);
        
        for i = 1:length(tasks)
            task_bids = bids(:, i);
            task_bids = task_bids(task_bids > 0);
            
            if length(task_bids) > 1
                task_min = min(task_bids);
                task_max = max(task_bids);
                spread = task_max - task_min;
                task_spreads(i) = spread;
            else
                task_spreads(i) = 0;
            end
        end
        
        % Sort tasks by spread
        [sorted_spreads, sort_idx] = sort(task_spreads, 'descend');
        
        % Print top 5 tasks by spread
        for i = 1:min(5, length(sort_idx))
            task_id = sort_idx(i);
            task_bids = bids(:, task_id);
            task_bids = task_bids(task_bids > 0);
            
            if length(task_bids) > 1
                task_min = min(task_bids);
                task_max = max(task_bids);
                spread = task_max - task_min;
                
                % Get assigned robot
                assigned_robot = auction_data.assignment(task_id);
                if assigned_robot == 0
                    assigned = 'None';
                else
                    assigned = sprintf('%d', assigned_robot);
                end
                
                fprintf('%d\t%.2f\t%.2f\t%.2f\t%s\n', ...
                        task_id, task_min, task_max, spread, assigned);
            end
        end
    else
        fprintf('No bids have been placed yet.\n');
    end
    
    fprintf('\n');
end

function [critical_path, task_depths] = findCriticalPath(tasks)
    % FINDCRITICALPATH Find the critical path in the task dependency graph
    %
    % Parameters:
    %   tasks - Array of task structures
    %
    % Returns:
    %   critical_path - Array of task IDs on the critical path
    %   task_depths - Depth of each task in the dependency graph
    
    num_tasks = length(tasks);
    
    % Initialize task depths
    task_depths = zeros(1, num_tasks);
    
    % Calculate max path length to each task (depth)
    for i = 1:num_tasks
        if task_depths(i) == 0  % If not already calculated
            task_depths(i) = calculateTaskDepth(i, tasks, task_depths);
        end
    end
    
    % Find the maximum depth (longest path)
    [max_depth, max_depth_idx] = max(task_depths);
    
    % Trace back the critical path from the deepest task
    critical_path = traceCriticalPath(max_depth_idx, tasks, task_depths);
    
    % Reverse the path to get start-to-end ordering
    critical_path = fliplr(critical_path);
    
    fprintf('Identified critical path: ');
    fprintf('%d ', critical_path);
    fprintf('\n');
end

function depth = calculateTaskDepth(task_id, tasks, memo)
    % Recursive function to calculate the maximum path length ending at this task
    
    % If already calculated, return memoized value
    if memo(task_id) > 0
        depth = memo(task_id);
        return;
    end
    
    % No prerequisites means depth of 1
    if isempty(tasks(task_id).prerequisites)
        depth = 1;
    else
        % Calculate max depth from prerequisites
        max_prereq_depth = 0;
        for i = 1:length(tasks(task_id).prerequisites)
            prereq_id = tasks(task_id).prerequisites(i);
            prereq_depth = calculateTaskDepth(prereq_id, tasks, memo);
            max_prereq_depth = max(max_prereq_depth, prereq_depth);
        end
        depth = max_prereq_depth + 1;
    end
    
    % Store in memoization table
    memo(task_id) = depth;
end

function path = traceCriticalPath(end_task, tasks, task_depths)
    % Trace the critical path from end to start
    
    path = end_task;
    current = end_task;
    
    while ~isempty(tasks(current).prerequisites)
        max_depth = 0;
        max_prereq = 0;
        
        % Find prerequisite with maximum depth
        for i = 1:length(tasks(current).prerequisites)
            prereq_id = tasks(current).prerequisites(i);
            if task_depths(prereq_id) > max_depth
                max_depth = task_depths(prereq_id);
                max_prereq = prereq_id;
            end
        end
        
        if max_prereq > 0
            path = [max_prereq, path];
            current = max_prereq;
        else
            break;
        end
    end
end