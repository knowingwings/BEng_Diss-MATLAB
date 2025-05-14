function utils = auction_utils()
    % AUCTION_UTILS - Returns function handles for auction-related functions
    utils = struct(...
        'initializeAuctionData', @local_initializeAuctionData, ...
        'distributedAuctionStep', @local_distributedAuctionStep, ...
        'calculateBid', @local_calculateBid, ...
        'calculateFutureBid', @local_calculateFutureBid, ... % NEW FUNCTION
        'initiateRecovery', @local_initiateRecovery, ...
        'runAuctionSimulation', @local_runAuctionSimulation, ...
        'analyzeTaskAllocation', @local_analyzeTaskAllocation, ...
        'analyzeBidDistribution', @local_analyzeBidDistribution, ...
        'resetPricesForBlockedTasks', @local_resetPricesForBlockedTasks, ...
        'printTaskStatus', @local_printTaskStatus, ...
        'progressTaskExecution', @local_progressTaskExecution ... % NEW FUNCTION
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
    
    % IMPROVED: Find tasks that will soon be available based on current progress
    future_available_tasks = [];
    for i = 1:length(tasks)
        if ~ismember(i, available_tasks) && auction_data.completion_status(i) == 0
            % Check if prerequisites are in progress and close to completion
            prerequisites_met = true;
            for j = 1:length(tasks(i).prerequisites)
                prereq = tasks(i).prerequisites(j);
                if auction_data.completion_status(prereq) == 0
                    % If prerequisite is assigned and more than 70% complete, consider this task
                    if assignment(prereq) > 0 && auction_data.task_progress(prereq) >= 0.7 * tasks(prereq).execution_time
                        continue;
                    else
                        prerequisites_met = false;
                        break;
                    end
                end
            end
            if prerequisites_met
                future_available_tasks = [future_available_tasks, i];
            end
        end
    end
    
    % Combine currently available and future available tasks for bidding
    bidding_tasks = union(available_tasks, future_available_tasks);
    
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
                    bid = local_calculateBid(robots(r), tasks(task_idx), prices(task_idx), params);
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
                
                % Calculate when this task can be started based on prerequisites
                earliest_start_time = auction_data.simulation_time;
                for j = 1:length(tasks(task_idx).prerequisites)
                    prereq = tasks(task_idx).prerequisites(j);
                    if auction_data.completion_status(prereq) == 0
                        % If prerequisite is assigned, use its projected completion time
                        if assignment(prereq) > 0
                            prerequisite_completion = auction_data.projected_completion_time(prereq);
                            earliest_start_time = max(earliest_start_time, prerequisite_completion);
                        else
                            % If prerequisite isn't assigned, use a heuristic estimate
                            earliest_start_time = max(earliest_start_time, auction_data.simulation_time + tasks(prereq).execution_time);
                        end
                    end
                end
                
                % Update robot workloads
                if isfield(tasks, 'execution_time')
                    task_time = tasks(task_idx).execution_time;
                else
                    task_time = 1;  % Default execution time
                end
                
                % Update robot available time and projected completion time
                auction_data.robot_available_time(max_bidder) = max(auction_data.robot_available_time(max_bidder), earliest_start_time) + task_time;
                auction_data.projected_completion_time(task_idx) = auction_data.robot_available_time(max_bidder);
                
                utils = utils_manager();
                robot_utils = utils.robot();
                robots = robot_utils.updateRobotWorkload(robots, old_assignment, max_bidder, task_idx, task_time);
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
                    % Increment progress by time step
                    auction_data.task_progress(i) = auction_data.task_progress(i) + params.time_step;
                    
                    % If task has been executed for its required time, mark as completed
                    if auction_data.task_progress(i) >= tasks(i).execution_time
                        auction_data.completion_status(i) = 1;
                        fprintf('Task %d completed by Robot %d\n', i, robot_id);
                        
                        % Update robot's completed tasks list
                        utils = utils_manager();
                        robot_utils = utils.robot();
                        robots = robot_utils.updateRobotCompletedTasks(robots, i, robot_id);
                        
                        % Update robot available time
                        auction_data.robot_available_time(robot_id) = auction_data.simulation_time;
                    end
                end
            end
        end
    end
end

function bid = local_calculateBid(robot, task, current_price, params)
    % CALCULATEBID Calculate bid value for a task
    %
    % Parameters:
    %   robot - Robot structure
    %   task - Task structure
    %   current_price - Current price of the task
    %   params - Algorithm parameters
    %
    % Returns:
    %   bid - Calculated bid value
    
    % Calculate capability match with proper normalization
    capability_match = dot(robot.capabilities, task.capabilities_required) / ...
                      (norm(robot.capabilities) * norm(task.capabilities_required));
    
    % Calculate distance-based cost with exponential decay
    distance = norm(robot.position - task.position);
    distance_cost = params.alpha(3) * (1 - exp(-distance/2));
    
    % Ensure workload factor is effective even at zero workload
    workload_factor = params.alpha(2) / (1 + robot.workload);
    
    % IMPROVED: Add value for tasks on critical path
    critical_path_bonus = 0;
    if isfield(task, 'on_critical_path') && task.on_critical_path
        critical_path_bonus = params.alpha(4) * 2;  % Double importance for critical path tasks
    end
    
    % Calculate bid value with proper weighting
    bid = params.alpha(1) * capability_match + ...
          workload_factor - ...
          distance_cost + ...
          params.alpha(4) * (1 / (task.execution_time + 0.1)) + ...
          critical_path_bonus;
    
    % Ensure bid exceeds current price by at least epsilon
    if bid <= current_price
        bid = current_price + params.epsilon;
    end
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
    earliest_available_time = auction_data.simulation_time;
    
    for prereq = task.prerequisites
        if auction_data.completion_status(prereq) == 0
            % Get progress on this prerequisite
            if auction_data.assignment(prereq) > 0
                % Calculate remaining time
                remaining_time = tasks(prereq).execution_time - auction_data.task_progress(prereq);
                prereq_completion_time = auction_data.simulation_time + remaining_time;
                earliest_available_time = max(earliest_available_time, prereq_completion_time);
            else
                % If prerequisite isn't assigned, use a pessimistic estimate
                earliest_available_time = max(earliest_available_time, auction_data.simulation_time + tasks(prereq).execution_time);
            end
        end
    end
    
    % Calculate time discount factor (value decreases with wait time)
    time_to_availability = max(0, earliest_available_time - auction_data.simulation_time);
    time_discount = exp(-0.1 * time_to_availability);  % Exponential decay with time
    
    % Apply discount to bid
    discounted_bid = base_bid * time_discount;
    
    % Ensure minimum bid
    if discounted_bid <= current_price
        discounted_bid = current_price + params.epsilon;
    end
    
    bid = discounted_bid;
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
    
    % ADDITION: Add time step to parameters if not present
    if ~isfield(params, 'time_step')
        params.time_step = 0.1; % Default time step
    end
    
    % Maximum iterations
    max_iterations = 1000;
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
        
        % Simulate one step of the distributed auction algorithm
        [auction_data, new_assignments, messages] = local_distributedAuctionStep(auction_data, robots, tasks, available_tasks, params);
        metrics.messages = metrics.messages + messages;
        
        % Update performance metrics
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
                
                % Convergence metric - FIXED: Check for valid data range
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
        
        % IMPROVED: Check completion of critical path tasks
        critical_path_completed = true;
        for i = 1:length(critical_path)
            if critical_path(i) <= length(tasks) && auction_data.completion_status(critical_path(i)) == 0
                critical_path_completed = false;
                break;
            end
        end
        
        % IMPROVED: Better convergence criteria
        % Convergence when:
        % 1. All tasks are completed or assigned, or
        % 2. Assignments haven't changed for several iterations AND
        %    all available tasks are assigned
        % 3. Simulation time has passed a threshold
        
        % Check if all tasks are completed or assigned
        all_tasks_handled = all(auction_data.completion_status == 1 | auction_data.assignment > 0);
        
        % Check if all available tasks are assigned
        available_tasks_assigned = true;
        for i = 1:length(available_tasks)
            if auction_data.assignment(available_tasks(i)) == 0
                available_tasks_assigned = false;
                break;
            end
        end
        
        % Check for simulation timeout
        simulation_timeout = auction_data.simulation_time >= 60; % 60 second timeout
        
        % Determine if we've converged
        if (all_tasks_handled && unchanged_iterations >= 5) || 
           (available_tasks_assigned && unchanged_iterations >= 10) || 
           simulation_timeout
            if all_tasks_handled
                fprintf('Auction algorithm converged - all tasks assigned or completed\n');
            elseif available_tasks_assigned
                fprintf('Auction algorithm converged - all available tasks assigned (stable for %d iterations)\n', unchanged_iterations);
            else
                fprintf('Auction algorithm terminated - simulation timeout reached\n');
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
        
        % Break if we go too long without progress
        if iter > 50 && unchanged_iterations > 30 && sum(auction_data.completion_status) == 0
            fprintf('Auction algorithm terminated - no progress after %d iterations\n', unchanged_iterations);
            break;
        end
    end
    
    % Trim history matrices to actual size
    metrics.price_history = metrics.price_history(:, 1:iter);
    metrics.assignment_history = metrics.assignment_history(:, 1:iter);
    metrics.convergence_history = metrics.convergence_history(1:iter);
    
    % Final analysis
    fprintf('\n--- Final Task Allocation ---\n');
    local_analyzeTaskAllocation(auction_data, tasks);
    
    % Calculate makespan
    robot_utils = utils.robot();
    metrics.makespan = robot_utils.calculateMakespan(auction_data.assignment, tasks, robots);
    metrics.optimal_makespan = robot_utils.calculateOptimalMakespan(tasks, robots);
    metrics.optimality_gap = abs(metrics.makespan - metrics.optimal_makespan);
    
    % Store final state
    metrics.final_assignment = auction_data.assignment;
    metrics.final_prices = auction_data.prices;
    metrics.completion_status = auction_data.completion_status;
    metrics.task_oscillation_count = auction_data.task_oscillation_count;
    
    % Print makespan information
    fprintf('Makespan: %.2f (Optimal: %.2f, Gap: %.2f)\n', ...
            metrics.makespan, metrics.optimal_makespan, metrics.optimality_gap);
    fprintf('Total task oscillations: %d\n', sum(auction_data.task_oscillation_count));
    
    % Check for unassigned tasks
    unassigned_count = sum(auction_data.assignment == 0);
    if unassigned_count > 0
        fprintf('Final result has %d unassigned tasks.\n', unassigned_count);
    end
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