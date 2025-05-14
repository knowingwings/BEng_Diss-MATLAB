function utils = auction_utils()
    % AUCTION_UTILS - Returns function handles for auction-related functions
    utils = struct(...
        'initializeAuctionData', @local_initializeAuctionData, ...
        'distributedAuctionStep', @local_distributedAuctionStep, ...
        'calculateBid', @local_calculateBid, ...
        'calculateFutureBid', @local_calculateFutureBid, ...
        'initiateRecovery', @local_initiateRecovery, ...
        'runAuctionSimulation', @local_runAuctionSimulation, ...
        'analyzeTaskAllocation', @local_analyzeTaskAllocation, ...
        'analyzeBidDistribution', @local_analyzeBidDistribution, ...
        'resetPricesForBlockedTasks', @local_resetPricesForBlockedTasks, ...
        'printTaskStatus', @local_printTaskStatus, ...
        'progressTaskExecution', @local_progressTaskExecution ...
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
    
    % FIX: Add logging for available tasks
    if ~isempty(bidding_tasks)
        fprintf('Current bidding tasks: ');
        fprintf('%d ', bidding_tasks);
        fprintf('\n');
    end
    
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
                
                % FIX: Log the assignment
                fprintf('Task %d assigned to Robot %d (bid: %.2f, price: %.2f)\n', ...
                       task_idx, max_bidder, max_bid, prices(task_idx));
                
                % Update robot workload using utils_manager
                utils = utils_manager();
                robot_utils = utils.robot;
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
                    % Increment progress by time step
                    auction_data.task_progress(i) = auction_data.task_progress(i) + params.time_step;
                    
                    % If task has been executed for its required time, mark as completed
                    if auction_data.task_progress(i) >= tasks(i).execution_time
                        auction_data.completion_status(i) = 1;
                        
                        % Log task completion
                        fprintf('Task %d completed by Robot %d\n', i, robot_id);
                        
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
    
    % FIX: Improve workload factor calculation
    workload_factor = params.alpha(2) * (1 / (robot.workload + 1));
    
    % IMPROVED: Add value for tasks on critical path and for tasks with dependencies
    critical_path_bonus = 0;
    dependency_bonus = 0;
    
    if isfield(task, 'on_critical_path') && task.on_critical_path
        critical_path_bonus = params.alpha(4) * 3;  % Triple importance for critical path tasks
    end
    
    % Add bonus for tasks with no prerequisites (to get the process started)
    if ~isfield(task, 'prerequisites') || isempty(task.prerequisites)
        dependency_bonus = params.alpha(5) * 2;
    end
    
    % FIX: Add time efficiency factor - reward shorter tasks
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
                if isfield(task, 'execution_time')
                    remaining_time = task.execution_time - auction_data.task_progress(prereq);
                    prereq_completion_time = auction_data.simulation_time + remaining_time;
                    earliest_available_time = max(earliest_available_time, prereq_completion_time);
                end
            else
                % If prerequisite isn't assigned, use a pessimistic estimate
                if isfield(task, 'execution_time')
                    earliest_available_time = max(earliest_available_time, auction_data.simulation_time + task.execution_time);
                end
            end
        end
    end
    
    % Calculate time discount factor (value decreases with wait time)
    time_to_availability = max(0, earliest_available_time - auction_data.simulation_time);
    
    % FIX: Use gentler time discount to encourage future task bidding
    time_discount = exp(-0.05 * time_to_availability);  % Slower exponential decay with time
    
    % Apply discount to bid
    discounted_bid = base_bid * time_discount;
    
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
    end
    
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
    
    % ADDITION: Add time step to parameters if not present
    if ~isfield(params, 'time_step')
        params.time_step = 0.1; % Default time step
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
        
        % IMPROVED: Better convergence criteria
        % FIX: Completely changed convergence criteria
        
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
    fprintf('ID\tAssigned\tProgress\tPrice\t\tPrerequisites\n');
    fprintf('------------------------------------------------------------------\n');
    
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
        
        % Print task information
        fprintf('%d\t%s\t\t%s\t\t%.2f\t\t%s\n', ...
                i, status, progress, auction_data.prices(i), prereq_str);
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
    fprintf('Robot\tTask Count\tExecution Time\tUtilization\n');
    fprintf('------------------------------------------------------\n');
    
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
        
        fprintf('%d\t%d\t\t%.2f\t\t%.1f%%\n', ...
                robot_id, task_count, robot_execution_time, utilization);
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