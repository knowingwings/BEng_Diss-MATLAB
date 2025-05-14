function utils = auction_utils()
    % AUCTION_UTILS - Returns function handles for auction-related functions
    utils = struct(...
        'initializeAuctionData', @local_initializeAuctionData, ...
        'distributedAuctionStep', @local_distributedAuctionStep, ...
        'calculateBid', @local_calculateBid, ...
        'initiateRecovery', @local_initiateRecovery, ...
        'runAuctionSimulation', @local_runAuctionSimulation, ...
        'analyzeTaskAllocation', @local_analyzeTaskAllocation, ...
        'analyzeBidDistribution', @local_analyzeBidDistribution, ...
        'resetPricesForBlockedTasks', @local_resetPricesForBlockedTasks, ...
        'printTaskStatus', @local_printTaskStatus ...
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
    
    % ADDITION: Initialize task progress tracking
    auction_data.task_progress = zeros(length(tasks), 1);
    
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
    
    % Phase 1: Bidding
    % Each robot bids on available tasks
    for r = 1:length(robots)
        % Skip failed robots
        if isfield(robots, 'failed') && robots(r).failed
            continue;
        end
        
        % For each available task
        for i = 1:length(available_tasks)
            task_idx = available_tasks(i);
            
            % Skip tasks that are already completed
            if auction_data.completion_status(task_idx) == 1
                continue;
            end
            
            % Calculate bid for this task
            if auction_data.recovery_mode && isfield(params, 'beta')
                % Adjust bid calculation during recovery
                bid = local_calculateRecoveryBid(robots(r), tasks(task_idx), prices(task_idx), params);
            else
                bid = local_calculateBid(robots(r), tasks(task_idx), prices(task_idx), params);
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
    for i = 1:length(available_tasks)
        task_idx = available_tasks(i);
        
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
                
                % Update robot workloads
                if isfield(tasks, 'execution_time')
                    task_time = tasks(task_idx).execution_time;
                else
                    task_time = 1;  % Default execution time
                end
                
                robot_utils = robot_utils();
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
    
    % ADDITION: Task completion simulation
    % Add this section after task assignments have been updated
    for i = 1:length(tasks)
        if auction_data.assignment(i) > 0 && auction_data.completion_status(i) == 0 && ismember(i, available_tasks)
            robot_id = auction_data.assignment(i);
            
            % If robot is not failed and task was assigned for enough iterations, mark as completed
            if robot_id <= length(robots) && ~(isfield(robots, 'failed') && robots(robot_id).failed)
                if ~isfield(auction_data, 'task_progress')
                    auction_data.task_progress = zeros(size(auction_data.assignment));
                end
                
                % Increment progress by time step (normally this would be actual execution progress)
                auction_data.task_progress(i) = auction_data.task_progress(i) + params.time_step;
                
                % If task has been executed for its required time, mark as completed
                if auction_data.task_progress(i) >= tasks(i).execution_time
                    auction_data.completion_status(i) = 1;
                    fprintf('Task %d completed by Robot %d\n', i, robot_id);
                    
                    % Update robot's completed tasks list
                    robot_utils = robot_utils();
                    robots = robot_utils.updateRobotCompletedTasks(robots, i, robot_id);
                end
            end
        end
    end
end

function bid = local_calculateBid(robot, task, current_price, params)
    % CALCULATEBID Calculate bid value for a task
    
    % FIXED: Only bid on valid tasks
    if ~isfield(task, 'id') || task.id == 0
        bid = 0;
        return;
    end
    
    % Calculate capability match
    capability_match = dot(robot.capabilities, task.capabilities_required);
    
    % Calculate distance-based cost
    distance = norm(robot.position - task.position);
    distance_cost = distance * params.alpha(3);
    
    % Calculate workload balancing factor
    % FIXED: Better workload balancing 
    workload_factor = 1 / (1 + robot.workload);
    
    % ADDITION: Consider task dependencies
    dependency_bonus = 0;
    if isfield(task, 'prerequisites') && ~isempty(task.prerequisites)
        % Give bonus for tasks that extend existing work
        for prereq_id = task.prerequisites
            if isfield(robot, 'completed_tasks') && ismember(prereq_id, robot.completed_tasks)
                dependency_bonus = params.alpha(5);
                break;
            end
        end
    end
    
    % Calculate bid value
    bid = params.alpha(1) * capability_match + ...
          params.alpha(2) * workload_factor - ...
          distance_cost + ...
          params.alpha(4) * (1 / task.execution_time) + ...
          dependency_bonus;
    
    % If bid is less than current price plus epsilon, don't bid
    if bid <= current_price + params.epsilon
        bid = 0;
    end
end

function bid = local_calculateRecoveryBid(robot, task, current_price, params)
    % CALCULATERECOVERYBID Calculate bid value during recovery
    
    % Use regular bid calculation as a base
    bid = local_calculateBid(robot, task, current_price, params);
    
    % Apply additional recovery-specific factors
    if bid > 0
        % Add recovery priority based on beta parameters
        bid = bid * params.beta(1);
        
        % Give higher priority to tasks that were assigned to the failed robot
        if isfield(params, 'failed_robot') && isfield(auction_data, 'initial_assignment')
            if auction_data.initial_assignment(task.id) == params.failed_robot
                bid = bid * params.beta(2);
            end
        end
    end
end

function auction_data = local_initiateRecovery(auction_data, robots, tasks, failed_robot)
    % INITIATERECOVERY Initiate the recovery process after a robot failure
    
    % Set recovery mode
    auction_data.recovery_mode = true;
    auction_data.failed_robot = failed_robot;
    
    % Store initial assignment if not already stored
    if ~isfield(auction_data, 'initial_assignment') || isempty(auction_data.initial_assignment)
        auction_data.initial_assignment = auction_data.assignment;
    end
    
    % Release all tasks assigned to the failed robot
    for i = 1:length(tasks)
        if auction_data.assignment(i) == failed_robot
            auction_data.assignment(i) = 0;  % Unassign the task
            
            % Reset task price to encourage reassignment
            auction_data.prices(i) = 0;
            
            % Reset task progress
            if isfield(auction_data, 'task_progress')
                auction_data.task_progress(i) = 0;
            end
        end
    end
    
    fprintf('Recovery mode initiated for Robot %d failure. %d tasks need reassignment.\n', ...
            failed_robot, sum(auction_data.initial_assignment == failed_robot));
end

function auction_data = local_resetPricesForBlockedTasks(auction_data, tasks, available_tasks)
    % RESETPRICESFORBLOCKEDTASKS Reset prices for tasks that are blocked by dependencies
    % This helps avoid price inflation for unavailable tasks
    
    for i = 1:length(tasks)
        % If task is not available and not completed, reset its price
        if ~ismember(i, available_tasks) && auction_data.completion_status(i) == 0
            auction_data.prices(i) = 0;
        end
    end
end

function [metrics, converged] = local_runAuctionSimulation(params, env, robots, tasks, visualize)
    % RUNAUCTIONSIMULATION Run the complete auction simulation
    
    % Load utilities
    task_utils = task_utils();
    
    % Initialize auction data
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
    metrics.failure_time = 0;
    metrics.makespan = 0;
    metrics.optimal_makespan = 0;
    metrics.task_oscillation_count = zeros(length(tasks), 1);
    metrics.failed_task_count = 0;
    
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
    
    % Track simulation time
    simulation_time = 0;
    
    % Main simulation loop
    for iter = 1:max_iterations
        metrics.iterations = iter;
        simulation_time = simulation_time + params.time_step;
        
        % Check for robot failure
        if iter == params.failure_time && ~isempty(params.failed_robot)
            fprintf('Robot %d has failed at iteration %d\n', params.failed_robot, iter);
            robots(params.failed_robot).failed = true;
            metrics.failure_time = iter;  % Store failure time in metrics
            
            % Count tasks assigned to failed robot
            metrics.failed_task_count = sum(auction_data.assignment == params.failed_robot);
            
            % Initiate recovery process
            auction_data = local_initiateRecovery(auction_data, robots, tasks, params.failed_robot);
        end
        
        % FIXED: Reset prices for blocked tasks
        auction_data = local_resetPricesForBlockedTasks(auction_data, tasks, available_tasks);
        
        % ADDITION: Update available tasks based on newly completed tasks
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
            metrics.convergence_history(iter) = NaN;
            unchanged_iterations = 0;
        end
        
        % Update visualization if requested
        if visualize && (mod(iter, 5) == 0 || iter < 10)
            subplot(2, 3, [1, 4]);
            env_utils.visualizeEnvironment(env, robots, tasks, auction_data);
            title(sprintf('Environment (Iteration %d, Time: %.1f s)', iter, simulation_time));
            
            % Print detailed status
            local_printTaskStatus(auction_data, tasks, robots);
            
            % Update other plots
            if iter > 1
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
                env_utils.plotConvergence(metrics.convergence_history(1:iter));
                title('Convergence Metric');
                
                % Workload distribution
                subplot(2, 3, 6);
                env_utils.plotWorkload(metrics.assignment_history(:, iter), tasks, robots);
                title('Current Workload Distribution');
            end
            
            drawnow;
            pause(0.01);
        end
        
        % FIXED: Better convergence criteria
        % Convergence when: 
        % 1. Assignments haven't changed for several iterations AND
        % 2. Either all available tasks are assigned OR system has been stuck for many iterations
        if unchanged_iterations >= 10
            % Check if all available tasks are assigned
            unassigned_available = sum(ismember(available_tasks, find(auction_data.assignment == 0)));
            
            if unassigned_available == 0
                fprintf('Auction algorithm converged - all available tasks assigned (stable for %d iterations)\n', unchanged_iterations);
                converged = true;
                break;
            elseif unchanged_iterations >= 30
                fprintf('Auction algorithm converged but %d available tasks remain unassigned\n', unassigned_available);
                converged = true;
                break;
            end
        end
        
        % Additional exit condition: all tasks are completed or assigned
        if all(auction_data.completion_status == 1 | auction_data.assignment > 0)
            fprintf('Auction algorithm converged - all tasks assigned or completed\n');
            converged = true;
            break;
        end
        
        % Update recovery time if in recovery mode
        if auction_data.recovery_mode && metrics.recovery_time == 0
            % Check if all tasks have been reassigned
            tasks_to_reassign = find(auction_data.initial_assignment == auction_data.failed_robot);
            if all(auction_data.assignment(tasks_to_reassign) ~= auction_data.failed_robot)
                metrics.recovery_time = iter - metrics.failure_time;
                fprintf('Recovery completed after %d iterations\n', metrics.recovery_time);
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
        if iter > 50 && all(unchanged_iterations > 30)
            fprintf('Auction algorithm terminated - no progress after %d iterations\n', unchanged_iterations);
            break;
        end
    end
    
    % Trim history matrices to actual size
    metrics.price_history = metrics.price_history(:, 1:iter);
    metrics.assignment_history = metrics.assignment_history(:, 1:iter);
    
    % Final analysis
    fprintf('\n--- Final Task Allocation ---\n');
    local_analyzeTaskAllocation(auction_data, tasks);
    
    % Calculate makespan
    robot_utils = robot_utils();
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

function local_analyzeTaskAllocation(auction_data, tasks)
    % ANALYZETASKALLOCATION Analyze the current task allocation
    
    % Get task assignments
    assignment = auction_data.assignment;
    
    % Count assigned tasks per robot
    unique_robots = unique(assignment);
    unique_robots = unique_robots(unique_robots > 0);  % Remove unassigned (0)
    
    % Print task allocation
    fprintf('Task Allocation:');
    for r = unique_robots'
        fprintf(' R%d: [', r);
        assigned_tasks = find(assignment == r);
        fprintf('%d ', assigned_tasks);
        fprintf(']');
    end
    fprintf('\n');
    
    % Calculate workload per robot
    workloads = zeros(1, max(unique_robots));
    
    for r = unique_robots'
        assigned_tasks = find(assignment == r);
        for t = assigned_tasks'
            if isfield(tasks, 'execution_time')
                workloads(r) = workloads(r) + tasks(t).execution_time;
            else
                workloads(r) = workloads(r) + 1;  % Default execution time
            end
        end
    end
    
    % Print workload distribution
    fprintf('Workload:');
    for r = 1:length(workloads)
        if r <= length(workloads)
            fprintf(' R%d=%.2f,', r, workloads(r));
        end
    end
    
    % Calculate workload imbalance ratio
    if length(workloads) > 1 && min(workloads) > 0
        ratio = max(workloads) / min(workloads);
        fprintf(' Ratio=%.2f', ratio);
    elseif length(workloads) > 1
        fprintf(' Ratio=Inf');
    end
    fprintf('\n');
    
    % Check for unassigned tasks
    unassigned = find(assignment == 0);
    if ~isempty(unassigned)
        fprintf('WARNING: %d tasks remain unassigned!\n', length(unassigned));
        
        % Display unassigned tasks with unassigned iterations
        if isfield(auction_data, 'assignment_count')
            fprintf('Unassigned tasks:');
            display_count = min(length(unassigned), 10);
            for i = 1:display_count
                t = unassigned(i);
                fprintf(' T%d (unassigned for %d iterations)', t, auction_data.assignment_count(t));
            end
            fprintf('\n');
        end
    end
    
    % Check for tasks stuck in oscillation
    if isfield(auction_data, 'task_oscillation_count')
        oscillating_tasks = find(auction_data.task_oscillation_count > 0);
        if ~isempty(oscillating_tasks)
            fprintf('Task oscillations:');
            for t = oscillating_tasks'
                fprintf(' T%d:%d', t, auction_data.task_oscillation_count(t));
            end
            fprintf('\n');
        end
    end
end

function local_analyzeBidDistribution(auction_data, robots, tasks)
    % ANALYZEBIDDISTRIBUTION Analyze the bid distribution
    
    % Check if we have bid data
    if ~isfield(auction_data, 'bids') || size(auction_data.bids, 1) ~= length(robots) || size(auction_data.bids, 2) ~= length(tasks)
        fprintf('No bid data available for analysis.\n');
        return;
    end
    
    % Calculate bid statistics
    bids = auction_data.bids;
    
    % Average bid per robot
    avg_bids_per_robot = mean(bids, 2);
    
    % Average bid per task
    avg_bids_per_task = mean(bids, 1);
    
    % Print statistics
    fprintf('\n--- Bid Distribution Analysis ---\n');
    
    % Robot bid statistics
    fprintf('Average bid per robot:\n');
    for r = 1:length(robots)
        fprintf('  Robot %d: %.2f\n', r, avg_bids_per_robot(r));
    end
    
    % Task bid statistics
    fprintf('Top 5 most bid-on tasks:\n');
    [~, top_tasks] = sort(avg_bids_per_task, 'descend');
    for i = 1:min(5, length(top_tasks))
        t = top_tasks(i);
        fprintf('  Task %d: %.2f (assigned to Robot %d, price: %.2f)\n', ...
                t, avg_bids_per_task(t), auction_data.assignment(t), auction_data.prices(t));
    end
    
    % Low bid tasks
    fprintf('Tasks with no bids:\n');
    no_bid_tasks = find(sum(bids > 0, 1) == 0);
    if ~isempty(no_bid_tasks)
        for t = no_bid_tasks
            fprintf('  Task %d (price: %.2f)\n', t, auction_data.prices(t));
        end
    else
        fprintf('  None\n');
    end
end

function local_printTaskStatus(auction_data, tasks, robots)
    % PRINTTASKSTATUS Print detailed status of all tasks
    fprintf('Task status:\n');
    for i = 1:length(tasks)
        if auction_data.completion_status(i) == 1
            status = 'Completed';
        elseif auction_data.assignment(i) > 0
            robot_id = auction_data.assignment(i);
            if robot_id <= length(robots) && ~(isfield(robots, 'failed') && robots(robot_id).failed)
                if isfield(auction_data, 'task_progress')
                    progress = auction_data.task_progress(i) / tasks(i).execution_time * 100;
                    status = sprintf('Assigned to R%d (%.1f%% complete)', robot_id, progress);
                else
                    status = sprintf('Assigned to R%d', robot_id);
                end
            else
                status = 'Assigned to failed robot';
            end
        else
            status = 'Unassigned';
        end
        fprintf('  T%d (%s): %s (Price: %.2f)\n', i, tasks(i).name, status, auction_data.prices(i));
        
        % Print dependencies
        if isfield(tasks(i), 'prerequisites') && ~isempty(tasks(i).prerequisites)
            fprintf('    Prerequisites: ');
            for j = 1:length(tasks(i).prerequisites)
                prereq = tasks(i).prerequisites(j);
                if auction_data.completion_status(prereq) == 1
                    fprintf('%d✓ ', prereq);
                else
                    fprintf('%d✗ ', prereq);
                end
            end
            fprintf('\n');
        end
    end
    fprintf('\n');
end