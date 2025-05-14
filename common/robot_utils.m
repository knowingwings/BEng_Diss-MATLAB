function utils = robot_utils()
    % ROBOT_UTILS - Returns function handles for robot-related functions
    utils = struct(...
        'createRobots', @local_createRobots, ...
        'calculateMakespan', @local_calculateMakespan, ...
        'calculateOptimalMakespan', @local_calculateOptimalMakespan, ...
        'updateRobotWorkload', @local_updateRobotWorkload ...
    );
end

function robots = local_createRobots(num_robots, env)
    % CREATEROBOTS Creates robot structs with initial positions and capabilities
    robots = struct([]);
    
    % IMPROVED: More balanced capability distribution
    % Generate capabilities with controlled differences to ensure better load balancing
    base_capabilities = [1.0, 1.0, 1.0, 1.0, 1.0];
    
    for i = 1:num_robots
        robots(i).id = i;
        % Position robots at opposite corners
        if i == 1
            robots(i).position = [0.5, 0.5];
        else
            robots(i).position = [env.width-0.5, env.height-0.5];
        end
        
        % IMPROVED: Create complementary capabilities between robots
        if i == 1
            % First robot slightly better at capabilities 1, 3, 5
            variation = [0.1, -0.1, 0.1, -0.1, 0.1];
        else
            % Second robot slightly better at capabilities 2, 4
            variation = [-0.1, 0.1, -0.1, 0.1, -0.1];
        end
        
        robots(i).capabilities = base_capabilities + variation;
        
        % Normalize capabilities
        robots(i).capabilities = robots(i).capabilities / norm(robots(i).capabilities) * 2;
        
        robots(i).workload = 0;
        robots(i).failed = false;
    end
    
    % Print capabilities for diagnostics
    fprintf('Robot capabilities:\n');
    for i = 1:num_robots
        fprintf('Robot %d: [%.2f %.2f %.2f %.2f %.2f]\n', i, robots(i).capabilities);
    end
end

function makespan = local_calculateMakespan(assignment, tasks, robots)
    % CALCULATEMAKESPAN Calculate the makespan (maximum completion time among robots)
    
    % Calculate total execution time for each robot
    robot_times = zeros(1, length(robots));
    
    % IMPROVED: Handle task dependencies in makespan calculation
    % First, determine the execution order based on dependencies
    execution_order = local_determineExecutionOrder(tasks);
    
    % Track start and finish times for each task
    start_times = zeros(1, length(tasks));
    finish_times = zeros(1, length(tasks));
    
    % Process tasks in the determined order
    for task_idx = execution_order
        robot_id = assignment(task_idx);
        
        if robot_id > 0 && ~robots(robot_id).failed
            % Find earliest start time based on prerequisites
            earliest_start = 0;
            if isfield(tasks, 'prerequisites') && ~isempty(tasks(task_idx).prerequisites)
                for prereq = tasks(task_idx).prerequisites
                    earliest_start = max(earliest_start, finish_times(prereq));
                end
            end
            
            % Find earliest time this robot is available
            robot_available_time = robot_times(robot_id);
            
            % Task starts at the later of prerequisite completion or robot availability
            start_time = max(earliest_start, robot_available_time);
            
            % Calculate finish time
            if isfield(tasks, 'execution_time')
                finish_time = start_time + tasks(task_idx).execution_time;
            else
                finish_time = start_time + 1;  % Default execution time
            end
            
            % Update robot's busy time
            robot_times(robot_id) = finish_time;
            
            % Record start and finish times
            start_times(task_idx) = start_time;
            finish_times(task_idx) = finish_time;
        end
    end
    
    % Makespan is the maximum finish time across all tasks
    makespan = max(finish_times);
    
    % Handle edge case where no tasks are assigned
    if makespan == 0 && ~all(assignment == 0)
        warning('Unusual makespan calculation: No execution time found despite assignments');
    end
end

function execution_order = local_determineExecutionOrder(tasks)
    % DETERMINEEXECUTIONORDER Determines execution order respecting dependencies
    num_tasks = length(tasks);
    
    % Create dependency graph
    dependency_matrix = zeros(num_tasks, num_tasks);
    
    for i = 1:num_tasks
        if isfield(tasks, 'prerequisites') && ~isempty(tasks(i).prerequisites)
            for prereq = tasks(i).prerequisites
                if prereq <= num_tasks
                    dependency_matrix(prereq, i) = 1;  % prereq must complete before i
                end
            end
        end
    end
    
    % Compute in-degree for each task (number of prerequisites)
    in_degree = sum(dependency_matrix, 1);
    
    % Start with tasks that have no prerequisites
    queue = find(in_degree == 0);
    execution_order = [];
    
    % Process queue until empty (topological sort)
    while ~isempty(queue)
        % Get a task with no remaining prerequisites
        current_task = queue(1);
        queue(1) = [];
        
        % Add to execution order
        execution_order = [execution_order, current_task];
        
        % Find tasks that depend on the current task
        dependent_tasks = find(dependency_matrix(current_task, :) > 0);
        
        % Reduce in-degree of dependent tasks
        for dependent = dependent_tasks
            in_degree(dependent) = in_degree(dependent) - 1;
            
            % If dependent task has no more prerequisites, add to queue
            if in_degree(dependent) == 0
                queue = [queue, dependent];
            end
        end
    end
    
    % If not all tasks are in the execution order, there might be cycles
    if length(execution_order) < num_tasks
        warning('Task dependency graph might contain cycles. Adding remaining tasks in numerical order.');
        missing_tasks = setdiff(1:num_tasks, execution_order);
        execution_order = [execution_order, missing_tasks];
    end
end

function path_length = local_calculateCriticalPathLength(task_id, tasks, memo)
    % CALCULATECRITICALPATHHLENGTH Helper function to calculate the critical path length
    % starting from a task. Uses memoization to avoid recalculating paths.
    
    % Check if we've already calculated this path length
    if isKey(memo, task_id)
        path_length = memo(task_id);
        return;
    end
    
    % If no prerequisites, just return the task's execution time
    if ~isfield(tasks, 'prerequisites') || isempty(tasks(task_id).prerequisites)
        if isfield(tasks, 'execution_time')
            path_length = tasks(task_id).execution_time;
        else
            path_length = 1;  % Default execution time
        end
    else
        % Otherwise, recursively find the longest path through prerequisites
        max_prereq_length = 0;
        for i = 1:numel(tasks(task_id).prerequisites)
            prereq_id = tasks(task_id).prerequisites(i);
            prereq_length = local_calculateCriticalPathLength(prereq_id, tasks, memo);
            max_prereq_length = max(max_prereq_length, prereq_length);
        end
        
        if isfield(tasks, 'execution_time')
            path_length = max_prereq_length + tasks(task_id).execution_time;
        else
            path_length = max_prereq_length + 1;  % Default execution time
        end
    end
    
    % Store result in memoization table
    memo(task_id) = path_length;
end

function makespan = local_calculateOptimalMakespan(tasks, robots)
    % CALCULATEOPTIMALMAKESPAN Calculate the optimal makespan (lower bound)
    
    % Count active robots
    active_robots = sum(~[robots.failed]);
    
    % Get total execution time
    total_time = 0;
    if isfield(tasks, 'execution_time')
        total_time = sum([tasks.execution_time]);
    else
        total_time = length(tasks);  % Default execution time of 1 per task
    end
    
    % Perfect load balancing makespan (lower bound)
    if active_robots > 0
        balanced_makespan = total_time / active_robots;
    else
        % Handle case where all robots have failed
        balanced_makespan = total_time;
    end
    
    % Also consider the longest single task - this can't be split
    max_task_time = 0;
    if isfield(tasks, 'execution_time')
        max_task_time = max([tasks.execution_time]);
    else
        max_task_time = 1;  % Default execution time
    end
    
    % IMPROVED: Calculate critical path length using memoization
    % Use a container map for memoization to avoid redundant calculations
    memo = containers.Map('KeyType', 'double', 'ValueType', 'double');
    
    critical_path_length = 0;
    for i = 1:length(tasks)
        path_length = local_calculateCriticalPathLength(i, tasks, memo);
        critical_path_length = max(critical_path_length, path_length);
    end
    
    % IMPROVED: Better handling of task dependencies
    % Analyze strongly connected components in the task graph
    dependency_graph = zeros(length(tasks));
    
    for i = 1:length(tasks)
        if isfield(tasks, 'prerequisites') && ~isempty(tasks(i).prerequisites)
            for prereq = tasks(i).prerequisites
                if prereq <= length(tasks)
                    dependency_graph(prereq, i) = 1;
                end
            end
        end
    end
    
    % The optimal makespan can't be less than either the balanced makespan,
    % the longest task, or the critical path length
    makespan = max([balanced_makespan, max_task_time, critical_path_length]);
end

function robots = local_updateRobotWorkload(robots, old_assignment, new_assignment, task_index, task_time)
    % UPDATEROBOTWORKLOAD Update robot workload after task reassignment
    
    % If task was previously assigned, reduce workload for that robot
    if old_assignment > 0 && old_assignment <= length(robots)
        robots(old_assignment).workload = robots(old_assignment).workload - task_time;
    end
    
    % Add workload to newly assigned robot
    if new_assignment > 0 && new_assignment <= length(robots)
        robots(new_assignment).workload = robots(new_assignment).workload + task_time;
    end
end