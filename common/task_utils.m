function utils = task_utils()
    % TASK_UTILS - Returns function handles for task-related functions
    utils = struct(...
        'createTasks', @local_createTasks, ...
        'addTaskDependencies', @local_addTaskDependencies, ...
        'findAvailableTasks', @local_findAvailableTasks, ...
        'findTasksWithPrereqsInProgress', @local_findTasksWithPrereqsInProgress, ...
        'calculateTaskCriticality', @local_calculateTaskCriticality, ...
        'calculateDependencyDepth', @local_calculateDependencyDepth, ...
        'isOnCriticalPath', @local_isOnCriticalPath, ...
        'calculateEarliestStartTime', @local_calculateEarliestStartTime, ...
        'sortTasksByDependency', @local_sortTasksByDependency, ...
        'calculateDependencyLevels', @local_calculateDependencyLevels, ...
        'buildDependencyGraph', @local_buildDependencyGraph ...
    );
end

function tasks = local_createTasks(num_tasks, env)
    % CREATETASKS Creates task structs with positions and requirements
    tasks = struct([]);
    
    % IMPROVED: Create more diverse task grid with reduced clustering
    grid_size = ceil(sqrt(num_tasks * 1.5));  % Make grid 1.5x larger than minimal size
    grid_points = local_getGridPoints(grid_size, env);
    
    % Randomly select grid points without replacement to ensure better distribution
    selected_indices = randperm(length(grid_points), num_tasks);
    selected_positions = grid_points(selected_indices, :);
    
    % IMPROVED: Generate capabilities with meaningful patterns for better load balancing
    % Create 3 types of tasks with different capability patterns
    capability_patterns = [
        0.8, 0.4, 0.6, 0.2, 0.3;  % Type 1: Strong in capabilities 1, 3
        0.3, 0.9, 0.2, 0.8, 0.4;  % Type 2: Strong in capabilities 2, 4
        0.5, 0.5, 0.7, 0.5, 0.9   % Type 3: Strong in capability 5, balanced otherwise
    ];
    
    for i = 1:num_tasks
        tasks(i).id = i;
        
        % Assign position from the grid
        tasks(i).position = selected_positions(i, :);
        
        % Assign capability requirements using one of the patterns with small variations
        pattern_idx = mod(i-1, size(capability_patterns, 1)) + 1;
        base_capabilities = capability_patterns(pattern_idx, :);
        
        % Add small random variations
        variation = 0.1 * (rand(1, 5) - 0.5);
        
        tasks(i).capabilities_required = base_capabilities + variation;
        
        % Normalize capabilities to unit length
        tasks(i).capabilities_required = tasks(i).capabilities_required / norm(tasks(i).capabilities_required);
        
        % IMPROVED: More varied execution times based on capability pattern
        % Each pattern has a slightly different execution time distribution
        if pattern_idx == 1
            tasks(i).execution_time = 3 + 2 * rand();  % 3-5 time units
        elseif pattern_idx == 2
            tasks(i).execution_time = 4 + 2 * rand();  % 4-6 time units
        else
            tasks(i).execution_time = 3 + 3 * rand();  % 3-6 time units
        end
        
        tasks(i).prerequisites = [];  % Will be filled in addTaskDependencies
        
        % NEW: Add task criticality flags (to be set later)
        tasks(i).on_critical_path = false;
        
        % NEW: Add planned start and end times
        tasks(i).planned_start_time = 0;
        tasks(i).planned_end_time = 0;
    end
end

function grid_points = local_getGridPoints(grid_size, env)
    % Helper function to generate a grid of points evenly distributed in the environment
    x_range = linspace(0.5, env.width - 0.5, grid_size);
    y_range = linspace(0.5, env.height - 0.5, grid_size);
    
    [X, Y] = meshgrid(x_range, y_range);
    grid_points = [X(:), Y(:)];
end

function tasks = local_addTaskDependencies(tasks, probability)
    % ADDTASKDEPENDENCIES Add prerequisites to create a directed acyclic graph
    %
    % Parameters:
    %   tasks      - Array of task structures
    %   probability - (Optional) Probability of adding a dependency (default: 0.3)
    %
    % Returns:
    %   tasks      - Updated array of task structures with prerequisites
    
    num_tasks = length(tasks);
    
    % Set default probability
    if nargin < 2
        probability = 0.3;
    end
    
    % IMPROVED: Reduce dependency probability for better parallelization
    probability = 0.2;  % Lower than default to enable more tasks to run in parallel
    
    % IMPROVED: Create a more structured dependency tree
    % Extract x-coordinates for proper sorting
    x_coords = zeros(num_tasks, 1);
    for i = 1:num_tasks
        x_coords(i) = tasks(i).position(1);  % Just use x-coordinate 
    end
    [~, sorted_idx] = sort(x_coords);
    
    % IMPROVED: Create multiple parallel task chains instead of a single critical path
    num_chains = 3;  % Create 3 parallel chains
    chain_assignments = mod(1:num_tasks, num_chains) + 1;
    
    % IMPROVED: Ensure we don't create cycles by only allowing dependencies i -> j 
    % where i < j in sorted order, and preferring same-chain dependencies
    for j = 2:num_tasks
        task_idx = sorted_idx(j);
        task_chain = chain_assignments(task_idx);
        
        % Consider only tasks that come before in the sorted order
        potential_prereqs = sorted_idx(1:j-1);
        
        % Limit the maximum number of prerequisites to avoid overconstraining
        max_prereqs = min(2, length(potential_prereqs));
        
        % Prioritize adding dependencies within the same chain
        same_chain_prereqs = potential_prereqs(chain_assignments(potential_prereqs) == task_chain);
        
        % Choose at most one prerequisite from the same chain
        if ~isempty(same_chain_prereqs) && rand() < probability * 1.5  % Higher probability for same chain
            chosen_prereq = same_chain_prereqs(end);  % Choose the latest one in the chain
            tasks(task_idx).prerequisites = [tasks(task_idx).prerequisites, chosen_prereq];
        end
        
        % Maybe add a cross-chain dependency with lower probability
        if rand() < probability * 0.6 && length(tasks(task_idx).prerequisites) < max_prereqs
            other_chain_prereqs = setdiff(potential_prereqs, same_chain_prereqs);
            if ~isempty(other_chain_prereqs)
                random_idx = randi(length(other_chain_prereqs));
                cross_prereq = other_chain_prereqs(random_idx);
                
                % Avoid adding duplicates
                if ~ismember(cross_prereq, tasks(task_idx).prerequisites)
                    tasks(task_idx).prerequisites = [tasks(task_idx).prerequisites, cross_prereq];
                end
            end
        end
    end
    
    % IMPROVED: Detect and remove any transitive dependencies
    for i = 1:num_tasks
        prereqs = tasks(i).prerequisites;
        
        % Remove transitive dependencies
        for j = prereqs
            for k = tasks(j).prerequisites
                % If prereq j has its own prereq k, and k is also a direct prereq of i,
                % then k is a transitive prereq and can be removed from i's direct prereqs
                if ismember(k, prereqs)
                    tasks(i).prerequisites = tasks(i).prerequisites(tasks(i).prerequisites ~= k);
                end
            end
        end
    end
    
    % NEW: Calculate dependency levels for each task
    dependency_levels = local_calculateDependencyLevels(tasks);
    
    % NEW: Set dependency level for each task
    for i = 1:num_tasks
        tasks(i).dependency_level = dependency_levels(i);
    end
    
    % NEW: Find and mark the critical path
    [critical_path, ~] = local_findCriticalPath(tasks);
    
    % Mark tasks on critical path
    for i = 1:length(critical_path)
        task_id = critical_path(i);
        if task_id <= num_tasks
            tasks(task_id).on_critical_path = true;
        end
    end
    
    % IMPROVED: Ensure first task (if has a specific name) has no prerequisites
    if ~isempty(tasks) && isfield(tasks, 'name') && strcmp(tasks(1).name, 'chassis_positioning')
        tasks(1).prerequisites = [];
    end
end

function available_tasks = local_findAvailableTasks(tasks, completed_tasks)
    % FINDAVAILABLETASKS Find tasks that are available for assignment 
    %                   (all prerequisites completed)
    %
    % Parameters:
    %   tasks           - Array of task structures
    %   completed_tasks - List of completed task IDs
    %
    % Returns:
    %   available_tasks - List of available task IDs
    
    available_tasks = [];
    
    for i = 1:length(tasks)
        % Skip completed tasks
        if ismember(i, completed_tasks)
            continue;
        end
        
        % Check prerequisites
        prerequisites_met = true;
        if isfield(tasks(i), 'prerequisites') && ~isempty(tasks(i).prerequisites)
            for j = 1:length(tasks(i).prerequisites)
                prereq = tasks(i).prerequisites(j);
                if ~ismember(prereq, completed_tasks)
                    prerequisites_met = false;
                    break;
                end
            end
        end
        
        % If no prerequisites or all prerequisites are met, task is available
        if prerequisites_met
            available_tasks = [available_tasks, i];
        end
    end
    
    % IMPROVED: If no tasks are available, find tasks with the fewest missing prerequisites
    if isempty(available_tasks) && ~isempty(tasks)
        min_missing_prereqs = inf;
        candidate_tasks = [];
        
        for i = 1:length(tasks)
            if ~ismember(i, completed_tasks)
                % Count missing prerequisites
                missing_prereqs = 0;
                if isfield(tasks(i), 'prerequisites')
                    for j = 1:length(tasks(i).prerequisites)
                        prereq = tasks(i).prerequisites(j);
                        if ~ismember(prereq, completed_tasks)
                            missing_prereqs = missing_prereqs + 1;
                        end
                    end
                end
                
                if missing_prereqs < min_missing_prereqs
                    min_missing_prereqs = missing_prereqs;
                    candidate_tasks = [i];
                elseif missing_prereqs == min_missing_prereqs
                    candidate_tasks = [candidate_tasks, i];
                end
            end
        end
        
        % Add tasks with fewest missing prerequisites
        if min_missing_prereqs < inf
            available_tasks = candidate_tasks;
            fprintf('No fully available tasks. Using %d tasks with %d missing prerequisites.\n', ...
                   length(available_tasks), min_missing_prereqs);
        end
    end
end

function potential_tasks = local_findTasksWithPrereqsInProgress(tasks, completed_tasks, in_progress_tasks, task_progress)
    % FINDTASKSWITHPREREQSINPROGRESS Find tasks that will soon be available
    %                               because their prerequisites are close to completion
    %
    % Parameters:
    %   tasks           - Array of task structures
    %   completed_tasks - List of completed task IDs
    %   in_progress_tasks - List of tasks currently being executed
    %   task_progress   - Progress of each task (0-1 scale)
    %
    % Returns:
    %   potential_tasks - List of tasks that may soon be available
    
    potential_tasks = [];
    
    % Progress threshold to consider a prerequisite "almost done"
    PROGRESS_THRESHOLD = 0.7;
    
    for i = 1:length(tasks)
        % Skip completed tasks and already available tasks
        if ismember(i, completed_tasks) || ismember(i, local_findAvailableTasks(tasks, completed_tasks))
            continue;
        end
        
        % Check prerequisites
        all_prereqs_near_completion = true;
        if isfield(tasks(i), 'prerequisites') && ~isempty(tasks(i).prerequisites)
            for j = 1:length(tasks(i).prerequisites)
                prereq = tasks(i).prerequisites(j);
                
                % If prerequisite is completed, it's fine
                if ismember(prereq, completed_tasks)
                    continue;
                end
                
                % If prerequisite is in progress and almost done, it's fine
                if ismember(prereq, in_progress_tasks) && task_progress(prereq) >= PROGRESS_THRESHOLD
                    continue;
                end
                
                % Otherwise, not all prerequisites are near completion
                all_prereqs_near_completion = false;
                break;
            end
        end
        
        % If all prerequisites are completed or near completion, add to potential tasks
        if all_prereqs_near_completion
            potential_tasks = [potential_tasks, i];
        end
    end
end

function criticality = local_calculateTaskCriticality(task_id, tasks)
    % CALCULATETASKCRITICALITY Calculate the criticality of a task based on dependencies
    %
    % Criticality is measured by how many other tasks depend on this task,
    % directly or indirectly.
    %
    % Parameters:
    %   task_id      - ID of the task to evaluate
    %   tasks        - Array of task structures
    %
    % Returns:
    %   criticality  - Criticality score (higher means more critical)
    
    % IMPROVED: Count both direct and indirect dependencies
    % Initialize criticality with direct dependencies
    direct_dependencies = 0;
    for i = 1:length(tasks)
        if ismember(task_id, tasks(i).prerequisites)
            direct_dependencies = direct_dependencies + 1;
        end
    end
    
    % Count indirect dependencies (tasks that depend on tasks that depend on this task)
    indirect_dependencies = 0;
    for i = 1:length(tasks)
        if ismember(task_id, tasks(i).prerequisites)
            % For each task that depends directly on task_id,
            % count how many tasks depend on it
            for j = 1:length(tasks)
                if ismember(i, tasks(j).prerequisites)
                    indirect_dependencies = indirect_dependencies + 1;
                end
            end
        end
    end
    
    % Combine direct and indirect dependencies with weights
    criticality = direct_dependencies + 0.5 * indirect_dependencies;
    
    % IMPROVED: Add bonus for tasks on critical path
    if isfield(tasks(task_id), 'on_critical_path') && tasks(task_id).on_critical_path
        criticality = criticality * 1.5;  % 50% bonus for critical path tasks
    end
end

function dependency_depth = local_calculateDependencyDepth(task_id, tasks)
    % CALCULATEDEPENDENCYDEPTH Calculate the depth of a task in the dependency graph
    %
    % Depth is measured by the longest chain of prerequisites leading to this task.
    %
    % Parameters:
    %   task_id          - ID of the task to evaluate
    %   tasks            - Array of task structures
    %
    % Returns:
    %   dependency_depth - Depth in the dependency graph
    
    % IMPROVED: Use memoization to avoid recalculating depths
    persistent depth_memo;
    
    % Initialize memoization table if not already done
    if isempty(depth_memo) || numel(depth_memo) ~= length(tasks)
        depth_memo = zeros(1, length(tasks));
    end
    
    % Check if depth is already calculated
    if depth_memo(task_id) > 0
        dependency_depth = depth_memo(task_id);
        return;
    end
    
    if isempty(tasks(task_id).prerequisites)
        dependency_depth = 0;
    else
        % Calculate max depth from prerequisites
        depths = zeros(1, length(tasks(task_id).prerequisites));
        for i = 1:length(tasks(task_id).prerequisites)
            prereq_id = tasks(task_id).prerequisites(i);
            depths(i) = local_calculateDependencyDepth(prereq_id, tasks);
        end
        dependency_depth = 1 + max(depths);
    end
    
    % Store in memoization table
    depth_memo(task_id) = dependency_depth;
end

function is_critical = local_isOnCriticalPath(task_id, tasks)
    % ISONCRITICALPATH Determine if a task is on the critical path
    %
    % A task is on the critical path if it has the highest
    % combination of dependency depth and dependent tasks.
    %
    % Parameters:
    %   task_id      - ID of the task to evaluate
    %   tasks        - Array of task structures
    %
    % Returns:
    %   is_critical  - Boolean indicating if task is on critical path
    
    % IMPROVED: Better critical path detection
    % Calculate all task depths and criticalities
    all_depths = zeros(1, length(tasks));
    all_criticalities = zeros(1, length(tasks));
    
    for i = 1:length(tasks)
        all_depths(i) = local_calculateDependencyDepth(i, tasks);
        all_criticalities(i) = local_calculateTaskCriticality(i, tasks);
    end
    
    % Calculate critical scores as weighted sum of depth and criticality
    critical_scores = all_depths + 0.5 * all_criticalities;
    
    % Find all tasks on the critical path (longest path through the graph)
    max_depth = max(all_depths);
    potential_critical_tasks = find(all_depths == max_depth);
    
    % Among tasks with max depth, select those with highest criticality
    if ~isempty(potential_critical_tasks)
        max_score = max(critical_scores(potential_critical_tasks));
        critical_tasks = find(critical_scores >= max_score * 0.9);  % Include near-critical tasks
        
        is_critical = ismember(task_id, critical_tasks);
    else
        is_critical = false;
    end
    
    % IMPROVED: Directly check the on_critical_path flag if available
    if isfield(tasks(task_id), 'on_critical_path')
        is_critical = tasks(task_id).on_critical_path;
    end
end

function earliest_start_time = local_calculateEarliestStartTime(task_id, tasks, completed_tasks, execution_times)
    % CALCULATEEARLIESTSTARTIME Calculate the earliest possible start time for a task
    %
    % Parameters:
    %   task_id - ID of the task
    %   tasks - Array of task structures
    %   completed_tasks - List of completed task IDs
    %   execution_times - Array of execution times for each task
    %
    % Returns:
    %   earliest_start_time - Earliest possible start time for the task
    
    % Initialize earliest start time to 0
    earliest_start_time = 0;
    
    % Check if task has prerequisites
    if isfield(tasks(task_id), 'prerequisites') && ~isempty(tasks(task_id).prerequisites)
        for prereq = tasks(task_id).prerequisites
            % Skip completed prerequisites
            if ismember(prereq, completed_tasks)
                continue;
            end
            
            % For each uncompleted prerequisite, recursively find its earliest finish time
            prereq_start_time = local_calculateEarliestStartTime(prereq, tasks, completed_tasks, execution_times);
            prereq_finish_time = prereq_start_time + execution_times(prereq);
            
            % Update earliest start time
            earliest_start_time = max(earliest_start_time, prereq_finish_time);
        end
    end
end

function sorted_tasks = local_sortTasksByDependency(tasks)
    % SORTTASKSBYDEPENDENCY Sort tasks in order of dependency (topological sort)
    %
    % Parameters:
    %   tasks - Array of task structures
    %
    % Returns:
    %   sorted_tasks - Array of task IDs in dependency order
    
    num_tasks = length(tasks);
    
    % Calculate indegree (number of prerequisites) for each task
    indegree = zeros(num_tasks, 1);
    for i = 1:num_tasks
        if isfield(tasks(i), 'prerequisites')
            indegree(i) = length(tasks(i).prerequisites);
        end
    end
    
    % Start with tasks that have no prerequisites
    queue = find(indegree == 0);
    sorted_tasks = [];
    
    % Perform topological sort
    while ~isempty(queue)
        current = queue(1);
        queue(1) = [];
        
        % Add to sorted list
        sorted_tasks = [sorted_tasks, current];
        
        % For each task that depends on this one
        for i = 1:num_tasks
            if isfield(tasks(i), 'prerequisites') && ismember(current, tasks(i).prerequisites)
                % Decrement indegree
                indegree(i) = indegree(i) - 1;
                
                % If all prerequisites are processed, add to queue
                if indegree(i) == 0
                    queue = [queue, i];
                end
            end
        end
    end
    
    % Check for cycles
    if length(sorted_tasks) < num_tasks
        warning('Task dependency graph contains cycles. Some tasks were not sorted.');
        
        % Add remaining tasks
        missing_tasks = setdiff(1:num_tasks, sorted_tasks);
        sorted_tasks = [sorted_tasks, missing_tasks];
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

function graph = local_buildDependencyGraph(tasks)
    % BUILDDEPENDENCYGRAPH Build a graph representation of task dependencies
    %
    % Parameters:
    %   tasks - Array of task structures
    %
    % Returns:
    %   graph - Structure containing adjacency matrix and other graph properties
    
    num_tasks = length(tasks);
    
    % Initialize adjacency matrix
    adjacency = zeros(num_tasks, num_tasks);
    
    % Fill adjacency matrix based on prerequisites
    for i = 1:num_tasks
        if isfield(tasks(i), 'prerequisites') && ~isempty(tasks(i).prerequisites)
            for prereq = tasks(i).prerequisites
                if prereq <= num_tasks
                    adjacency(prereq, i) = 1;  % Edge from prereq to i
                end
            end
        end
    end
    
    % Calculate indegree and outdegree
    indegree = sum(adjacency, 1)';
    outdegree = sum(adjacency, 2);
    
    % Find source and sink nodes
    source_nodes = find(indegree == 0);
    sink_nodes = find(outdegree == 0);
    
    % Create graph structure
    graph = struct(...
        'adjacency', adjacency, ...
        'indegree', indegree, ...
        'outdegree', outdegree, ...
        'source_nodes', source_nodes, ...
        'sink_nodes', sink_nodes, ...
        'num_nodes', num_tasks ...
    );
    
    % Calculate dependency levels
    levels = local_calculateDependencyLevels(tasks);
    graph.levels = levels;
    
    % Find critical path
    [critical_path, ~] = local_findCriticalPath(tasks);
    graph.critical_path = critical_path;
end

function [critical_path, task_depths] = local_findCriticalPath(tasks)
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