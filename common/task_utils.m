function utils = task_utils()
    % TASK_UTILS - Returns function handles for task-related functions
    utils = struct(...
        'createTasks', @local_createTasks, ...
        'addTaskDependencies', @local_addTaskDependencies, ...
        'findAvailableTasks', @local_findAvailableTasks, ...
        'calculateTaskCriticality', @local_calculateTaskCriticality, ...
        'calculateDependencyDepth', @local_calculateDependencyDepth, ...
        'isOnCriticalPath', @local_isOnCriticalPath ...
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
            tasks(i).execution_time = 5 + 3 * rand();  % 5-8 time units
        elseif pattern_idx == 2
            tasks(i).execution_time = 7 + 3 * rand();  % 7-10 time units
        else
            tasks(i).execution_time = 6 + 4 * rand();  % 6-10 time units
        end
        
        tasks(i).prerequisites = [];  % Will be filled in addTaskDependencies
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
    
    % IMPROVED: Create a more structured dependency tree
    % Extract x-coordinates for proper sorting
    x_coords = zeros(num_tasks, 1);
    for i = 1:num_tasks
        x_coords(i) = tasks(i).position(1);  % Just use x-coordinate 
    end
    [~, sorted_idx] = sort(x_coords);
    
    % IMPROVED: Ensure we don't create cycles by only allowing dependencies i -> j where i < j in sorted order
    for j = 2:num_tasks
        task_idx = sorted_idx(j);
        
        % Consider only tasks that come before in the sorted order
        potential_prereqs = sorted_idx(1:j-1);
        
        % Limit the maximum number of prerequisites to avoid overconstraining
        max_prereqs = min(3, length(potential_prereqs));
        
        % Randomly select prerequisites with the given probability
        for i = 1:length(potential_prereqs)
            if rand() < probability && length(tasks(task_idx).prerequisites) < max_prereqs
                prereq_idx = potential_prereqs(i);
                
                % Avoid adding duplicates
                if ~ismember(prereq_idx, tasks(task_idx).prerequisites)
                    tasks(task_idx).prerequisites = [tasks(task_idx).prerequisites, prereq_idx];
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
        % IMPROVED: Check if task is already in completed_tasks, if so, skip it
        if ismember(i, completed_tasks)
            continue;
        end
        
        % Check if all prerequisites are completed
        if isempty(tasks(i).prerequisites) || all(ismember(tasks(i).prerequisites, completed_tasks))
            available_tasks = [available_tasks, i];
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
        depths = zeros(1, length(tasks(task_id).prerequisites));
        for i = 1:length(tasks(task_id).prerequisites)
            prereq_id = tasks(task_id).prerequisites(i);
            depths(i) = local_calculateDependencyDepth(prereq_id, tasks);
        end
        dependency_depth = 1 + max(depths);
    end
    
    % Store calculated depth for future use
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
end