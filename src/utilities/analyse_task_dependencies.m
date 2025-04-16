function [critical_path, task_priorities] = analyse_task_dependencies(tasks, execution_times)
    % ANALYS_TASK_DEPENDENCIES Performs task dependency analysis and critical path identification
    %
    % Inputs:
    %   tasks           - Task information matrix
    %   execution_times - Vector of task execution times
    %
    % Outputs:
    %   critical_path    - Indices of tasks on the critical path
    %   task_priorities  - Priority values for each task
    %
    % This implementation is based on the Critical Path Method (CPM)
    
    % Initialise variables
    num_tasks = size(tasks, 1);
    dependency_matrix = zeros(num_tasks, num_tasks);
    
    % Build adjacency matrix for dependency graph
    for i = 1:num_tasks
        dependencies = tasks(i, 8:10);
        dependencies = dependencies(dependencies > 0);
        
        for j = 1:length(dependencies)
            dep_id = dependencies(j);
            dependency_matrix(dep_id, i) = 1;  % dep_id is a prerequisite for i
        end
    end
    
    % Forward pass - calculate earliest start times
    earliest_start = zeros(num_tasks, 1);
    earliest_finish = zeros(num_tasks, 1);
    
    for i = 1:num_tasks
        % Find all predecessors
        predecessors = find(dependency_matrix(:, i));
        
        if isempty(predecessors)
            earliest_start(i) = 0;
        else
            earliest_start(i) = max(earliest_finish(predecessors));
        end
        
        earliest_finish(i) = earliest_start(i) + execution_times(i);
    end
    
    % Backward pass - calculate latest start times
    latest_finish = zeros(num_tasks, 1);
    latest_start = zeros(num_tasks, 1);
    
    % Initialise latest finish for tasks with no successors
    for i = num_tasks:-1:1
        successors = find(dependency_matrix(i, :));
        
        if isempty(successors)
            latest_finish(i) = max(earliest_finish);  % Project completion time
        else
            latest_finish(i) = min(latest_start(successors));
        end
        
        latest_start(i) = latest_finish(i) - execution_times(i);
    end
    
    % Calculate slack
    slack = latest_start - earliest_start;
    
    % Identify critical path (tasks with zero slack)
    critical_path = find(slack < 0.001);  % Use small threshold to handle floating-point issues
    
    % Calculate task depths
    depths = zeros(num_tasks, 1);
    for i = 1:num_tasks
        depths(i) = calculate_depth(i, dependency_matrix);
    end
    
    % Calculate number of successors for each task
    num_successors = sum(dependency_matrix, 2);
    
    % Calculate priorities
    % Lower values indicate higher priority
    % Components: 
    % - Slack (normalised) - Lower slack means higher priority
    % - Depth (normalised) - Deeper tasks (more predecessors) get higher priority
    % - Number of successors - Tasks with more dependents get higher priority
    task_priorities = zeros(num_tasks, 1);
    
    % Normalise factors
    max_slack = max(slack) + 0.001;  % Avoid division by zero
    max_depth = max(depths) + 0.001;
    max_successors = max(num_successors) + 0.001;
    
    for i = 1:num_tasks
        slack_factor = slack(i) / max_slack;
        depth_factor = 1 - (depths(i) / max_depth);  % Invert so deeper tasks have higher priority
        successor_factor = 1 - (num_successors(i) / max_successors);  % Invert so tasks with more successors have higher priority
        
        % Weight factors
        task_priorities(i) = 0.5 * slack_factor + 0.3 * depth_factor + 0.2 * successor_factor;
    end
    
    % Ensure tasks on critical path have highest priority
    task_priorities(critical_path) = 0;
    
    % Display critical path information
    if nargout == 0
        % Only display if no output arguments
        disp('Task Dependency Analysis:');
        disp('-------------------------');
        disp(['Critical Path: ', mat2str(critical_path')]);
        disp('Task Details:');
        disp('  Task | ES  | EF  | LS  | LF  | Slack | Depth | Priority');
        disp('  --------------------------------------------------');
        for i = 1:num_tasks
            on_cp = ismember(i, critical_path);
            cp_marker = ' ';
            if on_cp
                cp_marker = '*';
            end
            
            fprintf('  %2d%s  | %3.1f | %3.1f | %3.1f | %3.1f | %5.1f | %5d | %7.3f\n', ...
                i, cp_marker, earliest_start(i), earliest_finish(i), ...
                latest_start(i), latest_finish(i), slack(i), depths(i), task_priorities(i));
        end
        disp('  --------------------------------------------------');
        disp('  * = Critical Path Task');
    end
    
    end
    
    function depth = calculate_depth(task_id, dependency_matrix)
    % Calculate the depth of a task in the dependency graph
    % Depth = longest path from any root task to this task
    
    predecessors = find(dependency_matrix(:, task_id));
    
    if isempty(predecessors)
        depth = 1;  % Root task
    else
        predecessor_depths = zeros(length(predecessors), 1);
        for i = 1:length(predecessors)
            predecessor_depths(i) = calculate_depth(predecessors(i), dependency_matrix);
        end
        depth = 1 + max(predecessor_depths);
    end
    end