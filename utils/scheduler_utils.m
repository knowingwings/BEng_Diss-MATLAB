function utils = scheduler_utils()
    % SCHEDULER_UTILS - Utilities for task scheduling and execution
    
    % Create utils structure
    utils = struct();
    
    % Add function handles
    utils.generateSchedule = @generateSchedule;
    utils.updateSchedule = @updateSchedule;
    utils.executeSchedule = @executeSchedule;
    utils.visualizeSchedule = @visualizeSchedule;
    
    function schedule = generateSchedule(assignment, tasks, robots)
        % Generate a schedule based on task assignments
        num_tasks = length(tasks);
        num_robots = length(robots);
        
        % Create dependency matrix
        dep_matrix = zeros(num_tasks, num_tasks);
        for i = 1:num_tasks
            if isfield(tasks, 'prerequisites') && ~isempty(tasks(i).prerequisites)
                dep_matrix(tasks(i).prerequisites, i) = 1;
            end
        end
        
        % Initialize schedule
        schedule = struct();
        schedule.robot_schedules = cell(num_robots, 1);
        schedule.task_start_times = zeros(num_tasks, 1);
        schedule.task_finish_times = zeros(num_tasks, 1);
        schedule.robot_positions = zeros(num_robots, 2);
        schedule.makespan = 0;
        schedule.critical_path = [];
        
        % Get topological ordering of tasks
        [topo_order, schedule.critical_path] = getTopologicalOrder(dep_matrix, tasks);
        
        % Calculate earliest start times
        for i = 1:num_robots
            schedule.robot_schedules{i} = [];
            schedule.robot_positions(i, :) = robots(i).position;
        end
        
        current_time = zeros(num_robots, 1);
        
        % Assign tasks in topological order
        for task_idx = topo_order
            robot_id = assignment(task_idx);
            
            if robot_id == 0
                % Task is unassigned
                continue;
            end
            
            % Check prerequisites
            ready_time = 0;
            prereqs = find(dep_matrix(:, task_idx));
            for p = prereqs'
                ready_time = max(ready_time, schedule.task_finish_times(p));
            end
            
            % Calculate travel time
            travel_time = calculateTravelTime(schedule.robot_positions(robot_id, :), tasks(task_idx).position);
            
            % Calculate start time
            start_time = max(current_time(robot_id), ready_time) + travel_time;
            
            % Get task duration
            if isfield(tasks, 'execution_time')
                duration = tasks(task_idx).execution_time;
            else
                duration = 1; % Default
            end
            
            % Special handling for collaborative tasks
            if isfield(tasks, 'collaborative') && tasks(task_idx).collaborative
                % Find partner robot
                partner_robot = 3 - robot_id; % Assumes dual robots (1 and 2)
                
                % Synchronize start times
                partner_ready_time = current_time(partner_robot) + calculateTravelTime(schedule.robot_positions(partner_robot, :), tasks(task_idx).position);
                start_time = max(start_time, partner_ready_time);
                
                % Both robots are occupied during collaborative task
                current_time(partner_robot) = start_time + duration;
                schedule.robot_positions(partner_robot, :) = tasks(task_idx).position;
            end
            
            % Update schedule
            schedule.task_start_times(task_idx) = start_time;
            schedule.task_finish_times(task_idx) = start_time + duration;
            current_time(robot_id) = start_time + duration;
            schedule.robot_positions(robot_id, :) = tasks(task_idx).position;
            
            % Add task to robot's schedule
            schedule.robot_schedules{robot_id} = [schedule.robot_schedules{robot_id}; task_idx];
        end
        
        % Calculate makespan
        schedule.makespan = max(schedule.task_finish_times);
        
        return;
    end
    
    function schedule = updateSchedule(schedule, task_idx, new_start_time)
        % Update the schedule with a new start time for a task
        old_start = schedule.task_start_times(task_idx);
        delta = new_start_time - old_start;
        
        if delta <= 0
            % No need to update if new start time is earlier
            return;
        end
        
        % Get task duration
        duration = schedule.task_finish_times(task_idx) - schedule.task_start_times(task_idx);
        
        % Update this task
        schedule.task_start_times(task_idx) = new_start_time;
        schedule.task_finish_times(task_idx) = new_start_time + duration;
        
        % Find dependent tasks
        num_tasks = length(schedule.task_start_times);
        dep_matrix = zeros(num_tasks, num_tasks);
        for i = 1:num_tasks
            if isfield(tasks, 'prerequisites') && ~isempty(tasks(i).prerequisites)
                dep_matrix(tasks(i).prerequisites, i) = 1;
            end
        end
        
        dependents = find(dep_matrix(task_idx, :));
        
        % Recursively update dependent tasks
        for dep = dependents
            if schedule.task_start_times(dep) < schedule.task_finish_times(task_idx)
                schedule = updateSchedule(schedule, dep, schedule.task_finish_times(task_idx));
            end
        end
        
        % Update makespan
        schedule.makespan = max(schedule.task_finish_times);
        
        return;
    end
    
    function [status, schedule] = executeSchedule(schedule, current_time, robots, tasks)
        % Execute the schedule up to the current time
        status = struct();
        status.completed_tasks = [];
        status.active_tasks = [];
        status.ready_tasks = [];
        
        % Check each task
        num_tasks = length(schedule.task_start_times);
        for i = 1:num_tasks
            if schedule.task_start_times(i) > 0 % Assigned task
                if current_time >= schedule.task_finish_times(i)
                    % Task completed
                    status.completed_tasks = [status.completed_tasks, i];
                elseif current_time >= schedule.task_start_times(i)
                    % Task active
                    status.active_tasks = [status.active_tasks, i];
                elseif all(ismember(tasks(i).prerequisites, status.completed_tasks))
                    % Task ready to start
                    status.ready_tasks = [status.ready_tasks, i];
                end
            end
        end
        
        % Update robot positions based on active tasks
        for task_idx = status.active_tasks
            robot_id = 0;
            for r = 1:length(schedule.robot_schedules)
                if ismember(task_idx, schedule.robot_schedules{r})
                    robot_id = r;
                    break;
                end
            end
            
            if robot_id > 0 && robot_id <= length(robots)
                % Move robot to task position
                robots(robot_id).position = tasks(task_idx).position;
            end
        end
        
        return;
    end
    
    function visualizeSchedule(schedule, tasks, robots)
        % Visualize the schedule as a Gantt chart
        figure('Name', 'Schedule Visualization', 'Position', [100, 100, 1000, 500]);
        
        num_robots = length(schedule.robot_schedules);
        num_tasks = length(schedule.task_start_times);
        
        % Colors for different robots
        robot_colors = {'b', 'r', 'g', 'm', 'c'};
        
        % Plot Gantt chart
        hold on;
        
        for r = 1:num_robots
            robot_tasks = schedule.robot_schedules{r};
            
            for i = 1:length(robot_tasks)
                task_idx = robot_tasks(i);
                start_time = schedule.task_start_times(task_idx);
                finish_time = schedule.task_finish_times(task_idx);
                duration = finish_time - start_time;
                
                % Determine color based on robot
                color = robot_colors{mod(r-1, length(robot_colors))+1};
                
                % Special handling for collaborative tasks
                is_collaborative = false;
                if isfield(tasks, 'collaborative')
                    is_collaborative = tasks(task_idx).collaborative;
                end
                
                if is_collaborative
                    % Use a pattern fill for collaborative tasks
                    h = rectangle('Position', [start_time, r-0.4, duration, 0.8], ...
                                  'FaceColor', color, ...
                                  'EdgeColor', 'k', ...
                                  'LineWidth', 1.5, ...
                                  'LineStyle', '--');
                else
                    % Regular task
                    h = rectangle('Position', [start_time, r-0.4, duration, 0.8], ...
                                  'FaceColor', color, ...
                                  'EdgeColor', 'k', ...
                                  'LineWidth', 1);
                end
                
                % Add task number
                text(start_time + duration/2, r, sprintf('T%d', task_idx), ...
                     'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'middle', ...
                     'Color', 'w', ...
                     'FontWeight', 'bold');
            end
        end
        
        % Add critical path highlight
        if isfield(schedule, 'critical_path') && ~isempty(schedule.critical_path)
            for i = 1:length(schedule.critical_path)
                task_idx = schedule.critical_path(i);
                
                % Find which robot is assigned this task
                for r = 1:num_robots
                    if ismember(task_idx, schedule.robot_schedules{r})
                        start_time = schedule.task_start_times(task_idx);
                        finish_time = schedule.task_finish_times(task_idx);
                        duration = finish_time - start_time;
                        
                        % Highlight critical path with a thicker border
                        rectangle('Position', [start_time, r-0.4, duration, 0.8], ...
                                  'EdgeColor', 'k', ...
                                  'LineWidth', 3, ...
                                  'LineStyle', '-', ...
                                  'FaceColor', 'none');
                        break;
                    end
                end
            end
        end
        
        % Plot prerequisites as arrows
        for i = 1:num_tasks
            if isfield(tasks, 'prerequisites') && ~isempty(tasks(i).prerequisites)
                for prereq = tasks(i).prerequisites
                    % Find robot assignments
                    r_prereq = 0;
                    r_task = 0;
                    
                    for r = 1:num_robots
                        if ismember(prereq, schedule.robot_schedules{r})
                            r_prereq = r;
                        end
                        if ismember(i, schedule.robot_schedules{r})
                            r_task = r;
                        end
                    end
                    
                    if r_prereq > 0 && r_task > 0
                        prereq_end = schedule.task_finish_times(prereq);
                        task_start = schedule.task_start_times(i);
                        
                        % Draw arrow from prerequisite to dependent task
                        arrow([prereq_end, r_prereq], [task_start, r_task], ...
                              'Length', 10, 'Width', 1, 'EdgeColor', [0.5 0.5 0.5], ...
                              'FaceColor', [0.5 0.5 0.5]);
                    end
                end
            end
        end
        
        % Set plot properties
        ylim([0.5, num_robots+0.5]);
        xlim([0, schedule.makespan*1.1]);
        yticks(1:num_robots);
        yticklabels(arrayfun(@(i) sprintf('Robot %d', i), 1:num_robots, 'UniformOutput', false));
        grid on;
        title('Schedule Gantt Chart');
        xlabel('Time');
        ylabel('Robot');
        
        % Add legend
        legend_handles = [];
        legend_labels = {};
        
        for r = 1:min(num_robots, length(robot_colors))
            h = rectangle('Position', [0, 0, 1, 1], 'Visible', 'off', ...
                         'FaceColor', robot_colors{r});
            legend_handles = [legend_handles, h];
            legend_labels{end+1} = sprintf('Robot %d', r);
        end
        
        % Add collaborative task to legend if any
        collab_exists = false;
        if isfield(tasks, 'collaborative')
            collab_exists = any([tasks.collaborative]);
        end
        
        if collab_exists
            h = rectangle('Position', [0, 0, 1, 1], 'Visible', 'off', ...
                         'EdgeColor', 'k', 'LineStyle', '--', 'LineWidth', 1.5);
            legend_handles = [legend_handles, h];
            legend_labels{end+1} = 'Collaborative Task';
        end
        
        % Add critical path to legend
        if isfield(schedule, 'critical_path') && ~isempty(schedule.critical_path)
            h = rectangle('Position', [0, 0, 1, 1], 'Visible', 'off', ...
                         'EdgeColor', 'k', 'LineWidth', 3);
            legend_handles = [legend_handles, h];
            legend_labels{end+1} = 'Critical Path';
        end
        
        legend(legend_handles, legend_labels, 'Location', 'northeastoutside');
        
        hold off;
    end
    
    % Helper functions
    function travel_time = calculateTravelTime(start_pos, end_pos)
        % Calculate travel time between two positions
        distance = norm(start_pos - end_pos);
        max_speed = 0.5; % m/s - adjust based on robot capabilities
        travel_time = distance / max_speed;
    end
    
    function [topo_order, critical_path] = getTopologicalOrder(dep_matrix, tasks)
        % Get topological ordering of tasks respecting dependencies
        num_tasks = size(dep_matrix, 1);
        visited = false(num_tasks, 1);
        topo_order = [];
        
        % Visit each node
        for i = 1:num_tasks
            if ~visited(i)
                [visited, topo_order] = topologicalVisit(i, visited, topo_order, dep_matrix);
            end
        end
        
        % Reverse order (we built it backwards)
        topo_order = fliplr(topo_order);
        
        % Calculate critical path
        if nargin >= 2
            [~, critical_path] = calculateCriticalPath(tasks, dep_matrix);
        else
            critical_path = [];
        end
    end
    
    function [visited, topo_order] = topologicalVisit(node, visited, topo_order, dep_matrix)
        visited(node) = true;
        
        % Visit all dependents
        dependents = find(dep_matrix(node, :));
        for i = dependents
            if ~visited(i)
                [visited, topo_order] = topologicalVisit(i, visited, topo_order, dep_matrix);
            end
        end
        
        % Add current node
        topo_order = [topo_order, node];
    end
    
    function [earliest_finish, critical_path] = calculateCriticalPath(tasks, dep_matrix)
        num_tasks = length(tasks);
        
        % Calculate earliest start times (forward pass)
        earliest_start = zeros(num_tasks, 1);
        earliest_finish = zeros(num_tasks, 1);
        
        % Forward pass
        topo_order = getTopologicalOrder(dep_matrix);
        
        for i = topo_order
            % Find prerequisites
            prereqs = find(dep_matrix(:, i));
            
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
        makespan = max(earliest_finish);
        latest_finish = makespan * ones(num_tasks, 1);
        latest_start = zeros(num_tasks, 1);
        
        % Backward pass
        reverse_topo = fliplr(topo_order);
        
        for i = reverse_topo
            % Find tasks that depend on this one
            dependents = find(dep_matrix(i, :));
            
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
        critical_path = find(slack < 0.001);  % Near-zero slack
    end
    
    % Helper function for drawing arrows
    function h = arrow(start, stop, varargin)
        % Define arrow properties
        p = inputParser;
        addParameter(p, 'Length', 5, @isnumeric);
        addParameter(p, 'Width', 2, @isnumeric);
        addParameter(p, 'EdgeColor', 'k', @(x)ischar(x)||isnumeric(x));
        addParameter(p, 'FaceColor', 'k', @(x)ischar(x)||isnumeric(x));
        parse(p, varargin{:});
        
        % Extract inputs
        x1 = start(1);
        y1 = start(2);
        x2 = stop(1);
        y2 = stop(2);
        
        % Draw the line
        h1 = line([x1, x2], [y1, y2], 'Color', p.Results.EdgeColor);
        
        % Calculate the arrow angle
        angle = atan2(y2 - y1, x2 - x1);
        
        % Define arrow head shape
        len = p.Results.Length;
        width = p.Results.Width;
        
        % Calculate arrow head coordinates
        x3 = x2 - len * cos(angle - pi/6);
        y3 = y2 - len * sin(angle - pi/6);
        x4 = x2 - len * cos(angle + pi/6);
        y4 = y2 - len * sin(angle + pi/6);
        
        % Draw arrow head
        h2 = patch([x2, x3, x4], [y2, y3, y4], p.Results.FaceColor, ...
                   'EdgeColor', p.Results.EdgeColor);
        
        % Return handle
        h = [h1, h2];
    end
    
    return;
end