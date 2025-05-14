% enhanced_visualization.m
% Enhanced visualization tools for distributed auction algorithm
% Provides functions to create more informative and interactive visualizations

function utils = enhanced_visualization()
    % ENHANCED_VISUALIZATION - Returns function handles for enhanced visualization
    utils = struct(...
        'visualizeAuctionProcess', @local_visualizeAuctionProcess, ...
        'createInteractiveTaskGraph', @local_createInteractiveTaskGraph, ...
        'visualizeRobotWorkspaces', @local_visualizeRobotWorkspaces, ...
        'createBidValueHeatmap', @local_createBidValueHeatmap, ...
        'visualizeBiddingHistory', @local_visualizeBiddingHistory, ...
        'createTaskAssignmentAnimation', @local_createTaskAssignmentAnimation, ...
        'visualizeUtilityLandscape', @local_visualizeUtilityLandscape, ...
        'createPerformanceDashboard', @local_createPerformanceDashboard ...
    );
end

function local_visualizeAuctionProcess(auction_data, tasks, robots, selected_tasks)
    % VISUALIZEAUCTIONPROCESS - Create a detailed visualization of the auction process
    %
    % Parameters:
    %   auction_data - Auction data structure
    %   tasks - Array of task structures
    %   robots - Array of robot structures
    %   selected_tasks - (Optional) List of task IDs to highlight
    
    fprintf('Creating enhanced auction process visualization...\n');
    
    % Set default for selected tasks
    if nargin < 4
        selected_tasks = [];
    end
    
    % Create figure
    figure('Name', 'Auction Process Visualization', 'Position', [50, 50, 1200, 700]);
    
    % Create 3x2 layout
    subplot(2, 3, 1);
    
    % 1. Current task allocation visualization
    % Draw environment boundary (assumed 4x4)
    env_width = 4;
    env_height = 4;
    rectangle('Position', [0, 0, env_width, env_height], 'EdgeColor', 'k', 'LineWidth', 2);
    hold on;
    
    % Draw tasks
    for i = 1:length(tasks)
        % Determine task color based on assignment
        if auction_data.assignment(i) == 0
            color = [0.7, 0.7, 0.7];  % Gray for unassigned
            marker = 's';
        else
            robot_id = auction_data.assignment(i);
            if robot_id > length(robots) || (isfield(robots, 'failed') && robots(robot_id).failed)
                color = [1.0, 0.3, 0.3];  % Red for tasks assigned to failed robot
                marker = 'x';
            else
                robot_colors = {'b', 'g', 'r', 'm', 'c'};  % Colors for different robots
                color = robot_colors{mod(robot_id-1, length(robot_colors))+1};
                marker = 'o';
            end
        end
        
        % Highlight selected tasks
        if ismember(i, selected_tasks)
            edge_color = 'r';
            line_width = 2;
            marker_size = 10;
        else
            edge_color = 'k';
            line_width = 1;
            marker_size = 8;
        end
        
        % Plot task
        plot(tasks(i).position(1), tasks(i).position(2), marker, ...
             'Color', color, 'MarkerSize', marker_size, 'LineWidth', line_width, ...
             'MarkerEdgeColor', edge_color);
        
        % Add task ID
        text(tasks(i).position(1) + 0.1, tasks(i).position(2) + 0.1, ...
             sprintf('%d', i), 'FontSize', 8);
    end
    
    % Draw robots
    for i = 1:length(robots)
        if isfield(robots, 'failed') && robots(i).failed
            % Failed robot
            plot(robots(i).position(1), robots(i).position(2), 'rx', 'MarkerSize', 12, 'LineWidth', 2);
        else
            % Active robot
            robot_colors = {'b', 'g', 'r', 'm', 'c'};
            color = robot_colors{mod(i-1, length(robot_colors))+1};
            plot(robots(i).position(1), robots(i).position(2), 'p', ...
                 'Color', color, 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', color);
        end
        
        % Add robot ID
        text(robots(i).position(1) + 0.15, robots(i).position(2) + 0.15, ...
             sprintf('R%d', i), 'FontSize', 10, 'FontWeight', 'bold');
        
        % Draw lines to assigned tasks
        assigned_tasks = find(auction_data.assignment == i);
        for j = 1:length(assigned_tasks)
            task_id = assigned_tasks(j);
            line([robots(i).position(1), tasks(task_id).position(1)], ...
                 [robots(i).position(2), tasks(task_id).position(2)], ...
                 'Color', color, 'LineStyle', '--', 'LineWidth', 1);
        end
    end
    
    % Add axis labels and title
    axis([0, env_width, 0, env_height]);
    axis equal;
    grid on;
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    title('Current Task Allocation');
    
    % 2. Price evolution visualization
    subplot(2, 3, 2);
    
    % Get price history data
    if isfield(auction_data, 'all_utilities') && ~isempty(auction_data.all_utilities)
        % If detailed history is available, use it
        num_iterations = size(auction_data.all_utilities, 3);
        price_history = zeros(length(tasks), num_iterations);
        
        % Extract price history from the full history
        for i = 1:num_iterations
            if i <= size(auction_data.all_utilities, 3)
                for j = 1:length(tasks)
                    if j <= size(auction_data.all_utilities, 2)
                        % Find maximum bid for this task at this iteration
                        bids = auction_data.all_utilities(:, j, i);
                        max_bid = max(bids);
                        price_history(j, i) = max_bid;
                    end
                end
            end
        end
    else if isfield(auction_data, 'price_history') && ~isempty(auction_data.price_history)
            % Use provided price history
            price_history = auction_data.price_history;
        else
            % No history available, just use current prices
            price_history = auction_data.prices;
        end
    end
    
    % Plot price evolution
    if size(price_history, 2) > 1
        % Multiple iterations available
        for i = 1:size(price_history, 1)
            plot(price_history(i, :), 'LineWidth', 1.5);
            hold on;
        end
        
        xlabel('Iteration');
        ylabel('Task Price');
        title('Price Evolution');
        grid on;
        
        % Highlight selected tasks in legend
        if ~isempty(selected_tasks)
            legend_entries = cell(length(selected_tasks), 1);
            for i = 1:length(selected_tasks)
                legend_entries{i} = sprintf('Task %d', selected_tasks(i));
            end
            legend(legend_entries, 'Location', 'eastoutside');
        end
    else
        % Only current prices available
        bar(price_history);
        xlabel('Task ID');
        ylabel('Price');
        title('Current Task Prices');
        grid on;
    end
    
    % 3. Utility landscape for robots
    subplot(2, 3, 3);
    
    % Create utility matrix
    utility_matrix = zeros(length(robots), length(tasks));
    
    % Fill in current utilities if available
    if isfield(auction_data, 'utilities') && ~isempty(auction_data.utilities)
        utility_matrix = auction_data.utilities;
    else
        % Estimate utilities from bids and prices
        if isfield(auction_data, 'bids') && ~isempty(auction_data.bids)
            for i = 1:length(robots)
                for j = 1:length(tasks)
                    utility_matrix(i, j) = auction_data.bids(i, j) - auction_data.prices(j);
                end
            end
        end
    end
    
    % Create heatmap
    imagesc(utility_matrix);
    colormap('jet');
    colorbar;
    
    % Add labels
    xlabel('Task ID');
    ylabel('Robot ID');
    title('Utility Landscape');
    
    % Add axis labels
    xticks(1:length(tasks));
    yticks(1:length(robots));
    
    % 4. Workload distribution
    subplot(2, 3, 4);
    
    % Calculate workload for each robot
    workloads = zeros(1, length(robots));
    for i = 1:length(robots)
        assigned_tasks = find(auction_data.assignment == i);
        for j = 1:length(assigned_tasks)
            task_id = assigned_tasks(j);
            if isfield(tasks, 'execution_time')
                workloads(i) = workloads(i) + tasks(task_id).execution_time;
            else
                workloads(i) = workloads(i) + 1;  % Default weight of 1
            end
        end
    end
    
    % Create bar plot
    bar(workloads);
    
    % Add labels
    xlabel('Robot ID');
    ylabel('Total Workload');
    title('Workload Distribution');
    grid on;
    
    % 5. Task dependencies visualization
    subplot(2, 3, 5);
    
    % Create graph adjacency matrix
    num_tasks = length(tasks);
    adjacency = zeros(num_tasks, num_tasks);
    
    % Fill in dependencies
    for i = 1:num_tasks
        if isfield(tasks(i), 'prerequisites') && ~isempty(tasks(i).prerequisites)
            for j = 1:length(tasks(i).prerequisites)
                prereq = tasks(i).prerequisites(j);
                if prereq <= num_tasks
                    adjacency(prereq, i) = 1;
                end
            end
        end
    end
    
    % Plot as a directed graph
    try
        G = digraph(adjacency);
        
        % Get task IDs as node labels
        node_labels = cell(num_tasks, 1);
        for i = 1:num_tasks
            node_labels{i} = num2str(i);
        end
        
        % Create node colors based on assignment
        node_colors = cell(num_tasks, 1);
        for i = 1:num_tasks
            if auction_data.assignment(i) == 0
                node_colors{i} = [0.7, 0.7, 0.7];  % Gray for unassigned
            else
                robot_id = auction_data.assignment(i);
                robot_colormap = [0, 0, 1; 0, 0.8, 0; 1, 0, 0; 1, 0, 1; 0, 1, 1];
                if robot_id > size(robot_colormap, 1)
                    robot_id = mod(robot_id-1, size(robot_colormap, 1)) + 1;
                end
                node_colors{i} = robot_colormap(robot_id, :);
            end
        end
        
        % Highlight selected tasks
        highlight_nodes = selected_tasks;
        
        % Draw graph
        h = plot(G, 'NodeLabel', node_labels, 'NodeColor', cell2mat(node_colors));
        
        % Highlight selected nodes
        if ~isempty(highlight_nodes)
            highlight(h, highlight_nodes, 'NodeColor', 'r', 'MarkerSize', 8);
        end
        
        % Add labels
        title('Task Dependencies');
    catch
        % If graph toolbox not available, show adjacency matrix
        imagesc(adjacency);
        colormap('gray');
        
        % Add labels
        xlabel('Task ID');
        ylabel('Prerequisite Task ID');
        title('Task Dependencies');
        
        % Add axis ticks
        xticks(1:num_tasks);
        yticks(1:num_tasks);
    end
    
    % 6. Auction statistics
    subplot(2, 3, 6);
    
    % Calculate statistics
    unassigned_count = sum(auction_data.assignment == 0);
    assigned_count = length(tasks) - unassigned_count;
    assignment_rate = assigned_count / length(tasks) * 100;
    
    % Calculate workload imbalance
    if length(robots) > 1
        active_robots = find(~[robots.failed]);
        if ~isempty(active_robots) && length(active_robots) > 1
            active_workloads = workloads(active_robots);
            workload_min = min(active_workloads);
            workload_max = max(active_workloads);
            workload_imbalance = (workload_max - workload_min) / workload_max * 100;
        else
            workload_imbalance = 0;
        end
    else
        workload_imbalance = 0;
    end
    
    % Create text for statistics
    stats_text = {
        sprintf('Total tasks: %d', length(tasks)),
        sprintf('Assigned tasks: %d (%.1f%%)', assigned_count, assignment_rate),
        sprintf('Unassigned tasks: %d (%.1f%%)', unassigned_count, 100 - assignment_rate),
        sprintf('Workload imbalance: %.1f%%', workload_imbalance)
    };
    
    % Add oscillation statistics if available
    if isfield(auction_data, 'task_oscillation_count') && ~isempty(auction_data.task_oscillation_count)
        max_oscillations = max(auction_data.task_oscillation_count);
        avg_oscillations = mean(auction_data.task_oscillation_count);
        
        stats_text{end+1} = sprintf('Max task oscillations: %d', max_oscillations);
        stats_text{end+1} = sprintf('Avg task oscillations: %.1f', avg_oscillations);
    end
    
    % Add recovery information if applicable
    if isfield(auction_data, 'recovery_mode') && auction_data.recovery_mode
        stats_text{end+1} = '';
        stats_text{end+1} = 'RECOVERY MODE ACTIVE';
        
        if isfield(auction_data, 'failure_assignment') && ~isempty(auction_data.failure_assignment)
            failed_tasks = sum(auction_data.failure_assignment == auction_data.failed_robot);
            reassigned_tasks = sum(auction_data.assignment ~= auction_data.failed_robot & ...
                                   auction_data.failure_assignment == auction_data.failed_robot);
            reassignment_rate = reassigned_tasks / failed_tasks * 100;
            
            stats_text{end+1} = sprintf('Failed tasks: %d', failed_tasks);
            stats_text{end+1} = sprintf('Reassigned: %d (%.1f%%)', reassigned_tasks, reassignment_rate);
        end
    end
    
    % Create text display
    text(0.5, 0.5, stats_text, 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle', 'FontSize', 10);
    axis off;
    title('Auction Statistics');
    
    % Adjust layout
    set(gcf, 'Color', 'w');
    sgtitle('Enhanced Auction Process Visualization', 'FontSize', 14);
    
    fprintf('Enhanced auction process visualization created.\n');
end

function local_createInteractiveTaskGraph(tasks, auction_data)
    % CREATEINTERACTIVETASKGRAPH - Create an interactive task dependency graph
    %
    % Parameters:
    %   tasks - Array of task structures
    %   auction_data - Auction data structure
    
    fprintf('Creating interactive task dependency graph...\n');
    
    % Check for required graph toolbox
    if ~exist('graph', 'dir') && ~exist('digraph', 'file')
        fprintf('Warning: Graph toolbox not available. Creating static graph instead.\n');
        local_createStaticTaskGraph(tasks, auction_data);
        return;
    end
    
    % Create figure
    figure('Name', 'Interactive Task Dependency Graph', 'Position', [100, 100, 900, 700]);
    
    % Create graph adjacency matrix
    num_tasks = length(tasks);
    adjacency = zeros(num_tasks, num_tasks);
    
    % Fill in dependencies
    for i = 1:num_tasks
        if isfield(tasks(i), 'prerequisites') && ~isempty(tasks(i).prerequisites)
            for j = 1:length(tasks(i).prerequisites)
                prereq = tasks(i).prerequisites(j);
                if prereq <= num_tasks
                    adjacency(prereq, i) = 1;
                end
            end
        end
    end
    
    % Create directed graph
    G = digraph(adjacency);
    
    % Create node labels
    node_labels = cell(num_tasks, 1);
    for i = 1:num_tasks
        if isfield(tasks(i), 'name')
            % Use task name if available
            node_labels{i} = sprintf('%d: %s', i, tasks(i).name);
        else
            % Just use task ID
            node_labels{i} = sprintf('%d', i);
        end
    end
    
    % Create node colors based on assignment
    node_colors = ones(num_tasks, 3) * 0.7;  % Default gray
    
    for i = 1:num_tasks
        if auction_data.assignment(i) > 0
            % Use robot color for assigned tasks
            robot_id = auction_data.assignment(i);
            robot_colormap = [0, 0, 1; 0, 0.8, 0; 1, 0, 0; 1, 0, 1; 0, 1, 1];
            color_idx = mod(robot_id-1, size(robot_colormap, 1)) + 1;
            node_colors(i, :) = robot_colormap(color_idx, :);
        end
    end
    
    % Create graph plot
    h = plot(G, 'NodeLabel', node_labels, 'NodeColor', node_colors, ...
             'EdgeColor', [0.5, 0.5, 0.5], 'LineWidth', 1.5, ...
             'Marker', 'o', 'MarkerSize', 8, 'NodeFontSize', 8);
    
    % Set layout algorithm
    layout(h, 'layered');
    
    % Add title and axis labels
    title('Interactive Task Dependency Graph', 'FontSize', 14);
    
    % Add legend for robot assignments
    robot_colormap = [0, 0, 1; 0, 0.8, 0; 1, 0, 0; 1, 0, 1; 0, 1, 1; 0.7, 0.7, 0.7];
    legend_entries = cell(size(robot_colormap, 1), 1);
    legend_entries{end} = 'Unassigned';
    
    % Find unique robot assignments
    unique_robots = unique(auction_data.assignment);
    unique_robots = unique_robots(unique_robots > 0);
    
    for i = 1:min(length(unique_robots), size(robot_colormap, 1)-1)
        legend_entries{i} = sprintf('Robot %d', unique_robots(i));
    end
    
    % Create legend handles
    legend_handles = zeros(size(robot_colormap, 1), 1);
    for i = 1:size(robot_colormap, 1)
        legend_handles(i) = plot(NaN, NaN, 'o', 'MarkerFaceColor', robot_colormap(i, :), ...
                                'MarkerEdgeColor', 'k', 'MarkerSize', 8);
        hold on;
    end
    
    % Add legend
    legend(legend_handles, legend_entries, 'Location', 'eastoutside');
    
    % Add task completion status if available
    if isfield(auction_data, 'completion_status') && ~isempty(auction_data.completion_status)
        completed_tasks = find(auction_data.completion_status == 1);
        if ~isempty(completed_tasks)
            highlight(h, completed_tasks, 'NodeColor', [0, 0.7, 0]);
        end
    end
    
    % Add right-click menu for node selection
    h.NodeLabel = '';  % Hide node labels initially
    
    % Add UI controls
    % Toggle for node labels
    uicontrol('Style', 'checkbox', 'String', 'Show Labels', ...
              'Position', [20, 20, 100, 20], 'Value', 0, ...
              'Callback', @(src, event) toggleLabels(src, h, node_labels));
    
    % Toggle for layout algorithm
    uicontrol('Style', 'popupmenu', 'String', {'layered', 'force', 'circle', 'subspace'}, ...
              'Position', [20, 50, 100, 20], 'Value', 1, ...
              'Callback', @(src, event) changeLayout(src, h));
    
    % Dropdown for highlighting critical paths
    uicontrol('Style', 'pushbutton', 'String', 'Highlight Critical Path', ...
              'Position', [20, 80, 150, 20], ...
              'Callback', @(src, event) highlightCriticalPath(h, tasks));
    
    % Button to reset view
    uicontrol('Style', 'pushbutton', 'String', 'Reset View', ...
              'Position', [20, 110, 100, 20], ...
              'Callback', @(src, event) resetView(h));
    
    fprintf('Interactive task dependency graph created.\n');
    
    % Callback functions
    function toggleLabels(src, h, labels)
        if src.Value
            h.NodeLabel = labels;
        else
            h.NodeLabel = '';
        end
    end
    
    function changeLayout(src, h)
        layouts = {'layered', 'force', 'circle', 'subspace'};
        selected = layouts{src.Value};
        layout(h, selected);
    end
    
    function highlightCriticalPath(h, tasks)
        % Calculate critical path
        % This is a simplified version - in a real implementation, you would
        % use a more sophisticated algorithm
        
        % Create directed graph for analysis
        adjacency = zeros(length(tasks));
        
        for i = 1:length(tasks)
            if isfield(tasks(i), 'prerequisites') && ~isempty(tasks(i).prerequisites)
                for j = 1:length(tasks(i).prerequisites)
                    prereq = tasks(i).prerequisites(j);
                    if prereq <= length(tasks)
                        adjacency(prereq, i) = 1;
                    end
                end
            end
        end
        
        % Find start and end nodes
        start_nodes = find(sum(adjacency, 1) == 0);
        end_nodes = find(sum(adjacency, 2) == 0);
        
        % Use depth-first search to find longest path
        max_path_length = 0;
        critical_path = [];
        
        for start_node = start_nodes'
            [path_length, path] = findLongestPath(adjacency, start_node, end_nodes, tasks);
            
            if path_length > max_path_length
                max_path_length = path_length;
                critical_path = path;
            end
        end
        
        % Highlight critical path
        if ~isempty(critical_path)
            % Reset all nodes first
            highlight(h, 'NodeColor', h.NodeColor);
            
            % Highlight nodes on critical path
            highlight(h, critical_path, 'NodeColor', 'r');
            
            % Highlight edges on critical path
            for i = 1:length(critical_path)-1
                edge_idx = findedge(h.Graph, critical_path(i), critical_path(i+1));
                if edge_idx > 0
                    highlight(h, edge_idx, 'EdgeColor', 'r', 'LineWidth', 2);
                end
            end
        end
    end
    
    function resetView(h)
        % Reset highlights and layout
        layout(h, 'layered');
        highlight(h, 'NodeColor', h.NodeColor);
        highlight(h, 'EdgeColor', [0.5, 0.5, 0.5], 'LineWidth', 1.5);
    end
    
    function [max_length, max_path] = findLongestPath(adjacency, start, end_nodes, tasks)
        % Recursive function to find longest path using DFS
        max_length = 0;
        max_path = start;
        
        % Check if this is an end node
        if ismember(start, end_nodes)
            if isfield(tasks, 'execution_time')
                max_length = tasks(start).execution_time;
            else
                max_length = 1;
            end
            return;
        end
        
        % Find all outgoing edges
        successors = find(adjacency(start, :) == 1);
        
        for succ = successors
            % Recursive call
            [path_length, path] = findLongestPath(adjacency, succ, end_nodes, tasks);
            
            % Add execution time of current task
            if isfield(tasks, 'execution_time')
                path_length = path_length + tasks(start).execution_time;
            else
                path_length = path_length + 1;
            end
            
            % Check if this is the longest path so far
            if path_length > max_length
                max_length = path_length;
                max_path = [start, path];
            end
        end
    end
end

function local_createStaticTaskGraph(tasks, auction_data)
    % Fallback function for when the graph toolbox is not available
    
    % Create figure
    figure('Name', 'Task Dependency Graph', 'Position', [100, 100, 800, 600]);
    
    % Create adjacency matrix
    num_tasks = length(tasks);
    adjacency = zeros(num_tasks, num_tasks);
    
    % Fill in dependencies
    for i = 1:num_tasks
        if isfield(tasks(i), 'prerequisites') && ~isempty(tasks(i).prerequisites)
            for j = 1:length(tasks(i).prerequisites)
                prereq = tasks(i).prerequisites(j);
                if prereq <= num_tasks
                    adjacency(prereq, i) = 1;
                end
            end
        end
    end
    
    % Plot adjacency matrix as a heatmap
    imagesc(adjacency);
    colormap('gray');
    
    % Add labels
    xlabel('Task ID');
    ylabel('Prerequisite Task ID');
    title('Task Dependencies (Adjacency Matrix)');
    
    % Add axis ticks
    xticks(1:num_tasks);
    yticks(1:num_tasks);
    
    % Add grid
    grid on;
    
    % Add task names to ticks if available
    if isfield(tasks, 'name')
        task_names = cell(num_tasks, 1);
        for i = 1:num_tasks
            task_names{i} = sprintf('%d: %s', i, tasks(i).name);
        end
        xticklabels(task_names);
        yticklabels(task_names);
        xtickangle(45);
    end
    
    % Create a second subplot for task assignments
    subplot(1, 2, 2);
    
    % Create task assignment visualization
    bar(auction_data.assignment);
    
    % Add labels
    xlabel('Task ID');
    ylabel('Assigned Robot ID');
    title('Current Task Assignments');
    
    % Add grid
    grid on;
    
    % Add axis ticks
    xticks(1:num_tasks);
    yticks(0:max(auction_data.assignment));
    
    % Adjust layout
    set(gcf, 'Color', 'w');
end

function local_visualizeRobotWorkspaces(robots, tasks, env, auction_data)
    % VISUALIZEROBOTWORKSPACES - Visualize robot workspaces and task assignments
    %
    % Parameters:
    %   robots - Array of robot structures
    %   tasks - Array of task structures
    %   env - Environment structure
    %   auction_data - Auction data structure
    
    fprintf('Creating robot workspace visualization...\n');
    
    % Create figure
    figure('Name', 'Robot Workspaces', 'Position', [100, 100, 1000, 800]);
    
    % Create grid for workspace calculation
    grid_resolution = 0.1;
    x_range = 0:grid_resolution:env.width;
    y_range = 0:grid_resolution:env.height;
    [X, Y] = meshgrid(x_range, y_range);
    
    % Calculate workspace cost for each robot
    workspace_costs = zeros(length(robots), length(y_range), length(x_range));
    
    for r = 1:length(robots)
        robot_pos = robots(r).position;
        
        % Calculate distance from robot to each grid point
        for i = 1:length(y_range)
            for j = 1:length(x_range)
                grid_pos = [x_range(j), y_range(i)];
                
                % Calculate Euclidean distance
                distance = norm(robot_pos - grid_pos);
                
                % Convert to cost
                workspace_costs(r, i, j) = distance;
            end
        end
    end
    
    % Calculate optimal workspace allocation
    [min_cost, best_robot] = min(workspace_costs, [], 1);
    
    % FIX: Make sure dimensions match by squeezing best_robot to 2D
    best_robot = squeeze(best_robot);
    
    % Create mask for each robot's optimal workspace
    robot_masks = zeros(length(robots), size(best_robot, 1), size(best_robot, 2));
    
    for r = 1:length(robots)
        % FIX: Create the masks correctly with matching dimensions
        robot_masks(r, :, :) = (best_robot == r);
    end
    
    % Create colormap for visualization
    robot_colors = [
        0, 0, 1;    % Blue
        0, 0.8, 0;  % Green
        1, 0, 0;    % Red
        1, 0, 1;    % Magenta
        0, 1, 1;    % Cyan
        0.8, 0.4, 0; % Orange
        0.5, 0, 0.5; % Purple
        0, 0.5, 0.5; % Teal
    ];
    
    % Ensure we have enough colors
    if length(robots) > size(robot_colors, 1)
        robot_colors = [robot_colors; rand(length(robots) - size(robot_colors, 1), 3)];
    end
    
    % Create combined workspace image
    workspace_img = zeros(size(best_robot, 1), size(best_robot, 2), 3);
    
    for r = 1:length(robots)
        if ~(isfield(robots, 'failed') && robots(r).failed)
            % For each channel (R, G, B)
            for c = 1:3
                channel = workspace_img(:, :, c);
                % FIX: Properly reshape the robot_mask
                robot_mask = squeeze(robot_masks(r, :, :));
                channel(robot_mask == 1) = robot_colors(r, c) * 0.4; % Make workspace semi-transparent
                workspace_img(:, :, c) = channel;
            end
        end
    end
    
    % Plot workspace image
    imagesc([0, env.width], [0, env.height], workspace_img);
    set(gca, 'YDir', 'normal');
    hold on;
    
    % Plot environment boundary
    rectangle('Position', [0, 0, env.width, env.height], 'EdgeColor', 'k', 'LineWidth', 2);
    
    % Plot robots
    for r = 1:length(robots)
        if isfield(robots, 'failed') && robots(r).failed
            % Failed robot
            plot(robots(r).position(1), robots(r).position(2), 'rx', 'MarkerSize', 12, 'LineWidth', 2);
        else
            % Active robot
            plot(robots(r).position(1), robots(r).position(2), 'o', 'MarkerSize', 10, 'LineWidth', 2, ...
                 'MarkerFaceColor', robot_colors(r, :), 'MarkerEdgeColor', 'k');
        end
        
        % Add robot ID
        text(robots(r).position(1) + 0.15, robots(r).position(2) + 0.15, ...
             sprintf('R%d', r), 'FontSize', 10, 'FontWeight', 'bold');
    end
    
    % Plot tasks
    for t = 1:length(tasks)
        % Determine marker style based on assignment
        if auction_data.assignment(t) == 0
            % Unassigned task
            marker = 's';
            marker_color = [0.7, 0.7, 0.7];
            edge_color = 'k';
        else
            % Assigned task
            marker = 's';
            robot_id = auction_data.assignment(t);
            
            if robot_id > length(robots) || (isfield(robots, 'failed') && robots(robot_id).failed)
                marker_color = [1, 0.3, 0.3];  % Red for failed robot
                edge_color = 'r';
            else
                marker_color = robot_colors(robot_id, :);
                edge_color = 'k';
            end
        end
        
        % Plot task
        plot(tasks(t).position(1), tasks(t).position(2), marker, ...
             'MarkerSize', 8, 'LineWidth', 1, ...
             'MarkerFaceColor', marker_color, 'MarkerEdgeColor', edge_color);
        
        % Add task ID
        text(tasks(t).position(1) + 0.1, tasks(t).position(2) + 0.1, ...
             sprintf('%d', t), 'FontSize', 8);
        
        % Draw line to assigned robot
        if auction_data.assignment(t) > 0
            robot_id = auction_data.assignment(t);
            if robot_id <= length(robots) && ~(isfield(robots, 'failed') && robots(robot_id).failed)
                line([tasks(t).position(1), robots(robot_id).position(1)], ...
                     [tasks(t).position(2), robots(robot_id).position(2)], ...
                     'Color', robot_colors(robot_id, :), 'LineStyle', '--', 'LineWidth', 1);
            end
        end
    end
    
    % Add axis labels and title
    axis equal;
    axis([0, env.width, 0, env.height]);
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    title('Robot Workspaces and Task Assignments');
    
    % Add legend
    legend_handles = zeros(length(robots), 1);
    legend_labels = cell(length(robots), 1);
    
    for r = 1:length(robots)
        if isfield(robots, 'failed') && robots(r).failed
            legend_handles(r) = plot(NaN, NaN, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
            legend_labels{r} = sprintf('Robot %d (Failed)', r);
        else
            legend_handles(r) = plot(NaN, NaN, 'o', 'MarkerSize', 8, 'LineWidth', 1.5, ...
                                    'MarkerFaceColor', robot_colors(r, :), 'MarkerEdgeColor', 'k');
            legend_labels{r} = sprintf('Robot %d Workspace', r);
        end
    end
    
    legend(legend_handles, legend_labels, 'Location', 'eastoutside');
    
    % Add colorbar for distance
    colorbar_axes = axes('Position', [0.92, 0.2, 0.02, 0.6]);
    colormap(colorbar_axes, 'jet');
    c = colorbar(colorbar_axes);
    c.Label.String = 'Distance (m)';
    axis(colorbar_axes, 'off');
    
    fprintf('Robot workspace visualization created.\n');
end

function local_createBidValueHeatmap(auction_data, tasks, robots)
    % CREATEBIDVALUEHEATMAP - Create a heatmap of bid values for each robot-task pair
    %
    % Parameters:
    %   auction_data - Auction data structure
    %   tasks - Array of task structures
    %   robots - Array of robot structures
    
    fprintf('Creating bid value heatmap...\n');
    
    % Create figure
    figure('Name', 'Bid Value Heatmap', 'Position', [100, 100, 1000, 600]);
    
    % Extract bid values
    if isfield(auction_data, 'bids') && ~isempty(auction_data.bids)
        bid_values = auction_data.bids;
    else
        % No bid values available
        fprintf('No bid values available in auction data.\n');
        text(0.5, 0.5, 'No bid values available', 'HorizontalAlignment', 'center', 'FontSize', 14);
        axis off;
        return;
    end
    
    % Create subplot layout
    % 1. Main heatmap
    subplot(1, 2, 1);
    
    % Create heatmap
    imagesc(bid_values);
    
    % Use custom colormap - white to blue for positive values
    colormap(jet);
    colorbar;
    
    % Add labels
    xlabel('Task ID');
    ylabel('Robot ID');
    title('Bid Values');
    
    % Add task names if available
    if isfield(tasks, 'name')
        task_labels = cell(length(tasks), 1);
        for i = 1:length(tasks)
            task_labels{i} = sprintf('%d: %s', i, tasks(i).name);
        end
        xticks(1:length(tasks));
        xticklabels(task_labels);
        xtickangle(45);
    else
        xticks(1:length(tasks));
    end
    
    % Add robot labels
    yticks(1:length(robots));
    
    % Add grid
    grid on;
    
    % 2. Utility heatmap (bids - prices)
    subplot(1, 2, 2);
    
    % Calculate utilities
    if isfield(auction_data, 'utilities') && ~isempty(auction_data.utilities)
        utilities = auction_data.utilities;
    else
        utilities = zeros(size(bid_values));
        for i = 1:size(bid_values, 1)
            for j = 1:size(bid_values, 2)
                utilities(i, j) = bid_values(i, j) - auction_data.prices(j);
            end
        end
    end
    
    % Create heatmap
    imagesc(utilities);
    
    % Use custom colormap - blue-white-red for utilities
    % Blue for negative, white for zero, red for positive
    utility_colormap = [
        linspace(0, 1, 64)', linspace(0, 1, 64)', ones(64, 1);  % Blue to white
        ones(64, 1), linspace(1, 0, 64)', linspace(1, 0, 64)'   % White to red
    ];
    colormap(gca, utility_colormap);
    colorbar;
    
    % Add labels
    xlabel('Task ID');
    ylabel('Robot ID');
    title('Utility Values (Bids - Prices)');
    
    % Add task names if available
    if isfield(tasks, 'name')
        xticks(1:length(tasks));
        xticklabels(task_labels);
        xtickangle(45);
    else
        xticks(1:length(tasks));
    end
    
    % Add robot labels
    yticks(1:length(robots));
    
    % Add grid
    grid on;
    
    % Add current assignments overlay
    hold on;
    for j = 1:length(tasks)
        if auction_data.assignment(j) > 0
            robot_id = auction_data.assignment(j);
            rectangle('Position', [j-0.5, robot_id-0.5, 1, 1], ...
                      'EdgeColor', 'k', 'LineWidth', 2, 'LineStyle', '-');
        end
    end
    
    % Add suptitle
    sgtitle('Bid and Utility Visualization', 'FontSize', 14);
    
    fprintf('Bid value heatmap created.\n');
end

function local_visualizeBiddingHistory(auction_data, tasks)
    % VISUALIZEBIDDINGHISTORY - Create a visualization of the bidding history
    %
    % Parameters:
    %   auction_data - Auction data structure
    %   tasks - Array of task structures
    
    fprintf('Creating bidding history visualization...\n');
    
    % Check if we have the required data
    if ~isfield(auction_data, 'price_history') || ~isfield(auction_data, 'assignment_history')
        fprintf('No bidding history available in auction data.\n');
        figure('Name', 'Bidding History');
        text(0.5, 0.5, 'No bidding history available', 'HorizontalAlignment', 'center', 'FontSize', 14);
        axis off;
        return;
    end
    
    % Get history data
    price_history = auction_data.price_history;
    assignment_history = auction_data.assignment_history;
    
    % Create figure
    figure('Name', 'Bidding History', 'Position', [100, 100, 1200, 800]);
    
    % Calculate number of iterations
    num_iterations = size(price_history, 2);
    
    % FIX: Make sure we don't try to plot more tasks than we have price history for
    num_tasks = min(length(tasks), size(price_history, 1));
    
    % Calculate layout
    subplot_rows = min(5, num_tasks);
    subplot_cols = ceil(num_tasks / subplot_rows);
    
    % Plot tasks
    for i = 1:num_tasks
        subplot(subplot_rows, subplot_cols, i);
        
        % Plot price history for this task
        yyaxis left;
        plot(1:num_iterations, price_history(i, :), 'b-', 'LineWidth', 1.5);
        ylabel('Price');
        
        % Plot assignment history for this task
        yyaxis right;
        stairs(1:num_iterations, assignment_history(i, :), 'r-', 'LineWidth', 1.5);
        ylim([0, max(assignment_history(i, :)) + 1]);
        yticks(0:max(assignment_history(:)));
        ylabel('Robot Assignment');
        
        % Add labels
        xlabel('Iteration');
        if isfield(tasks, 'name') && i <= length(tasks)
            title(sprintf('Task %d: %s', i, tasks(i).name));
        else
            title(sprintf('Task %d', i));
        end
        
        grid on;
    end
    
    % Add overall title
    sgtitle('Bidding History by Task', 'FontSize', 14);
    
    % Create a second figure for composite view
    figure('Name', 'Composite Bidding History', 'Position', [100, 100, 1200, 600]);
    
    % 1. Price evolution for all tasks
    subplot(2, 1, 1);
    
    for i = 1:num_tasks
        plot(1:num_iterations, price_history(i, :), 'LineWidth', 1.5);
        hold on;
    end
    
    xlabel('Iteration');
    ylabel('Price');
    title('Price Evolution for All Tasks');
    grid on;
    
    % Add legend
    if num_tasks <= 10
        legend_entries = cell(num_tasks, 1);
        for i = 1:num_tasks
            if isfield(tasks, 'name') && i <= length(tasks)
                legend_entries{i} = sprintf('Task %d: %s', i, tasks(i).name);
            else
                legend_entries{i} = sprintf('Task %d', i);
            end
        end
        legend(legend_entries, 'Location', 'eastoutside');
    end
    
    % 2. Assignment changes
    subplot(2, 1, 2);
    
    % Calculate number of assignment changes at each iteration
    assignment_changes = zeros(1, num_iterations);
    
    for i = 2:num_iterations
        assignment_changes(i) = sum(assignment_history(:, i) ~= assignment_history(:, i-1));
    end
    
    % Plot assignment changes
    bar(assignment_changes);
    
    xlabel('Iteration');
    ylabel('Number of Assignment Changes');
    title('Assignment Changes per Iteration');
    grid on;
    
    fprintf('Bidding history visualization created.\n');
end

function local_createTaskAssignmentAnimation(auction_data, tasks, robots, env)
    % CREATETASKASSIGNMENTANIMATION - Create an animation of the task assignment process
    %
    % Parameters:
    %   auction_data - Auction data structure
    %   tasks - Array of task structures
    %   robots - Array of robot structures
    %   env - Environment structure
    
    fprintf('Creating task assignment animation...\n');
    
    % Check if we have the required data
    if ~isfield(auction_data, 'assignment_history')
        fprintf('No assignment history available in auction data.\n');
        figure('Name', 'Task Assignment Animation');
        text(0.5, 0.5, 'No assignment history available', 'HorizontalAlignment', 'center', 'FontSize', 14);
        axis off;
        return;
    end
    
    % Get history data
    assignment_history = auction_data.assignment_history;
    
    % Calculate number of iterations
    num_iterations = size(assignment_history, 2);
    
    % Create figure
    fig = figure('Name', 'Task Assignment Animation', 'Position', [100, 100, 800, 700]);
    
    % Create robot colormap
    robot_colors = [
        0.7, 0.7, 0.7;  % Gray for unassigned
        0, 0, 1;        % Blue
        0, 0.8, 0;      % Green
        1, 0, 0;        % Red
        1, 0, 1;        % Magenta
        0, 1, 1;        % Cyan
        0.8, 0.4, 0;    % Orange
        0.5, 0, 0.5;    % Purple
    ];
    
    % Ensure we have enough colors
    if max(assignment_history(:)) + 1 > size(robot_colors, 1)
        robot_colors = [robot_colors; rand(max(assignment_history(:)) + 1 - size(robot_colors, 1), 3)];
    end
    
    % Create animation controls
    play_button = uicontrol('Style', 'pushbutton', 'String', 'Play', ...
                          'Position', [350, 20, 50, 30], ...
                          'Callback', @playAnimation);
    
    slider = uicontrol('Style', 'slider', 'Min', 1, 'Max', num_iterations, ...
                      'Value', 1, 'SliderStep', [1/num_iterations, 10/num_iterations], ...
                      'Position', [150, 20, 200, 20], ...
                      'Callback', @updateFrame);
    
    iteration_text = uicontrol('Style', 'text', 'String', 'Iteration: 1', ...
                             'Position', [50, 20, 100, 20]);
    
    % Create status display
    status_text = uicontrol('Style', 'text', 'String', '', ...
                           'Position', [410, 20, 300, 20], ...
                           'HorizontalAlignment', 'left');
    
    % Initialize animation
    is_playing = false;
    current_iteration = 1;
    
    % Draw initial frame
    drawFrame(current_iteration);
    
    fprintf('Task assignment animation created. Use the controls to play/pause and navigate.\n');
    
    % Nested function to draw a specific frame
    function drawFrame(iteration)
        % Clear axes
        clf;
        
        % Recreate controls
        play_button = uicontrol('Style', 'pushbutton', 'String', 'Play', ...
                              'Position', [350, 20, 50, 30], ...
                              'Callback', @playAnimation);
        
        slider = uicontrol('Style', 'slider', 'Min', 1, 'Max', num_iterations, ...
                          'Value', iteration, 'SliderStep', [1/num_iterations, 10/num_iterations], ...
                          'Position', [150, 20, 200, 20], ...
                          'Callback', @updateFrame);
        
        iteration_text = uicontrol('Style', 'text', ...
                                 'String', sprintf('Iteration: %d', iteration), ...
                                 'Position', [50, 20, 100, 20]);
        
        status_text = uicontrol('Style', 'text', 'String', '', ...
                               'Position', [410, 20, 300, 20], ...
                               'HorizontalAlignment', 'left');
        
        % Get current assignments
        current_assignments = assignment_history(:, iteration);
        
        % Calculate changes from previous iteration
        if iteration > 1
            prev_assignments = assignment_history(:, iteration-1);
            changed_tasks = find(current_assignments ~= prev_assignments);
            
            if ~isempty(changed_tasks)
                status_str = sprintf('Changes: ');
                for i = 1:min(3, length(changed_tasks))
                    task_id = changed_tasks(i);
                    old_robot = prev_assignments(task_id);
                    new_robot = current_assignments(task_id);
                    status_str = [status_str, sprintf('Task %d: R%d→R%d ', task_id, old_robot, new_robot)];
                end
                
                if length(changed_tasks) > 3
                    status_str = [status_str, sprintf('(+%d more)', length(changed_tasks) - 3)];
                end
                
                status_text.String = status_str;
            else
                status_text.String = 'No changes in this iteration';
            end
        end
        
        % Create axes
        axes_handle = axes('Position', [0.1, 0.15, 0.8, 0.8]);
        
        % Draw environment boundary
        rectangle('Position', [0, 0, env.width, env.height], 'EdgeColor', 'k', 'LineWidth', 2);
        hold on;
        
        % Draw robots
        for r = 1:length(robots)
            if isfield(robots, 'failed') && robots(r).failed
                % Failed robot
                plot(robots(r).position(1), robots(r).position(2), 'rx', 'MarkerSize', 12, 'LineWidth', 2);
            else
                % Active robot
                robot_color = robot_colors(r+1, :); % +1 to skip unassigned color
                plot(robots(r).position(1), robots(r).position(2), 'o', 'MarkerSize', 12, 'LineWidth', 2, ...
                     'MarkerFaceColor', robot_color, 'MarkerEdgeColor', 'k');
            end
            
            % Add robot ID
            text(robots(r).position(1) + 0.15, robots(r).position(2) + 0.15, ...
                 sprintf('R%d', r), 'FontSize', 10, 'FontWeight', 'bold');
        end
        
        % Draw tasks
        for t = 1:length(tasks)
            % Get assigned robot (0 for unassigned)
            robot_id = current_assignments(t);
            
            % Set color based on assignment
            if robot_id == 0
                color = robot_colors(1, :);  % Gray for unassigned
            else
                color = robot_colors(robot_id+1, :);  % +1 to skip unassigned color
            end
            
            % Check if this task changed in this iteration
            if iteration > 1 && current_assignments(t) ~= assignment_history(t, iteration-1)
                edge_color = 'r';
                line_width = 2;
                marker_size = 10;
            else
                edge_color = 'k';
                line_width = 1;
                marker_size = 8;
            end
            
            % Plot task
            plot(tasks(t).position(1), tasks(t).position(2), 's', ...
                 'MarkerSize', marker_size, 'LineWidth', line_width, ...
                 'MarkerFaceColor', color, 'MarkerEdgeColor', edge_color);
            
            % Add task ID
            text(tasks(t).position(1) + 0.1, tasks(t).position(2) + 0.1, ...
                 sprintf('%d', t), 'FontSize', 8);
            
            % Draw line to assigned robot
            if robot_id > 0 && robot_id <= length(robots)
                if ~(isfield(robots, 'failed') && robots(robot_id).failed)
                    line([tasks(t).position(1), robots(robot_id).position(1)], ...
                         [tasks(t).position(2), robots(robot_id).position(2)], ...
                         'Color', color, 'LineStyle', '--', 'LineWidth', 1);
                end
            end
        end
        
        % Add axis labels and title
        axis equal;
        axis([0, env.width, 0, env.height]);
        xlabel('X Position (m)');
        ylabel('Y Position (m)');
        title(sprintf('Task Assignment - Iteration %d of %d', iteration, num_iterations));
        
        % Create legend
        legend_handles = [];
        legend_labels = {};
        
        % Unassigned tasks
        h = plot(NaN, NaN, 's', 'MarkerSize', 8, 'LineWidth', 1, ...
                'MarkerFaceColor', robot_colors(1, :), 'MarkerEdgeColor', 'k');
        legend_handles = [legend_handles, h];
        legend_labels{end+1} = 'Unassigned';
        
        % Add robots to legend
        for r = 1:length(robots)
            if isfield(robots, 'failed') && robots(r).failed
                h = plot(NaN, NaN, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
                legend_handles = [legend_handles, h];
                legend_labels{end+1} = sprintf('Robot %d (Failed)', r);
            else
                h = plot(NaN, NaN, 'o', 'MarkerSize', 8, 'LineWidth', 1.5, ...
                        'MarkerFaceColor', robot_colors(r+1, :), 'MarkerEdgeColor', 'k');
                legend_handles = [legend_handles, h];
                legend_labels{end+1} = sprintf('Robot %d', r);
            end
        end
        
        % Add legend for changed tasks
        h = plot(NaN, NaN, 's', 'MarkerSize', 8, 'LineWidth', 2, ...
                'MarkerFaceColor', [0.7, 0.7, 0.7], 'MarkerEdgeColor', 'r');
        legend_handles = [legend_handles, h];
        legend_labels{end+1} = 'Changed in this iteration';
        
        legend(legend_handles, legend_labels, 'Location', 'eastoutside');
        
        % Display workload distribution
        workloads = zeros(1, length(robots));
        for r = 1:length(robots)
            assigned_tasks = find(current_assignments == r);
            for t = 1:length(assigned_tasks)
                task_id = assigned_tasks(t);
                if isfield(tasks, 'execution_time')
                    workloads(r) = workloads(r) + tasks(task_id).execution_time;
                else
                    workloads(r) = workloads(r) + 1;
                end
            end
        end
        
        % Add workload text
        workload_str = 'Workload: ';
        for r = 1:length(robots)
            workload_str = [workload_str, sprintf('R%d=%.1f ', r, workloads(r))];
        end
        
        annotation('textbox', [0.1, 0.025, 0.8, 0.05], 'String', workload_str, ...
                  'EdgeColor', 'none', 'HorizontalAlignment', 'center');
        
        drawnow;
    end
    
    % Callback for the slider
    function updateFrame(src, ~)
        current_iteration = round(src.Value);
        drawFrame(current_iteration);
    end
    
    % Callback for the play button
    function playAnimation(~, ~)
        is_playing = ~is_playing;
        
        if is_playing
            play_button.String = 'Pause';
            
            % Start animation loop
            while is_playing && current_iteration < num_iterations
                current_iteration = current_iteration + 1;
                slider.Value = current_iteration;
                drawFrame(current_iteration);
                pause(0.2);  % Control animation speed
                
                % Check if figure is still open
                if ~isvalid(fig)
                    return;
                end
            end
            
            % Reset button when animation finishes
            play_button.String = 'Play';
            is_playing = false;
        else
            play_button.String = 'Play';
        end
    end
end

function local_visualizeUtilityLandscape(auction_data, tasks, robots)
    % VISUALIZEUTILITYLANDSCAPE - Create a 3D visualization of the utility landscape
    %
    % Parameters:
    %   auction_data - Auction data structure
    %   tasks - Array of task structures
    %   robots - Array of robot structures
    
    fprintf('Creating utility landscape visualization...\n');
    
    % Check if we have the required data
    if ~isfield(auction_data, 'utilities') || ~isfield(auction_data, 'bids')
        fprintf('No utility or bid data available in auction data.\n');
        figure('Name', 'Utility Landscape');
        text(0.5, 0.5, 'No utility or bid data available', 'HorizontalAlignment', 'center', 'FontSize', 14);
        axis off;
        return;
    end
    
    % Get data
    utilities = auction_data.utilities;
    bids = auction_data.bids;
    prices = auction_data.prices;
    
    % Create figure
    figure('Name', 'Utility Landscape', 'Position', [100, 100, 1200, 700]);
    
    % Set up 3D plot
    subplot(1, 2, 1);
    
    % Create meshgrid for robots and tasks
    [X, Y] = meshgrid(1:size(utilities, 2), 1:size(utilities, 1));
    
    % Create 3D surface plot
    surf(X, Y, utilities);
    
    % Set colormap
    colormap('jet');
    colorbar;
    
    % Add labels
    xlabel('Task ID');
    ylabel('Robot ID');
    zlabel('Utility Value');
    title('3D Utility Landscape');
    
    % Set view angle
    view(45, 30);
    
    % Add grid
    grid on;
    
    % Highlight maximum utility for each task
    hold on;
    
    for j = 1:size(utilities, 2)
        [max_util, max_robot] = max(utilities(:, j));
        
        % Plot maximum utility point
        plot3(j, max_robot, max_util, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
        
        % Draw line to the surface
        line([j, j], [max_robot, max_robot], [0, max_util], 'Color', 'r', 'LineStyle', '--');
    end
    
    % Highlight current assignments
    for j = 1:length(tasks)
        if auction_data.assignment(j) > 0
            robot_id = auction_data.assignment(j);
            util_val = utilities(robot_id, j);
            
            % Plot assigned point
            plot3(j, robot_id, util_val, 'g*', 'MarkerSize', 10, 'LineWidth', 2);
        end
    end
    
    % Create legend
    legend('Utility Surface', 'Maximum Utility', 'Current Assignment', 'Location', 'best');
    
    % Create decomposition plot (bids - prices)
    subplot(1, 2, 2);
    
    % Select a subset of tasks if there are many
    if size(utilities, 2) > 10
        selected_tasks = round(linspace(1, size(utilities, 2), 10));
    else
        selected_tasks = 1:size(utilities, 2);
    end
    
    % Plot data for each selected task
    robot_colors = lines(size(utilities, 1));
    
    for j = selected_tasks
        % Create price line
        x_vals = (0:0.1:1.5) * max(max(bids(:, j)));
        y_vals = ones(size(x_vals)) * prices(j);
        
        % Plot price line
        plot(x_vals, y_vals, 'k--', 'LineWidth', 1.5);
        hold on;
        
        % Plot bid points for each robot
        for i = 1:size(utilities, 1)
            bid_val = bids(i, j);
            utility_val = utilities(i, j);
            
            % Skip if no bid
            if bid_val == 0
                continue;
            end
            
            % Plot point
            h = plot(bid_val, prices(j), 'o', 'MarkerSize', 8, 'LineWidth', 1.5, ...
                    'MarkerFaceColor', robot_colors(i, :), 'MarkerEdgeColor', 'k');
            
            % Draw utility line
            line([prices(j), bid_val], [prices(j), prices(j)], 'Color', robot_colors(i, :), 'LineWidth', 1);
            
            % Add utility text
            if utility_val > 0.1
                text(bid_val, prices(j) + 0.05, sprintf('U=%.2f', utility_val), ...
                     'FontSize', 8, 'Color', robot_colors(i, :));
            end
            
            % Highlight if this is the current assignment
            if auction_data.assignment(j) == i
                plot(bid_val, prices(j), 'p', 'MarkerSize', 12, 'LineWidth', 1.5, ...
                     'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'g');
            end
        end
        
        % Add task label
        if isfield(tasks, 'name')
            text(max(x_vals) * 0.8, prices(j) + 0.1, sprintf('Task %d: %s', j, tasks(j).name), ...
                 'FontSize', 8);
        else
            text(max(x_vals) * 0.8, prices(j) + 0.1, sprintf('Task %d', j), 'FontSize', 8);
        end
    end
    
    % Add labels
    xlabel('Bid Value');
    ylabel('Price');
    title('Bid-Price Decomposition');
    grid on;
    
    % Create robot legend
    legend_handles = zeros(size(utilities, 1), 1);
    legend_labels = cell(size(utilities, 1), 1);
    
    for i = 1:size(utilities, 1)
        legend_handles(i) = plot(NaN, NaN, 'o', 'MarkerSize', 8, 'LineWidth', 1.5, ...
                                'MarkerFaceColor', robot_colors(i, :), 'MarkerEdgeColor', 'k');
        hold on;
        
        if isfield(robots, 'name')
            legend_labels{i} = sprintf('Robot %d: %s', i, robots(i).name);
        else
            legend_labels{i} = sprintf('Robot %d', i);
        end
    end
    
    % Add legend
    legend(legend_handles, legend_labels, 'Location', 'best');
    
    % Add annotation to explain utility
    annotation('textbox', [0.65, 0.02, 0.3, 0.05], ...
               'String', 'Utility = Bid - Price', ...
               'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    
    fprintf('Utility landscape visualization created.\n');
end

function local_createPerformanceDashboard(metrics, params, auction_data, tasks, robots)
    % CREATEPERFORMANCEDASHBOARD - Create a comprehensive performance dashboard
    %
    % Parameters:
    %   metrics - Performance metrics structure
    %   params - Algorithm parameters structure
    %   auction_data - Auction data structure
    %   tasks - Array of task structures
    %   robots - Array of robot structures
    
    fprintf('Creating performance dashboard...\n');
    
    % Create figure
    figure('Name', 'Performance Dashboard', 'Position', [50, 50, 1200, 800]);
    
    % 1. Parameter summary
    subplot(3, 3, 1);
    
    % Create parameter text
    param_text = {
        '\bf{Algorithm Parameters}',
        sprintf('Epsilon (ε): %.3f', params.epsilon),
        sprintf('Alpha: [%s]', num2str(params.alpha, '%.1f ')),
        sprintf('Gamma: %.2f', params.gamma),
        sprintf('Lambda: %.2f', params.lambda),
        '',
        '\bf{Communication Parameters}',
        sprintf('Comm delay: %d iterations', params.comm_delay),
        sprintf('Packet loss: %.1f%%', params.packet_loss_prob * 100)
    };
    
    % Add failure parameters if applicable
    if isfield(params, 'failure_time') && ~isinf(params.failure_time)
        param_text{end+1} = '';
        param_text{end+1} = '\bf{Failure Scenario}';
        param_text{end+1} = sprintf('Failure time: %d iterations', params.failure_time);
        param_text{end+1} = sprintf('Failed robot: %d', params.failed_robot);
    end
    
    % Display parameters
    text(0.05, 0.95, param_text, 'VerticalAlignment', 'top', 'FontSize', 9);
    axis off;
    title('Algorithm Configuration');
    
    % 2. Key performance metrics
    subplot(3, 3, 2);
    
    % Extract key metrics
    if isfield(metrics, 'iterations')
        iterations = metrics.iterations;
    else
        iterations = NaN;
    end
    
    if isfield(metrics, 'messages')
        messages = metrics.messages;
    else
        messages = NaN;
    end
    
    if isfield(metrics, 'makespan')
        makespan = metrics.makespan;
    else
        makespan = NaN;
    end
    
    if isfield(metrics, 'optimal_makespan')
        optimal_makespan = metrics.optimal_makespan;
    else
        optimal_makespan = NaN;
    end
    
    if isfield(metrics, 'optimality_gap')
        optimality_gap = metrics.optimality_gap;
    else
        optimality_gap = NaN;
    end
    
    if isfield(metrics, 'theoretical_gap_bound')
        theoretical_gap_bound = metrics.theoretical_gap_bound;
    else
        theoretical_gap_bound = 2 * params.epsilon;
    end
    
    % Create metrics text
    metrics_text = {
        '\bf{Convergence Performance}',
        sprintf('Iterations: %d', iterations),
        sprintf('Messages: %d', messages),
        sprintf('Messages per task: %.1f', messages / length(tasks)),
        '',
        '\bf{Solution Quality}',
        sprintf('Makespan: %.2f', makespan),
        sprintf('Optimal makespan: %.2f', optimal_makespan),
        sprintf('Optimality gap: %.2f (%.1f%%)', optimality_gap, optimality_gap / optimal_makespan * 100),
        sprintf('Theoretical bound (2ε): %.2f', theoretical_gap_bound)
    };
    
    % Add recovery metrics if applicable
    if isfield(metrics, 'recovery_time') && metrics.recovery_time > 0
        metrics_text{end+1} = '';
        metrics_text{end+1} = '\bf{Recovery Performance}';
        metrics_text{end+1} = sprintf('Recovery time: %d iterations', metrics.recovery_time);
        
        if isfield(metrics, 'theoretical_recovery_bound')
            metrics_text{end+1} = sprintf('Theoretical bound: %.1f iterations', metrics.theoretical_recovery_bound);
        end
        
        if isfield(metrics, 'failed_task_count')
            metrics_text{end+1} = sprintf('Failed tasks: %d', metrics.failed_task_count);
        end
    end
    
    % Display metrics
    text(0.05, 0.95, metrics_text, 'VerticalAlignment', 'top', 'FontSize', 9);
    axis off;
    title('Performance Metrics');
    
    % 3. Convergence history
    subplot(3, 3, 3);
    
    if isfield(metrics, 'convergence_history') && ~isempty(metrics.convergence_history)
        plot(metrics.convergence_history, 'LineWidth', 1.5);
        xlabel('Iteration');
        ylabel('Assignment Changes');
        title('Convergence History');
        grid on;
        
        % Add failure time line if applicable
        if isfield(metrics, 'failure_time') && metrics.failure_time > 0
            hold on;
            xline(metrics.failure_time, 'r--', 'LineWidth', 1.5);
            
            if isfield(metrics, 'recovery_time') && metrics.recovery_time > 0
                xline(metrics.failure_time + metrics.recovery_time, 'g--', 'LineWidth', 1.5);
                legend('Changes', 'Failure', 'Recovery Complete', 'Location', 'best');
            else
                legend('Changes', 'Failure', 'Location', 'best');
            end
        end
    else
        text(0.5, 0.5, 'Convergence history not available', 'HorizontalAlignment', 'center');
        axis off;
    end
    
    % 4. Task allocation visualization
    subplot(3, 3, [4, 7]);
    
    % Calculate robot assignments
    task_counts = zeros(1, length(robots));
    for i = 1:length(robots)
        task_counts(i) = sum(auction_data.assignment == i);
    end
    
    % Calculate workloads
    workloads = zeros(1, length(robots));
    for i = 1:length(robots)
        assigned_tasks = find(auction_data.assignment == i);
        for j = 1:length(assigned_tasks)
            task_id = assigned_tasks(j);
            if isfield(tasks, 'execution_time')
                workloads(i) = workloads(i) + tasks(task_id).execution_time;
            else
                workloads(i) = workloads(i) + 1;
            end
        end
    end
    
    % Create bar plot
    bar_data = [task_counts; workloads]';
    bar(bar_data);
    
    % Add labels
    xlabel('Robot ID');
    ylabel('Count / Workload');
    title('Task Allocation');
    legend('Task Count', 'Workload', 'Location', 'best');
    grid on;
    
    % Add robot labels
    xticks(1:length(robots));
    
    % Calculate workload imbalance
    active_robots = find(~[robots.failed]);
    if ~isempty(active_robots) && length(active_robots) > 1
        active_workloads = workloads(active_robots);
        workload_min = min(active_workloads);
        workload_max = max(active_workloads);
        workload_mean = mean(active_workloads);
        workload_imbalance = (workload_max - workload_min) / workload_max * 100;
        workload_cv = std(active_workloads) / workload_mean * 100;  % Coefficient of variation
        
        % Add imbalance annotation
        imbalance_text = sprintf('Workload imbalance: %.1f%%\nCoefficient of variation: %.1f%%', ...
                                workload_imbalance, workload_cv);
        annotation('textbox', [0.2, 0.33, 0.2, 0.05], 'String', imbalance_text, ...
                   'EdgeColor', 'none', 'FontSize', 9);
    end
    
    % 5. Price evolution
    subplot(3, 3, [5, 8]);
    
    if isfield(metrics, 'price_history') && ~isempty(metrics.price_history)
        % Plot price history for all tasks
        price_history = metrics.price_history;
        
        for i = 1:size(price_history, 1)
            plot(price_history(i, :), 'LineWidth', 1.5);
            hold on;
        end
        
        xlabel('Iteration');
        ylabel('Price');
        title('Price Evolution');
        grid on;
        
        % Add failure time line if applicable
        if isfield(metrics, 'failure_time') && metrics.failure_time > 0
            xline(metrics.failure_time, 'r--', 'LineWidth', 1.5);
            
            if isfield(metrics, 'recovery_time') && metrics.recovery_time > 0
                xline(metrics.failure_time + metrics.recovery_time, 'g--', 'LineWidth', 1.5);
            end
        end
    else
        text(0.5, 0.5, 'Price history not available', 'HorizontalAlignment', 'center');
        axis off;
    end
    
    % 6. Task oscillations
    subplot(3, 3, 6);
    
    if isfield(auction_data, 'task_oscillation_count') && ~isempty(auction_data.task_oscillation_count)
        bar(auction_data.task_oscillation_count);
        xlabel('Task ID');
        ylabel('Oscillation Count');
        title('Task Reassignment Frequency');
        grid on;
        
        % Add high oscillation threshold
        if max(auction_data.task_oscillation_count) > 3
            hold on;
            yline(3, 'r--', 'LineWidth', 1.5);
            legend('Oscillations', 'Threshold', 'Location', 'best');
        end
    else
        text(0.5, 0.5, 'Task oscillation data not available', 'HorizontalAlignment', 'center');
        axis off;
    end
    
    % 7. Theoretical bounds
    subplot(3, 3, 9);
    
    % Prepare bounds data
    num_tasks = length(tasks);
    
    % Calculate theoretical complexity bound
    b_max = max(params.alpha);
    epsilon = params.epsilon;
    theoretical_bound = num_tasks^2 * b_max / epsilon;
    
    % Calculate theoretical gap bound
    theoretical_gap = 2 * epsilon;
    
    % Create theoretical bound plot
    bounds_data = [iterations / theoretical_bound, optimality_gap / theoretical_gap];
    bar(bounds_data);
    
    % Add labels
    xticklabels({'Convergence Ratio', 'Optimality Ratio'});
    ylabel('Actual / Theoretical');
    title('Theoretical Bound Tightness');
    grid on;
    
    % Add threshold line
    hold on;
    yline(1, 'r--', 'LineWidth', 1.5);
    
    % Add annotation with formulas
    formula_text = {
        sprintf('Theoretical convergence: O(K² · bₘₐₓ/ε) = O(%d² · %.2f/%.2f) ≈ %.1f', ...
                num_tasks, b_max, epsilon, theoretical_bound),
        sprintf('Theoretical optimality gap: 2ε = 2 × %.2f = %.2f', epsilon, theoretical_gap)
    };
    
    annotation('textbox', [0.6, 0.05, 0.35, 0.05], 'String', formula_text, ...
               'EdgeColor', 'none', 'FontSize', 9);
    
    % Add overall title
    sgtitle('Performance Dashboard', 'FontSize', 16);
    
    fprintf('Performance dashboard created.\n');
end