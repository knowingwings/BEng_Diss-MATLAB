function visualise_simulation_results(robot_states, tasks, task_assignments, task_prices, ...
    task_start_times, task_finish_times, messages_sent, metrics)
% VISUALIsE_SIMULATION_RESULTS Creates comprehensive visualisations of simulation results
%
% Inputs:
%   robot_states      - Array of robot state structures
%   tasks             - Matrix of task information
%   task_assignments  - Final task assignments
%   task_prices       - Final task prices
%   task_start_times  - Start times for each task
%   task_finish_times - Finish times for each task
%   messages_sent     - Number of messages sent over time
%   metrics           - Performance metrics structure
%
% This function creates multiple visualisations to analyse simulation results:
% 1. Task assignment and dependency visualisation
% 2. Gantt chart of task execution
% 3. Task price evolution
% 4. Performance metrics radar chart
% 5. Communication analysis

% Create figure
figure('Name', 'Distributed Auction Simulation Results', 'Position', [100, 100, 1200, 800]);

% 1. Task Assignment Visualisation
subplot(2, 3, 1);
hold on;

% Plot robots
colors = {'b', 'r'};
markers = {'o', 's'};
for i = 1:length(robot_states)
plot(robot_states(i).position(1), robot_states(i).position(2), markers{i}, 'MarkerSize', 10, ...
'MarkerFaceColor', colors{i}, 'DisplayName', ['Robot ', num2str(i)]);
end

% Plot tasks and assignments
task_colors = {'m', 'c', 'g', 'y', 'k'};
for i = 1:size(tasks, 1)
task_color = task_colors{mod(i-1, length(task_colors))+1};

% Determine marker based on collaborative flag
if tasks(i, 11) > 0  % Collaborative task
marker = 'd';  % Diamond for collaborative tasks
else
marker = 'p';  % Pentagon for standard tasks
end

% Plot task
plot(tasks(i, 1), tasks(i, 2), marker, 'MarkerSize', 8, ...
'MarkerFaceColor', task_color, 'DisplayName', ['Task ', num2str(i)]);

% Add task number label
text(tasks(i, 1)+0.2, tasks(i, 2)+0.2, num2str(i), 'FontSize', 10);

% Show assignment
robot_id = task_assignments(i);
if robot_id > 0 && robot_id <= 2  % Standard assignment to one robot
line([robot_states(robot_id).position(1), tasks(i, 1)], ...
[robot_states(robot_id).position(2), tasks(i, 2)], ...
'Color', colors{robot_id}, 'LineStyle', '--');
elseif robot_id == 3  % Collaborative task (both robots)
for r = 1:2
line([robot_states(r).position(1), tasks(i, 1)], ...
[robot_states(r).position(2), tasks(i, 2)], ...
'Color', colors{r}, 'LineStyle', ':');
end
end
end

% Plot task dependencies
for i = 1:size(tasks, 1)
dependencies = tasks(i, 8:10);
dependencies = dependencies(dependencies > 0);

for j = 1:length(dependencies)
dep_id = dependencies(j);
arrow_x = [tasks(dep_id, 1), tasks(i, 1)];
arrow_y = [tasks(dep_id, 2), tasks(i, 2)];
% Use quiver for arrows if arrow function is not available
quiver(arrow_x(1), arrow_y(1), arrow_x(2)-arrow_x(1), arrow_y(2)-arrow_y(1), 0, ...
'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'MaxHeadSize', 0.5);
end
end

title('Task Assignments and Dependencies');
xlabel('X Position');
ylabel('Y Position');
legend('show', 'Location', 'eastoutside');
grid on;
axis equal;
hold off;

% 2. Gantt Chart
subplot(2, 3, [2 3]);
hold on;

% Create a Gantt chart of task executions
% Sort tasks by start time
valid_tasks = find(task_start_times > 0);
[sorted_times, idx] = sort(task_start_times(valid_tasks));
sorted_tasks = valid_tasks(idx);

% Plot bars
for i = 1:length(sorted_tasks)
task_id = sorted_tasks(i);
if task_finish_times(task_id) > 0
% Determine color based on assignment
if task_assignments(task_id) <= 2
bar_color = colors{task_assignments(task_id)};
else  % Collaborative task
bar_color = [0.5 0 0.5];  % Purple for collaborative tasks
end

% Plot bar for task execution
duration = task_finish_times(task_id) - task_start_times(task_id);
h = barh(task_id, duration, 0.5);
set(h, 'FaceColor', bar_color, 'EdgeColor', 'none');

% Set bar position to start time
x = get(h, 'XData');
set(h, 'XData', x + task_start_times(task_id) - min(x));

% Add task label
text(task_start_times(task_id) + duration/2, task_id, ['Task ', num2str(task_id)], ...
'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
end
end

title('Task Execution Timeline (Gantt Chart)');
xlabel('Time');
ylabel('Task ID');
grid on;
hold off;

% 3. Task Price Evolution
subplot(2, 3, 4);

% Plot the price evolution
bar(task_prices, 'FaceColor', [0.3 0.6 0.9]);
hold on;
% Add task assignment labels
for i = 1:length(task_prices)
if task_assignments(i) > 0
if task_assignments(i) <= 2
text(i, task_prices(i) + 0.05*max(task_prices), ['R', num2str(task_assignments(i))], ...
'HorizontalAlignment', 'center', 'Color', colors{task_assignments(i)}, 'FontWeight', 'bold');
else
text(i, task_prices(i) + 0.05*max(task_prices), 'R1+R2', ...
'HorizontalAlignment', 'center', 'Color', [0.5 0 0.5], 'FontWeight', 'bold');
end
end
end
hold off;

title('Final Task Prices and Assignments');
xlabel('Task ID');
ylabel('Price');
grid on;

% 4. Metrics Summary
subplot(2, 3, 5);
hold on;

% Create a radar chart of key metrics
metrics_labels = {'Makespan', 'Utilisation', 'Load Balance', 'Msg/Task', 'CP Utilisation', 'Waiting Time'};
metrics_values = [metrics.makespan/100, mean(metrics.robot_utilisation), ...
metrics.load_balance, metrics.messages_per_task/10, ...
metrics.critical_path_utilisation, metrics.avg_waiting_time/10];

% Normalise values to 0-1 range for radar chart
metrics_values = metrics_values / max(metrics_values);

% Plot radar chart
angles = linspace(0, 2*pi, length(metrics_labels)+1);
angles = angles(1:end-1);

% Create circular grid
for grid_val = 0.2:0.2:1
plot(grid_val*sin(linspace(0, 2*pi, 100)), grid_val*cos(linspace(0, 2*pi, 100)), 'Color', [0.8 0.8 0.8]);
end

% Plot spokes and labels
for i = 1:length(angles)
plot([0, sin(angles(i))], [0, cos(angles(i))], 'Color', [0.8 0.8 0.8]);
text(1.1*sin(angles(i)), 1.1*cos(angles(i)), metrics_labels{i}, 'HorizontalAlignment', 'center');
end

% Plot metrics
x_vals = [metrics_values.*sin(angles), metrics_values(1)*sin(angles(1))];
y_vals = [metrics_values.*cos(angles), metrics_values(1)*cos(angles(1))];
plot(x_vals, y_vals, 'r-', 'LineWidth', 2);

% Fill radar chart
patch(x_vals, y_vals, 'r', 'FaceAlpha', 0.3);

title('Performance Metrics');
axis equal;
axis off;
hold off;

% 5. Communication Analysis
subplot(2, 3, 6);

% Create a bar chart of messages sent over time
if ~isempty(messages_sent) && any(messages_sent > 0)
bar(messages_sent);
title('Communication Overhead (Messages)');
xlabel('Time Step');
ylabel('Messages Sent');
grid on;
else
% If no message data, show metrics summary table
metrics_table = {'Metric', 'Value';
'Makespan', num2str(metrics.makespan, '%.2f');
'Avg. Utilisation', num2str(mean(metrics.robot_utilisation)*100, '%.1f%%');
'Load Balance', num2str(metrics.load_balance*100, '%.1f%%');
'Total Messages', num2str(metrics.total_messages);
'Messages/Task', num2str(metrics.messages_per_task, '%.2f');
'CP Utilisation', num2str(metrics.critical_path_utilisation*100, '%.1f%%');
'Avg. Waiting Time', num2str(metrics.avg_waiting_time, '%.2f')};

% Create table visual
axis off;
title('Performance Metrics Summary');

% Display table data
for i = 1:size(metrics_table, 1)
if i == 1
text(0.1, 1-i*0.14, metrics_table{i,1}, 'FontWeight', 'bold');
text(0.6, 1-i*0.14, metrics_table{i,2}, 'FontWeight', 'bold');
else
text(0.1, 1-i*0.14, metrics_table{i,1});
text(0.6, 1-i*0.14, metrics_table{i,2});
end
end
end

% Create separate figure for critical path analysis
figure('Name', 'Critical Path Analysis', 'Position', [150, 150, 800, 400]);

% Get critical path information
[critical_path, task_priorities] = analyse_task_dependencies(tasks, task_finish_times - task_start_times);

% Create directed graph visualisation
subplot(1, 2, 1);
hold on;

% Create connectivity matrix
adj_matrix = zeros(size(tasks, 1));
for i = 1:size(tasks, 1)
dependencies = tasks(i, 8:10);
dependencies = dependencies(dependencies > 0);

for j = 1:length(dependencies)
adj_matrix(dependencies(j), i) = 1;
end
end

% Plot directed graph
[x, y] = gplot(adj_matrix, [tasks(:,1), tasks(:,2)]);
plot(x, y, 'k-', 'LineWidth', 1);

% Plot nodes
for i = 1:size(tasks, 1)
if ismember(i, critical_path)
% Critical path tasks
scatter(tasks(i,1), tasks(i,2), 100, 'r', 'filled', 'o');
else
% Regular tasks
scatter(tasks(i,1), tasks(i,2), 60, 'b', 'filled', 'o');
end
text(tasks(i,1)+0.2, tasks(i,2), num2str(i), 'FontSize', 10);
end

title('Task Dependency Graph (Critical Path in Red)');
xlabel('X Position');
ylabel('Y Position');
axis equal;
grid on;
hold off;

% Plot task priorities
subplot(1, 2, 2);
bar(task_priorities);
hold on;
% Highlight critical path tasks
for i = 1:length(critical_path)
bar(critical_path(i), task_priorities(critical_path(i)), 'r');
end
hold off;
title('Task Priorities (Critical Path in Red)');
xlabel('Task ID');
ylabel('Priority Value (Lower = Higher Priority)');
grid on;

end