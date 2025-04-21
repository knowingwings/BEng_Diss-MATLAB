function utils = environment_utils()
    % ENVIRONMENT_UTILS - Returns function handles for environment-related functions
    % 
    % This file contains utility functions for creating and visualizing the
    % environment for the distributed auction algorithm verification model.
    %
    % Returns:
    %   utils - Structure containing the following function handles:
    %           createEnvironment     - Creates a rectangular environment
    %           visualizeEnvironment  - Visualizes the environment and tasks
    %           arrow                 - Draws an arrow
    %           plotTaskPrices        - Plots task prices over iterations
    %           plotAssignments       - Plots task assignments over iterations
    %           plotConvergence       - Plots convergence metric
    %           plotWorkload          - Plots workload distribution
    
    % Return a struct of function handles
    utils = struct(...
        'createEnvironment', @createEnvironment, ...
        'visualizeEnvironment', @visualizeEnvironment, ...
        'arrow', @arrow, ...
        'plotTaskPrices', @plotTaskPrices, ...
        'plotAssignments', @plotAssignments, ...
        'plotConvergence', @plotConvergence, ...
        'plotWorkload', @plotWorkload ...
    );
    
    % Define nested functions
    function env = createEnvironment(width, height)
        % CREATEENVIRONMENT Creates a rectangular environment for the simulation
        %
        % Parameters:
        %   width  - Width of the environment in meters
        %   height - Height of the environment in meters
        %
        % Returns:
        %   env    - Environment structure with the following fields:
        %            width    - Width of the environment
        %            height   - Height of the environment
        %            obstacles - List of obstacles (empty in basic configuration)
        
        env.width = width;
        env.height = height;
        env.obstacles = [];  % Can add obstacles if needed
    end
    
    function visualizeEnvironment(env, robots, tasks, auction_data)
        % VISUALIZEENVIRONMENT Visualizes the environment, robots, tasks, and current assignments
        %
        % Parameters:
        %   env          - Environment structure
        %   robots       - Array of robot structures
        %   tasks        - Array of task structures
        %   auction_data - Auction algorithm data structure
        
        % Clear current axes
        cla;
        
        % Plot environment boundary
        rectangle('Position', [0, 0, env.width, env.height], 'EdgeColor', 'k', 'LineWidth', 2);
        hold on;
        
        % Plot robots
        for i = 1:length(robots)
            if robots(i).failed
                color = 'r';  % Red for failed robots
                marker = 'x';
            else
                color = 'b';  % Blue for active robots
                marker = 'o';
            end
            plot(robots(i).position(1), robots(i).position(2), marker, 'Color', color, 'MarkerSize', 12, 'LineWidth', 2);
            text(robots(i).position(1) + 0.1, robots(i).position(2) + 0.1, sprintf('R%d', i), 'FontSize', 12);
        end
        
        % Plot tasks
        for i = 1:length(tasks)
            % Different color based on assignment
            if auction_data.assignment(i) == 0
                color = 'k';  % Black for unassigned
            elseif auction_data.completion_status(i) == 1
                color = 'g';  % Green for completed
            else
                colors = 'bmcry';  % Different colors for different robots
                color = colors(mod(auction_data.assignment(i)-1, length(colors))+1);
            end
            
            plot(tasks(i).position(1), tasks(i).position(2), 's', 'Color', color, 'MarkerSize', 8, 'LineWidth', 2);
            text(tasks(i).position(1) + 0.1, tasks(i).position(2) + 0.1, sprintf('T%d (%.1f)', i, tasks(i).execution_time), 'FontSize', 10);
            
            % Draw lines for assignments
            if auction_data.assignment(i) > 0 && auction_data.assignment(i) <= length(robots) && ~robots(auction_data.assignment(i)).failed
                robot_pos = robots(auction_data.assignment(i)).position;
                line([robot_pos(1), tasks(i).position(1)], [robot_pos(2), tasks(i).position(2)], ...
                     'Color', color, 'LineStyle', '--', 'LineWidth', 1.5);
            end
        end
        
        % Draw task dependencies as directed arrows (using modified arrow function)
        for i = 1:length(tasks)
            for j = tasks(i).prerequisites
                % Instead of using the arrow function, draw a simple line with an arrow head
                p0 = [tasks(j).position(1), tasks(j).position(2)];
                p1 = [tasks(i).position(1), tasks(i).position(2)];
                
                % Calculate vector for arrow direction
                dp = p1 - p0;
                dp_length = norm(dp);
                
                % Make arrow shorter to avoid overlapping with markers
                if dp_length > 0.3
                    p1 = p0 + 0.9 * dp;
                end
                
                % Draw the arrow with a light gray color
                arrow_color = [0.6, 0.6, 0.6];
                quiver(p0(1), p0(2), p1(1)-p0(1), p1(2)-p0(2), 0, 'Color', arrow_color, 'LineWidth', 1);
            end
        end
        
        % Set axis properties
        axis([0, env.width, 0, env.height]);
        axis equal;
        grid on;
        xlabel('X (m)');
        ylabel('Y (m)');
        
        hold off;
    end
    
    function h = arrow(p0, p1, varargin)
        % ARROW Draw an arrow from point p0 to point p1
        %
        % Parameters:
        %   p0      - Starting point [x, y]
        %   p1      - Ending point [x, y]
        %   varargin - Additional arguments for the quiver function
        %
        % Returns:
        %   h       - Handle to the arrow object
        
        dp = p1 - p0;
        length = norm(dp);
        
        % Make arrow shorter to avoid overlapping with markers
        if length > 0.3
            p1 = p0 + 0.9 * dp;
        end
        
        % Note: quiver doesn't support EdgeColor - use Color instead
        % Convert any EdgeColor parameter to Color
        varargin_modified = varargin;
        for i = 1:2:length(varargin)
            if strcmpi(varargin{i}, 'EdgeColor') || strcmpi(varargin{i}, 'FaceColor')
                % Find the value associated with EdgeColor/FaceColor
                if i+1 <= length(varargin)
                    color_value = varargin{i+1};
                    % Replace with Color parameter
                    varargin_modified{i} = 'Color';
                    varargin_modified{i+1} = color_value;
                end
            end
        end
        
        % Draw the arrow with the modified parameters
        h = quiver(p0(1), p0(2), p1(1)-p0(1), p1(2)-p0(2), 0, varargin_modified{:});
    end
    
    % Plotting utilities
    function plotTaskPrices(price_history)
        % PLOTTASKPRICES Plot task prices over iterations
        %
        % Parameters:
        %   price_history - Matrix of task prices over iterations
        
        % Clear current axes
        cla;
        
        % Plot price for each task
        for i = 1:size(price_history, 1)
            plot(price_history(i, :), 'LineWidth', 1.5);
            hold on;
        end
        
        % Set labels and grid
        xlabel('Iteration');
        ylabel('Price');
        grid on;
        legend(arrayfun(@(i) sprintf('Task %d', i), 1:size(price_history, 1), 'UniformOutput', false), ...
               'Location', 'eastoutside');
        
        hold off;
    end
    
    function plotAssignments(assignment_history, num_robots)
        % PLOTASSIGNMENTS Plot task assignments over iterations
        %
        % Parameters:
        %   assignment_history - Matrix of task assignments over iterations
        %   num_robots         - Number of robots
        
        % Clear current axes
        cla;
        
        % Create colormap
        cmap = [0.8, 0.8, 0.8; jet(num_robots)];  % Gray for unassigned, then colors for robots
        
        % Plot assignment for each task
        for i = 1:size(assignment_history, 1)
            % Add 1 to assignments so that unassigned (0) maps to color 1
            plot(assignment_history(i, :) + 1, 'LineWidth', 1.5);
            hold on;
        end
        
        % Set labels and grid
        xlabel('Iteration');
        ylabel('Assigned Robot');
        grid on;
        ylim([0.5, num_robots+1.5]);
        yticks(1:num_robots+1);
        yticklabels(['Unassigned', arrayfun(@(i) sprintf('Robot %d', i), 1:num_robots, 'UniformOutput', false)]);
        legend(arrayfun(@(i) sprintf('Task %d', i), 1:size(assignment_history, 1), 'UniformOutput', false), ...
               'Location', 'eastoutside');
        
        hold off;
    end
    
    function plotConvergence(convergence_history)
        % PLOTCONVERGENCE Plot convergence metric over iterations
        %
        % Parameters:
        %   convergence_history - Vector of convergence metrics over iterations
        
        % Clear current axes
        cla;
        
        % Plot convergence metric
        plot(convergence_history, 'LineWidth', 2);
        
        % Set labels and grid
        xlabel('Iteration');
        ylabel('Number of Changes');
        grid on;
        
        hold off;
    end
    
    function plotWorkload(final_assignment, tasks, robots)
        % PLOTWORKLOAD Plot final workload distribution
        %
        % Parameters:
        %   final_assignment - Vector of final task assignments
        %   tasks           - Array of task structures
        %   robots          - Array of robot structures
        
        % Clear current axes
        cla;
        
        % Calculate workload for each robot
        workloads = zeros(1, length(robots));
        for i = 1:length(final_assignment)
            if final_assignment(i) > 0
                workloads(final_assignment(i)) = workloads(final_assignment(i)) + tasks(i).execution_time;
            end
        end
        
        % Plot workload for each robot
        bar(workloads);
        
        % Set labels and grid
        xlabel('Robot');
        ylabel('Total Execution Time');
        grid on;
        xticks(1:length(robots));
        
        hold off;
    end
    
    end % End of main function