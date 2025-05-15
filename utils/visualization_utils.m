function utils = visualization_utils()
    % VISUALIZATION_UTILS - Utilities for visualizing results
    
    % Create utils structure
    utils = struct();
    
    % Add function handles
    utils.plotConvergence = @plotConvergence;
    utils.plotAssignments = @plotAssignments;
    utils.plotPrices = @plotPrices;
    utils.plotWorkload = @plotWorkload;
    utils.plotMakespan = @plotMakespan;
    utils.plotFactorEffects = @plotFactorEffects;
    utils.plotRecovery = @plotRecovery;
    utils.plotHeatmap = @plotHeatmap;
    utils.plotParameterTrajectory = @plotParameterTrajectory;
    utils.saveAllFigures = @saveAllFigures;
    
    function h = plotConvergence(convergence_history, failure_time, recovery_time)
        % Plot convergence metric over iterations
        h = figure('Name', 'Convergence History', 'Position', [100, 100, 800, 600]);
        
        plot(convergence_history, 'LineWidth', 2);
        title('Convergence History');
        xlabel('Iteration');
        ylabel('Number of Changes');
        grid on;
        
        % Add vertical lines for failure and recovery if provided
        if nargin >= 2 && ~isempty(failure_time) && failure_time < Inf
            hold on;
            xline(failure_time, 'r--', 'LineWidth', 2);
            
            if nargin >= 3 && ~isempty(recovery_time) && recovery_time > 0
                xline(failure_time + recovery_time, 'g--', 'LineWidth', 2);
                legend('Changes', 'Failure', 'Recovery Complete');
            else
                legend('Changes', 'Failure');
            end
            
            hold off;
        end
    end
    
    function h = plotAssignments(assignment_history, collaborative_tasks)
        % Plot task assignments over time
        h = figure('Name', 'Task Assignments', 'Position', [100, 100, 1000, 600]);
        
        % Plot assignment heatmap
        imagesc(assignment_history);
        colormap(jet);
        colorbar;
        
        title('Task Assignments Over Time');
        xlabel('Iteration');
        ylabel('Task');
        
        % Highlight collaborative tasks if provided
        if nargin >= 2 && ~isempty(collaborative_tasks)
            hold on;
            for i = 1:length(collaborative_tasks)
                if collaborative_tasks(i)
                    % Add a marker to indicate collaborative tasks
                    plot(zeros(1, size(assignment_history, 2)), i * ones(1, size(assignment_history, 2)), 'w*');
                end
            end
            hold off;
        end
        
        grid on;
    end
    
    function h = plotPrices(price_history)
        % Plot task prices over time
        h = figure('Name', 'Task Prices', 'Position', [100, 100, 800, 600]);
        
        plot(price_history', 'LineWidth', 1.5);
        title('Task Prices Over Time');
        xlabel('Iteration');
        ylabel('Price');
        
        % Add legend with task numbers
        legend_str = cell(size(price_history, 1), 1);
        for i = 1:size(price_history, 1)
            legend_str{i} = sprintf('Task %d', i);
        end
        legend(legend_str, 'Location', 'eastoutside');
        
        grid on;
    end
    
    function h = plotWorkload(robot_loads, robot_labels)
        % Plot workload distribution
        h = figure('Name', 'Workload Distribution', 'Position', [100, 100, 600, 400]);
        
        bar(robot_loads);
        title('Robot Workload Distribution');
        xlabel('Robot');
        ylabel('Total Execution Time');
        
        if nargin >= 2 && ~isempty(robot_labels)
            xticks(1:length(robot_loads));
            xticklabels(robot_labels);
        end
        
        grid on;
    end
    
    function h = plotMakespan(task_counts, makespans, optimal_makespans, factor_name)
        % Plot makespan vs. a factor
        h = figure('Name', 'Makespan Analysis', 'Position', [100, 100, 800, 600]);
        
        plot(task_counts, makespans, 'o-', 'LineWidth', 2);
        hold on;
        
        if nargin >= 3 && ~isempty(optimal_makespans)
            plot(task_counts, optimal_makespans, 's--', 'LineWidth', 2);
            legend('Achieved Makespan', 'Optimal Makespan');
        end
        
        if nargin >= 4 && ~isempty(factor_name)
            xlabel(factor_name);
        else
            xlabel('Number of Tasks');
        end
        
        ylabel('Makespan');
        title('Makespan Analysis');
        grid on;
        
        hold off;
    end
    
    function h = plotFactorEffects(factor_values, response_means, response_stds, factor_name, response_name)
        % Plot effect of a factor on a response variable
        h = figure('Name', 'Factor Effects', 'Position', [100, 100, 800, 600]);
        
        errorbar(factor_values, response_means, response_stds, 'o-', 'LineWidth', 2);
        
        if nargin >= 4 && ~isempty(factor_name)
            xlabel(factor_name);
        else
            xlabel('Factor Value');
        end
        
        if nargin >= 5 && ~isempty(response_name)
            ylabel(response_name);
        else
            ylabel('Response');
        end
        
        title('Factor Effect Analysis');
        grid on;
    end
    
    function h = plotRecovery(task_counts, recovery_times, theoretical_bounds)
        % Plot recovery times and theoretical bounds
        h = figure('Name', 'Recovery Analysis', 'Position', [100, 100, 800, 600]);
        
        plot(task_counts, recovery_times, 'o-', 'LineWidth', 2);
        hold on;
        
        if nargin >= 3 && ~isempty(theoretical_bounds)
            plot(task_counts, theoretical_bounds, 's--', 'LineWidth', 2);
            legend('Actual Recovery Time', 'Theoretical Bound');
        end
        
        xlabel('Number of Tasks');
        ylabel('Recovery Time (iterations)');
        title('Recovery Time Analysis');
        grid on;
        
        hold off;
    end
    
    function h = plotHeatmap(data, row_labels, col_labels, title_text)
        % Plot heatmap
        h = figure('Name', 'Heatmap', 'Position', [100, 100, 800, 600]);
        
        imagesc(data);
        colormap(jet);
        colorbar;
        
        if nargin >= 2 && ~isempty(row_labels)
            yticks(1:length(row_labels));
            yticklabels(row_labels);
        end
        
        if nargin >= 3 && ~isempty(col_labels)
            xticks(1:length(col_labels));
            xticklabels(col_labels);
        end
        
        if nargin >= 4 && ~isempty(title_text)
            title(title_text);
        else
            title('Heatmap');
        end
        
        grid on;
    end
    
    function h = plotParameterTrajectory(parameter_history, parameter_names)
        % Plot parameter trajectory during optimization
        h = figure('Name', 'Parameter Trajectory', 'Position', [100, 100, 1000, 800]);
        
        num_params = size(parameter_history, 2);
        
        for i = 1:num_params
            subplot(ceil(num_params/2), 2, i);
            plot(parameter_history(:, i), 'o-', 'LineWidth', 1.5);
            
            if nargin >= 2 && length(parameter_names) >= i
                title(parameter_names{i});
                ylabel(parameter_names{i});
            else
                title(sprintf('Parameter %d', i));
                ylabel(sprintf('Parameter %d', i));
            end
            
            xlabel('Iteration');
            grid on;
        end
    end
    
    function saveAllFigures(directory)
        % Save all open figures to a directory
        if nargin < 1 || isempty(directory)
            directory = './figures';
        end
        
        % Create directory if it doesn't exist
        if ~exist(directory, 'dir')
            mkdir(directory);
        end
        
        % Get all figure handles
        figures = findall(0, 'Type', 'figure');
        
        % Save each figure
        for i = 1:length(figures)
            fig = figures(i);
            fig_name = get(fig, 'Name');
            
            % Generate filename from figure name or number
            if isempty(fig_name)
                filename = sprintf('figure_%d.png', fig.Number);
            else
                % Replace spaces and special characters
                filename = strrep(fig_name, ' ', '_');
                filename = strrep(filename, ':', '_');
                filename = strrep(filename, '.', '_');
                filename = [filename '.png'];
            end
            
            % Save figure
            saveas(fig, fullfile(directory, filename));
        end
        
        fprintf('Saved %d figures to %s\n', length(figures), directory);
    end
    
    return;
end