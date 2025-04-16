function [recovery_tasks, updated_assignments] = failure_recovery(robot_states, task_info, task_assignments, task_progress, heartbeat_signals, last_heartbeat_times, current_time, robot_id)
    % FAILURE_RECOVERY Implements the failure recovery mechanism
    %
    % Inputs:
    %   robot_states        - States of all robots in the system
    %   task_info           - Information about all tasks
    %   task_assignments    - Current task assignments
    %   task_progress       - Progress of each task (0-1)
    %   heartbeat_signals   - Latest heartbeat signals from all robots
    %   last_heartbeat_times - Time of last heartbeat from each robot
    %   current_time        - Current simulation time
    %   robot_id            - ID of the robot running this function
    %
    % Outputs:
    %   recovery_tasks      - Tasks that need to be recovered
    %   updated_assignments - Updated task assignments after recovery
    %
    % This implementation is based on failure recovery approach described in:
    % Zavlanos, M.M., Spesivtsev, L., Pappas, G.J. (2008). "A distributed auction 
    % algorithm for the assignment problem." IEEE Conference on Decision and Control.
    
    % Parameters
    heartbeat_threshold = 3.0;  % Time threshold for missing heartbeats (seconds)
    progress_threshold = 0.05;  % Minimum expected progress per minute
    progress_check_period = 60;  % Check progress every 60 seconds
    
    % Initialize outputs
    recovery_tasks = [];
    updated_assignments = task_assignments;
    
    % Get IDs of all robots except self
    num_robots = length(robot_states);
    other_robot_ids = setdiff(1:num_robots, robot_id);
    
    % Check for failures in other robots
    for i = 1:length(other_robot_ids)
        other_robot_id = other_robot_ids(i);
        
        % Failure detection logic
        failed = false;
        
        % Method 1: Heartbeat monitoring
        if heartbeat_signals(other_robot_id) == 0 && ...
           (current_time - last_heartbeat_times(other_robot_id) > heartbeat_threshold)
            failed = true;
            disp(['Robot ', num2str(robot_id), ' detected failure of Robot ', num2str(other_robot_id), ' via heartbeat monitoring.']);
        end
        
        % Method 2: Progress monitoring
        for task_idx = 1:length(task_assignments)
            if task_assignments(task_idx) == other_robot_id
                % Check if progress has stalled
                if task_progress(task_idx) > 0 && task_progress(task_idx) < 1
                    expected_progress = task_progress(task_idx) + progress_threshold*(current_time/progress_check_period);
                    if task_progress(task_idx) < expected_progress
                        failed = true;
                        disp(['Robot ', num2str(robot_id), ' detected failure of Robot ', num2str(other_robot_id), ' via progress monitoring on Task ', num2str(task_idx), '.']);
                        break;
                    end
                end
            end
        end
        
        % If failure detected, initiate recovery
        if failed
            % Identify tasks assigned to failed robot
            failed_tasks = find(task_assignments == other_robot_id);
            
            % Calculate recovery bids for each task
            task_bids = zeros(length(failed_tasks), 2);  % [task_id, bid_value]
            
            for j = 1:length(failed_tasks)
                task_id = failed_tasks(j);
                
                % Special recovery bid calculation
                % Factors for recovery bid
                progress_factor = 1 - task_progress(task_id);  % Prioritize less completed tasks
                
                % Calculate number of dependent tasks (criticality)
                dependents = 0;
                for k = 1:size(task_info, 1)
                    dependencies = task_info(k, 8:10);
                    if any(dependencies == task_id)
                        dependents = dependents + 1;
                    end
                end
                criticality_factor = dependents / size(task_info, 1);
                
                % Calculate urgency based on critical path
                [critical_path, ~] = analyze_task_dependencies(task_info, ones(size(task_info, 1), 1));
                urgency_factor = ismember(task_id, critical_path) * 0.5;
                
                % Calculate recovery bid
                recovery_bid = 10*progress_factor + 8*criticality_factor + 5*urgency_factor;
                
                % Add to recovery tasks and bids
                task_bids(j, :) = [task_id, recovery_bid];
            end
            
            % Sort tasks by bid value (descending)
            [~, idx] = sort(task_bids(:, 2), 'descend');
            sorted_tasks = task_bids(idx, 1);
            
            % Store recovered tasks
            recovery_tasks = [recovery_tasks; sorted_tasks];
            
            % Reassign tasks to this robot (in real system, would use modified auction)
            for j = 1:length(sorted_tasks)
                task_id = sorted_tasks(j);
                
                % Check capability match
                robot_capabilities = robot_states(robot_id).capabilities;
                task_capabilities = task_info(task_id, 3:7);
                
                robot_cap_norm = robot_capabilities / norm(robot_capabilities);
                task_cap_norm = task_capabilities / norm(task_capabilities);
                capability_match = dot(robot_cap_norm, task_cap_norm);
                
                % If capability match is sufficient, assign task to this robot
                if capability_match > 0.4  % Threshold for capability match
                    updated_assignments(task_id) = robot_id;
                    disp(['Robot ', num2str(robot_id), ' recovering Task ', num2str(task_id), ' (capability match: ', num2str(capability_match), ').']);
                else
                    % Cannot be assigned due to insufficient capabilities
                    updated_assignments(task_id) = 0;  % Mark as unassigned
                    disp(['Robot ', num2str(robot_id), ' cannot recover Task ', num2str(task_id), ' (insufficient capabilities).']);
                end
            end
        end
    end
    
    end