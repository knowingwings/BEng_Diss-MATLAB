function [sync_signals] = collaborative_task_coordinator(task_info, task_assignments, task_progress, incoming_sync, robot_id, current_time)
    % COLLABORATIVE_TASK_COORDINATOR Implements the leader-follower coordination for collaborative tasks
    %
    % Inputs:
    %   task_info           - Information about all tasks
    %   task_assignments    - Current task assignments
    %   task_progress       - Progress of each task (0-1)
    %   incoming_sync       - Incoming synchronisation signals from other robots
    %   robot_id            - ID of the robot running this function
    %   current_time        - Current simulation time
    %
    % Outputs:
    %   sync_signals        - Outgoing synchronisation signals
    %
    % This implementation is based on the leader-follower paradigm described in:
    % Mathematical Foundations of Decentralised Control for Dual Mobile Manipulators
    
    % Parameters
    sync_timeout = 5.0;  % Maximum time to wait for synchronisation (seconds)
    
    % Extract collaborative tasks
    num_tasks = size(task_info, 1);
    collab_tasks = [];
    for i = 1:num_tasks
        if task_info(i, 11) > 0  % Check collaborative flag
            collab_tasks = [collab_tasks, i];
        end
    end
    
    % Initialise sync signals
    % Format: [task_id, sync_phase, timestamp, robot_id]
    % Sync phases: 
    % 1 = leader requesting sync
    % 2 = follower acknowledging
    % 3 = leader confirming
    % 4 = follower executing
    % 5 = task complete
    sync_signals = zeros(num_tasks, 4);
    
    % Process incoming sync signals
    if ~isempty(incoming_sync)
        for i = 1:size(incoming_sync, 1)
            if incoming_sync(i, 1) > 0  % Valid sync signal
                task_id = incoming_sync(i, 1);
                sync_phase = incoming_sync(i, 2);
                sender_time = incoming_sync(i, 3);
                sender_id = incoming_sync(i, 4);
                
                % Check if this is a collaborative task this robot is involved in
                if ismember(task_id, collab_tasks) && (task_assignments(task_id) == robot_id || task_assignments(task_id) == 3)
                    % Determine if this robot is leader or follower
                    % For simulation, assume robot with lower ID is leader
                    is_leader = (robot_id < sender_id);
                    
                    if ~is_leader
                        % Follower responds to leader's sync signals
                        if sync_phase == 1  % Leader requesting sync
                            % Check if prerequisites are complete and robot is ready
                            if is_robot_ready_for_task(task_id, task_info, task_assignments, task_progress, robot_id)
                                % Send acknowledgment
                                sync_signals(task_id, :) = [task_id, 2, current_time, robot_id];
                                disp(['Robot ', num2str(robot_id), ' (follower) acknowledging sync request for Task ', num2str(task_id)]);
                            end
                        elseif sync_phase == 3  % Leader confirming execution
                            % Task is synchronised, execute joint action
                            sync_signals(task_id, :) = [task_id, 4, current_time, robot_id];
                            disp(['Robot ', num2str(robot_id), ' (follower) confirming execution for Task ', num2str(task_id)]);
                        end
                    else
                        % Leader processes follower's responses
                        if sync_phase == 2  % Follower acknowledged
                            % Check if leader is also ready
                            if is_robot_ready_for_task(task_id, task_info, task_assignments, task_progress, robot_id)
                                % Confirm synchronised execution
                                sync_signals(task_id, :) = [task_id, 3, current_time, robot_id];
                                disp(['Robot ', num2str(robot_id), ' (leader) confirming sync for Task ', num2str(task_id)]);
                            end
                        elseif sync_phase == 4  % Follower confirmed execution
                            % Both robots are executing the task
                            if task_progress(task_id) >= 1.0
                                % Task is complete
                                sync_signals(task_id, :) = [task_id, 5, current_time, robot_id];
                                disp(['Robot ', num2str(robot_id), ' (leader) marking Task ', num2str(task_id), ' as complete']);
                            end
                        end
                    end
                end
            end
        end
    end
    
    % Generate new sync signals for collaborative tasks this robot is assigned to
    for i = 1:length(collab_tasks)
        task_id = collab_tasks(i);
        
        % Check if this is a collaborative task assigned to this robot
        if task_assignments(task_id) == robot_id || task_assignments(task_id) == 3
            % Determine if this robot should be leader
            % For simulation, assume robot with lower ID is leader
            other_robot_id = 3 - robot_id;  % If robot_id is 1, other is 2, and vice versa
            is_leader = (robot_id < other_robot_id);
            
            % If leader and no active sync signal, initiate sync
            if is_leader && all(sync_signals(task_id, :) == 0) && task_progress(task_id) < 1.0
                % Check if task is available (all prerequisites complete)
                if is_task_available(task_id, task_info, task_assignments, task_progress)
                    % Send initial sync request
                    sync_signals(task_id, :) = [task_id, 1, current_time, robot_id];
                    disp(['Robot ', num2str(robot_id), ' (leader) initiating sync request for Task ', num2str(task_id)]);
                end
            end
        end
    end
    
    end
    
    function ready = is_robot_ready_for_task(task_id, task_info, task_assignments, task_progress, robot_id)
    % Check if the robot is ready to perform the task
    
    % Check if all prerequisites are complete
    prereqs_complete = is_task_available(task_id, task_info, task_assignments, task_progress);
    
    % In a real system, would also check:
    % - Robot proximity to task location
    % - Appropriate end-effector configuration
    % - Resource availability
    
    % For simulation, just check prerequisites
    ready = prereqs_complete;
    end
    
    function available = is_task_available(task_id, task_info, task_assignments, task_progress)
    % Check if a task is available (all prerequisites complete)
    
    % Extract prerequisites
    dependencies = task_info(task_id, 8:10);
    dependencies = dependencies(dependencies > 0);
    
    % Check if all prerequisites are completed
    all_complete = true;
    for i = 1:length(dependencies)
        dep_id = dependencies(i);
        if task_progress(dep_id) < 1.0
            all_complete = false;
            break;
        end
    end
    
    available = all_complete;
    end