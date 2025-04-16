function [updated_state] = consensus_protocol(local_state, received_states, last_update_time, current_time, gamma, lambda)
    % CONSENSUS_PROTOCOL Implements the time-weighted consensus protocol
    %
    % Inputs:
    %   local_state      - Current state information of the local robot
    %   received_states  - State information received from other robots
    %   last_update_time - Time of last update from each robot
    %   current_time     - Current simulation time
    %   gamma            - Base weight factor for consensus (typically 0.3-0.7)
    %   lambda           - Information decay rate parameter
    %
    % Outputs:
    %   updated_state    - Updated state after applying consensus
    %
    % This implementation is based on the time-weighted consensus protocol described in:
    % Olfati-Saber, R., Fax, J.A., Murray, R.M. (2007). "Consensus and Cooperation 
    % in Networked Multi-Agent Systems." Proceedings of the IEEE, 95(1), 215-233.
    
    % Initialise updated state as a copy of local state
    updated_state = local_state;
    
    % If no received states, return local state unchanged
    if isempty(received_states)
        return;
    end
    
    % Process received state information from each robot
    robot_ids = fieldnames(received_states);
    
    for i = 1:length(robot_ids)
        robot_id = robot_ids{i};
        other_state = received_states.(robot_id);
        
        % Calculate time since last update from this robot
        time_diff = current_time - last_update_time.(robot_id);
        
        % Check if the update is recent enough to be considered
        if time_diff <= 10  % Only consider updates from last 10 seconds
            % Calculate weight based on time decay
            weight = gamma * exp(-lambda * time_diff);
            
            % Apply weighted averaging to each state component
            state_fields = fieldnames(local_state);
            
            for j = 1:length(state_fields)
                field = state_fields{j};
                
                % Only update fields that are present in both states
                if isfield(other_state, field)
                    % Handle different data types appropriately
                    if isnumeric(local_state.(field)) && isnumeric(other_state.(field))
                        % For numeric data, apply weighted average
                        updated_state.(field) = local_state.(field) + ...
                            weight * (other_state.(field) - local_state.(field));
                    elseif iscell(local_state.(field)) && iscell(other_state.(field))
                        % For cell arrays, update cell by cell if they have same size
                        if numel(local_state.(field)) == numel(other_state.(field))
                            for k = 1:numel(local_state.(field))
                                if isnumeric(local_state.(field){k}) && isnumeric(other_state.(field){k})
                                    updated_state.(field){k} = local_state.(field){k} + ...
                                        weight * (other_state.(field){k} - local_state.(field){k});
                                end
                            end
                        end
                    elseif isstruct(local_state.(field)) && isstruct(other_state.(field))
                        % For nested structures, recursively apply consensus
                        updated_state.(field) = consensus_protocol(local_state.(field), ...
                            struct(robot_id, other_state.(field)), ...
                            struct(robot_id, last_update_time.(robot_id)), ...
                            current_time, gamma, lambda);
                    end
                end
            end
        end
    end
    
    end
    
    function [updated_vector] = weighted_consensus_vector(local_vector, other_vector, weight)
    % Apply weighted consensus to numeric vectors
    
    % Check if vectors are compatible
    if numel(local_vector) ~= numel(other_vector)
        updated_vector = local_vector;
        return;
    end
    
    % Apply weighted averaging element by element
    updated_vector = local_vector + weight * (other_vector - local_vector);
    end
    
    function [updated_matrix] = weighted_consensus_matrix(local_matrix, other_matrix, weight)
    % Apply weighted consensus to numeric matrices
    
    % Check if matrices are compatible
    if ~isequal(size(local_matrix), size(other_matrix))
        updated_matrix = local_matrix;
        return;
    end
    
    % Apply weighted averaging element by element
    updated_matrix = local_matrix + weight * (other_matrix - local_matrix);
    end