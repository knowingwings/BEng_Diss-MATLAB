function utils = consensus_utils()
    % CONSENSUS_UTILS - Utilities for time-weighted consensus protocol
    
    % Create utils structure
    utils = struct();
    
    % Add function handles
    utils.runConsensusUpdate = @runConsensusUpdate;
    utils.calculateConsensusError = @calculateConsensusError;
    utils.visualizeConsensus = @visualizeConsensus;
    
    function state_vectors = runConsensusUpdate(state_vectors, robots, last_update_time, params)
        % Run a single iteration of the time-weighted consensus update
        
        num_robots = length(robots);
        
        % Update communication timestamps
        last_update_time = last_update_time + 1;
        
        % For each pair of robots, attempt information exchange
        for i = 1:num_robots
            if robots(i).failed
                continue;
            end
            
            % Get robot i's state vector
            x_i = state_vectors{i};
            
            % For each other robot j
            for j = setdiff(1:num_robots, i)
                if robots(j).failed
                    continue;
                end
                
                % Check communication conditions
                dist_ij = norm(robots(i).position - robots(j).position);
                
                % Check if within communication range
                if isfield(params, 'comm_range') && dist_ij > params.comm_range
                    continue;
                end
                
                % Check packet loss probability
                if isfield(params, 'packet_loss_prob') && rand() < params.packet_loss_prob
                    continue;  % Packet lost
                end
                
                % Get robot j's state vector with delay
                delay_ij = 0;
                if isfield(params, 'comm_delay') && params.comm_delay > 0
                    delay_ij = params.comm_delay;
                end
                
                % For simplicity in simulation, we just use the current state
                % In a real system, this would use a delayed state vector
                x_j = state_vectors{j};
                
                % Calculate time-decaying weight
                time_diff = last_update_time(i, j);
                weight = params.gamma * exp(-params.lambda * time_diff);
                
                % Update state with time-weighted consensus
                state_vectors{i} = x_i + weight * (x_j - x_i);
                
                % Reset last update time
                last_update_time(i, j) = 0;
            end
        end
        
        return;
    end
    
    function error = calculateConsensusError(state_vectors, robots)
        % Calculate the consensus error among robot state vectors
        
        error = 0;
        active_robots = find(~[robots.failed]);
        
        if length(active_robots) >= 2
            for i = active_robots
                for j = active_robots
                    if i < j
                        error = error + norm(state_vectors{i} - state_vectors{j});
                    end
                end
            end
        end
        
        return;
    end
    
    function visualizeConsensus(consensus_errors, params)
        % Visualize consensus error convergence
        
        figure('Name', 'Consensus Error Convergence', 'Position', [100, 100, 800, 600]);
        
        % Plot consensus error
        semilogy(consensus_errors, 'LineWidth', 2);
        hold on;
        
        % Calculate theoretical convergence rate
        if isfield(params, 'gamma')
            iterations = length(consensus_errors);
            initial_error = consensus_errors(1);
            theoretical_rate = -log(1 - 2*params.gamma);
            
            theoretical_curve = initial_error * exp(-theoretical_rate * (0:iterations-1));
            semilogy(theoretical_curve, '--r', 'LineWidth', 2);
            
            legend('Actual Error', 'Theoretical Bound', 'Location', 'northeast');
        end
        
        title('Consensus Error Convergence');
        xlabel('Iteration');
        ylabel('Error (log scale)');
        grid on;
        
        % Display convergence rate
        if length(consensus_errors) > 10
            actual_rate = -log(consensus_errors(end) / consensus_errors(1)) / length(consensus_errors);
            
            text_str = sprintf('Theoretical Rate: %.4f\nActual Rate: %.4f', theoretical_rate, actual_rate);
            text(0.7*iterations, 0.5*consensus_errors(1), text_str, 'FontSize', 12);
        end
    end
end