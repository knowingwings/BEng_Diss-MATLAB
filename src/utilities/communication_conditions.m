function [delay, packet_loss] = communication_conditions(robot1_pos, robot2_pos, current_time, environmental_factors)
    % COMMUNICATION_CONDITIONS Models time-varying communication conditions between robots
    %
    % Inputs:
    %   robot1_pos           - Position of robot 1 [x, y]
    %   robot2_pos           - Position of robot 2 [x, y]
    %   current_time         - Current simulation time
    %   environmental_factors - Optional parameter for environment effects (0-10 scale)
    %
    % Outputs:
    %   delay       - Communication delay in seconds
    %   packet_loss - Probability of packet loss (0-1)
    %
    % This function simulates realistic communication conditions based on robot distance,
    % time-varying network conditions, and environmental factors. It models phenomena like
    % interference, bandwidth constraints, and network congestion.
    
    % Parameters
    base_delay = 0.1;        % Base delay in seconds
    max_delay = 0.5;         % Maximum delay in seconds
    base_loss_prob = 0.05;   % Base packet loss probability
    max_loss_prob = 0.5;     % Maximum packet loss probability
    
    % Calculate distance between robots
    distance = norm(robot1_pos - robot2_pos);
    
    % Distance-based delay component
    % Normalise by 10m reference distance
    distance_factor = min(1, distance / 10);
    
    % Time-varying component (simulate network congestion periods)
    % Creates a sinusoidal pattern with period of 5 minutes
    % Environmental factor component (obstacles, interference, etc.)
    if nargin > 3
        env_factor = min(1, environmental_factors / 10);
    else
        env_factor = 0;
    end
    
    % Calculate delay as weighted combination of factors
    delay = base_delay + (max_delay - base_delay) * (0.5 * distance_factor + 0.3 * time_factor + 0.2 * env_factor);
    
    % Calculate packet loss probability
    packet_loss = base_loss_prob + (max_loss_prob - base_loss_prob) * (0.6 * distance_factor + 0.2 * time_factor + 0.2 * env_factor);
    
    % Ensure values are within bounds
    delay = min(max_delay, max(base_delay, delay));
    packet_loss = min(max_loss_prob, max(base_loss_prob, packet_loss));
    
    % Add random variation (to simulate jitter and network randomness)
    delay = delay * (1 + 0.1 * randn());  % Add random variation of ±10%
    packet_loss = max(0, min(1, packet_loss * (1 + 0.1 * randn())));  % Keep within [0,1]
    
    % For debugging
    if nargout == 0
        disp(['Distance between robots: ', num2str(distance), 'm']);
        disp(['Distance factor: ', num2str(distance_factor)]);
        disp(['Time factor: ', num2str(time_factor)]);
        disp(['Environmental factor: ', num2str(env_factor)]);
        disp(['Resulting delay: ', num2str(delay), 's']);
        disp(['Resulting packet loss probability: ', num2str(packet_loss*100), '%']);
    end
    
    end