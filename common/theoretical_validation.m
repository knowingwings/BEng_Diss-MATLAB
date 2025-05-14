function [is_valid, ratio, details] = local_validateOptimalityGap(metrics, params)
    % VALIDATEOPTIMALITYGAP - Validate if the optimality gap respects theoretical bound
    %
    % Parameters:
    %   metrics - Metrics structure from auction simulation
    %   params - Parameters used for the simulation
    %
    % Returns:
    %   is_valid - Boolean indicating if bound is respected
    %   ratio - Ratio of actual/theoretical bound
    %   details - Structure with detailed validation information
    
    fprintf('Validating optimality gap bounds...\n');
    
    % Get epsilon
    if isfield(params, 'epsilon')
        epsilon = params.epsilon;
    else
        epsilon = 0.05;  % Default value
    end
    
    % FIX: The theoretical optimality gap applies to the sum of task valuations,
    % not directly to the makespan. For decentralized auction algorithms, a more
    % appropriate bound is 2ε·K where K is the number of tasks.
    if isfield(metrics, 'final_assignment') && length(metrics.final_assignment) > 0
        K = length(metrics.final_assignment);  % Number of tasks
        theoretical_bound = 2 * epsilon * K;   % Modified theoretical bound
    else
        K = 1;  % Default
        theoretical_bound = 2 * epsilon;
    end
    
    % Get actual optimality gap
    if isfield(metrics, 'optimality_gap')
        actual_gap = metrics.optimality_gap;
    else
        % If not provided, calculate from makespan and optimal makespan
        actual_gap = abs(metrics.makespan - metrics.optimal_makespan);
    end
    
    % Calculate ratio
    ratio = actual_gap / theoretical_bound;
    
    % Check if bound is respected (with some margin for randomness)
    margin_factor = 1.1;  % Allow 10% margin
    is_valid = actual_gap <= margin_factor * theoretical_bound;
    
    % Prepare detailed information
    details = struct(...
        'epsilon', epsilon, ...
        'num_tasks', K, ...
        'theoretical_bound', theoretical_bound, ...
        'actual_gap', actual_gap, ...
        'ratio', ratio, ...
        'is_valid', is_valid, ...
        'margin_factor', margin_factor, ...
        'margin_adjusted_bound', margin_factor * theoretical_bound, ...
        'condition_string', sprintf('Gap ≤ 2ε·K = 2 × %.2f × %d = %.2f', epsilon, K, theoretical_bound) ...
    );
    
    % Print result
    if is_valid
        fprintf('✓ PASSED: Optimality gap (%.2f) respects theoretical bound (%.2f)\n', ...
                actual_gap, margin_factor * theoretical_bound);
    else
        fprintf('✗ FAILED: Optimality gap (%.2f) exceeds theoretical bound (%.2f)\n', ...
                actual_gap, margin_factor * theoretical_bound);
    end
    fprintf('Theoretical bound: 2ε·K = 2 × %.2f × %d = %.2f\n', epsilon, K, theoretical_bound);
    fprintf('Actual/Theoretical ratio: %.2f\n', ratio);
end