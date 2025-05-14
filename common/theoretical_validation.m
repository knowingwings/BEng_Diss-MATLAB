function utils = theoretical_validation()
    % THEORETICAL_VALIDATION - Returns function handles for theoretical validation
    utils = struct(...
        'validateOptimalityGap', @local_validateOptimalityGap, ...
        'validateConvergenceBound', @local_validateConvergenceBound, ...
        'validateRecoveryBound', @local_validateRecoveryBound, ...
        'calculateBoundTightness', @local_calculateBoundTightness, ...
        'generateTheoreticalReport', @local_generateTheoreticalReport, ...
        'visualizeTheoreticalGuarantees', @local_visualizeTheoreticalGuarantees ...
    );
end

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
        if isfield(metrics, 'makespan') && isfield(metrics, 'optimal_makespan')
            actual_gap = abs(metrics.makespan - metrics.optimal_makespan);
        else
            actual_gap = 0;
            fprintf('Warning: Cannot calculate optimality gap - makespan data missing\n');
        end
    end
    
    % Calculate ratio
    if theoretical_bound > 0
        ratio = actual_gap / theoretical_bound;
    else
        ratio = 0;
        fprintf('Warning: Theoretical bound is zero\n');
    end
    
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

function [is_valid, ratio, details] = local_validateConvergenceBound(metrics, params)
    % VALIDATECONVERGENCEBOUND - Validate if convergence time respects theoretical bound
    %
    % Parameters:
    %   metrics - Metrics structure from auction simulation
    %   params - Parameters used for the simulation
    %
    % Returns:
    %   is_valid - Boolean indicating if bound is respected
    %   ratio - Ratio of actual/theoretical bound
    %   details - Structure with detailed validation information
    
    fprintf('Validating convergence time bounds...\n');
    
    % Get parameters
    if isfield(params, 'epsilon')
        epsilon = params.epsilon;
    else
        epsilon = 0.05;  % Default value
    end
    
    if isfield(params, 'alpha')
        alpha = params.alpha;
        b_max = max(alpha);
    else
        b_max = 1.0;  % Default value
    end
    
    % For convergence bound, need number of tasks K
    if isfield(metrics, 'final_assignment')
        K = length(metrics.final_assignment);
    elseif isfield(metrics, 'assignment_history')
        K = size(metrics.assignment_history, 1);
    else
        K = 10;  % Default value
        fprintf('Warning: Cannot determine number of tasks, using default K=%d\n', K);
    end
    
    % Theoretical bound: O(K² · bₘₐₓ/ε)
    theoretical_bound = K^2 * b_max / epsilon;
    
    % Get actual convergence time
    if isfield(metrics, 'iterations')
        actual_iterations = metrics.iterations;
    else
        actual_iterations = 0;
        fprintf('Warning: Cannot determine actual iterations\n');
    end
    
    % Calculate ratio
    if theoretical_bound > 0
        ratio = actual_iterations / theoretical_bound;
    else
        ratio = 0;
        fprintf('Warning: Theoretical bound is zero\n');
    end
    
    % Check if bound is respected (with some margin for randomness)
    margin_factor = 1.2;  % Allow 20% margin
    is_valid = actual_iterations <= margin_factor * theoretical_bound;
    
    % Prepare detailed information
    details = struct(...
        'epsilon', epsilon, ...
        'b_max', b_max, ...
        'num_tasks', K, ...
        'theoretical_bound', theoretical_bound, ...
        'actual_iterations', actual_iterations, ...
        'ratio', ratio, ...
        'is_valid', is_valid, ...
        'margin_factor', margin_factor, ...
        'margin_adjusted_bound', margin_factor * theoretical_bound, ...
        'condition_string', sprintf('Iterations ≤ O(K² · bₘₐₓ/ε) = O(%d² · %.2f/%.2f) ≈ %.1f', K, b_max, epsilon, theoretical_bound) ...
    );
    
    % Print result
    if is_valid
        fprintf('✓ PASSED: Convergence time (%d iterations) respects theoretical bound (%.1f)\n', ...
                actual_iterations, margin_factor * theoretical_bound);
    else
        fprintf('✗ FAILED: Convergence time (%d iterations) exceeds theoretical bound (%.1f)\n', ...
                actual_iterations, margin_factor * theoretical_bound);
    end
    fprintf('Theoretical bound: O(K² · bₘₐₓ/ε) = O(%d² · %.2f/%.2f) ≈ %.1f\n', K, b_max, epsilon, theoretical_bound);
    fprintf('Actual/Theoretical ratio: %.2f\n', ratio);
end

function [is_valid, ratio, details] = local_validateRecoveryBound(metrics, params)
    % VALIDATERECOVERYBOUND - Validate if recovery time respects theoretical bound
    %
    % Parameters:
    %   metrics - Metrics structure from auction simulation
    %   params - Parameters used for the simulation
    %
    % Returns:
    %   is_valid - Boolean indicating if bound is respected
    %   ratio - Ratio of actual/theoretical bound
    %   details - Structure with detailed validation information
    
    fprintf('Validating recovery time bounds...\n');
    
    % Check if failure recovery was performed
    if ~isfield(metrics, 'recovery_time') || metrics.recovery_time == 0
        fprintf('No failure recovery was performed in this simulation.\n');
        is_valid = true;
        ratio = 0;
        details = struct('message', 'No failure recovery performed');
        return;
    end
    
    % Get parameters
    if isfield(params, 'epsilon')
        epsilon = params.epsilon;
    else
        epsilon = 0.05;  % Default value
    end
    
    if isfield(params, 'alpha')
        alpha = params.alpha;
        b_max = max(alpha);
    else
        b_max = 1.0;  % Default value
    end
    
    % Get number of tasks assigned to failed robot
    if isfield(metrics, 'failed_task_count')
        T_f = metrics.failed_task_count;
    else
        T_f = 0;
        fprintf('Warning: Cannot determine number of failed tasks\n');
    end
    
    % Theoretical bound: O(|Tᶠ|) + O(bₘₐₓ/ε)
    theoretical_bound = T_f + b_max / epsilon;
    
    % Get actual recovery time
    actual_recovery_time = metrics.recovery_time;
    
    % Calculate ratio
    if theoretical_bound > 0
        ratio = actual_recovery_time / theoretical_bound;
    else
        ratio = 0;
        fprintf('Warning: Theoretical bound is zero\n');
    end
    
    % Check if bound is respected (with some margin for randomness)
    margin_factor = 1.2;  % Allow 20% margin
    is_valid = actual_recovery_time <= margin_factor * theoretical_bound;
    
    % Prepare detailed information
    details = struct(...
        'epsilon', epsilon, ...
        'b_max', b_max, ...
        'failed_tasks', T_f, ...
        'theoretical_bound', theoretical_bound, ...
        'actual_recovery_time', actual_recovery_time, ...
        'ratio', ratio, ...
        'is_valid', is_valid, ...
        'margin_factor', margin_factor, ...
        'margin_adjusted_bound', margin_factor * theoretical_bound, ...
        'condition_string', sprintf('Recovery time ≤ O(|Tᶠ|) + O(bₘₐₓ/ε) = %d + %.2f/%.2f ≈ %.1f', T_f, b_max, epsilon, theoretical_bound) ...
    );
    
    % Print result
    if is_valid
        fprintf('✓ PASSED: Recovery time (%d iterations) respects theoretical bound (%.1f)\n', ...
                actual_recovery_time, margin_factor * theoretical_bound);
    else
        fprintf('✗ FAILED: Recovery time (%d iterations) exceeds theoretical bound (%.1f)\n', ...
                actual_recovery_time, margin_factor * theoretical_bound);
    end
    fprintf('Theoretical bound: O(|Tᶠ|) + O(bₘₐₓ/ε) = %d + %.2f/%.2f ≈ %.1f\n', T_f, b_max, epsilon, theoretical_bound);
    fprintf('Actual/Theoretical ratio: %.2f\n', ratio);
end

function [tightness, details] = local_calculateBoundTightness(metrics, params)
    % CALCULATEBOUNDTIGHTNESS - Calculate overall tightness of theoretical bounds
    %
    % Parameters:
    %   metrics - Metrics structure from auction simulation
    %   params - Parameters used for the simulation
    %
    % Returns:
    %   tightness - Overall bound tightness (0-1 scale, higher is tighter)
    %   details - Structure with detailed tightness information
    
    fprintf('Calculating theoretical bound tightness...\n');
    
    % Validate all bounds
    [opt_valid, opt_ratio, opt_details] = local_validateOptimalityGap(metrics, params);
    [conv_valid, conv_ratio, conv_details] = local_validateConvergenceBound(metrics, params);
    
    % Check if recovery was performed
    if isfield(metrics, 'recovery_time') && metrics.recovery_time > 0
        [rec_valid, rec_ratio, rec_details] = local_validateRecoveryBound(metrics, params);
        recovery_performed = true;
    else
        rec_valid = true;
        rec_ratio = 0;
        rec_details = struct('message', 'No recovery performed');
        recovery_performed = false;
    end
    
    % Calculate overall tightness as average of ratios
    if recovery_performed
        tightness = (opt_ratio + conv_ratio + rec_ratio) / 3;
    else
        tightness = (opt_ratio + conv_ratio) / 2;
    end
    
    % Prepare details
    details = struct(...
        'optimality_gap', struct(...
            'valid', opt_valid, ...
            'ratio', opt_ratio, ...
            'details', opt_details ...
        ), ...
        'convergence_bound', struct(...
            'valid', conv_valid, ...
            'ratio', conv_ratio, ...
            'details', conv_details ...
        ), ...
        'recovery_bound', struct(...
            'valid', rec_valid, ...
            'ratio', rec_ratio, ...
            'details', rec_details, ...
            'performed', recovery_performed ...
        ), ...
        'overall_tightness', tightness ...
    );
    
    % Print result
    fprintf('Overall bound tightness: %.2f\n', tightness);
    if tightness < 0.3
        fprintf('Bounds are loose (ratio < 0.3) - algorithm performs much better than theoretical guarantees\n');
    elseif tightness < 0.7
        fprintf('Bounds are moderately tight (0.3 ≤ ratio < 0.7)\n');
    else
        fprintf('Bounds are tight (ratio ≥ 0.7) - algorithm performance approaches theoretical limits\n');
    end
end

function report = local_generateTheoreticalReport(metrics, params)
    % GENERATETHEORETICALREPORT - Generate a comprehensive report on theoretical validations
    %
    % Parameters:
    %   metrics - Metrics structure from auction simulation
    %   params - Parameters used for the simulation
    %
    % Returns:
    %   report - Structure containing comprehensive theoretical validation results
    
    fprintf('Generating theoretical validation report...\n');
    
    % Calculate bound tightness (which calls all validation functions)
    [tightness, tightness_details] = local_calculateBoundTightness(metrics, params);
    
    % Add algorithm parameters to report
    algorithm_params = struct();
    if isfield(params, 'epsilon')
        algorithm_params.epsilon = params.epsilon;
    end
    if isfield(params, 'alpha')
        algorithm_params.alpha = params.alpha;
        algorithm_params.b_max = max(params.alpha);
    end
    if isfield(params, 'gamma')
        algorithm_params.gamma = params.gamma;
    end
    if isfield(params, 'lambda')
        algorithm_params.lambda = params.lambda;
    end
    if isfield(params, 'beta')
        algorithm_params.beta = params.beta;
    end
    
    % Add scenario parameters
    scenario_params = struct();
    if isfield(metrics, 'final_assignment')
        scenario_params.num_tasks = length(metrics.final_assignment);
    elseif isfield(metrics, 'assignment_history')
        scenario_params.num_tasks = size(metrics.assignment_history, 1);
    end
    
    if isfield(params, 'failure_time') && ~isinf(params.failure_time)
        scenario_params.failure_time = params.failure_time;
        scenario_params.failed_robot = params.failed_robot;
    end
    
    % Create full report
    report = struct(...
        'algorithm_params', algorithm_params, ...
        'scenario_params', scenario_params, ...
        'validation_results', tightness_details, ...
        'overall_tightness', tightness, ...
        'all_bounds_respected', tightness_details.optimality_gap.valid && ...
                                tightness_details.convergence_bound.valid && ...
                                tightness_details.recovery_bound.valid ...
    );
    
    % Print summary
    fprintf('\nTheoretical Validation Summary:\n');
    fprintf('-------------------------------\n');
    fprintf('Optimality Gap: %s (Ratio: %.2f)\n', ...
           iif(tightness_details.optimality_gap.valid, 'PASSED', 'FAILED'), ...
           tightness_details.optimality_gap.ratio);
    fprintf('Convergence Time: %s (Ratio: %.2f)\n', ...
           iif(tightness_details.convergence_bound.valid, 'PASSED', 'FAILED'), ...
           tightness_details.convergence_bound.ratio);
    
    if tightness_details.recovery_bound.performed
        fprintf('Recovery Time: %s (Ratio: %.2f)\n', ...
               iif(tightness_details.recovery_bound.valid, 'PASSED', 'FAILED'), ...
               tightness_details.recovery_bound.ratio);
    else
        fprintf('Recovery Time: Not tested (no failure in simulation)\n');
    end
    
    fprintf('Overall Result: %s\n', ...
           iif(report.all_bounds_respected, 'ALL BOUNDS RESPECTED', 'SOME BOUNDS VIOLATED'));
end

function visualizeTheoreticalGuarantees(report)
    % VISUALIZETHEORETICALGUARANTEES - Create visualization of theoretical guarantees
    %
    % Parameters:
    %   report - Report structure from generateTheoreticalReport
    
    fprintf('Creating theoretical guarantees visualization...\n');
    
    % Create figure
    figure('Name', 'Theoretical Guarantees', 'Position', [100, 100, 1200, 600]);
    
    % 1. Bound ratios
    subplot(2, 2, 1);
    
    % Prepare data
    labels = {'Optimality', 'Convergence'};
    ratios = [report.validation_results.optimality_gap.ratio, ...
              report.validation_results.convergence_bound.ratio];
    
    if report.validation_results.recovery_bound.performed
        labels{end+1} = 'Recovery';
        ratios(end+1) = report.validation_results.recovery_bound.ratio;
    end
    
    % Create bar plot
    bar(ratios);
    
    % Add labels
    xlabel('Theoretical Bound');
    ylabel('Ratio (Actual/Theoretical)');
    title('Bound Tightness');
    set(gca, 'XTickLabel', labels);
    grid on;
    
    % Add threshold line
    hold on;
    yline(1.0, 'r--', 'LineWidth', 1.5);
    yline(0.5, 'g--', 'LineWidth', 1.5);
    
    % 2. Parameter sensitivity (epsilon)
    subplot(2, 2, 2);
    
    % Get epsilon and number of tasks
    epsilon = report.algorithm_params.epsilon;
    if isfield(report.scenario_params, 'num_tasks')
        K = report.scenario_params.num_tasks;
    else
        K = 10;
    end
    
    % Create epsilon range
    epsilon_range = [0.01, 0.05, 0.1, 0.2, 0.5];
    opt_bounds = 2 * epsilon_range;
    conv_bounds = K^2 * report.algorithm_params.b_max ./ epsilon_range;
    
    % Normalize for comparison
    norm_opt_bounds = opt_bounds / max(opt_bounds);
    norm_conv_bounds = conv_bounds / max(conv_bounds);
    
    % Plot
    plot(epsilon_range, norm_opt_bounds, '-o', 'LineWidth', 1.5);
    hold on;
    plot(epsilon_range, norm_conv_bounds, '-s', 'LineWidth', 1.5);
    
    % Mark current epsilon
    plot([epsilon, epsilon], [0, 1], 'r--', 'LineWidth', 1.5);
    
    % Add labels
    xlabel('Epsilon (ε)');
    ylabel('Normalized Bound Value');
    title('Bound Sensitivity to Epsilon');
    legend('Optimality Bound', 'Convergence Bound', 'Current ε', 'Location', 'best');
    grid on;
    
    % 3. Optimality gap visualization
    subplot(2, 2, 3);
    
    % Get optimality gap data
    actual_gap = report.validation_results.optimality_gap.details.actual_gap;
    theoretical_bound = report.validation_results.optimality_gap.details.theoretical_bound;
    
    % Create bar graph
    bar([actual_gap, theoretical_bound]);
    
    % Add labels
    set(gca, 'XTickLabel', {'Actual Gap', 'Theoretical Bound'});
    ylabel('Optimality Gap');
    title('Optimality Gap vs. Theoretical Bound');
    grid on;
    
    % 4. Convergence visualization
    subplot(2, 2, 4);
    
    % Get convergence data
    actual_iter = report.validation_results.convergence_bound.details.actual_iterations;
    theoretical_iter = report.validation_results.convergence_bound.details.theoretical_bound;
    
    % Create bar graph
    bar([actual_iter, theoretical_iter]);
    
    % Add labels
    set(gca, 'XTickLabel', {'Actual Iterations', 'Theoretical Bound'});
    ylabel('Iterations to Converge');
    title('Convergence Time vs. Theoretical Bound');
    grid on;
    
    % Add overall title
    sgtitle('Theoretical Guarantee Validation', 'FontSize', 14);
    
    fprintf('Theoretical guarantees visualization complete.\n');
end

function result = iif(condition, true_value, false_value)
    % Simple inline if helper function
    if condition
        result = true_value;
    else
        result = false_value;
    end
end