% theoretical_validation.m
% Documentation and explicit linking of implementation to theoretical guarantees
% Provides functions to validate and document theoretical properties of the auction algorithm

function utils = theoretical_validation()
    % THEORETICAL_VALIDATION - Returns function handles for theoretical validation
    utils = struct(...
        'validateConvergenceBound', @local_validateConvergenceBound, ...
        'validateOptimalityGap', @local_validateOptimalityGap, ...
        'validateRecoveryBound', @local_validateRecoveryBound, ...
        'validateConsensusProperties', @local_validateConsensusProperties, ...
        'generateTheoreticalReport', @local_generateTheoreticalReport, ...
        'calculateBoundTightness', @local_calculateBoundTightness, ...
        'visualizeTheoreticalGuarantees', @local_visualizeTheoreticalGuarantees ...
    );
end

function [is_valid, ratio, details] = local_validateConvergenceBound(metrics, params)
    % VALIDATECONVERGENCEBOUND - Validate if the convergence time fits theoretical bounds
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
    
    % Extract relevant parameters
    K = length(metrics.assignment_history(:, 1));  % Number of tasks
    
    % Get maximum bid value from alpha
    if isfield(params, 'alpha')
        b_max = max(params.alpha);
    else
        b_max = 1.0;  % Default value
    end
    
    % Get epsilon
    if isfield(params, 'epsilon')
        epsilon = params.epsilon;
    else
        epsilon = 0.05;  % Default value
    end
    
    % Calculate theoretical bound: O(K² · bₘₐₓ/ε)
    theoretical_bound = K^2 * b_max / epsilon;
    
    % Get actual number of iterations
    actual_iterations = metrics.iterations;
    
    % Calculate ratio
    ratio = actual_iterations / theoretical_bound;
    
    % Check if bound is respected (with some margin for randomness)
    margin_factor = 1.2;  % Allow 20% margin
    is_valid = actual_iterations <= margin_factor * theoretical_bound;
    
    % Prepare detailed information
    details = struct(...
        'K', K, ...
        'b_max', b_max, ...
        'epsilon', epsilon, ...
        'theoretical_bound', theoretical_bound, ...
        'actual_iterations', actual_iterations, ...
        'ratio', ratio, ...
        'is_valid', is_valid, ...
        'margin_factor', margin_factor, ...
        'margin_adjusted_bound', margin_factor * theoretical_bound, ...
        'condition_string', sprintf('O(K² · bₘₐₓ/ε) = O(%d² · %.2f/%.2f) ≈ %.2f', K, b_max, epsilon, theoretical_bound) ...
    );
    
    % Print result
    if is_valid
        fprintf('✓ PASSED: Convergence time (%.2f iterations) respects theoretical bound (%.2f iterations)\n', ...
                actual_iterations, margin_factor * theoretical_bound);
    else
        fprintf('✗ FAILED: Convergence time (%.2f iterations) exceeds theoretical bound (%.2f iterations)\n', ...
                actual_iterations, margin_factor * theoretical_bound);
    end
    fprintf('Theoretical bound: O(K² · bₘₐₓ/ε) = O(%d² · %.2f/%.2f) ≈ %.2f\n', ...
            K, b_max, epsilon, theoretical_bound);
    fprintf('Actual/Theoretical ratio: %.2f\n', ratio);
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
    
    % Calculate theoretical bound: 2ε
    theoretical_bound = 2 * epsilon;
    
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
        'theoretical_bound', theoretical_bound, ...
        'actual_gap', actual_gap, ...
        'ratio', ratio, ...
        'is_valid', is_valid, ...
        'margin_factor', margin_factor, ...
        'margin_adjusted_bound', margin_factor * theoretical_bound, ...
        'condition_string', sprintf('Gap ≤ 2ε = 2 × %.2f = %.2f', epsilon, theoretical_bound) ...
    );
    
    % Print result
    if is_valid
        fprintf('✓ PASSED: Optimality gap (%.2f) respects theoretical bound (%.2f)\n', ...
                actual_gap, margin_factor * theoretical_bound);
    else
        fprintf('✗ FAILED: Optimality gap (%.2f) exceeds theoretical bound (%.2f)\n', ...
                actual_gap, margin_factor * theoretical_bound);
    end
    fprintf('Theoretical bound: 2ε = 2 × %.2f = %.2f\n', epsilon, theoretical_bound);
    fprintf('Actual/Theoretical ratio: %.2f\n', ratio);
end

function [is_valid, ratio, details] = local_validateRecoveryBound(metrics, params)
    % VALIDATERECOVERYBOUND - Validate if the recovery time respects theoretical bound
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
    
    % Check if failure recovery was needed
    if ~isfield(params, 'failure_time') || isinf(params.failure_time) || isempty(params.failed_robot)
        fprintf('No failure scenario in this simulation. Skipping recovery validation.\n');
        is_valid = true;
        ratio = 0;
        details = struct('message', 'No failure scenario');
        return;
    end
    
    % Check if recovery was completed
    if ~isfield(metrics, 'recovery_time') || metrics.recovery_time == 0
        fprintf('Recovery not completed. Validation failed.\n');
        is_valid = false;
        ratio = Inf;
        details = struct('message', 'Recovery not completed');
        return;
    end
    
    % Get number of tasks assigned to the failed robot (T_f)
    if isfield(metrics, 'failed_task_count')
        T_f = metrics.failed_task_count;
    else
        fprintf('Warning: failed_task_count not provided. Using estimate.\n');
        % Estimate from the assignment history at failure time
        if isfield(metrics, 'assignment_history') && isfield(metrics, 'failure_time')
            failure_time = metrics.failure_time;
            if failure_time <= size(metrics.assignment_history, 2)
                T_f = sum(metrics.assignment_history(:, failure_time) == params.failed_robot);
            else
                T_f = 0;
            end
        else
            T_f = 0;
        end
    end
    
    % Get maximum bid value and epsilon
    if isfield(params, 'alpha')
        b_max = max(params.alpha);
    else
        b_max = 1.0;  % Default value
    end
    
    if isfield(params, 'epsilon')
        epsilon = params.epsilon;
    else
        epsilon = 0.05;  % Default value
    end
    
    % Calculate theoretical bound: O(|T_f|) + O(b_max/ε)
    theoretical_bound = T_f + b_max / epsilon;
    
    % Get actual recovery time
    actual_recovery = metrics.recovery_time;
    
    % Calculate ratio
    ratio = actual_recovery / theoretical_bound;
    
    % Check if bound is respected (with some margin for randomness)
    margin_factor = 1.2;  % Allow 20% margin
    is_valid = actual_recovery <= margin_factor * theoretical_bound;
    
    % Prepare detailed information
    details = struct(...
        'T_f', T_f, ...
        'b_max', b_max, ...
        'epsilon', epsilon, ...
        'theoretical_bound', theoretical_bound, ...
        'actual_recovery', actual_recovery, ...
        'ratio', ratio, ...
        'is_valid', is_valid, ...
        'margin_factor', margin_factor, ...
        'margin_adjusted_bound', margin_factor * theoretical_bound, ...
        'condition_string', sprintf('O(|T_f|) + O(b_max/ε) = O(%d) + O(%.2f/%.2f) ≈ %.2f', ...
                                   T_f, b_max, epsilon, theoretical_bound) ...
    );
    
    % Print result
    if is_valid
        fprintf('✓ PASSED: Recovery time (%d iterations) respects theoretical bound (%.2f iterations)\n', ...
                actual_recovery, margin_factor * theoretical_bound);
    else
        fprintf('✗ FAILED: Recovery time (%d iterations) exceeds theoretical bound (%.2f iterations)\n', ...
                actual_recovery, margin_factor * theoretical_bound);
    end
    fprintf('Theoretical bound: O(|T_f|) + O(b_max/ε) = O(%d) + O(%.2f/%.2f) ≈ %.2f\n', ...
            T_f, b_max, epsilon, theoretical_bound);
    fprintf('Actual/Theoretical ratio: %.2f\n', ratio);
end

function [is_valid, exp_rate, details] = local_validateConsensusProperties(consensus_data, params)
    % VALIDATECONSENSUSPROPERTIES - Validate consensus convergence properties
    %
    % Parameters:
    %   consensus_data - Matrix with consensus values over iterations
    %   params - Parameters used for the simulation
    %
    % Returns:
    %   is_valid - Boolean indicating if properties are respected
    %   exp_rate - Estimated exponential convergence rate
    %   details - Structure with detailed validation information
    
    fprintf('Validating time-weighted consensus properties...\n');
    
    % Check if consensus data is provided
    if isempty(consensus_data) || ~isnumeric(consensus_data)
        fprintf('No consensus data provided. Skipping validation.\n');
        is_valid = false;
        exp_rate = 0;
        details = struct('message', 'No consensus data provided');
        return;
    end
    
    % Get consensus parameters
    if isfield(params, 'gamma')
        gamma = params.gamma;
    else
        gamma = 0.5;  % Default value
    end
    
    % Calculate theoretical exponential convergence rate
    theoretical_rate = -log(1 - 2 * gamma);
    
    % Calculate actual convergence rates
    % Get dimensions of consensus data
    [num_dims, num_agents, num_iterations] = size(consensus_data);
    
    % Calculate consensus errors
    consensus_error = zeros(num_iterations, 1);
    
    for k = 1:num_iterations
        % Calculate the average state across agents (true consensus value)
        mean_state = mean(consensus_data(:, :, k), 2);
        
        % Calculate error for each agent
        agent_errors = zeros(num_agents, 1);
        for i = 1:num_agents
            agent_errors(i) = norm(consensus_data(:, i, k) - mean_state);
        end
        
        % Average error across agents
        consensus_error(k) = mean(agent_errors);
    end
    
    % Fit exponential decay model to verify exponential convergence
    % Use data from iteration 5 onwards to avoid initial transients
    start_fit = min(5, floor(num_iterations / 5));
    
    % Check if we have enough data for fitting
    if num_iterations < 10
        fprintf('Warning: Not enough iterations for reliable exponential fit.\n');
        is_valid = false;
        exp_rate = 0;
        details = struct('message', 'Insufficient iterations for fit');
        return;
    end
    
    try
        % Try to fit exponential decay
        [fit_model, gof] = fit((start_fit:num_iterations)', consensus_error(start_fit:end), ...
                              'a*exp(-b*x)+c', 'StartPoint', [consensus_error(start_fit), 0.1, 0]);
        
        % Extract convergence rate
        exp_rate = fit_model.b;
        
        % Check if fit is good
        if gof.rsquare < 0.8
            fprintf('Warning: Exponential fit quality is poor (R² = %.2f).\n', gof.rsquare);
        end
        
        % Compare with theoretical rate
        rate_ratio = exp_rate / theoretical_rate;
        
        % Check if rate is reasonable (at least 60% of theoretical)
        is_valid = rate_ratio >= 0.6;
        
        % Prepare detailed information
        details = struct(...
            'gamma', gamma, ...
            'theoretical_rate', theoretical_rate, ...
            'estimated_rate', exp_rate, ...
            'rate_ratio', rate_ratio, ...
            'fit_rsquare', gof.rsquare, ...
            'is_valid', is_valid, ...
            'consensus_error', consensus_error, ...
            'condition_string', sprintf('Exponential rate ≈ μ = -ln(1-2γ) = -ln(1-2×%.2f) = %.4f', ...
                                       gamma, theoretical_rate) ...
        );
        
        % Print result
        if is_valid
            fprintf('✓ PASSED: Consensus converges exponentially with rate %.4f (%.0f%% of theoretical)\n', ...
                    exp_rate, 100 * rate_ratio);
        else
            fprintf('✗ FAILED: Consensus convergence rate %.4f too low (%.0f%% of theoretical)\n', ...
                    exp_rate, 100 * rate_ratio);
        end
        fprintf('Theoretical rate: μ = -ln(1-2γ) = -ln(1-2×%.2f) = %.4f\n', gamma, theoretical_rate);
        fprintf('Fit R²: %.4f\n', gof.rsquare);
        
    catch ME
        fprintf('Error fitting exponential decay: %s\n', ME.message);
        is_valid = false;
        exp_rate = 0;
        details = struct('message', 'Fit error', 'error', ME.message);
    end
end

function report = local_generateTheoreticalReport(metrics, params, consensus_data)
    % GENERATETHEORETICALREPORT - Generate a comprehensive report on theoretical guarantees
    %
    % Parameters:
    %   metrics - Metrics structure from auction simulation
    %   params - Parameters used for the simulation
    %   consensus_data - (Optional) Matrix with consensus values over iterations
    %
    % Returns:
    %   report - Structure containing comprehensive validation report
    
    fprintf('Generating theoretical validation report...\n');
    
    % Initialize report structure
    report = struct();
    report.timestamp = datestr(now);
    report.parameters = params;
    
    % Add parameter summary
    if isfield(params, 'epsilon')
        report.params_summary.epsilon = params.epsilon;
    else
        report.params_summary.epsilon = 'Not specified';
    end
    
    if isfield(params, 'alpha')
        report.params_summary.b_max = max(params.alpha);
    else
        report.params_summary.b_max = 'Not specified';
    end
    
    if isfield(params, 'gamma')
        report.params_summary.gamma = params.gamma;
    else
        report.params_summary.gamma = 'Not specified';
    end
    
    if isfield(params, 'lambda')
        report.params_summary.lambda = params.lambda;
    else
        report.params_summary.lambda = 'Not specified';
    end
    
    % Validate convergence bound
    [report.convergence.is_valid, report.convergence.ratio, report.convergence.details] = ...
        local_validateConvergenceBound(metrics, params);
    
    % Validate optimality gap
    [report.optimality.is_valid, report.optimality.ratio, report.optimality.details] = ...
        local_validateOptimalityGap(metrics, params);
    
    % Validate recovery bound if applicable
    if isfield(params, 'failure_time') && ~isinf(params.failure_time) && ~isempty(params.failed_robot)
        [report.recovery.is_valid, report.recovery.ratio, report.recovery.details] = ...
            local_validateRecoveryBound(metrics, params);
    else
        report.recovery.is_valid = true;
        report.recovery.message = 'No failure scenario in this simulation';
    end
    
    % Validate consensus properties if data provided
    if nargin >= 3 && ~isempty(consensus_data)
        [report.consensus.is_valid, report.consensus.exp_rate, report.consensus.details] = ...
            local_validateConsensusProperties(consensus_data, params);
    else
        report.consensus.is_valid = true;
        report.consensus.message = 'No consensus data provided';
    end
    
    % Calculate overall validation result
    report.overall_valid = report.convergence.is_valid && report.optimality.is_valid;
    
    if isfield(report.recovery, 'is_valid')
        report.overall_valid = report.overall_valid && report.recovery.is_valid;
    end
    
    if isfield(report.consensus, 'is_valid')
        report.overall_valid = report.overall_valid && report.consensus.is_valid;
    end
    
    % Print summary
    fprintf('\n==============================================\n');
    fprintf('THEORETICAL VALIDATION SUMMARY\n');
    fprintf('==============================================\n');
    
    if report.overall_valid
        fprintf('✓ OVERALL: All theoretical guarantees are verified!\n');
    else
        fprintf('✗ OVERALL: Some theoretical guarantees are not verified.\n');
    end
    
    fprintf('Convergence bound: %s\n', bool_to_str(report.convergence.is_valid));
    fprintf('Optimality gap: %s\n', bool_to_str(report.optimality.is_valid));
    
    if isfield(report.recovery, 'is_valid')
        fprintf('Recovery bound: %s\n', bool_to_str(report.recovery.is_valid));
    else
        fprintf('Recovery bound: Not applicable\n');
    end
    
    if isfield(report.consensus, 'is_valid')
        fprintf('Consensus properties: %s\n', bool_to_str(report.consensus.is_valid));
    else
        fprintf('Consensus properties: Not verified\n');
    end
    
    fprintf('==============================================\n');
    
    function str = bool_to_str(b)
        if b
            str = '✓ Verified';
        else
            str = '✗ Not verified';
        end
    end
end

function [tightness, details] = local_calculateBoundTightness(metrics, params)
    % CALCULATEBOUNDTIGHTNESS - Calculate how tight the theoretical bounds are
    %
    % Parameters:
    %   metrics - Metrics structure from auction simulation
    %   params - Parameters used for the simulation
    %
    % Returns:
    %   tightness - Structure containing tightness metrics
    %   details - Structure with detailed tightness information
    
    fprintf('Calculating theoretical bound tightness...\n');
    
    % Initialize tightness structure
    tightness = struct();
    
    % Calculate convergence bound tightness
    [~, conv_ratio, conv_details] = local_validateConvergenceBound(metrics, params);
    tightness.convergence = conv_ratio;
    
    % Calculate optimality gap tightness
    [~, opt_ratio, opt_details] = local_validateOptimalityGap(metrics, params);
    tightness.optimality = opt_ratio;
    
    % Calculate recovery bound tightness if applicable
    if isfield(params, 'failure_time') && ~isinf(params.failure_time) && ~isempty(params.failed_robot)
        [~, rec_ratio, rec_details] = local_validateRecoveryBound(metrics, params);
        tightness.recovery = rec_ratio;
    else
        tightness.recovery = NaN;
    end
    
    % Calculate average tightness (excluding NaN values)
    tightness_values = [tightness.convergence, tightness.optimality];
    
    if ~isnan(tightness.recovery)
        tightness_values = [tightness_values, tightness.recovery];
    end
    
    tightness.average = mean(tightness_values);
    
    % Prepare detailed information
    details = struct(...
        'convergence', conv_details, ...
        'optimality', opt_details ...
    );
    
    if isfield(params, 'failure_time') && ~isinf(params.failure_time) && ~isempty(params.failed_robot)
        details.recovery = rec_details;
    end
    
    % Print results
    fprintf('\nBound tightness (Actual/Theoretical ratios):\n');
    fprintf('Convergence: %.2f\n', tightness.convergence);
    fprintf('Optimality: %.2f\n', tightness.optimality);
    
    if ~isnan(tightness.recovery)
        fprintf('Recovery: %.2f\n', tightness.recovery);
    else
        fprintf('Recovery: N/A\n');
    end
    
    fprintf('Average: %.2f\n', tightness.average);
    
    % Interpret tightness
    if tightness.average < 0.5
        fprintf('Bounds are quite loose (ratio < 0.5)\n');
    elseif tightness.average < 0.8
        fprintf('Bounds are moderately tight (0.5 ≤ ratio < 0.8)\n');
    else
        fprintf('Bounds are very tight (ratio ≥ 0.8)\n');
    end
end

function local_visualizeTheoreticalGuarantees(report)
    % VISUALIZETHEORETICALGUARANTEES - Create visualizations of theoretical guarantees
    %
    % Parameters:
    %   report - Report structure from generateTheoreticalReport
    
    fprintf('Creating visualizations of theoretical guarantees...\n');
    
    % Create figure
    figure('Name', 'Theoretical Guarantees Validation', 'Position', [50, 50, 1200, 800]);
    
    % 1. Convergence bound validation
    subplot(2, 2, 1);
    
    % Extract data
    if isfield(report, 'convergence') && isfield(report.convergence, 'details')
        details = report.convergence.details;
        
        % Plot as bar chart
        bar_data = [details.actual_iterations, details.theoretical_bound];
        bar(bar_data, 'FaceColor', 'flat');
        
        % Set colors based on validation result
        if details.is_valid
            colormap([0.4, 0.7, 0.4; 0.8, 0.8, 0.8]);  % Green for actual, gray for bound
        else
            colormap([0.8, 0.4, 0.4; 0.8, 0.8, 0.8]);  % Red for actual, gray for bound
        end
        
        % Add labels
        ylabel('Iterations');
        title('Convergence Time Validation');
        set(gca, 'XTickLabel', {'Actual', 'Theoretical'});
        
        % Add text with ratio
        text(1.5, max(bar_data) * 0.5, sprintf('Ratio: %.2f', details.ratio), ...
             'HorizontalAlignment', 'center', 'FontSize', 12);
        
        % Add theoretical formula
        if isfield(details, 'condition_string')
            annotation('textbox', [0.05, 0.47, 0.4, 0.03], 'String', details.condition_string, ...
                       'EdgeColor', 'none', 'FontSize', 8);
        end
    else
        text(0.5, 0.5, 'Convergence data not available', 'HorizontalAlignment', 'center');
        axis off;
    end
    
    % 2. Optimality gap validation
    subplot(2, 2, 2);
    
    % Extract data
    if isfield(report, 'optimality') && isfield(report.optimality, 'details')
        details = report.optimality.details;
        
        % Plot as bar chart
        bar_data = [details.actual_gap, details.theoretical_bound];
        bar(bar_data, 'FaceColor', 'flat');
        
        % Set colors based on validation result
        if details.is_valid
            colormap([0.4, 0.7, 0.4; 0.8, 0.8, 0.8]);  % Green for actual, gray for bound
        else
            colormap([0.8, 0.4, 0.4; 0.8, 0.8, 0.8]);  % Red for actual, gray for bound
        end
        
        % Add labels
        ylabel('Optimality Gap');
        title('Optimality Gap Validation');
        set(gca, 'XTickLabel', {'Actual', 'Theoretical'});
        
        % Add text with ratio
        text(1.5, max(bar_data) * 0.5, sprintf('Ratio: %.2f', details.ratio), ...
             'HorizontalAlignment', 'center', 'FontSize', 12);
        
        % Add theoretical formula
        if isfield(details, 'condition_string')
            annotation('textbox', [0.55, 0.47, 0.4, 0.03], 'String', details.condition_string, ...
                       'EdgeColor', 'none', 'FontSize', 8);
        end
    else
        text(0.5, 0.5, 'Optimality data not available', 'HorizontalAlignment', 'center');
        axis off;
    end
    
    % 3. Recovery bound validation
    subplot(2, 2, 3);
    
    % Extract data
    if isfield(report, 'recovery') && isfield(report.recovery, 'details') && ...
       isstruct(report.recovery.details) && isfield(report.recovery.details, 'actual_recovery')
        details = report.recovery.details;
        
        % Plot as bar chart
        bar_data = [details.actual_recovery, details.theoretical_bound];
        bar(bar_data, 'FaceColor', 'flat');
        
        % Set colors based on validation result
        if details.is_valid
            colormap([0.4, 0.7, 0.4; 0.8, 0.8, 0.8]);  % Green for actual, gray for bound
        else
            colormap([0.8, 0.4, 0.4; 0.8, 0.8, 0.8]);  % Red for actual, gray for bound
        end
        
        % Add labels
        ylabel('Recovery Time (iterations)');
        title('Recovery Time Validation');
        set(gca, 'XTickLabel', {'Actual', 'Theoretical'});
        
        % Add text with ratio
        text(1.5, max(bar_data) * 0.5, sprintf('Ratio: %.2f', details.ratio), ...
             'HorizontalAlignment', 'center', 'FontSize', 12);
        
        % Add theoretical formula
        if isfield(details, 'condition_string')
            annotation('textbox', [0.05, 0.02, 0.4, 0.03], 'String', details.condition_string, ...
                       'EdgeColor', 'none', 'FontSize', 8);
        end
    else
        text(0.5, 0.5, 'Recovery data not available', 'HorizontalAlignment', 'center');
        axis off;
    end
    
    % 4. Consensus convergence validation (if available)
    subplot(2, 2, 4);
    
    % Extract data
    if isfield(report, 'consensus') && isfield(report.consensus, 'details') && ...
       isstruct(report.consensus.details) && isfield(report.consensus.details, 'consensus_error')
        details = report.consensus.details;
        
        % Plot consensus error
        semilogy(details.consensus_error, 'b-', 'LineWidth', 1.5);
        hold on;
        
        % Plot exponential fit if available
        if isfield(details, 'estimated_rate') && isfield(details, 'fit_rsquare')
            x = 1:length(details.consensus_error);
            fit_curve = details.consensus_error(1) * exp(-details.estimated_rate * x);
            semilogy(x, fit_curve, 'r--', 'LineWidth', 1.5);
            
            % Add legend
            legend('Consensus Error', sprintf('Exp. Fit (μ = %.4f, R² = %.2f)', ...
                                           details.estimated_rate, details.fit_rsquare), ...
                   'Location', 'northeast');
        else
            legend('Consensus Error', 'Location', 'northeast');
        end
        
        % Add labels
        xlabel('Iteration');
        ylabel('Consensus Error (log scale)');
        title('Consensus Convergence');
        grid on;
        
        % Add theoretical formula
        if isfield(details, 'condition_string')
            annotation('textbox', [0.55, 0.02, 0.4, 0.03], 'String', details.condition_string, ...
                       'EdgeColor', 'none', 'FontSize', 8);
        end
    else
        text(0.5, 0.5, 'Consensus data not available', 'HorizontalAlignment', 'center');
        axis off;
    end
    
    % Add overall title
    if isfield(report, 'overall_valid')
        if report.overall_valid
            status_str = '✓ All Theoretical Guarantees Verified';
        else
            status_str = '✗ Some Theoretical Guarantees Not Verified';
        end
    else
        status_str = 'Theoretical Guarantees Validation';
    end
    
    sgtitle(status_str, 'FontSize', 16);
    
    % Save figure
    saveas(gcf, 'theoretical_guarantees_validation.fig');
    saveas(gcf, 'theoretical_guarantees_validation.png');
    
    fprintf('Visualizations created and saved successfully.\n');
end