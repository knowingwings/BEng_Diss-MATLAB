% statistical_analysis.m
% Statistical analysis tools for the distributed auction algorithm
% Implements ANOVA and regression modeling for parameter sensitivity analysis

function utils = statistical_analysis()
    % STATISTICAL_ANALYSIS - Returns function handles for statistical analysis
    utils = struct(...
        'runParameterSensitivityANOVA', @local_runParameterSensitivityANOVA, ...
        'parameterRegressionModeling', @local_parameterRegressionModeling, ...
        'processExperimentalResults', @local_processExperimentalResults, ...
        'calculateConfidenceIntervals', @local_calculateConfidenceIntervals, ...
        'generateFullFactorialDesign', @local_generateFullFactorialDesign, ...
        'visualizeExperimentalResults', @local_visualizeExperimentalResults, ...
        'exportAnalysisResults', @local_exportAnalysisResults ...
    );
end

function results = local_runParameterSensitivityANOVA(experiment_data)
    % RUNPARAMETERSENSITIVITYANOVA - Perform ANOVA on parameter sensitivity data
    %
    % Parameters:
    %   experiment_data - Structure containing experimental results with fields:
    %     - factors: Cell array of factor names
    %     - factor_levels: Cell array of factor levels
    %     - responses: Structure of response variables
    %       - iterations: Convergence iterations for each experimental run
    %       - makespan: Makespan for each experimental run
    %       - optimality_gap: Optimality gap for each experimental run
    %       - messages: Message count for each experimental run
    %
    % Returns:
    %   results - Structure containing ANOVA results for each response variable
    
    fprintf('Running ANOVA analysis for parameter sensitivity...\n');
    
    % Extract response variables
    response_names = fieldnames(experiment_data.responses);
    
    % Initialize results structure
    results = struct();
    results.anova_tables = struct();
    results.significant_factors = struct();
    results.interaction_effects = struct();
    
    % Get factor information
    factors = experiment_data.factors;
    
    % Loop through each response variable
    for i = 1:length(response_names)
        response_name = response_names{i};
        fprintf('  Analyzing response variable: %s\n', response_name);
        
        % Extract response data
        y = experiment_data.responses.(response_name);
        
        % Convert factor matrix to categorical arrays for ANOVA
        X = experiment_data.factor_matrix;
        X_categorical = cell(1, size(X, 2));
        
        for j = 1:size(X, 2)
            % Convert numeric factors to categorical
            X_categorical{j} = categorical(X(:, j));
        end
        
        % Construct formula string for ANOVA
        formula = [response_name ' ~ '];
        
        % Add main effects
        for j = 1:length(factors)
            if j > 1
                formula = [formula ' + '];
            end
            formula = [formula factors{j}];
        end
        
        % Add two-way interactions
        for j = 1:length(factors)-1
            for k = j+1:length(factors)
                formula = [formula ' + ' factors{j} '*' factors{k}];
            end
        end
        
        % Run ANOVA
        try
            [p, tbl, stats] = anovan(y, X_categorical, 'model', 'full', ...
                                     'varnames', factors, 'display', 'off');
            
            % Store ANOVA table
            results.anova_tables.(response_name) = tbl;
            
            % Extract significant factors (p < 0.05)
            sig_idx = find(p < 0.05);
            sig_factors = cell(length(sig_idx), 1);
            sig_p_values = zeros(length(sig_idx), 1);
            
            for j = 1:length(sig_idx)
                if sig_idx(j) <= length(factors)
                    % Main effect
                    sig_factors{j} = factors{sig_idx(j)};
                else
                    % Interaction effect
                    interaction_idx = sig_idx(j) - length(factors);
                    interaction_term = tbl{sig_idx(j), 1};
                    sig_factors{j} = interaction_term;
                end
                sig_p_values(j) = p(sig_idx(j));
            end
            
            % Store significant factors
            results.significant_factors.(response_name).factors = sig_factors;
            results.significant_factors.(response_name).p_values = sig_p_values;
            
            % Calculate interaction plots if significant interactions exist
            if any(sig_idx > length(factors))
                fprintf('    Significant interactions detected. Generating interaction plots.\n');
                results.interaction_effects.(response_name) = stats;
            end
            
            fprintf('    ANOVA completed successfully. Found %d significant factors.\n', length(sig_idx));
        catch ME
            fprintf('    Error running ANOVA: %s\n', ME.message);
            results.anova_tables.(response_name) = [];
            results.significant_factors.(response_name).factors = {};
            results.significant_factors.(response_name).p_values = [];
        end
    end
    
    fprintf('ANOVA analysis completed.\n');
end

function models = local_parameterRegressionModeling(experiment_data)
    % PARAMETERREGRESSIONMODELING - Perform regression modeling on parameter sensitivity data
    %
    % Parameters:
    %   experiment_data - Structure containing experimental results
    %
    % Returns:
    %   models - Structure containing regression models for each response variable
    
    fprintf('Running regression modeling for parameter sensitivity...\n');
    
    % Extract response variables
    response_names = fieldnames(experiment_data.responses);
    
    % Initialize models structure
    models = struct();
    
    % Get factor information
    factors = experiment_data.factors;
    X = experiment_data.factor_matrix;
    
    % Loop through each response variable
    for i = 1:length(response_names)
        response_name = response_names{i};
        fprintf('  Building regression model for: %s\n', response_name);
        
        % Extract response data
        y = experiment_data.responses.(response_name);
        
        % Try different model types and select the best one
        models.(response_name) = struct();
        
        % 1. Linear model
        try
            mdl_linear = fitlm(X, y, 'linear');
            models.(response_name).linear = mdl_linear;
            fprintf('    Linear model R²: %.4f\n', mdl_linear.Rsquared.Ordinary);
        catch ME
            fprintf('    Error fitting linear model: %s\n', ME.message);
            models.(response_name).linear = [];
        end
        
        % 2. Interactions model
        try
            mdl_interactions = fitlm(X, y, 'interactions');
            models.(response_name).interactions = mdl_interactions;
            fprintf('    Interactions model R²: %.4f\n', mdl_interactions.Rsquared.Ordinary);
        catch ME
            fprintf('    Error fitting interactions model: %s\n', ME.message);
            models.(response_name).interactions = [];
        end
        
        % 3. Quadratic model (if enough data points)
        if size(X, 1) > 3 * size(X, 2)^2
            try
                mdl_quadratic = fitlm(X, y, 'quadratic');
                models.(response_name).quadratic = mdl_quadratic;
                fprintf('    Quadratic model R²: %.4f\n', mdl_quadratic.Rsquared.Ordinary);
            catch ME
                fprintf('    Error fitting quadratic model: %s\n', ME.message);
                models.(response_name).quadratic = [];
            end
        else
            fprintf('    Insufficient data for quadratic model.\n');
            models.(response_name).quadratic = [];
        end
        
        % 4. Stepwise regression to find optimal model
        try
            mdl_stepwise = stepwiselm(X, y, 'Upper', 'quadratic', 'Criterion', 'aic');
            models.(response_name).stepwise = mdl_stepwise;
            fprintf('    Stepwise model R²: %.4f\n', mdl_stepwise.Rsquared.Ordinary);
            
            % Get significant terms
            terms = mdl_stepwise.CoefficientNames;
            p_values = mdl_stepwise.Coefficients.pValue;
            significant_terms = {};
            significant_p = [];
            
            for j = 2:length(terms)  % Skip intercept
                if p_values(j) < 0.05
                    significant_terms{end+1} = terms{j};
                    significant_p(end+1) = p_values(j);
                end
            end
            
            models.(response_name).significant_terms = significant_terms;
            models.(response_name).significant_p = significant_p;
            
            fprintf('    Stepwise model found %d significant terms.\n', length(significant_terms));
        catch ME
            fprintf('    Error fitting stepwise model: %s\n', ME.message);
            models.(response_name).stepwise = [];
            models.(response_name).significant_terms = {};
            models.(response_name).significant_p = [];
        end
        
        % Select best model based on adjusted R²
        r_squared = [0, 0, 0, 0];  % [linear, interactions, quadratic, stepwise]
        
        if ~isempty(models.(response_name).linear)
            r_squared(1) = models.(response_name).linear.Rsquared.Adjusted;
        end
        
        if ~isempty(models.(response_name).interactions)
            r_squared(2) = models.(response_name).interactions.Rsquared.Adjusted;
        end
        
        if ~isempty(models.(response_name).quadratic)
            r_squared(3) = models.(response_name).quadratic.Rsquared.Adjusted;
        end
        
        if ~isempty(models.(response_name).stepwise)
            r_squared(4) = models.(response_name).stepwise.Rsquared.Adjusted;
        end
        
        [max_r2, best_idx] = max(r_squared);
        model_types = {'linear', 'interactions', 'quadratic', 'stepwise'};
        
        if max_r2 > 0
            models.(response_name).best_model = model_types{best_idx};
            fprintf('    Best model: %s (Adjusted R²: %.4f)\n', model_types{best_idx}, max_r2);
        else
            models.(response_name).best_model = 'none';
            fprintf('    No satisfactory model found.\n');
        end
    end
    
    fprintf('Regression modeling completed.\n');
end

function processed_data = local_processExperimentalResults(raw_results)
    % PROCESSEXPERIMENTALRESULTS - Process raw experimental results to prepare for analysis
    %
    % Parameters:
    %   raw_results - Cell array of experimental results, each cell containing:
    %     - metrics: Metrics from one experimental run
    %     - params: Parameters used for the run
    %
    % Returns:
    %   processed_data - Structure ready for statistical analysis
    
    fprintf('Processing experimental results...\n');
    
    % Determine factor names and levels from first experiment
    first_params = raw_results{1}.params;
    param_names = fieldnames(first_params);
    
    % Determine which parameters were varied
    factor_values = cell(length(raw_results), length(param_names));
    
    % Extract all parameter values
    for i = 1:length(raw_results)
        for j = 1:length(param_names)
            param_name = param_names{j};
            if isfield(raw_results{i}.params, param_name)
                param_value = raw_results{i}.params.(param_name);
                
                % Convert to numeric if possible
                if isnumeric(param_value)
                    factor_values{i, j} = param_value;
                elseif islogical(param_value)
                    factor_values{i, j} = double(param_value);
                else
                    factor_values{i, j} = param_value;
                end
            else
                factor_values{i, j} = NaN;
            end
        end
    end
    
    % Determine which factors were varied
    varied_factors = false(1, length(param_names));
    factor_levels = cell(1, length(param_names));
    
    for j = 1:length(param_names)
        % Extract unique values for this parameter
        if all(cellfun(@isnumeric, factor_values(:, j)))
            % Handle array parameters by extracting a representative value
            unique_values = [];
            for i = 1:size(factor_values, 1)
                if isnumeric(factor_values{i, j}) && ~isempty(factor_values{i, j})
                    if numel(factor_values{i, j}) > 1
                        % For arrays, use the mean or first element as representative
                        unique_values = [unique_values; mean(factor_values{i, j})];
                    else
                        unique_values = [unique_values; factor_values{i, j}];
                    end
                end
            end
            unique_values = unique(unique_values);
        else
            unique_values = unique(factor_values(:, j));
        end
        
        % If more than one unique value, this parameter was varied
        if length(unique_values) > 1
            varied_factors(j) = true;
            factor_levels{j} = unique_values;
        end
    end
    
    % Keep only varied factors
    factors = param_names(varied_factors);
    factor_levels = factor_levels(varied_factors);
    
    % Extract factor matrix
    factor_matrix = zeros(length(raw_results), sum(varied_factors));
    col_idx = 0;
    
    for j = 1:length(param_names)
        if varied_factors(j)
            col_idx = col_idx + 1;
            
            for i = 1:length(raw_results)
                if isnumeric(factor_values{i, j})
                    % FIX: Handle array parameters by taking their mean or first element
                    if numel(factor_values{i, j}) > 1
                        factor_matrix(i, col_idx) = mean(factor_values{i, j});
                        % Alternative: factor_matrix(i, col_idx) = factor_values{i, j}(1);
                    else
                        factor_matrix(i, col_idx) = factor_values{i, j};
                    end
                else
                    % If factor is categorical, convert to numeric levels
                    level_idx = find(strcmp(factor_values{i, j}, factor_levels{col_idx}));
                    factor_matrix(i, col_idx) = level_idx;
                end
            end
        end
    end
    
    % Extract response variables
    % Determine which metrics to analyze
    first_metrics = raw_results{1}.metrics;
    metric_names = fieldnames(first_metrics);
    
    % Initialize response data
    responses = struct();
    
    for j = 1:length(metric_names)
        metric_name = metric_names{j};
        responses.(metric_name) = zeros(length(raw_results), 1);
        
        for i = 1:length(raw_results)
            if isfield(raw_results{i}.metrics, metric_name)
                metric_value = raw_results{i}.metrics.(metric_name);
                
                % Convert to numeric if necessary
                if isnumeric(metric_value) && isscalar(metric_value)
                    responses.(metric_name)(i) = metric_value;
                else
                    responses.(metric_name)(i) = NaN;
                end
            else
                responses.(metric_name)(i) = NaN;
            end
        end
    end
    
    % Create processed data structure
    processed_data = struct();
    processed_data.factors = factors;
    processed_data.factor_levels = factor_levels;
    processed_data.factor_matrix = factor_matrix;
    processed_data.responses = responses;
    processed_data.raw_results = raw_results;
    
    fprintf('Processed %d experimental runs with %d factors and %d response variables.\n', ...
            length(raw_results), length(factors), length(metric_names));
end

function ci_results = local_calculateConfidenceIntervals(experiment_data, confidence_level)
    % CALCULATECONFIDENCEINTERVALS - Calculate confidence intervals for response variables
    %
    % Parameters:
    %   experiment_data - Structure containing experimental data
    %   confidence_level - Confidence level (default: 0.95 for 95% CI)
    %
    % Returns:
    %   ci_results - Structure containing confidence intervals for each response
    
    if nargin < 2
        confidence_level = 0.95;
    end
    
    fprintf('Calculating %.0f%% confidence intervals...\n', confidence_level * 100);
    
    % Extract response variables
    response_names = fieldnames(experiment_data.responses);
    
    % Initialize results structure
    ci_results = struct();
    
    % Calculate confidence intervals for each response
    for i = 1:length(response_names)
        response_name = response_names{i};
        fprintf('  Calculating CIs for: %s\n', response_name);
        
        % Extract response data
        y = experiment_data.responses.(response_name);
        
        % Get factor matrix
        X = experiment_data.factor_matrix;
        
        % Calculate overall mean and confidence interval
        [mean_val, ci] = local_calculate_mean_ci(y, confidence_level);
        
        ci_results.(response_name).overall.mean = mean_val;
        ci_results.(response_name).overall.lower = ci(1);
        ci_results.(response_name).overall.upper = ci(2);
        
        % Calculate confidence intervals for each factor level
        for j = 1:size(X, 2)
            factor_name = experiment_data.factors{j};
            levels = unique(X(:, j));
            
            ci_results.(response_name).(factor_name) = struct();
            ci_results.(response_name).(factor_name).levels = levels;
            ci_results.(response_name).(factor_name).means = zeros(length(levels), 1);
            ci_results.(response_name).(factor_name).lower = zeros(length(levels), 1);
            ci_results.(response_name).(factor_name).upper = zeros(length(levels), 1);
            
            for k = 1:length(levels)
                level = levels(k);
                
                % Get data for this factor level
                idx = (X(:, j) == level);
                if sum(idx) > 1
                    [mean_val, ci] = local_calculate_mean_ci(y(idx), confidence_level);
                    
                    ci_results.(response_name).(factor_name).means(k) = mean_val;
                    ci_results.(response_name).(factor_name).lower(k) = ci(1);
                    ci_results.(response_name).(factor_name).upper(k) = ci(2);
                else
                    ci_results.(response_name).(factor_name).means(k) = NaN;
                    ci_results.(response_name).(factor_name).lower(k) = NaN;
                    ci_results.(response_name).(factor_name).upper(k) = NaN;
                end
            end
        end
    end
    
    fprintf('Confidence interval calculations completed.\n');
end

function [mean_val, ci] = local_calculate_mean_ci(data, confidence_level)
    % Helper function to calculate mean and confidence interval
    
    % Remove NaN values
    data = data(~isnan(data));
    
    % Calculate mean
    mean_val = mean(data);
    
    % Calculate standard error of the mean
    n = length(data);
    se = std(data) / sqrt(n);
    
    % Calculate t-value for desired confidence level
    alpha = 1 - confidence_level;
    t_crit = tinv(1 - alpha/2, n-1);
    
    % Calculate confidence interval
    margin = t_crit * se;
    ci = [mean_val - margin, mean_val + margin];
end

function design = local_generateFullFactorialDesign(factors, include_center_points)
    % GENERATEFULLFACTORIALDESIGN - Generate a full factorial experimental design
    %
    % Parameters:
    %   factors - Cell array with each cell containing a vector of factor levels
    %   include_center_points - Whether to include center points (default: false)
    %
    % Returns:
    %   design - Matrix with each row representing one experimental run
    
    if nargin < 2
        include_center_points = false;
    end
    
    fprintf('Generating full factorial design...\n');
    
    % Calculate number of factors
    num_factors = length(factors);
    
    % Calculate total number of runs
    num_runs = 1;
    for i = 1:num_factors
        num_runs = num_runs * length(factors{i});
    end
    
    fprintf('  Full factorial design with %d factors will have %d runs.\n', num_factors, num_runs);
    
    % Initialize design matrix
    design = zeros(num_runs, num_factors);
    
    % Generate full factorial design
    idx = 1;
    indices = ones(1, num_factors);
    
    while idx <= num_runs
        % Record current combination
        for j = 1:num_factors
            design(idx, j) = factors{j}(indices(j));
        end
        
        % Increment indices
        j = 1;
        while j <= num_factors
            indices(j) = indices(j) + 1;
            if indices(j) <= length(factors{j})
                break;
            else
                indices(j) = 1;
                j = j + 1;
            end
        end
        
        idx = idx + 1;
    end
    
    % Add center points if requested
    if include_center_points
        % Calculate center point for each factor
        center_point = zeros(1, num_factors);
        
        for i = 1:num_factors
            center_point(i) = mean(factors{i});
        end
        
        % Add center points to design
        design = [design; repmat(center_point, 3, 1)]; % Add 3 center points
        
        fprintf('  Added 3 center points for a total of %d runs.\n', size(design, 1));
    end
    
    fprintf('Full factorial design generated successfully.\n');
end

function local_visualizeExperimentalResults(experiment_data, anova_results, regression_models, ci_results)
    % VISUALIZEEXPERIMENTALRESULTS - Generate visualizations for experimental results
    %
    % Parameters:
    %   experiment_data - Structure containing experimental data
    %   anova_results - Structure containing ANOVA results
    %   regression_models - Structure containing regression models
    %   ci_results - Structure containing confidence intervals
    
    fprintf('Generating visualizations for experimental results...\n');
    
    % Extract response variables
    response_names = fieldnames(experiment_data.responses);
    
    % Loop through each response variable
    for i = 1:length(response_names)
        response_name = response_names{i};
        fprintf('  Creating plots for: %s\n', response_name);
        
        % Create figure for this response variable
        figure('Name', ['Analysis of ' response_name], 'Position', [100, 100, 1200, 800]);
        
        % 1. Main effects plot
        subplot(2, 2, 1);
        local_plot_main_effects(experiment_data, response_name, ci_results);
        title(['Main Effects for ' strrep(response_name, '_', ' ')]);
        
        % 2. Interaction effects plot (if available)
        subplot(2, 2, 2);
        if isfield(anova_results, 'interaction_effects') && ...
           isfield(anova_results.interaction_effects, response_name)
            local_plot_interaction_effects(experiment_data, response_name, anova_results);
            title(['Interaction Effects for ' strrep(response_name, '_', ' ')]);
        else
            text(0.5, 0.5, 'No significant interactions', 'HorizontalAlignment', 'center');
            title('Interaction Effects');
            axis off;
        end
        
        % 3. Residual plots (if regression model is available)
        subplot(2, 2, 3);
        if isfield(regression_models, response_name) && ...
           isfield(regression_models.(response_name), 'best_model') && ...
           ~strcmp(regression_models.(response_name).best_model, 'none')
            best_model_type = regression_models.(response_name).best_model;
            best_model = regression_models.(response_name).(best_model_type);
            
            % Plot residuals
            residuals = best_model.Residuals.Raw;
            predicted = best_model.Fitted;
            
            scatter(predicted, residuals, 50, 'filled');
            hold on;
            plot([min(predicted), max(predicted)], [0, 0], 'r--', 'LineWidth', 1.5);
            
            xlabel('Predicted Value');
            ylabel('Residual');
            title(['Residuals for ' strrep(response_name, '_', ' ')]);
            grid on;
        else
            text(0.5, 0.5, 'No regression model available', 'HorizontalAlignment', 'center');
            title('Residual Plot');
            axis off;
        end
        
        % 4. Normal probability plot of residuals (if regression model is available)
        subplot(2, 2, 4);
        if isfield(regression_models, response_name) && ...
           isfield(regression_models.(response_name), 'best_model') && ...
           ~strcmp(regression_models.(response_name).best_model, 'none')
            best_model_type = regression_models.(response_name).best_model;
            best_model = regression_models.(response_name).(best_model_type);
            
            % Normal probability plot
            residuals = best_model.Residuals.Raw;
            normplot(residuals);
            title(['Normal Probability Plot for ' strrep(response_name, '_', ' ')]);
        else
            text(0.5, 0.5, 'No regression model available', 'HorizontalAlignment', 'center');
            title('Normal Probability Plot');
            axis off;
        end
        
        % Add overall title
        sgtitle(['Statistical Analysis of ' strrep(response_name, '_', ' ')]);
        
        % Save figure
        saveas(gcf, [response_name '_analysis.png']);
        saveas(gcf, [response_name '_analysis.fig']);
    end
    
    fprintf('Visualizations generated and saved.\n');
end

function local_plot_main_effects(experiment_data, response_name, ci_results)
    % Helper function to plot main effects
    
    % Get factors
    factors = experiment_data.factors;
    
    % Number of factors
    num_factors = length(factors);
    
    % Calculate grid dimensions for subplots
    rows = ceil(sqrt(num_factors));
    cols = ceil(num_factors / rows);
    
    % Create main effects plots
    for i = 1:num_factors
        factor_name = factors{i};
        
        % Create subplot
        subplot(rows, cols, i);
        
        % Get factor levels and means
        levels = ci_results.(response_name).(factor_name).levels;
        means = ci_results.(response_name).(factor_name).means;
        lower = ci_results.(response_name).(factor_name).lower;
        upper = ci_results.(response_name).(factor_name).upper;
        
        % Sort by levels
        [levels, sort_idx] = sort(levels);
        means = means(sort_idx);
        lower = lower(sort_idx);
        upper = upper(sort_idx);
        
        % Plot means with error bars
        errorbar(levels, means, means - lower, upper - means, 'o-', 'LineWidth', 1.5, 'MarkerSize', 6);
        
        % Add labels
        xlabel(factor_name);
        ylabel(response_name);
        title(['Effect of ' factor_name]);
        grid on;
    end
end

function local_plot_interaction_effects(experiment_data, response_name, anova_results)
    % Helper function to plot interaction effects
    
    % Get factors
    factors = experiment_data.factors;
    
    % Get significant interactions
    if isfield(anova_results, 'significant_factors') && ...
       isfield(anova_results.significant_factors, response_name)
        sig_factors = anova_results.significant_factors.(response_name).factors;
        
        % Find interaction terms
        interaction_idx = find(cellfun(@(x) contains(x, '*'), sig_factors));
        
        if ~isempty(interaction_idx)
            % Get most significant interaction
            interaction_term = sig_factors{interaction_idx(1)};
            
            % Extract factor names
            parts = strsplit(interaction_term, '*');
            factor1 = strtrim(parts{1});
            factor2 = strtrim(parts{2});
            
            % Get factor indices
            idx1 = find(strcmp(factors, factor1));
            idx2 = find(strcmp(factors, factor2));
            
            if ~isempty(idx1) && ~isempty(idx2)
                % Get factor matrix
                X = experiment_data.factor_matrix;
                
                % Get unique levels for each factor
                levels1 = unique(X(:, idx1));
                levels2 = unique(X(:, idx2));
                
                % Initialize interaction matrix
                means = zeros(length(levels1), length(levels2));
                
                % Calculate mean response for each combination
                for i = 1:length(levels1)
                    for j = 1:length(levels2)
                        idx = (X(:, idx1) == levels1(i)) & (X(:, idx2) == levels2(j));
                        if any(idx)
                            means(i, j) = mean(experiment_data.responses.(response_name)(idx));
                        else
                            means(i, j) = NaN;
                        end
                    end
                end
                
                % Plot interaction
                if length(levels2) <= 7  % Use line plot for fewer levels
                    for i = 1:length(levels2)
                        plot(levels1, means(:, i), 'o-', 'LineWidth', 1.5, 'DisplayName', sprintf('%s = %.2f', factor2, levels2(i)));
                        hold on;
                    end
                    
                    xlabel(factor1);
                    ylabel(response_name);
                    legend('show');
                    title(['Interaction: ' factor1 ' × ' factor2]);
                    grid on;
                else  % Use heatmap for many levels
                    h = heatmap(levels2, levels1, means);
                    h.Title = ['Interaction: ' factor1 ' × ' factor2];
                    h.XLabel = factor2;
                    h.YLabel = factor1;
                    h.ColorbarTitle.String = response_name;
                end
            else
                text(0.5, 0.5, 'Error identifying interaction factors', 'HorizontalAlignment', 'center');
                axis off;
            end
        else
            text(0.5, 0.5, 'No significant interactions', 'HorizontalAlignment', 'center');
            axis off;
        end
    else
        text(0.5, 0.5, 'No significant interactions', 'HorizontalAlignment', 'center');
        axis off;
    end
end

function local_exportAnalysisResults(experiment_data, anova_results, regression_models, ci_results, filename)
    % EXPORTANALYSISRESULTS - Export statistical analysis results to MAT file
    %
    % Parameters:
    %   experiment_data - Structure containing experimental data
    %   anova_results - Structure containing ANOVA results
    %   regression_models - Structure containing regression models
    %   ci_results - Structure containing confidence intervals
    %   filename - Name of the output file (default: 'statistical_analysis_results.mat')
    
    if nargin < 5
        filename = 'statistical_analysis_results.mat';
    end
    
    fprintf('Exporting statistical analysis results to %s...\n', filename);
    
    % Create results structure
    results = struct();
    results.experiment_data = experiment_data;
    results.anova_results = anova_results;
    results.regression_models = regression_models;
    results.ci_results = ci_results;
    results.timestamp = datestr(now);
    
    % Save results
    save(filename, 'results');
    
    fprintf('Results exported successfully.\n');
end