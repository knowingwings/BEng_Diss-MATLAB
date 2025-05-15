function utils = statistical_analysis()
    % STATISTICAL_ANALYSIS - Utilities for statistical analysis of results
    
    % Create utils structure
    utils = struct();
    
    % Add function handles
    utils.calculateMean = @calculateMean;
    utils.calculateStd = @calculateStd;
    utils.calculateCI = @calculateCI;
    utils.runANOVA = @runANOVA;
    utils.runTTest = @runTTest;
    utils.checkNormality = @checkNormality;
    utils.calculateEffectSize = @calculateEffectSize;
    utils.createANOVATable = @createANOVATable;
    utils.plotBoxPlot = @plotBoxPlot;
    utils.plotBarPlot = @plotBarPlot;
    utils.plotErrorBars = @plotErrorBars;
    
    function mean_val = calculateMean(data, dim)
        % Calculate mean along dimension dim
        if nargin < 2
            dim = 1;
        end
        mean_val = mean(data, dim, 'omitnan');
    end
    
    function std_val = calculateStd(data, dim)
        % Calculate standard deviation along dimension dim
        if nargin < 2
            dim = 1;
        end
        std_val = std(data, 0, dim, 'omitnan');
    end
    
    function [ci_low, ci_high] = calculateCI(data, alpha, dim)
        % Calculate confidence interval along dimension dim
        if nargin < 2
            alpha = 0.05;
        end
        if nargin < 3
            dim = 1;
        end
        
        mean_val = mean(data, dim, 'omitnan');
        std_val = std(data, 0, dim, 'omitnan');
        n = sum(~isnan(data), dim);
        
        % Calculate t-value for the given alpha
        t_val = tinv(1 - alpha/2, n - 1);
        
        % Calculate confidence interval
        ci_low = mean_val - t_val .* std_val ./ sqrt(n);
        ci_high = mean_val + t_val .* std_val ./ sqrt(n);
    end
    
    function [p, tbl, stats] = runANOVA(data, groups)
        % Run one-way ANOVA
        [p, tbl, stats] = anova1(data, groups);
    end
    
    function [h, p, ci, stats] = runTTest(data1, data2, alpha)
        % Run t-test
        if nargin < 3
            alpha = 0.05;
        end
        
        [h, p, ci, stats] = ttest2(data1, data2, 'Alpha', alpha);
    end
    
    function [h, p] = checkNormality(data)
        % Check normality using Lilliefors test
        [h, p] = lillietest(data);
    end
    
    function d = calculateEffectSize(data1, data2)
        % Calculate Cohen's d effect size
        mean1 = mean(data1, 'omitnan');
        mean2 = mean(data2, 'omitnan');
        
        std1 = std(data1, 0, 'omitnan');
        std2 = std(data2, 0, 'omitnan');
        
        % Pooled standard deviation
        n1 = sum(~isnan(data1));
        n2 = sum(~isnan(data2));
        
        pooled_std = sqrt(((n1-1)*std1^2 + (n2-1)*std2^2) / (n1 + n2 - 2));
        
        % Cohen's d
        d = abs(mean1 - mean2) / pooled_std;
    end
    
    function tbl = createANOVATable(factors, response, data)
        % Create ANOVA table for a factorial design
        num_factors = length(factors);
        factor_levels = zeros(num_factors, 1);
        
        for i = 1:num_factors
            factor_levels(i) = length(factors{i});
        end
        
        % Create the design matrix
        [X, labels] = createDesignMatrix(factors);
        
        % Run N-way ANOVA
        [~, tbl, ~] = anovan(data, X, 'varnames', labels, 'display', 'off');
    end
    
    function [X, labels] = createDesignMatrix(factors)
        % Create design matrix for N-way ANOVA
        num_factors = length(factors);
        num_samples = length(factors{1});
        
        X = cell(num_factors, num_samples);
        labels = cell(num_factors, 1);
        
        for i = 1:num_factors
            X{i} = factors{i};
            labels{i} = sprintf('Factor%d', i);
        end
    end
    
    function h = plotBoxPlot(data, group_labels, x_label, y_label, title_text)
        % Create box plot
        h = figure;
        boxplot(data, group_labels);
        xlabel(x_label);
        ylabel(y_label);
        title(title_text);
        grid on;
    end
    
    function h = plotBarPlot(means, errors, group_labels, x_label, y_label, title_text)
        % Create bar plot with error bars
        h = figure;
        bar(means);
        hold on;
        
        % Add error bars
        x = 1:length(means);
        errorbar(x, means, errors, '.k', 'LineWidth', 1.5);
        
        % Set x-axis labels
        if nargin >= 3 && ~isempty(group_labels)
            xticks(x);
            xticklabels(group_labels);
        end
        
        % Set axis labels and title
        if nargin >= 4 && ~isempty(x_label)
            xlabel(x_label);
        end
        
        if nargin >= 5 && ~isempty(y_label)
            ylabel(y_label);
        end
        
        if nargin >= 6 && ~isempty(title_text)
            title(title_text);
        end
        
        grid on;
        hold off;
    end
    
    function h = plotErrorBars(x, y, err, line_style, marker_style, color, label)
        % Plot data with error bars
        h = figure;
        
        if nargin < 4 || isempty(line_style)
            line_style = '-';
        end
        
        if nargin < 5 || isempty(marker_style)
            marker_style = 'o';
        end
        
        if nargin < 6 || isempty(color)
            color = 'b';
        end
        
        errorbar(x, y, err, [line_style marker_style], 'Color', color, 'LineWidth', 1.5);
        
        if nargin >= 7 && ~isempty(label)
            legend(label);
        end
        
        grid on;
    end
    
    return;
end