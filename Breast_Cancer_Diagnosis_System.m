clear all; close all; clc;
fprintf('========================================================================\n');
fprintf('========================================================================\n\n');

%% ===== Configuration =====
REAL_DATA_PATH = 'C:\Users\DJ\Desktop\breast+cancer+wisconsin+diagnostic (1)\wdbc.data';
K_FOLDS = 5;
LAMBDA_GRID = [0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1];
rng(2024);

%% ===== Data Loading=====
if ~exist(REAL_DATA_PATH, 'file')
    error('Data file not found!');
end

fprintf('Loading Wisconsin Breast Cancer Dataset...\n');
data = readtable(REAL_DATA_PATH, 'FileType', 'text', 'Delimiter', ',', 'ReadVariableNames', false);
y_all = double(strcmp(data{:,2}, 'M'));
X_all_raw = data{:,3:end}; 

feature_names = {
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', ...
    'smoothness_mean', 'compactness_mean', 'concavity_mean', ...
    'concave_pts_mean', 'symmetry_mean', 'fractal_dim_mean', ...
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', ...
    'smoothness_se', 'compactness_se', 'concavity_se', ...
    'concave_pts_se', 'symmetry_se', 'fractal_dim_se', ...
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', ...
    'smoothness_worst', 'compactness_worst', 'concavity_worst', ...
    'concave_pts_worst', 'symmetry_worst', 'fractal_dim_worst'
};

y = 2 * y_all - 1; 

fprintf('Dataset: %d samples, %d features\n', size(X_all_raw, 1), size(X_all_raw, 2));
fprintf('Malignant (Positive): %d | Benign (Negative): %d\n\n', sum(y_all == 1), sum(y_all == 0));

%% ===== Train-Test Split =====
idx_pos = find(y_all == 1);
idx_neg = find(y_all == 0);
n_train_pos = floor(length(idx_pos) * 0.8);
n_train_neg = floor(length(idx_neg) * 0.8);

idx_train = [idx_pos(1:n_train_pos); idx_neg(1:n_train_neg)];
idx_test = [idx_pos(n_train_pos+1:end); idx_neg(n_train_neg+1:end)];

idx_train = idx_train(randperm(length(idx_train)));
idx_test = idx_test(randperm(length(idx_test)));

X_train_raw = X_all_raw(idx_train, :);
y_train = y(idx_train);
X_test_raw = X_all_raw(idx_test, :);
y_test = y(idx_test);

fprintf('Train: %d | Test: %d\n\n', length(idx_train), length(idx_test));

%% ===== Core Training Function  =====
function w = train_model(X, y, lambda)
        n = size(X, 2);
    d = size(X, 1);
    grad_f = @(w) -(1/n) * X * (y ./ (1 + exp(y .* (w' * X)'))) + lambda * w;
    
    alpha = 6.0; beta = 1.5; q = 1.0; p = 3.5;
    t0 = 1.0; h = 0.01; N = 9900;
    epsilon = @(t) t^(-p);
    b_func = @(t) t^q;
    
    x = randn(d, 1) * 0.01;
    z = -beta * (grad_f(x) + epsilon(t0) * x);
    
    for k = 1:N
        t = t0 + (k-1)*h;
        g = grad_f(x) + epsilon(t)*x;
        k1_x = -beta*g - z;
        k1_z = alpha/t * k1_x + b_func(t)*g;
        
        x_t = x + h/2 * k1_x; z_t = z + h/2 * k1_z; t_t = t + h/2;
        g = grad_f(x_t) + epsilon(t_t)*x_t;
        k2_x = -beta*g - z_t;
        k2_z = alpha/t_t * k2_x + b_func(t_t)*g;
        
        x_t = x + h/2 * k2_x; z_t = z + h/2 * k2_z;
        g = grad_f(x_t) + epsilon(t_t)*x_t;
        k3_x = -beta*g - z_t;
        k3_z = alpha/t_t * k3_x + b_func(t_t)*g;
        
        x_t = x + h * k3_x; z_t = z + h * k3_z; t_t = t + h;
        g = grad_f(x_t) + epsilon(t_t)*x_t;
        k4_x = -beta*g - z_t;
        k4_z = alpha/t_t * k4_x + b_func(t_t)*g;
        
        x = x + h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x);
        z = z + h/6 * (k1_z + 2*k2_z + 2*k3_z + k4_z);
    end
    w = x;
end

%% ===== Enhanced K-Fold CV =====
function [mean_metrics, fold_metrics, fold_train_metrics, fold_conf_matrices, ...
          fold_data_dist] = kfold_cv_detailed(X_raw, y, lambda, K)
    
    n = size(X_raw, 1);
    y_bin = (y + 1) / 2;
    idx_pos = find(y_bin == 1);
    idx_neg = find(y_bin == 0);
    
    fold_pos = floor(length(idx_pos) / K);
    fold_neg = floor(length(idx_neg) / K);
    
    fold_metrics = zeros(K, 5);        
    fold_train_metrics = zeros(K, 5);  
    fold_conf_matrices = cell(K, 2);   
    fold_data_dist = zeros(K, 4);      
    
    for k = 1:K
        % 1. Determine Indices
        if k < K
            val_pos = idx_pos((k-1)*fold_pos + 1 : k*fold_pos);
            val_neg = idx_neg((k-1)*fold_neg + 1 : k*fold_neg);
        else
            val_pos = idx_pos((k-1)*fold_pos + 1 : end);
            val_neg = idx_neg((k-1)*fold_neg + 1 : end);
        end
        val_idx = [val_pos; val_neg];
        train_idx = setdiff(1:n, val_idx);
        
        % 2. Get Raw Subsets
        X_fold_train_raw = X_raw(train_idx, :);
        X_fold_val_raw = X_raw(val_idx, :);
        y_fold_train = y(train_idx);
        y_fold_val = y(val_idx);
        
        mu = mean(X_fold_train_raw, 1);
        sigma = std(X_fold_train_raw, 0, 1) + 1e-8;
        
        X_fold_train_norm = (X_fold_train_raw - mu) ./ sigma;
        X_fold_val_norm = (X_fold_val_raw - mu) ./ sigma; 
        
        % 4. Add Bias
        X_train_k = [ones(length(train_idx), 1), X_fold_train_norm]';
        X_val_k = [ones(length(val_idx), 1), X_fold_val_norm]';
        
        % Store distribution info (for plotting)
        y_train_bin = (y_fold_train + 1) / 2;
        y_val_bin = (y_fold_val + 1) / 2;
        fold_data_dist(k, :) = [sum(y_train_bin == 0), sum(y_train_bin == 1), ...
                                sum(y_val_bin == 0), sum(y_val_bin == 1)];
        
        % 5. Train
        w = train_model(X_train_k, y_fold_train, lambda);
        
        % 6. Evaluate (Validation)
        y_pred_val = sign(w' * X_val_k)';
        TP_val = sum((y_pred_val == 1) & (y_fold_val == 1));
        TN_val = sum((y_pred_val == -1) & (y_fold_val == -1));
        FP_val = sum((y_pred_val == 1) & (y_fold_val == -1));
        FN_val = sum((y_pred_val == -1) & (y_fold_val == 1));
        
        precision_val = TP_val / (TP_val + FP_val + 1e-8) * 100;
        recall_val = TP_val / (TP_val + FN_val + 1e-8) * 100;
        accuracy_val = (TP_val + TN_val) / length(y_fold_val) * 100;
        specificity_val = TN_val / (TN_val + FP_val + 1e-8) * 100;
        f1_val = 2 * precision_val * recall_val / (precision_val + recall_val + 1e-8);
        
        fold_metrics(k, :) = [precision_val, recall_val, accuracy_val, specificity_val, f1_val];
        fold_conf_matrices{k, 1} = [TP_val, FN_val; FP_val, TN_val];
        
        % 7. Evaluate (Training Partition)
        y_pred_train = sign(w' * X_train_k)';
        TP_train = sum((y_pred_train == 1) & (y_fold_train == 1));
        TN_train = sum((y_pred_train == -1) & (y_fold_train == -1));
        FP_train = sum((y_pred_train == 1) & (y_fold_train == -1));
        FN_train = sum((y_pred_train == -1) & (y_fold_train == 1));
        
        precision_train = TP_train / (TP_train + FP_train + 1e-8) * 100;
        recall_train = TP_train / (TP_train + FN_train + 1e-8) * 100;
        accuracy_train = (TP_train + TN_train) / length(y_fold_train) * 100;
        specificity_train = TN_train / (TN_train + FP_train + 1e-8) * 100;
        f1_train = 2 * precision_train * recall_train / (precision_train + recall_train + 1e-8);
        
        fold_train_metrics(k, :) = [precision_train, recall_train, accuracy_train, specificity_train, f1_train];
        fold_conf_matrices{k, 2} = [TP_train, FN_train; FP_train, TN_train];
    end
    
    mean_metrics = mean(fold_metrics, 1);
end

%% ===== Grid Search =====
fprintf('========================================================================\n');
fprintf('Grid Search with %d-Fold CV...\n', K_FOLDS);
fprintf('========================================================================\n\n');

n_lambda = length(LAMBDA_GRID);
cv_results = zeros(n_lambda, 3);

for i = 1:n_lambda
    lambda = LAMBDA_GRID(i);
    fprintf('  [%d/%d] λ=%.4f ... ', i, n_lambda, lambda);
    
    [mean_metrics, ~, ~, ~, ~] = kfold_cv_detailed(X_train_raw, y_train, lambda, K_FOLDS);
    cv_results(i, :) = [mean_metrics(3), mean_metrics(2), mean_metrics(4)];
    fprintf('Acc=%.2f%%, Recall=%.2f%%, Spec=%.2f%%\n', cv_results(i, 1), cv_results(i, 2), cv_results(i, 3));
end

[~, best_idx] = max(cv_results(:, 2)); 
best_lambda = LAMBDA_GRID(best_idx);
fprintf('\nBest λ = %.4f (Max CV Recall = %.2f%%)\n\n', best_lambda, cv_results(best_idx, 2));
%% ===== Detailed K-Fold Analysis =====
fprintf('========================================================================\n');
fprintf('Detailed K-Fold Analysis for λ=%.4f\n', best_lambda);
fprintf('========================================================================\n');

[mean_metrics_best, fold_metrics_best, fold_train_metrics_best, fold_conf_matrices, ...
 fold_data_dist] = kfold_cv_detailed(X_train_raw, y_train, best_lambda, K_FOLDS);

fprintf('\nValidation Set - Fold-wise Results:\n');
fprintf('Fold  | Precision | Recall  | Accuracy | Specificity | F1-Score\n');
fprintf('------|-----------|---------|----------|-------------|----------\n');
for k = 1:K_FOLDS
    fprintf('  %d   |   %.2f%%  | %.2f%% |  %.2f%%  |   %.2f%%    | %.2f%%\n', ...
        k, fold_metrics_best(k, 1), fold_metrics_best(k, 2), fold_metrics_best(k, 3), ...
        fold_metrics_best(k, 4), fold_metrics_best(k, 5));
end
fprintf('------|-----------|---------|----------|-------------|----------\n');
fprintf(' Avg  |   %.2f%%  | %.2f%% |  %.2f%%  |   %.2f%%    | %.2f%%\n', ...
    mean_metrics_best(1), mean_metrics_best(2), mean_metrics_best(3), ...
    mean_metrics_best(4), mean_metrics_best(5));
fprintf('========================================================================\n\n');

%% ===== Final Model Training =====
fprintf('Training final model with λ=%.4f...\n', best_lambda);

mu_final = mean(X_train_raw, 1);
sigma_final = std(X_train_raw, 0, 1) + 1e-8;

X_train_norm = (X_train_raw - mu_final) ./ sigma_final;
X_train = [ones(size(X_train_norm, 1), 1), X_train_norm]'; 


w_final = train_model(X_train, y_train, best_lambda);


X_test_norm = (X_test_raw - mu_final) ./ sigma_final;
X_test = [ones(size(X_test_norm, 1), 1), X_test_norm]';

% ===== Final Model - Training Set Evaluation =====
y_pred_train_final = sign(w_final' * X_train)';
TP_train_final = sum((y_pred_train_final == 1) & (y_train == 1));
TN_train_final = sum((y_pred_train_final == -1) & (y_train == -1));
FP_train_final = sum((y_pred_train_final == 1) & (y_train == -1));
FN_train_final = sum((y_pred_train_final == -1) & (y_train == 1));

accuracy_train_final = (TP_train_final + TN_train_final) / length(y_train) * 100;
recall_train_final = TP_train_final / (TP_train_final + FN_train_final) * 100;
specificity_train_final = TN_train_final / (TN_train_final + FP_train_final) * 100;
precision_train_final = TP_train_final / (TP_train_final + FP_train_final) * 100;
f1_train_final = 2 * precision_train_final * recall_train_final / (precision_train_final + recall_train_final);

% ===== Final Model - Test Set Evaluation =====
y_pred_test = sign(w_final' * X_test)';

y_proba = (1 ./ (1 + exp(-w_final' * X_test)))';

TP = sum((y_pred_test == 1) & (y_test == 1));
TN = sum((y_pred_test == -1) & (y_test == -1));
FP = sum((y_pred_test == 1) & (y_test == -1));
FN = sum((y_pred_test == -1) & (y_test == 1));

accuracy = (TP + TN) / length(y_test) * 100;
recall = TP / (TP + FN) * 100;
specificity = TN / (TN + FP) * 100;
precision = TP / (TP + FP) * 100;
f1 = 2 * precision * recall / (precision + recall);

fprintf('\n========================================================================\n');
fprintf('Final Model - Training Set Performance:\n');
fprintf('  Accuracy:    %.2f%%\n', accuracy_train_final);
fprintf('  Precision:   %.2f%%\n', precision_train_final);
fprintf('  Recall:      %.2f%%\n', recall_train_final);
fprintf('  Specificity: %.2f%%\n', specificity_train_final);
fprintf('  F1-Score:    %.2f%%\n', f1_train_final);
fprintf('========================================================================\n\n');

fprintf('========================================================================\n');
fprintf('Final Model - Test Set Performance:\n');
fprintf('  Accuracy:    %.2f%%\n', accuracy);
fprintf('  Precision:   %.2f%%\n', precision);
fprintf('  Recall:      %.2f%%\n', recall);
fprintf('  Specificity: %.2f%%\n', specificity);
fprintf('  F1-Score:    %.2f%%\n', f1);
fprintf('========================================================================\n\n');

%% ===== Feature Importance =====
feature_imp = abs(w_final(2:end));
[sorted_imp, idx_sorted] = sort(feature_imp, 'descend');
top_features = feature_names(idx_sorted(1:15));

%% ===== Visualization Setup =====
set(0, 'DefaultAxesFontName', 'Arial', 'DefaultAxesFontSize', 11, 'DefaultLineLineWidth', 1.5);

%% ===== Figure 5: Stratified Data Partition =====
fig1 = figure('Position', [50, 50, 800, 800], 'Color', 'white');
sgtitle('Stratified Data Partition Across 5-Folds', 'FontSize', 15, 'FontWeight', 'bold');

subplot(2, 1, 1);
x_pos = 1:K_FOLDS;
bar_width = 0.35;
b1 = bar(x_pos - bar_width/2, fold_data_dist(:, 1), bar_width, 'FaceColor', [0.3 0.5 0.8], ...
    'EdgeColor', 'k', 'LineWidth', 1.2);
hold on;
b2 = bar(x_pos + bar_width/2, fold_data_dist(:, 2), bar_width, 'FaceColor', [0.8 0.3 0.3], ...
    'EdgeColor', 'k', 'LineWidth', 1.2);
set(gca, 'XTick', x_pos, 'XTickLabel', arrayfun(@(k) sprintf('%d', k), 1:K_FOLDS, 'UniformOutput', false));
xlabel({'Fold'; '(a)'},'FontSize', 10);
ylabel('Number of Samples','FontSize', 10);
title('Training Partition', 'FontSize', 10);
legend({'Benign', 'Malignant'}, 'Location', 'northeast', 'FontSize', 10);
grid on;
box on;
ylim([0, max(fold_data_dist(:, 1:2), [], 'all') * 1.15]);

for k = 1:K_FOLDS
    text(k - bar_width/2, fold_data_dist(k, 1) + 5, num2str(fold_data_dist(k, 1)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    text(k + bar_width/2, fold_data_dist(k, 2) + 5, num2str(fold_data_dist(k, 2)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
end

subplot(2, 1, 2);
b1 = bar(x_pos - bar_width/2, fold_data_dist(:, 3), bar_width, 'FaceColor', [0.3 0.5 0.8], ...
    'EdgeColor', 'k', 'LineWidth', 1.2);
hold on;
b2 = bar(x_pos + bar_width/2, fold_data_dist(:, 4), bar_width, 'FaceColor', [0.8 0.3 0.3], ...
    'EdgeColor', 'k', 'LineWidth', 1.2);
set(gca, 'XTick', x_pos, 'XTickLabel', arrayfun(@(k) sprintf('%d', k), 1:K_FOLDS, 'UniformOutput', false));
xlabel({'Fold'; '(b)'},'FontSize', 10);
ylabel('Number of Samples','FontSize', 10);
title('Validation Partition', 'FontSize', 10);
legend({'Benign', 'Malignant'}, 'Location', 'northeast', 'FontSize', 10);
grid on;
box on;
ylim([0, max(fold_data_dist(:, 3:4), [], 'all') * 1.15]);

for k = 1:K_FOLDS
    text(k - bar_width/2, fold_data_dist(k, 3) + 2, num2str(fold_data_dist(k, 3)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    text(k + bar_width/2, fold_data_dist(k, 4) + 2, num2str(fold_data_dist(k, 4)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
end

%% ===== Figure 6: Performance Metrics Across Folds =====
fig2 = figure('Position', [100, 100, 1100, 800], 'Color', 'white');
sgtitle('Performance Metrics Across 5-Folds', 'FontSize', 15, 'FontWeight', 'bold');

metric_names = {'Precision', 'Recall', 'Accuracy', 'Specificity', 'F1-Score'};
x_pos_metrics = 1:5;
bar_width_metric = 0.12;

jet_colors = hsv(6);
colors_fold = [
    jet_colors(1, :);
    jet_colors(2, :);
    jet_colors(3, :);
    jet_colors(4, :);
    jet_colors(5, :);
    jet_colors(6, :)
];

subplot(2, 1, 1);
mean_train_metrics = mean(fold_train_metrics_best, 1);
hold on;
for k = 1:K_FOLDS
    bar(x_pos_metrics + (k-3)*bar_width_metric, fold_train_metrics_best(k, :), ...
        bar_width_metric, 'FaceColor', colors_fold(k, :), 'EdgeColor', 'k', 'LineWidth', 0.8);
end
bar(x_pos_metrics + 3*bar_width_metric, mean_train_metrics, bar_width_metric, ...
    'FaceColor', colors_fold(6, :), 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'XTick', x_pos_metrics, 'XTickLabel', metric_names);
xlabel({'Metrics'; '(a)'}, 'FontSize', 10);
ylabel('Performance (%)', 'FontSize', 10);
title('Training Partition Performance', 'FontSize', 10);
ylim([75, 100]);
grid on;
legend({'Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5', 'Average'}, ...
    'Location', 'northeast', 'FontSize', 10, 'NumColumns', 2);
box on;

subplot(2, 1, 2);
hold on;
for k = 1:K_FOLDS
    bar(x_pos_metrics + (k-3)*bar_width_metric, fold_metrics_best(k, :), ...
        bar_width_metric, 'FaceColor', colors_fold(k, :), 'EdgeColor', 'k', 'LineWidth', 0.8);
end
bar(x_pos_metrics + 3*bar_width_metric, mean_metrics_best, bar_width_metric, ...
    'FaceColor', colors_fold(6, :), 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'XTick', x_pos_metrics, 'XTickLabel', metric_names);
xlabel({'Metrics'; '(b)'}, 'FontSize', 10);
ylabel('Performance (%)', 'FontSize', 10);
title('Validation Partition Performance', 'FontSize', 10);
ylim([75, 100]);
grid on;
legend({'Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5', 'Average'}, ...
    'Location', 'northeast', 'FontSize', 10, 'NumColumns', 2);
box on;

%% ===== Figure 7: Confusion Matrices for All Folds =====
fig3 = figure('Position', [150, 150, 1600, 700], 'Color', 'white');
sgtitle('Confusion Matrices Across 5-Folds', 'FontSize', 15, 'FontWeight', 'bold');

for k = 1:K_FOLDS
    subplot(2, K_FOLDS, k);
    conf_mat = fold_conf_matrices{k, 2};
    imagesc(conf_mat);
    colormap(gca, [1 0.9 0.8; 0.9 0.7 0.5; 0.8 0.5 0.3; 0.7 0.3 0.2]);
    axis square;
    set(gca, 'XTick', 1:2, 'XTickLabel', {'Malignant', 'Benign'}, ...
        'YTick', 1:2, 'YTickLabel', {'Malignant', 'Benign'}, 'FontSize', 10);
    title(sprintf('Fold %d', k), 'FontSize', 10);
    if k == 1
        xlabel({'Predicted Class'; '(a)'}, 'FontSize', 10);
    elseif k == 2
        xlabel({'Predicted Class'; '(b)'}, 'FontSize', 10);
    elseif k == 3
        xlabel({'Predicted Class'; '(c)'}, 'FontSize', 10);
    elseif k == 4
        xlabel({'Predicted Class'; '(d)'}, 'FontSize', 10);
    else
        xlabel({'Predicted Class'; '(e)'}, 'FontSize', 10);
    end
    ylabel('True Class', 'FontSize', 10);
    
    for i = 1:2
        for j = 1:2
            val = conf_mat(i, j);
            text(j, i, num2str(val), 'HorizontalAlignment', 'center', ...
                'Color', ternary(val > max(conf_mat(:))/2, 'white', 'black'), ...
                'FontSize', 14, 'FontWeight', 'bold');
        end
    end
end

annotation('textbox', [0.05, 0.88, 0.9, 0.05], 'String', 'Training Partition', ...
    'FontSize', 13, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'EdgeColor', 'none', 'VerticalAlignment', 'middle');

for k = 1:K_FOLDS
    subplot(2, K_FOLDS, K_FOLDS + k);
    conf_mat = fold_conf_matrices{k, 1};
    imagesc(conf_mat);
    colormap(gca, [1 0.9 0.8; 0.9 0.7 0.5; 0.8 0.5 0.3; 0.7 0.3 0.2]);
    axis square;
    set(gca, 'XTick', 1:2, 'XTickLabel', {'Malignant', 'Benign'}, ...
        'YTick', 1:2, 'YTickLabel', {'Malignant', 'Benign'}, 'FontSize', 10);
    title(sprintf('Fold %d', k), 'FontSize', 10);
    if k == 1
        xlabel({'Predicted Class'; '(f)'}, 'FontSize', 10);
    elseif k == 2
        xlabel({'Predicted Class'; '(g)'}, 'FontSize', 10);
    elseif k == 3
        xlabel({'Predicted Class'; '(h)'}, 'FontSize', 10);
    elseif k == 4
        xlabel({'Predicted Class'; '(i)'}, 'FontSize', 10);
    else
        xlabel({'Predicted Class'; '(j)'}, 'FontSize', 10);
    end
    ylabel('True Class', 'FontSize', 10);
    
    for i = 1:2
        for j = 1:2
            val = conf_mat(i, j);
            text(j, i, num2str(val), 'HorizontalAlignment', 'center', ...
                'Color', ternary(val > max(conf_mat(:))/2, 'white', 'black'), ...
                'FontSize', 14, 'FontWeight', 'bold');
        end
    end
end

annotation('textbox', [0.05, 0.43, 0.9, 0.05], 'String', 'Validation Partition', ...
    'FontSize', 13, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'EdgeColor', 'none', 'VerticalAlignment', 'middle');

%% ===== Figure 8: Final Model Performance Comparison =====
fig4 = figure('Position', [200, 200, 1000, 400], 'Color', 'white');
sgtitle('Final Model Performance Metrics', 'FontSize', 15, 'FontWeight', 'bold');

train_metrics_final = [precision_train_final, recall_train_final, accuracy_train_final, ...
                       specificity_train_final, f1_train_final];
test_metrics_final = [precision, recall, accuracy, specificity, f1];

x_pos_final = 1:5;
bar_width_final = 0.35;

bar(x_pos_final - bar_width_final/2, train_metrics_final, bar_width_final, ...
    'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 1.2);
hold on;
bar(x_pos_final + bar_width_final/2, test_metrics_final, bar_width_final, ...
    'FaceColor', [0.9 0.4 0.3], 'EdgeColor', 'k', 'LineWidth', 1.2);

set(gca, 'XTick', x_pos_final, 'XTickLabel', metric_names);
ylabel('Performance (%)', 'FontSize', 12);
xlabel({'Metrics'}, 'FontSize', 12);
ylim([85, 101]);
grid on;
legend({'Training Set (454 samples)', 'Test Set (115 samples)'}, 'Location', 'northeast', 'FontSize', 8);
box on;

for i = 1:5
    text(i - bar_width_final/2, train_metrics_final(i) + 0.5, sprintf('%.1f%%', train_metrics_final(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    text(i + bar_width_final/2, test_metrics_final(i) + 0.5, sprintf('%.1f%%', test_metrics_final(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
end

%% ===== Figure 9: Final Model Confusion Matrices =====
fig5 = figure('Position', [250, 250, 900, 400], 'Color', 'white');
sgtitle('Final Model Confusion Matrices', 'FontSize', 15, 'FontWeight', 'bold');

subplot(1, 2, 1);
conf_mat_train = [TP_train_final, FN_train_final; FP_train_final, TN_train_final];
imagesc(conf_mat_train);
colormap(gca, [1 0.9 0.8; 0.9 0.7 0.5; 0.8 0.5 0.3; 0.7 0.3 0.2]);
axis square;
set(gca, 'XTick', 1:2, 'XTickLabel', {'Malignant', 'Benign'}, ...
    'YTick', 1:2, 'YTickLabel', {'Malignant', 'Benign'}, 'FontSize', 11);
title('Training Set (454 samples)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel({'Predicted Class'; '(a)'}, 'FontSize', 11);
ylabel('True Class', 'FontSize', 11);

for i = 1:2
    for j = 1:2
        val = conf_mat_train(i, j);
        text(j, i, num2str(val), 'HorizontalAlignment', 'center', ...
            'Color', ternary(val > max(conf_mat_train(:))/2, 'white', 'black'), ...
            'FontSize', 16, 'FontWeight', 'bold');
    end
end

subplot(1, 2, 2);
conf_mat_test = [TP, FN; FP, TN];
imagesc(conf_mat_test);
colormap(gca, [1 0.9 0.8; 0.9 0.7 0.5; 0.8 0.5 0.3; 0.7 0.3 0.2]);
axis square;
set(gca, 'XTick', 1:2, 'XTickLabel', {'Malignant', 'Benign'}, ...
    'YTick', 1:2, 'YTickLabel', {'Malignant', 'Benign'}, 'FontSize', 11);
title('Test Set (115 samples)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel({'Predicted Class'; '(b)'}, 'FontSize', 11);
ylabel('True Class', 'FontSize', 11);

for i = 1:2
    for j = 1:2
        val = conf_mat_test(i, j);
        text(j, i, num2str(val), 'HorizontalAlignment', 'center', ...
            'Color', ternary(val > max(conf_mat_test(:))/2, 'white', 'black'), ...
            'FontSize', 16, 'FontWeight', 'bold');
    end
end

%% ===== Figure 10: ROC, PR Curves and Feature Importance =====
fig6 = figure('Position', [300, 300, 1800, 500], 'Color', 'white');
sgtitle('Final model evaluation and interpretation', 'FontSize', 15, 'FontWeight', 'bold');

y_test_bin = (y_test + 1) / 2;
thresholds = 0:0.01:1;
fpr_curve = zeros(length(thresholds), 1);
tpr_curve = zeros(length(thresholds), 1);
precision_curve = zeros(length(thresholds), 1);
recall_curve = zeros(length(thresholds), 1);

for i = 1:length(thresholds)
    pred = (y_proba >= thresholds(i));
    tp = sum(pred & y_test_bin);
    fp = sum(pred & ~y_test_bin);
    fn = sum(~pred & y_test_bin);
    tn = sum(~pred & ~y_test_bin);
    
    tpr_curve(i) = tp / (tp + fn + 1e-8);
    fpr_curve(i) = fp / (fp + tn + 1e-8);
    
    if (tp + fp) > 0
        precision_curve(i) = tp / (tp + fp);
    else
        precision_curve(i) = 1;
    end
    recall_curve(i) = tp / (tp + fn + 1e-8);
end

subplot(1, 3, 1);
fpr_curve_plot = flipud(fpr_curve);
tpr_curve_plot = flipud(tpr_curve);
AUC_ROC = abs(trapz(fpr_curve_plot, tpr_curve_plot));

fill([0; fpr_curve_plot; 1; 0], [0; tpr_curve_plot; 0; 0], ...
    [0.7 0.85 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold on;

h_roc = plot(fpr_curve_plot, tpr_curve_plot, 'b-', 'LineWidth', 2.5);

h_random = plot([0 1], [0 1], 'k--', 'LineWidth', 2.5);

h_legend = [h_roc; h_random];  
legend_str = {'ROC Curve', 'Random Classifier (1)'};

threshold_points = [0.20, 0.50, 0.80];
colors = {'b', [1 0.5 0], 'r'};

for i = 1:length(threshold_points)
    thresh = threshold_points(i);
    idx = find(abs(thresholds - thresh) < 0.01, 1);
    if ~isempty(idx)
        h = plot(fpr_curve(idx), tpr_curve(idx), 'o', ...
            'MarkerSize', 10, 'MarkerFaceColor', colors{i}, ...
            'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
        h_legend = [h_legend; h];
        legend_str{end+1} = sprintf('Threshold=%.2f', thresh);
    end
end

xlabel({'FPR'; '(a)'},  'FontSize', 12);
ylabel('Recall',  'FontSize', 12);
title('ROC Curve', 'FontWeight', 'bold', 'FontSize', 13);
grid on;
axis square;
xlim([0 1]); ylim([0 1]);

text(0.7, 0.1, sprintf('AUC = %.3f', AUC_ROC), 'FontSize', 8, 'FontWeight', 'bold', ...
    'BackgroundColor', 'white', 'EdgeColor', 'k', 'LineWidth', 1.5, 'Margin', 5);

legend(h_legend, legend_str, 'Location', 'southwest', 'FontSize', 8);
box on;

subplot(1, 3, 2);
[recall_sorted, sort_idx] = sort(recall_curve);
precision_sorted = precision_curve(sort_idx);
AUC_PR = abs(trapz(recall_sorted, precision_sorted));

fill([0; recall_sorted(:); 1; 0], [0; precision_sorted(:); 0; 0], ...
    [1.0 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold on;

subplot(1, 3, 2);
[recall_sorted, sort_idx] = sort(recall_curve);
precision_sorted = precision_curve(sort_idx);
AUC_PR = abs(trapz(recall_sorted, precision_sorted));

fill([0; recall_sorted(:); 1; 0], [0; precision_sorted(:); 0; 0], ...
    [1.0 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold on;

subplot(1, 3, 2);
[recall_sorted, sort_idx] = sort(recall_curve);
precision_sorted = precision_curve(sort_idx);
AUC_PR = abs(trapz(recall_sorted, precision_sorted));

fill([0; recall_sorted(:); 1; 0], [0; precision_sorted(:); 0; 0], ...
    [1.0 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hold on;

h_pr = plot(recall_sorted, precision_sorted, 'r-', 'LineWidth', 2.5);

baseline = sum(y_test_bin) / length(y_test_bin);
h_baseline = plot([0 1], [baseline baseline], 'k--', 'LineWidth', 2.5);

h_legend_pr = [h_pr; h_baseline];
legend_str_pr = {'PR Curve', ...
                 sprintf('Random Classifier (2):\n malignant test samples /test')};

threshold_points = [0.20, 0.50, 0.80];
colors = {'b', [1 0.5 0], 'r'};

for i = 1:length(threshold_points)
    thresh = threshold_points(i);
    idx = find(abs(thresholds - thresh) < 0.01, 1);
    if ~isempty(idx)
        h = plot(recall_curve(idx), precision_curve(idx), 'o', ...
            'MarkerSize', 10, 'MarkerFaceColor', colors{i}, ...
            'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
        h_legend_pr = [h_legend_pr; h];
        legend_str_pr{end+1} = sprintf('Threshold=%.2f', thresh);
    end
end

xlabel({'Recall'; '(b)'}, 'FontSize', 12);
ylabel('Precision', 'FontSize', 12);
title('Precision-Recall Curve', 'FontWeight', 'bold', 'FontSize', 13);
grid on;
axis square;
xlim([0 1]); ylim([0 1]);

text(0.7, 0.1, sprintf('AUC = %.3f', AUC_PR), 'FontSize', 8, 'FontWeight', 'bold', ...
    'BackgroundColor', 'white', 'EdgeColor', 'k', 'LineWidth', 1.5, 'Margin', 5);

legend(h_legend_pr, legend_str_pr, 'Location', 'southwest', 'FontSize', 8);
box on;

subplot(1, 3, 3);
top_features_short = cell(15,1);
for i = 1:15
    name = top_features{i};
    name = strrep(name, '_mean', '_m');
    name = strrep(name, '_worst', '_w');
    name = strrep(name, '_se', '_s');
    name = strrep(name, 'concave_pts', 'conc_pt');
    name = strrep(name, 'fractal_dim', 'fract_d');
    top_features_short{i} = name;
end

barh(1:15, sorted_imp(1:15), 'FaceColor', [0.4 0.6 0.8], 'EdgeColor', 'k', 'LineWidth', 1.2);
set(gca, 'YDir', 'reverse', 'YTick', 1:15, 'YTickLabel', top_features_short, 'FontSize', 10);
xlabel({'Absolute Weight Value'; '(c)'}, 'FontSize', 10);
title('Top 15 Features', 'FontWeight', 'bold','FontSize', 10);
grid on;
box on;

%% ===== Helper function =====
function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end