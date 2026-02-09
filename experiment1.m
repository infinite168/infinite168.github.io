% ========================================================================
% Numerical experiments of four dynamical systems on a 2D problem
% Objective function: f(x,y) = (5x + y)^2
% ========================================================================
clear all; close all; clc;

%% ===== Optimization Parameters =====
alpha = 5.0;         
q = 1.5;             
p = 3.6;             
beta_val = 3.0;      

% Time parameters
t0 = 1.0;
T = 100;
h = 0.01;
N = floor((T-t0)/h);

% Create time array (for x-axis in plots)
t_array = t0 + (0:N) * h;

% Initial conditions
x0 = [10; 10];
v0 = [0; 0];

% Objective function
f = @(x) (5*x(1) + x(2))^2;
grad_f = @(x) 2*(5*x(1) + x(2)) * [5; 1];

% Time-dependent functions
epsilon = @(t) t^(-p);
b_func = @(t) t^q;

f_min = 0;

fprintf('===== Parameter Settings =====\n');
fprintf('alpha=%.1f, beta=%.1f, q=%.1f, p=%.1f\n', alpha, beta_val, q, p);
fprintf('Time range: t in [%.1f, %.1f], step size h=%.4f\n', t0, T, h);
fprintf('Initial point: [%.1f, %.1f]^T, f(x0)=%.1f\n\n', x0(1), x0(2), f(x0));

% Store results
store_interval = 5;
n_store = floor(N/store_interval) + 1;

trajectory_sys1 = zeros(2, n_store);
trajectory_sys2 = zeros(2, n_store);
trajectory_sys3 = zeros(2, n_store);
trajectory_sys4 = zeros(2, n_store);

f_vals_sys1 = zeros(N+1, 1);
f_vals_sys2 = zeros(N+1, 1);
f_vals_sys3 = zeros(N+1, 1);
f_vals_sys4 = zeros(N+1, 1);

%% ===== System 1: TR =====
fprintf('Running System 1: TR...\n');
x1 = x0;
v1 = v0;
f_vals_sys1(1) = f(x1);
trajectory_sys1(:, 1) = x1;
store_idx = 2;

for k = 1:N
    t = t0 + (k-1)*h;
    
    k1_x = v1;
    k1_v = -alpha/t * v1 - grad_f(x1) - epsilon(t)*x1;
    
    k2_x = v1 + h/2 * k1_v;
    k2_v = -alpha/(t+h/2) * (v1 + h/2*k1_v) - grad_f(x1 + h/2*k1_x) ...
           - epsilon(t+h/2)*(x1 + h/2*k1_x);
    
    k3_x = v1 + h/2 * k2_v;
    k3_v = -alpha/(t+h/2) * (v1 + h/2*k2_v) - grad_f(x1 + h/2*k2_x) ...
           - epsilon(t+h/2)*(x1 + h/2*k2_x);
    
    k4_x = v1 + h * k3_v;
    k4_v = -alpha/(t+h) * (v1 + h*k3_v) - grad_f(x1 + h*k3_x) ...
           - epsilon(t+h)*(x1 + h*k3_x);
    
    x1 = x1 + h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x);
    v1 = v1 + h/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v);
    
    f_vals_sys1(k+1) = f(x1);
    
    if mod(k, store_interval) == 0 && store_idx <= n_store
        trajectory_sys1(:, store_idx) = x1;
        store_idx = store_idx + 1;
    end
end

%% ===== System 2: TSTR =====
fprintf('Running System 2: TSTR...\n');
x2 = x0;
v2 = v0;
f_vals_sys2(1) = f(x2);
trajectory_sys2(:, 1) = x2;
store_idx = 2;

for k = 1:N
    t = t0 + (k-1)*h;
    
    k1_x = v2;
    k1_v = -alpha/t * v2 - b_func(t)*(grad_f(x2) + epsilon(t)*x2);
    
    k2_x = v2 + h/2 * k1_v;
    k2_v = -alpha/(t+h/2) * (v2 + h/2*k1_v) ...
           - b_func(t+h/2)*(grad_f(x2 + h/2*k1_x) + epsilon(t+h/2)*(x2 + h/2*k1_x));
    
    k3_x = v2 + h/2 * k2_v;
    k3_v = -alpha/(t+h/2) * (v2 + h/2*k2_v) ...
           - b_func(t+h/2)*(grad_f(x2 + h/2*k2_x) + epsilon(t+h/2)*(x2 + h/2*k2_x));
    
    k4_x = v2 + h * k3_v;
    k4_v = -alpha/(t+h) * (v2 + h*k3_v) ...
           - b_func(t+h)*(grad_f(x2 + h*k3_x) + epsilon(t+h)*(x2 + h*k3_x));
    
    x2 = x2 + h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x);
    v2 = v2 + h/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v);
    
    f_vals_sys2(k+1) = f(x2);
    
    if mod(k, store_interval) == 0 && store_idx <= n_store
        trajectory_sys2(:, store_idx) = x2;
        store_idx = store_idx + 1;
    end
end

%% ===== System 3: HTR =====
fprintf('Running System 3: HTR...\n');
x3 = x0;
y3 = -beta_val*(v0 + beta_val*(grad_f(x0) + epsilon(t0)*x0)) ...
     + (1 - beta_val*alpha/t0)*x0;
f_vals_sys3(1) = f(x3);
trajectory_sys3(:, 1) = x3;
store_idx = 2;

for k = 1:N
    t = t0 + (k-1)*h;
    
    k1_x = -beta_val*grad_f(x3) - beta_val*epsilon(t)*x3 ...
           + (1/beta_val - alpha/t)*x3 - (1/beta_val)*y3;
    k1_y = (1/beta_val - alpha/t + alpha*beta_val/t^2)*x3 - (1/beta_val)*y3;
    
    x_temp = x3 + h/2 * k1_x;
    y_temp = y3 + h/2 * k1_y;
    t_temp = t + h/2;
    k2_x = -beta_val*grad_f(x_temp) - beta_val*epsilon(t_temp)*x_temp ...
           + (1/beta_val - alpha/t_temp)*x_temp - (1/beta_val)*y_temp;
    k2_y = (1/beta_val - alpha/t_temp + alpha*beta_val/t_temp^2)*x_temp ...
           - (1/beta_val)*y_temp;
    
    x_temp = x3 + h/2 * k2_x;
    y_temp = y3 + h/2 * k2_y;
    k3_x = -beta_val*grad_f(x_temp) - beta_val*epsilon(t_temp)*x_temp ...
           + (1/beta_val - alpha/t_temp)*x_temp - (1/beta_val)*y_temp;
    k3_y = (1/beta_val - alpha/t_temp + alpha*beta_val/t_temp^2)*x_temp ...
           - (1/beta_val)*y_temp;
    
    x_temp = x3 + h * k3_x;
    y_temp = y3 + h * k3_y;
    t_temp = t + h;
    k4_x = -beta_val*grad_f(x_temp) - beta_val*epsilon(t_temp)*x_temp ...
           + (1/beta_val - alpha/t_temp)*x_temp - (1/beta_val)*y_temp;
    k4_y = (1/beta_val - alpha/t_temp + alpha*beta_val/t_temp^2)*x_temp ...
           - (1/beta_val)*y_temp;
    
    x3 = x3 + h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x);
    y3 = y3 + h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y);
    
    f_vals_sys3(k+1) = f(x3);
    
    if mod(k, store_interval) == 0 && store_idx <= n_store
        trajectory_sys3(:, store_idx) = x3;
        store_idx = store_idx + 1;
    end
end

%% ===== System 4: HTSTR =====
fprintf('Running System 4: HTSTR...\n');
x4 = x0;
u0 = v0;
z4 = -u0 - beta_val*(grad_f(x0) + epsilon(t0)*x0);
f_vals_sys4(1) = f(x4);
trajectory_sys4(:, 1) = x4;
store_idx = 2;

for k = 1:N
    t = t0 + (k-1)*h;
    
    grad_term4 = grad_f(x4) + epsilon(t)*x4;
    k1_x = -beta_val*grad_term4 - z4;
    k1_z = alpha/t * k1_x + b_func(t)*grad_term4;
    
    x_temp = x4 + h/2 * k1_x;
    z_temp = z4 + h/2 * k1_z;
    t_temp = t + h/2;
    grad_term4 = grad_f(x_temp) + epsilon(t_temp)*x_temp;
    k2_x = -beta_val*grad_term4 - z_temp;
    k2_z = alpha/t_temp * k2_x + b_func(t_temp)*grad_term4;
    
    x_temp = x4 + h/2 * k2_x;
    z_temp = z4 + h/2 * k2_z;
    grad_term4 = grad_f(x_temp) + epsilon(t_temp)*x_temp;
    k3_x = -beta_val*grad_term4 - z_temp;
    k3_z = alpha/t_temp * k3_x + b_func(t_temp)*grad_term4;
    
    x_temp = x4 + h * k3_x;
    z_temp = z4 + h * k3_z;
    t_temp = t + h;
    grad_term4 = grad_f(x_temp) + epsilon(t_temp)*x_temp;
    k4_x = -beta_val*grad_term4 - z_temp;
    k4_z = alpha/t_temp * k4_x + b_func(t_temp)*grad_term4;
    
    x4 = x4 + h/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x);
    z4 = z4 + h/6 * (k1_z + 2*k2_z + 2*k3_z + k4_z);
    
    f_vals_sys4(k+1) = f(x4);
    
    if mod(k, store_interval) == 0 && store_idx <= n_store
        trajectory_sys4(:, store_idx) = x4;
        store_idx = store_idx + 1;
    end
end

fprintf('\nAll systems completed!\n\n');

%% ===== Performance Analysis =====
fprintf('==================== Final Convergence Results ====================\n');
fprintf('System 1 (TR):         f(x(T)) = %.6e\n', f_vals_sys1(end));
fprintf('System 2 (TSTR):       f(x(T)) = %.6e\n', f_vals_sys2(end));
fprintf('System 3 (HTR):        f(x(T)) = %.6e\n', f_vals_sys3(end));
fprintf('System 4 (HTSTR):      f(x(T)) = %.6e\n', f_vals_sys4(end));
fprintf('=====================================================\n\n');

%% ===== Visualization 1: Convergence Curves =====
error_sys1 = abs(f_vals_sys1 - f_min);
error_sys2 = abs(f_vals_sys2 - f_min);
error_sys3 = abs(f_vals_sys3 - f_min);
error_sys4 = abs(f_vals_sys4 - f_min);

% Replace minimal values to avoid log(0)
error_sys1(error_sys1 < 1e-15) = 1e-15;
error_sys2(error_sys2 < 1e-15) = 1e-15;
error_sys3(error_sys3 < 1e-15) = 1e-15;
error_sys4(error_sys4 < 1e-15) = 1e-15;

figure('Position', [50, 100, 1200, 600]);
subplot(1,2,1)
semilogy(t_array, error_sys1, 'b-.', 'LineWidth', 2.5, 'DisplayName', 'TR');
hold on;
semilogy(t_array, error_sys2, 'r--', 'LineWidth', 2.5, 'DisplayName', 'TSTR');
semilogy(t_array, error_sys3, 'g:', 'LineWidth', 2.5, 'DisplayName', 'HTR');
semilogy(t_array, error_sys4, 'm-', 'LineWidth', 2.5, 'DisplayName', 'HTSTR');
hold off;

xlabel({'time $t$'; '(a)'}, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'latex');
ylabel('$f_1(x(t)) - \min f_1$', 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'latex'); 
title(sprintf('min $f_1(x) = (5x_1 - x_2)^2$'), 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'latex');
legend('Location', 'northeast', 'FontSize', 10);
set(gca, 'FontSize', 10);
ylim([1e-16, 1e4]);
xlim([t0, T]);
grid on;
box on;

%% ===== Visualization 2: 2D Trajectories =====
subplot(1,2,2)

% Create fine contour background
[X, Y] = meshgrid(linspace(-6, 11, 200), linspace(2, 16, 200));
Z = (5*X + Y).^2;

% Use logarithmic scale for contour levels
contour_levels = [0.1, 1, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200];

% Create colormap for contours
colors = hsv(length(contour_levels));

hold on;
% Plot colored contours and add labels
for i = 1:length(contour_levels)
    level = contour_levels(i);
    [C, h] = contour(X, Y, Z, [level, level], 'LineWidth', 1.5, ...
           'LineColor', colors(i,:), 'HandleVisibility', 'off');
    
    % Add numerical labels for each contour
    if ~isempty(C)
        clabel(C, h, 'FontSize', 9, 'Color', colors(i,:), 'LabelSpacing', 300);
    end
end

% Solution set: the line 5x + y = 0 (thick black line)
x_solution_set = linspace(-6, 11, 100);
y_solution_set = -5 * x_solution_set;
plot(x_solution_set, y_solution_set, 'k-', 'LineWidth', 3, ...
    'DisplayName', 'Solution Set');
% Plot trajectories
h1 = plot(trajectory_sys1(1,:), trajectory_sys1(2,:), 'b-.', 'LineWidth', 2.5, ...
    'DisplayName', 'TR');
h2 = plot(trajectory_sys2(1,:), trajectory_sys2(2,:), 'r--', 'LineWidth', 2.5, ...
    'DisplayName', 'TSTR');
h3 = plot(trajectory_sys3(1,:), trajectory_sys3(2,:), 'g:', 'LineWidth', 2.5, ...
    'DisplayName', 'HTR');
h4 = plot(trajectory_sys4(1,:), trajectory_sys4(2,:), 'm-', 'LineWidth', 2.5, ...
    'DisplayName', 'HTSTR');

% Add direction arrows
arrow_interval = floor(n_store/8);
for idx = arrow_interval:arrow_interval:n_store-arrow_interval
    quiver(trajectory_sys1(1,idx), trajectory_sys1(2,idx), ...
           trajectory_sys1(1,idx+1)-trajectory_sys1(1,idx), ...
           trajectory_sys1(2,idx+1)-trajectory_sys1(2,idx), ...
           0, 'b', 'LineWidth', 2, 'MaxHeadSize', 1.5, 'HandleVisibility', 'off');
    quiver(trajectory_sys2(1,idx), trajectory_sys2(2,idx), ...
           trajectory_sys2(1,idx+1)-trajectory_sys2(1,idx), ...
           trajectory_sys2(2,idx+1)-trajectory_sys2(2,idx), ...
           0, 'r', 'LineWidth', 2, 'MaxHeadSize', 1.5, 'HandleVisibility', 'off');
    quiver(trajectory_sys3(1,idx), trajectory_sys3(2,idx), ...
           trajectory_sys3(1,idx+1)-trajectory_sys3(1,idx), ...
           trajectory_sys3(2,idx+1)-trajectory_sys3(2,idx), ...
           0, 'g', 'LineWidth', 2, 'MaxHeadSize', 1.5, 'HandleVisibility', 'off');
    quiver(trajectory_sys4(1,idx), trajectory_sys4(2,idx), ...
           trajectory_sys4(1,idx+1)-trajectory_sys4(1,idx), ...
           trajectory_sys4(2,idx+1)-trajectory_sys4(2,idx), ...
           0, 'm', 'LineWidth', 2, 'MaxHeadSize', 1.5, 'HandleVisibility', 'off');
end

% Mark initial point
plot(x0(1), x0(2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'black', ...
    'LineWidth', 2, 'DisplayName', 'Initial Point');

% Mark end points
markers = {'s', 'd', '^', 'v'};
colors_marker = {'b', 'r', 'g', 'm'};
marker_names = {'Convergence Point (TR)', 'Convergence Point (TSTR)', ...
                'Convergence Point (HTR)', 'Convergence Point (HTSTR)'};

for i = 1:4
    if i == 1
        traj = trajectory_sys1;
    elseif i == 2
        traj = trajectory_sys2;
    elseif i == 3
        traj = trajectory_sys3;
    else
        traj = trajectory_sys4;
    end
    
    plot(traj(1,end), traj(2,end), markers{i}, ...
        'MarkerSize', 10, 'MarkerFaceColor', colors_marker{i}, ...
        'MarkerEdgeColor', colors_marker{i}, 'LineWidth', 2, ...
        'DisplayName', marker_names{i});
end

hold off;
grid on;

xlabel({'$x_1(t)$'; '(b)'}, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'latex');
ylabel('$x_2(t)$', 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'latex');
title('$\arg\min f_1 = \{(x_1, -5x_1)\}$', ...
      'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'latex');
legend('Location', 'northeast', 'FontSize', 12);
set(gca, 'FontSize', 10);
xlim([-6, 11]);
ylim([2, 16]);
box on;