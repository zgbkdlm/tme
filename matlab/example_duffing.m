% Performing TME filtering and smoothing on Stochastic Duffing Van Der Pol
% oscillator model
%
% x'' + x' - (a-x^2)x = xw(t)
% d[x1] = [       x2    ]dt   [0 ]
%  [x2]   [x1(a-x1^2)-x2]     [x1] dB
% y_k   =  H x_k + r_k
%
% References:
%
%     [1] Zheng Z., Toni K., Roland H., and Simo S., Taylor Moment Expansion
%     for Continuous-discrete Filtering and Smoothing. 
%
% Zheng Zhao @ 2019 Aalto University
% zz@zabemon.com 
%
% Copyright (c) 2018 Zheng Zhao
% 
% Verson 0.1, Dec 2018

% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or 
% any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
%

clear
clc
close all

rng('default')
rng(2019)
%% Simulate data

% Parameters of simulating model
alp = 2; 
T = 5;
dt = 0.01;

% Measurement noise and some random H
R = 0.01;
H = randn(1, 2);
Y = zeros(1, round(T / dt));

% Simulate
[xx, ff, LL, QQ, dt_sym, X, x0] = models.duffing_van_der(alp, T, dt, 1000);
for i = 1:size(Y, 2)
    Y(:, i) = H * X(:, i) + sqrt(R) * randn();
end

%% Giving TME estimators
% See our paper [1] for details

% TME expansion order
TME_order = 3;

% Give 1st, 2nd moment and covariance (symbolic)
[a, Sigma] = tools.TME(xx, ff, LL, QQ, ...
                       dt_sym, TME_order, 'simplify');
as = xx * a';  % For smoother use.

% Convery symbolic to anonoymous function
a = matlabFunction(a, 'Vars', {xx, dt_sym});
Sigma = matlabFunction(Sigma, 'Vars', {xx, dt_sym});
as = matlabFunction(as, 'Vars', {xx, dt_sym});

%% Perform TME Gaussian filter and smoother

% Initial value
m0 = x0;
P0 = 0.01 * eye(2);

% Additional integration steps
it_steps = 10;      
% Smoothing yes
smooth = 1;

% Perform TME filter and smoother
[MM, PP, MS, PS] = TME_GHKF(m0, P0, a, Sigma, as, H, R, Y, dt, it_steps, smooth);

%% Plot results and show RMSE
plot(X(1, :), X(2, :), 'LineWidth',2, 'Color', [0 0.4470 0.7410], ...
                        'DisplayName', 'True mean');
hold on
plot(MM(1, :), MM(2, :), 'LineWidth',2, 'Color', [0.9290 0.6940 0.1250], ...
                        'DisplayName', 'TME Filtering mean');
plot(MS(1, :), MS(2, :), 'LineWidth',2, 'Color', [0.8500 0.3250 0.0980], ...
                        'DisplayName', 'TME Smoothing mean');
xlabel('x1');
ylabel('x2');
title({'TME Gaussian filter and smoother for', 'Duffing van der Pol model'});
lgd = legend();

grid on
ax = gca;
ax.GridLineStyle = '--';
ax.GridAlpha = 0.2;

rmse_filter = tools.RMSE(X, MM);
rmse_smoother = tools.RMSE(X, MS);

fprintf('Filtering RMSE: [ %.4f %.4f ] \n', rmse_filter(1), rmse_filter(2))
fprintf('Smoothing RMSE: [ %.4f %.4f ] \n', rmse_smoother(1), rmse_smoother(2))


