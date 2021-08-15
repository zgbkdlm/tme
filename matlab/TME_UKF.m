function [MM, PP, UKF_MS, UKF_PS] = TME_UKF(m0, P0, a, Sigma, as, H, R, Y, dt, it_steps, smooth)
% TME Gaussian filter and smoother for continuous-discrete model with
% unscented transform
% 
% dx = f(x, t) dt + L(x, t) dB, 
% y  = h(x) + r,       r~ N(0, R)
%
% Input:
%   m0, P0:     Initial condition
%   a:          E[x_k | x_{k-1}] by TME
%   Sigma:      Cov[x_k | x_{k-1}] by TME
%   as:         x_k * a^T
%   h:          Measurement function
%   R:          Measurement covariance
%   Y:          Measurements 
%   dt:         Time interval between measurements
%   it_steps:   Integrations steps (additional prediction steps)
%   smooth:     1: Do smooth. 0: No smooth
%
% Return:
%   MM, PP:     Mean and covariance of filtering posterior
%   MS, PS:     Mean and covariance of smoothing posterior (if use smooth)
%   
% References:
%
%     [1] Zheng Zhao, Toni Karvonen, Roland Hostettler, Simo Särkkä, 
%         Taylor Moment Expansion for Continuous-discrete Filtering. 
%         IEEE Transactions on Automatic Control. 
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

%% Initialization and allocation
dim_x = size(m0, 1);
length_y = size(Y,2);

MM = zeros(size(m0,1), size(Y,2));
PP = zeros(size(P0,1), size(P0,2), length_y);
if smooth
    UKF_pred_store_m = zeros(size(m0,1), length_y*it_steps);
    UKF_pred_store_P = zeros(size(P0,1), size(P0,2), length_y*it_steps);
end
m = m0;
P = P0;

%% Initialize unscented sigma points
alpha = 1;
beta = 0;
kappa = 3 - dim_x;
lamd = alpha^2*(size(m0,1) + kappa) - dim_x;
%kappa = 0; % This is CKF

[WM, WC, c] = tools.ut_weights(dim_x, alpha, beta, kappa);
WM = WM';
WC = WC';

XI = [zeros(dim_x,1) eye(dim_x) -eye(dim_x)];
XI = sqrt(c) * XI;
        
sigp = chol(P)'*XI + repmat(m,1,size(XI,2)); 
ax = sigp;
Bx = zeros(dim_x, dim_x, 2 * dim_x);

%% Filtering pass
for k = 1:size(Y,2)
    
    % Prediction step
    ddt = dt / it_steps;
    for i = 1:it_steps
        sigp = chol(P)' * XI + repmat(m, 1, 2*dim_x+1);
        for j = 1:2*dim_x+1
            ax(:, j) = a(sigp(:, j), ddt);
            Bx(:, :, j) = Sigma(sigp(:, j), ddt) + ax(:, j) * ax(:, j)';
        end
        m = tools.weighted_sum(WM, ax);
        P = tools.weighted_sum(WM, Bx) - m * m';
        
        % Store  prediction for smoother
        if smooth
            UKF_pred_store_m(:, i+(k-1)*it_steps) = m;
            UKF_pred_store_P(:, :, i+(k-1)*it_steps) = P;
        end
    end
    
    % Update step
    v = Y(:, k) - H * m;
    S = H * P * H' + R;
    K = P * H' / S;
    m = m + K * v;
    P = P - K * S * K';
    
    MM(:,k) = m;
    PP(:,:,k) = P;
end

%% Smoothing pass
if smooth  % Perform smoothing?
    
    UKF_MS = UKF_pred_store_m;
    UKF_PS = UKF_pred_store_P;
    
    MS = m;
    PS = P;
    
    % The code below can be optimized
    for k = size(UKF_pred_store_m,2)-1:-1:1
        % Additional pred on mk and Pk with integration step, that is ddt
        MS = UKF_MS(:, k+1);
        PS = UKF_PS(:, :, k+1);
        if mod(k, it_steps) == 0
            % use the updated posterior
            m = MM(:, round(k/it_steps));
            P = PP(:, :, round(k/it_steps));
        else
            m = UKF_pred_store_m(:, k);
            P = UKF_pred_store_P(:, :, k);
        end
        ms_pred = UKF_pred_store_m(:, k+1);
        Ps_pred = UKF_pred_store_P(:, :, k+1);
        sigp = chol(P)' * XI + repmat(m, 1, 2*dim_x+1);   
        % Calculate cross-cov D
        for j=1:2*dim_x+1
            % Re-using Bx
            Bx(:, :, j) = as(sigp(:, j), ddt);
        end
        DS = tools.weighted_sum(WM, Bx) - m * ms_pred';
        % Smoothing Gain
        GS = DS / Ps_pred;
        % Smooth
        MS = m + GS * (MS - ms_pred);
        PS = P + GS * (PS - Ps_pred) * GS';
        
        % Store results
        UKF_MS(:, k) = MS;
        UKF_PS(:, :, k) = PS;
    end
    
    UKF_MS = UKF_MS(:, it_steps:it_steps:end);
    UKF_PS = UKF_PS(:, it_steps:it_steps:end);
    
end

