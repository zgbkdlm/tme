function [x, f, L, Q, dt, X, x0] = duffing_van_der(alp1, alp2, T, DT, int_steps, varargin)
%Duffing van der Pol oscillator model
% x'' + x' - (a-x^2)x = xw(t)
% d[x1] = [       x2    ]dt   [0 ]dB
%  [x2]   [x1(a-x1^2)-x2]     [x1]
%
% Simulate with Euler--Maruyama
%
% Input:
%	alp:          \alpha\geq 0. Model parameter
%	T:            Total time length
%	DT:           Time interval dt. Number of samples = T / DT
%	int_steps:    Number of interpolation steps. (10~10000) for better
%                 simulation accuracy
%
% Return:
%	x:            (sym) state vector
%   f:            (sym) drift functiom
%   L:            (sym) dispersion function
%   Q:            (sym, num) Diffusion matrix
%   dt:           (sym) \Delta t
%   X:            Simulations
%   x0:           Initial value
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

% Hyper-parameters
q = 1;

%% Build symbolic model for TME use. 
x = sym('x', [2, 1], 'real');
f = sym('f', 'real');
L = sym('L', 'real');
dt = sym('dt', 'real'); % avoid conflicton with symbolic dt here
Q = sym('Q', 'real');

f = [x(2); -x(2) * (alp2 + x(1)^2) + x(1) * (alp1 - x(1)^2)];
ff = @(u) [u(2); -u(2) * (alp2 + u(1)^2) + u(1) * (alp1 - u(1)^2)]; 
Q = q;
L = [0; x(1)]; 
LL = @(u) [0; u(1)];

%% Simulate data using Euler-Maruyama with very small DT
n_samples = round(T / DT);
x0 = [1; 0];

X = zeros(2, n_samples);

DDT = DT / int_steps;

xx = x0;
for j = 1:n_samples
    for k = 1:int_steps
        xx = xx + ff(xx) * DDT + LL(xx) * chol(Q * DDT)' * randn();
    end
    X(:, j) = xx;
end

end

