function [val] = RMSE(x1, x2, varargin)
% Give the Root Mean Square Error
% Input:
%     x1, x2: n*m matrix, where n and m is dim of state and observation.
%     norm:   "norm" will give normalized RMSE
% Output:
%     val:    RMSE with size n*1
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
if size(varargin, 1) ~= 0
    val = sqrt(1 / size(x1, 2) * sum((x1 - x2).^2, 2)) ./ (max(x2, [], 2) - min(x2, [], 2));
else
    val = sqrt((1 / size(x1, 2)) * sum((x1 - x2).^2, 2));
end

end
