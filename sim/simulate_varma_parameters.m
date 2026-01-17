function [Phi, Theta] = simulate_varma_parameters(p, q)

% inputs: - p: AR lag
%         - q: MA lag

% output: - Phi: AR coefficients (p lags)
%         - Theta: MA coefficients (q lags)

% Initialize cell arrays for AR and MA coefficients
Phi = cell(p, 1); Theta = cell(q, 1);

% Simulate AR coefficients
for i = 1:p
    Phi_matrix = 0.6+0.3*rand(1);
    % Vectorize the matrix
    Phi{i} = Phi_matrix(:);
end

% Simulate MA coefficients
for j = 1:q
    Theta_matrix = 0.05+0.25*rand(1);
    % Vectorize the matrix
    Theta{j} = Theta_matrix(:);
end
% Convert cell arrays to matrices
Phi = cell2mat(Phi);
Theta = cell2mat(Theta);
end