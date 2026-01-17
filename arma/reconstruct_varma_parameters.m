function [Phi, Theta] = reconstruct_varma_parameters(param, d, p, q)

% inputs: - param: Stacked parameter vector containing [Phi_init; Theta_init]
%         - d: Dimension of the time series
%         - p: Order of the AR component
%         - q: Order of the MA component

% outputs: - Phi: AR coefficient
%          - Theta: MA coefficient

% Determine the length of each Phi and Theta matrix
len_phi = d * d;
len_theta = d * d;

% Initialize cell arrays for Phi and Theta
Phi = cell(p, 1);
Theta = cell(q, 1);

% Reconstruct Phi matrices
for i = 1:p
    start_idx = (i - 1) * len_phi + 1;
    end_idx = i * len_phi;
    Phi{i} = reshape(param(start_idx:end_idx), d, d);
end

% Reconstruct Theta matrices
for j = 1:q
    start_idx = p * len_phi + (j - 1) * len_theta + 1;
    end_idx = p * len_phi + j * len_theta;
    Theta{j} = reshape(param(start_idx:end_idx), d, d);
end