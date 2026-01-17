function [b,c] = simulate_garch_param(N)

% input: - N: dimension (number of time series)

% ouputs: - b: autoregressive GARCH parameter
%         - c: MA GARCH parameter (shock term)

gamma = true;
while(gamma)
    b = 0.7 + (0.95-0.7)*rand(1,N);
    c = 0.02 + (0.1-0.02)*rand(1,N);
    gamma = (any(b+c > 1));
end