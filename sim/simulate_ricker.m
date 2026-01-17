function y = simulate_ricker(theta,T,dist)

% Ricker model DGP

% inputs: - theta: model parameter
%         - T: number of time points
%         - dist: type of distribution: 'Gaussian' or 'Student'

% ouput: Ricker based generated data

logr = theta(1);
sigma = theta(2);
phi   = theta(3);

N = zeros(T,1);
N(1) = 1; % initial population

for t = 2:T
    switch dist
        case 'Gaussian'
            epsilon_error = randn(1);
        case 'Student'
            gamma = 3;
            epsilon_error = sqrt(gamma-2)*trnd(gamma)/sqrt(gamma);
    end
    N(t) = exp(logr + log(max(N(t-1),1e-8)) - N(t-1) + sigma*epsilon_error);
end
y = poissrnd(phi * N);
end