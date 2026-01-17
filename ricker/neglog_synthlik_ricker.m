function nll = neglog_synthlik_ricker(param,data,R)

% Synthetic likelihood for Ricker model
% inputs: - param: 3-dimensional vector of parameters [log_r, sigma, phi]
%         - data: T x 1 vector of observations
%         - R: number of simulated paths

% output: - nll: negative log likelihood function

log_r = param(1); sigma = param(2); phi   = param(3);
T = length(data);
% Observed probes
S_obs = probe_fun(data);
% Simulate synthetic datasets and probes
S_sim = zeros(R, length(S_obs));
for i = 1:R
    y_sim = generate_ricker([log_r, sigma, phi],T);
    S_sim(i,:) = probe_fun(y_sim);
end
% Regularized covariance
muS = mean(S_sim,1);
SigmaS = cov(S_sim) + 1e-6 * eye(length(S_obs));
% Negative synthetic log-likelihood
diffS = S_obs - muS;
nll = 0.5 * (diffS / SigmaS * diffS' + log(det(SigmaS)) + length(S_obs)*log(2*pi));
end

function S = probe_fun(y)
% Computation of phase-insensitive summary statistics
y = double(y(:));
S = [
    mean(y)
    var(y)
    mean(diff(y))
    var(diff(y))
    autocorr_lag(y,1)
    autocorr_lag(y,2)
    autocorr_lag(diff(y),1)
    mean(diff(y).^2)
    ]';
end

function y = generate_ricker(theta,T)
% Simulate Ricker model-based data
% theta = [log_r, sigma, phi]

logr = theta(1); sigma = theta(2); phi   = theta(3);
N = zeros(T,1); N(1) = 1; % initial population
for t = 2:T
    N(t) = exp(logr + log(max(N(t-1),1e-8)) - N(t-1) + sigma*randn(1));
end
y = poissrnd(phi * N);

end