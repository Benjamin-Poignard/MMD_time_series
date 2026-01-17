function loglik = NLts_particle_filter_loglik(param,data)

% inputs: - param : scalar parameter
%         - data : T x 1 vector of data

% output: loglikelihood function

Y = data(:); T = numel(Y); N = 5e3;

% initialize eps_0 ~ N(0,1)
eps_prev = randn(N,1);   % particles for eps_{t-1}
loglik = 0;

for t = 1:T
    % propagate: draw eps_t ~ N(0,1)
    eps_t = randn(N,1);
    
    % predicted observation
    Y_pred = eps_t + param*(eps_prev.^2);
    
    % weight: Gaussian kernel around observed Y_t
    % Use a small bandwidth for sharp likelihood (simulate delta)
    sigma_w = 0.01;  % small number to approximate deterministic observation
    lw = -0.5*((Y(t)-Y_pred).^2)/sigma_w^2 - log(sqrt(2*pi)*sigma_w);
    
    % increment log-likelihood (log-sum-exp)
    m = max(lw);
    inc = m + log(mean(exp(lw-m)));
    loglik = loglik + inc;
    
    % normalize weights
    w = exp(lw-m);
    W = w/sum(w);
    
    % systematic resampling
    idx = systematic_resample(W);
    
    % update particles
    eps_prev = eps_t(idx);
end
end


function idx = systematic_resample(W)
% Systematic resampling of particles
N = numel(W);
u0 = rand()/N;
u = u0 + (0:N-1)'/N;
cumW = cumsum(W);
idx = zeros(N,1);
i = 1; j = 1;
while i <= N
    while u(i) > cumW(j)
        j = j + 1;
    end
    idx(i) = j;
    i = i + 1;
end
end