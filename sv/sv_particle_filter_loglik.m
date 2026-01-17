function nll = sv_particle_filter_loglik(param,data)

% Particle filter-based likelihood for SV model following Pitt, Malik and
% Doucet (2014)

% inputs: - param: 3-dimensional vector of parameters
%         - data: T x 1 vector of observations

% output: - nll: negative log likelihood function

phi = param(1); sigma_e = param(2); sigma_y = param(3);
y = data(:); T = numel(y); N = 5e3;

% Particle initialization from stationary prior of h_0
sigma_h0 = sigma_e / sqrt(1 - phi^2);
h = sigma_h0 * randn(N,1); % particles for h_0
loglik = 0;

% Filtering loop
for t = 1:T
    % Propagate
    h = phi .* h + sigma_e .* randn(N,1);
    
    % Compute the weight by likelihood g(y_t | h_t)
    vy = (sigma_y^2) .* exp(h); % variance of y_t conditional on h_t
    lw = -0.5*( log(2*pi) + log(vy) + (y(t).^2)./vy ); % log-weights (unnormalized)
    
    % Incremental log-likelihood using log-sum-exp
    m = max(lw);
    inc = m + log( mean( exp(lw - m) ) ); % log( (1/N) * sum w )
    loglik = loglik + inc;
    
    % Normalize the weights
    w = exp(lw - m);
    W = w ./ sum(w);
    
    % Resample (systematic resampling)
    idx = systematic_resample(W);
    h = h(idx);
end
nll = -loglik;
end

function idx = systematic_resample(W)
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