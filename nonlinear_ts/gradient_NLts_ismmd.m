function param_est = gradient_NLts_ismmd(data,TT,p,N,sigma)

% Stochastic gradient descent for non-linear MA model to obtain ISMMD estimator

% inputs: - data: T x 1 vector containing the time series
%         - TT: T0 length as in ISMMD
%         - p: lag to construct \tilde{y}_i: x^{(i)}_{TT},...,x^{(i)}_{TT-p}
%         - N: length of the simulated vector \tilde{y}_i
%         - sigma: Gaussian kernel hyperparameter set by median heuristic

% output: ISMMD estimator

eta = 0.025; maxIt = 50e2; crit = 10^(-4); iter = 0;

% Initialization
param = 0.6+(0.95-0.6)*rand(1);
G = zeros(length(param),1); epsilon = 1e-6; param_old = param;

while iter<maxIt
    
    iter = iter+1;
    
    error_NLts = mvnrnd(zeros(N,1),eye(N),TT);
    gradient = Grad(@(x)NLts_ismmd(x,data,error_NLts,TT,p,N,sigma),param);
    G = G + gradient.^2;
    adaptive_alpha = eta./(sqrt(G+epsilon));
    param = param-gradient.*adaptive_alpha;
    
    % Verify parameter constraints
    param = min(max(param, 0.01), 1.4);
    
    error = norm(param-param_old);
    cond = (error<crit)&&(max(abs(gradient))<1e-02);
    
    if cond||(iter>maxIt)
        break;
    end
    param_old = param;
    
end
param_est = param;