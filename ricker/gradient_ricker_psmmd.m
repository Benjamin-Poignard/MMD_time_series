function param_est = gradient_ricker_psmmd(data,TT,p,sigma)

% Stochastic gradient descent for Ricker model to obtain PSMMD estimator

% inputs: - data: T x 1 vector containing the time series
%         - TT: length of simulated data for PSMMD
%         - p: lag to construct \tilde{y}_i: x_{i},...,x_{i-p}, i=1,...,TT
%         - sigma: Gaussian kernel hyperparameter set by median heuristic

% output: PSMMD estimator

eta = 0.025; maxIt = 50e2; crit = 10^(-4); iter = 0;

% Initialization
param = [log(4+(9-4)*rand(1));log(0.01+(0.1-0.01)*rand(1));log(4+(9-4)*rand(1))];
G = zeros(length(param),1); epsilon = 1e-6; param_old = param;

while iter<maxIt
    
    iter = iter+1;
    
    error_ricker = mvnrnd(zeros(TT,1),eye(TT));
    gradient = Grad(@(x)ricker_psmmd(x,data,error_ricker,TT,p,sigma),param);
    
    % Update the accumulated sum of squared gradients
    G = G + gradient.^2;
    adaptive_alpha = eta./(sqrt(G+epsilon));
    param = param-gradient.*adaptive_alpha;
    
    % Verify parameter constraints
    param(1) = min(max(param(1),log(4)),log(9));
    param(2) = min(max(param(2),log(0.01)),log(0.1));
    param(3) = min(max(param(3),log(4)),log(9));
    
    error = norm(param-param_old);
    cond = (error<crit);
    
    if cond||(iter>maxIt)
        break;
    end
    param_old = param;
    %[param; max(abs(gradient));error]
    
end
param_est = [param(1);exp(param(2));exp(param(3))];