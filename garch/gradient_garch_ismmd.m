function param_est = gradient_garch_ismmd(data,TT,p,N,sigma)

% Stochastic gradient descent for GARCH model to obtain ISMMD estimator

% inputs: - data: T x 1 vector containing the time series
%         - TT: T0 length as in ISMMD
%         - p: lag to construct \tilde{y}_i: x^{(i)}_{TT},...,x^{(i)}_{TT-p}
%         - N: length of the simulated vector \tilde{y}_i
%         - sigma: Gaussian kernel hyperparameter set by median heuristic

% output: ISMMD estimator

eta = 0.025;
maxIt = 50e2; crit = 10^(-4); iter = 0;

% Initialization
[b_init,c_init] = simulate_garch_param(1); a_const = 0.02+(0.08-0.02)*rand(1);
param = [a_const,b_init,c_init]';
G = zeros(length(param),1); epsilon = 1e-6; param_old = param;

while iter<maxIt
    
    iter = iter+1;
    
    error_garch = mvnrnd(zeros(N,1),eye(N),TT);
    gradient = Grad(@(x)garch_ismmd(x,data,error_garch,TT,p,N,sigma),param);
    
    if ~isreal(gradient) || any(isnan(gradient)) || any(isinf(gradient))
        param = project_param(param);  % Apply projection
        gradient = zeros(size(gradient));      % Skip update
    else
        
        % Update the accumulated sum of squared gradients
        G = G + gradient.^2;
        adaptive_alpha = eta./(sqrt(G+epsilon));
        
        param = param-gradient.*adaptive_alpha;
        
        % Verify parameter constraints
        param = project_param(param);
        
    end
    
    error = norm(param-param_old);
    cond = (error<crit)&&(max(abs(gradient))<1e-02);
    
    if cond||(iter>maxIt)
        break;
    end
    param_old = param;
    
end
param_est = param;