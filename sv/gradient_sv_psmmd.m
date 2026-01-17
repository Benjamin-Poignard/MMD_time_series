function param_est = gradient_sv_psmmd(data,TT,p,sigma)

% Stochastic gradient descent for SV model to obtain PSMMD estimator

% inputs: - data: T x 1 vector containing the time series
%         - TT: length of simulated data for PSMMD
%         - p: lag to construct \tilde{y}_i: x_{i},...,x_{i-p}, i=1,...,TT
%         - sigma: Gaussian kernel hyperparameter set by median heuristic

% output: PSMMD estimator

eta = 0.025; maxIt = 50e2; crit = 10^(-4); iter = 0;
% Initialization
param = [0.7+(0.95-0.7)*rand(1),0.01+(0.1-0.01)*rand(1),0.1+(0.05)*rand(1)]';
G = zeros(length(param),1); epsilon = 1e-6; param_old = param;

while iter<maxIt
    
    iter = iter+1;
    
    error1_sv = mvnrnd(zeros(TT,1),eye(TT)); error2_sv = mvnrnd(zeros(TT,1),eye(TT));
    gradient = Grad(@(x)sv_psmmd(x,data,error1_sv,error2_sv,TT,p,sigma),param);
    
    % Update the accumulated sum of squared gradients
    G = G + gradient.^2;
    adaptive_alpha = eta./(sqrt(G+epsilon));
    param = param-gradient.*adaptive_alpha;
    
    % Verify parameter constraints
    param(1) = min(max(param(1), 0.7), 0.99);  
    param(2) = min(max(param(2), eps), 0.3);   
    param(3) = min(max(param(3), eps), 0.3);
    
    error = norm(param-param_old);
    cond = (error<crit)&&(max(abs(gradient))<1e-02);
    
    if cond||(iter>maxIt)
        break;
    end
    param_old = param;
    
end
param_est = param;