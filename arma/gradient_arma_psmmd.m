function param_est = gradient_arma_psmmd(data,p_lag,q_lag,TT,p,sigma)

% Stochastic gradient descent for ARMA model to obtain PSMMD estimator

% inputs: - data: T x 1 vector containing the time series
%         - p_lag: AR order
%         - q_lag: MA order
%         - TT: length of simulated data for PSMMD
%         - p: lag to construct \tilde{y}_i: x_{i},...,x_{i-p}, i=1,...,TT
%         - sigma: Gaussian kernel hyperparameter set by median heuristic

% output: PSMMD estimator

eta = 0.025; maxIt = 50e2; crit = 10^(-4); iter = 0;

% Initialization
[Phi_init,Theta_init] = simulate_varma_parameters(p_lag,q_lag);
Sigma_init = 0.03+0.07*rand(1);
param = [Phi_init;Theta_init;Sigma_init];
G = zeros(length(param),1); epsilon = 1e-6; param_old = param;

while iter<maxIt

    iter = iter+1;

    error_arma = mvnrnd(zeros(TT,1),eye(TT));
    gradient = Grad(@(x)arma_psmmd(x,data,error_arma,p_lag,q_lag,TT,p,sigma),param);

    % Update the accumulated sum of squared gradients
    G = G + gradient.^2;
    adaptive_alpha = eta./(sqrt(G+epsilon));

    param = param-gradient.*adaptive_alpha;

    % Verify parameter constraints
    param(1) = min(max(param(1),0.6),0.99);
    param(2) = min(max(param(2),0.1),0.3);
    param(3) = min(max(param(3),0.02),0.3);

    error = norm(param-param_old);
    cond = (error<crit)&&(max(abs(gradient))<1e-02);

    if cond||(iter>maxIt)
        break;
    end
    param_old = param;

end
param_est = param;