function y = generate_arma_T(param,error,p_lag,q_lag,TT)

% Generation of ARMA paths

% inputs: - param: ARMA model (univariate) parameters
%         - error: 1 x TT vector for innovation u_t
%         - p_lag: AR order
%         - q_lag: MA order
%         - TT: T0 length as in ISMMD

% output: - y: simulated path

Phi = param(1); Theta = param(2); Sigma = param(3);
y = simulate_varma_sim(error,TT,p_lag,q_lag,Phi,Theta,Sigma);
