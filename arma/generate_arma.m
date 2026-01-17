function y = generate_arma(param,error,p_lag,q_lag,TT,N)

% Generation of ARMA paths

% inputs: - param: ARMA model (univariate) parameters
%         - error: TT x N matrix for innovation u_t
%         - p_lag: AR order
%         - q_lag: MA order
%         - TT: T0 length as in ISMMD
%         - N: length of the simulated vector \tilde{y}_i

% output: - y: simulated path

Phi = param(1); Theta = param(2); Sigma = param(3);
y = zeros(TT,N);
for kk = 1:N
   y(:,kk) = simulate_varma_sim(error(:,kk),TT,p_lag,q_lag,Phi,Theta,Sigma);
end