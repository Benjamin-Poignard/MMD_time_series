function loss = ricker_psmmd(param,data,error,TT,p,sigma)

% Compute MMD loss function for Ricker-based time series

% inputs: - param: vector of parameters
%         - data: T x 1 vector containing the time series
%         - error: 1 x TT vector for innovation u_t
%         - TT: length of simulated data for PSMMD
%         - p: lag to construct \tilde{y}_i: x_{i},...,x_{i-p}, i=1,...,TT
%         - sigma: Gaussian kernel hyperparameter set by median heuristic

% output: loss: MMD loss function

r = param(1); sigma_e = exp(param(2)); phi = exp(param(3));   

N_latent = zeros(TT,1); N_latent(1) = 1; E = exp(sigma_e*error);
for t = 2:TT
    N_latent(t) = r.*N_latent(t-1).*exp(-N_latent(t-1)).*E(t);
end
y = poissrnd(phi*N_latent); Y_sim = zeros(TT-p,p+1); Y_sim(:,1) = y(p+1:end);
for i=1:p
    Y_sim(:,i+1) = y(p-i+1:end-i);
end
loss = mmd(data,Y_sim,sigma);