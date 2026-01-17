function loss = ricker_ismmd(param,data,error,TT,p,N,sigma)

% Compute MMD loss function for Ricker-based time series

% inputs: - param: vector of parameters
%         - data: T x 1 vector containing the time series
%         - error: TT x N matrix for innovation u_t
%         - TT: T0 length as in ISMMD
%         - p: lag to construct \tilde{y}_i: x^{(i)}_{TT},...,x^{(i)}_{TT-p}
%         - N: length of the simulated vector \tilde{y}_i
%         - sigma: Gaussian kernel hyperparameter set by median heuristic

% output: loss: MMD loss function

r = exp(param(1)); sigma_e = exp(param(2)); phi = exp(param(3));   

N_latent = zeros(TT,N); N_latent(1,:) = ones(1,N); E = exp(sigma_e * error);
for t = 2:TT
    N_latent(t,:) = r.*N_latent(t-1,:).*exp(-N_latent(t-1,:)).*E(t,:);
end
y = poissrnd(phi*N_latent); Y_sim = zeros(N,p+1);
for oo=1:N
    Y_sim(oo,:) = flip(y(TT-p:end,oo)');
end
loss = mmd(data,Y_sim,sigma);