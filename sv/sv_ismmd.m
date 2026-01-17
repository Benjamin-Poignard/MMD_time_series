function loss = sv_ismmd(param,data,error1,error2,TT,p,N,sigma)

% Compute MMD loss function for SV-based time series

% inputs: - param: vector of parameters
%         - data: T x 1 vector containing the time series
%         - error1: innovation v_t
%         - error2: innovation \eta_t  ==> u_t = (v_t,\eta_t)'
%         - TT: T0 length as in ISMMD
%         - p: lag to construct \tilde{y}_i: x^{(i)}_{TT},...,x^{(i)}_{TT-p}
%         - N: length of the simulated vector \tilde{y}_i
%         - sigma: Gaussian kernel hyperparameter set by median heuristic

% output: loss: MMD loss function

b = param(1); sigma_e = param(2); sigma_y = param(3);
y = zeros(TT,N); var_sto = zeros(TT,N); var_sto(1,:) = var(data(:,1,1))*ones(1,N);
for t = 2:TT
    var_sto(t,:) = b*var_sto(t-1,:) + sigma_e*error1(t,:);
    y(t,:) = sigma_y.*exp(var_sto(t,:)/2).*error2(t,:);
end
Y_sim = zeros(N,p+1);
for oo=1:N
    Y_sim(oo,:) = flip(y(TT-p:end,oo)');
end
loss = mmd(data,Y_sim,sigma);