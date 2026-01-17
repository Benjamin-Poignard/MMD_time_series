%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Simulation experiments for ARMA model %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% True DGP:  Phi = 0.8, Psi = 0.15, Sigma_true = 0.05, T = 300
% Select the distribution of the innovation: dist = 'Gaussian' or 'Student'
addpath(genpath(pwd))
clear
clc
% Parameters
rng(1, 'twister' );
p_lag = 1; % AR order
q_lag = 1; % MA order

Sim = 100;
% AR coefficients
Phi = {0.8};
% MA coefficients
Psi = {0.15};
% Covariance matrix for the Gaussian white noise
Sigma_true = 0.05;
% Distribution of the true innovations
dist = 'Gaussian';
% dist = 'Student';

max_lag = 16;
param_ISMMD1 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_ISMMD2 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_PSMMD1 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_PSMMD2 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_MLE = zeros((p_lag+q_lag)+1,Sim);

for oo = 1:Sim
    T = 300; % Number of time points
    % Simulate the VARMA model
    Y_obs = simulate_varma(T,p_lag,q_lag,Phi,Psi,Sigma_true,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,2000,p,sigma);
    end
    
    model = arima(1,0,1); model.Constant = 0; EstMdl = estimate(model,Y_obs);
    param_MLE(:,oo) = [EstMdl.AR{1};EstMdl.MA{1};EstMdl.Variance];
end

%% True DGP:  Phi = 0.8, Psi = 0.15, Sigma_true = 0.05, T = 1000
% Select the distribution of the innovation: dist = 'Gaussian' or 'Student'
addpath(genpath(pwd))
clear
clc
% Parameters
rng(2, 'twister' );
p_lag = 1; % AR order
q_lag = 1; % MA order

Sim = 100;
% AR coefficients
Phi = {0.8};
% MA coefficients
Psi = {0.15};
% Covariance matrix for the Gaussian white noise
Sigma_true = 0.05;
% Distribution of the true innovations
dist = 'Gaussian';
% dist = 'Student';

max_lag = 16;
param_ISMMD1 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_ISMMD2 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_PSMMD1 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_PSMMD2 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_MLE = zeros((p_lag+q_lag)+1,Sim);


for oo = 1:Sim
    T = 1000; % Number of time points
    % Simulate the VARMA model
    Y_obs = simulate_varma(T,p_lag,q_lag,Phi,Psi,Sigma_true,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,2000,p,sigma);
    end
    
    model = arima(1,0,1); model.Constant = 0; EstMdl = estimate(model,Y_obs);
    param_MLE(:,oo) = [EstMdl.AR{1};EstMdl.MA{1};EstMdl.Variance];
end

%% True DGP:  Phi = 0.9, Psi = 0.08, Sigma_true = 0.03, T = 300
% Select the distribution of the innovation: dist = 'Gaussian' or 'Student'
addpath(genpath(pwd))
clear
clc
% Parameters
rng(3, 'twister' );
p_lag = 1; % AR order
q_lag = 1; % MA order

Sim = 100;
% AR coefficient
Phi = {0.9};
% MA coefficients
Psi = {0.08};
% Covariance matrix for the Gaussian white noise
Sigma_true = 0.03;
% Distribution of the true innovations
dist = 'Gaussian';
% dist = 'Student';

max_lag = 16;
param_ISMMD1 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_ISMMD2 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_PSMMD1 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_PSMMD2 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_MLE = zeros((p_lag+q_lag)+1,Sim);


for oo = 1:Sim
    T = 300; % Number of time points
    % Simulate the VARMA model
    Y_obs = simulate_varma(T,p_lag,q_lag,Phi,Psi,Sigma_true,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,2000,p,sigma);
    end
    
    model = arima(1,0,1); model.Constant = 0; EstMdl = estimate(model,Y_obs);
    param_MLE(:,oo) = [EstMdl.AR{1};EstMdl.MA{1};EstMdl.Variance];
end

%% True DGP:  Phi = 0.9, Psi = 0.08, Sigma_true = 0.03, T = 1000
% Select the distribution of the innovation: dist = 'Gaussian' or 'Student'
addpath(genpath(pwd))
clear
clc
% Parameters
rng(4, 'twister' );
p_lag = 1; % AR order
q_lag = 1; % MA order

Sim = 100;
% AR coefficients
Phi = {0.9};
% MA coefficients
Psi = {0.08};
% Covariance matrix for the Gaussian white noise
Sigma_true = 0.03;
% Distribution of the true innovations
dist = 'Gaussian';
% dist = 'Student';

max_lag = 16;
param_ISMMD1 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_ISMMD2 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_PSMMD1 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_PSMMD2 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_MLE = zeros((p_lag+q_lag)+1,Sim);


for oo = 1:Sim
    T = 1000; % Number of time points
    % Simulate the VARMA model
    Y_obs = simulate_varma(T,p_lag,q_lag,Phi,Psi,Sigma_true,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,2000,p,sigma);
    end
    
    model = arima(1,0,1); model.Constant = 0; EstMdl = estimate(model,Y_obs);
    param_MLE(:,oo) = [EstMdl.AR{1};EstMdl.MA{1};EstMdl.Variance];
end

%% \|\|_2-error, apply mean for averaging over the 100 batches
param_true = [Phi{1} Psi{1} Sigma_true]';
dist_NT = zeros(Sim,max_lag); dist_T = zeros(Sim,max_lag);
dist_NT2 = zeros(Sim,max_lag); dist_T2 = zeros(Sim,max_lag);
dist_MLE = zeros(Sim,1);
for oo=1:Sim
    for k = 1:max_lag
        dist_NT(oo,k) = norm(param_ISMMD1(:,k,oo)-param_true);
        dist_T(oo,k) = norm(param_PSMMD1(:,k,oo)-param_true);
        dist_NT2(oo,k) = norm(param_ISMMD2(:,k,oo)-param_true);
        dist_T2(oo,k) = norm(param_PSMMD2(:,k,oo)-param_true);
    end
    dist_MLE(oo) = norm(param_MLE(:,oo)-param_true);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Simulation experiments for ARMA model %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Stochastic GD ISMMD_sgd vs ISMMD_2 %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% True DGP:  Phi = 0.8, Psi = 0.15, Sigma_true = 0.05, T = 1000
% Select the distribution of the innovation: dist = 'Gaussian' or 'Student'
addpath(genpath(pwd))
clear
clc
% Parameters
rng(2, 'twister' );
p_lag = 1; % AR order
q_lag = 1; % MA order

Sim = 100;
% AR coefficients
Phi = {0.8};
% MA coefficients
Psi = {0.15};
% Covariance matrix for the Gaussian white noise
Sigma_true = 0.05;
% Distribution of the true innovations
dist = 'Student';

max_lag = 41;
param_ISMMD1 = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_PSMMD1 = zeros((p_lag+q_lag)+1,max_lag,Sim);

param_ISMMD1_det = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_PSMMD1_det = zeros((p_lag+q_lag)+1,max_lag,Sim);
param_MLE = zeros((p_lag+q_lag)+1,Sim);

for oo = 1:Sim
    T = 1000; % Number of time points
    % Simulate the VARMA model
    Y_obs = simulate_varma(T,p_lag,q_lag,Phi,Psi,Sigma_true,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,1000,sigma);
    param_PSMMD1(:,1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,1000,p,sigma);
    
    param_ISMMD1_det(:,1,oo) = gradient_arma_ismmd_det(Y_p,p_lag,q_lag,100,p,1000,sigma);
    param_PSMMD1_det(:,1,oo) = gradient_arma_psmmd_det(Y_p,p_lag,q_lag,1000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,1000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_arma_psmmd(Y_p,p_lag,q_lag,1000,p,sigma);
        
        param_ISMMD1_det(:,p+1,oo) = gradient_arma_ismmd_det(Y_p,p_lag,q_lag,100,p,1000,sigma);
        param_PSMMD1_det(:,p+1,oo) = gradient_arma_psmmd_det(Y_p,p_lag,q_lag,1000,p,sigma);
    end
    
    model = arima(1,0,1); model.Constant = 0; EstMdl = estimate(model,Y_obs);
    param_MLE(:,oo) = [EstMdl.AR{1};EstMdl.MA{1};EstMdl.Variance];
    
end