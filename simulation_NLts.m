%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Simulation experiments for Non-linear MA model %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% True DGP:  psi = 0.7, T = 300
% Select the distribution of the innovation: dist = 'Student' or 'Gaussian'
addpath(genpath(pwd))
clear
clc
rng(1, 'twister' );
Sim = 100; max_lag = 41;
param_ISMMD1 = zeros(1,max_lag,Sim); param_ISMMD2 = zeros(1,max_lag,Sim);
param_PSMMD1 = zeros(1,max_lag,Sim); param_PSMMD2 = zeros(1,max_lag,Sim);
param_mom = zeros(1,Sim);

param_true = 0.7;
dist = 'Student';
%dist = 'Gaussian';

for oo= 1:Sim
    % Non linear model simulation
    T = 300;
    Y_obs = simulate_NLts(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_NLts_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_NLts_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_NLts_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_NLts_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_NLts_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_NLts_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_NLts_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_NLts_psmmd(Y_p,2000,p,sigma);
    end
    param_mom(:,oo) = mean(Y_obs);
end

%% True DGP:  psi = 0.7, T = 1000
% Select the distribution of the innovation: dist = 'Student' or 'Gaussian'
addpath(genpath(pwd))
clear
clc
rng(2, 'twister' );
Sim = 100; max_lag = 41;
param_ISMMD1 = zeros(1,max_lag,Sim); param_ISMMD2 = zeros(1,max_lag,Sim);
param_PSMMD1 = zeros(1,max_lag,Sim); param_PSMMD2 = zeros(1,max_lag,Sim);
param_mom = zeros(1,Sim);

param_true = 0.7;
dist = 'Student';
%dist = 'Gaussian';

for oo= 1:Sim
    % Non linear model simulation
    T = 1000;
    Y_obs = simulate_NLts(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_NLts_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_NLts_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_NLts_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_NLts_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_NLts_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_NLts_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_NLts_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_NLts_psmmd(Y_p,2000,p,sigma);
    end
    param_mom(:,oo) = mean(Y_obs);
end

%% True DGP:  psi = 0.9, T = 300
% Select the distribution of the innovation: dist = 'Student' or 'Gaussian'
addpath(genpath(pwd))
clear
clc
rng(3, 'twister' );
Sim = 100; max_lag = 41;
param_ISMMD1 = zeros(1,max_lag,Sim); param_ISMMD2 = zeros(1,max_lag,Sim);
param_PSMMD1 = zeros(1,max_lag,Sim); param_PSMMD2 = zeros(1,max_lag,Sim);
param_mom = zeros(1,Sim);

param_true = 0.9;
dist = 'Student';
%dist = 'Gaussian';

for oo= 1:Sim
    % Non linear model simulation
    T = 300;
    Y_obs = simulate_NLts(param_true,T,dist);
    p=0;
    Y_p = zeros(T-p,p+1);
    
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_NLts_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_NLts_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_NLts_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_NLts_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_NLts_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_NLts_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_NLts_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_NLts_psmmd(Y_p,2000,p,sigma);
    end
    param_mom(:,oo) = mean(Y_obs);
end

%% True DGP:  psi = 0.9, T = 1000
% Select the distribution of the innovation: dist = 'Student' or 'Gaussian'
addpath(genpath(pwd))
clear
clc
rng(4, 'twister' );
Sim = 100; max_lag = 41;
param_ISMMD1 = zeros(1,max_lag,Sim); param_ISMMD2 = zeros(1,max_lag,Sim);
param_PSMMD1 = zeros(1,max_lag,Sim); param_PSMMD2 = zeros(1,max_lag,Sim);
param_mom = zeros(1,Sim);

param_true = 0.9;
dist = 'Student';
%dist = 'Gaussian';

for oo= 1:Sim
    % Non linear model simulation
    T = 1000;
    Y_obs = simulate_NLts(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_NLts_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_NLts_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_NLts_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_NLts_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_NLts_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_NLts_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_NLts_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_NLts_psmmd(Y_p,2000,p,sigma);
    end
    param_mom(:,oo) = mean(Y_obs);
end

%% \|\|_2-error, apply mean for averaging over the 100 batches
dist_NT = zeros(Sim,max_lag); dist_T = zeros(Sim,max_lag);
dist_NT2 = zeros(Sim,max_lag); dist_T2 = zeros(Sim,max_lag);
dist_mom = zeros(Sim,1);
for oo=1:Sim
    for k = 1:max_lag
        dist_NT(oo,k) = norm(param_ISMMD1(:,k,oo)-param_true);
        dist_T(oo,k) = norm(param_PSMMD1(:,k,oo)-param_true);
        dist_NT2(oo,k) = norm(param_ISMMD2(:,k,oo)-param_true);
        dist_T2(oo,k) = norm(param_PSMMD2(:,k,oo)-param_true);
    end
    dist_mom(oo) = norm(param_mom(:,oo)-param_PSMMD1rue);
end

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% Simulation experiments for Non-linear MA model %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Stochastic GD ISMMD_sgd vs ISMMD_2 %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% True DGP:  psi = 0.9, T = 1000
% Student case
addpath(genpath(pwd))
clear
clc
rng(4, 'twister' );
Sim = 100; max_lag = 41;
param_ISMMD1 = zeros(1,max_lag,Sim); param_ISMMD2 = zeros(1,max_lag,Sim);
param_PSMMD1 = zeros(1,max_lag,Sim); param_PSMMD2 = zeros(1,max_lag,Sim);

param_ISMMD1_det = zeros(1,max_lag,Sim); param_ISMMD2_det = zeros(1,max_lag,Sim);
param_PSMMD1_det = zeros(1,max_lag,Sim); param_PSMMD2_det = zeros(1,max_lag,Sim);

param_mom = zeros(1,Sim);

param_true = 0.9;
dist = 'Student';

for oo= 1:Sim
    % Non linear model simulation
    T = 1000;
    Y_obs = simulate_NLts(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);

    param_ISMMD1(:,1,oo) = gradient_NLts_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_NLts_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_NLts_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_NLts_psmmd(Y_p,2000,p,sigma);

    param_ISMMD1_det(:,1,oo) = gradient_NLts_ismmd_det(Y_p,100,p,1000,sigma);
    param_ISMMD2_det(:,1,oo) = gradient_NLts_ismmd_det(Y_p,100,p,2000,sigma);
    param_PSMMD1_det(:,1,oo) = gradient_NLts_psmmd_det(Y_p,1000,p,sigma);
    param_PSMMD2_det(:,1,oo) = gradient_NLts_psmmd_det(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);

	    param_ISMMD1(:,p+1,oo) = gradient_NLts_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_NLts_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_NLts_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_NLts_psmmd(Y_p,2000,p,sigma);

        param_ISMMD1_det(:,p+1,oo) = gradient_NLts_ismmd_det(Y_p,100,p,1000,sigma);
        param_ISMMD2_det(:,p+1,oo) = gradient_NLts_ismmd_det(Y_p,100,p,2000,sigma);
        param_PSMMD1_det(:,p+1,oo) = gradient_NLts_psmmd_det(Y_p,1000,p,sigma);
        param_PSMMD2_det(:,p+1,oo) = gradient_NLts_psmmd_det(Y_p,2000,p,sigma);
    end
    param_mom(:,oo) = mean(Y_obs);
end