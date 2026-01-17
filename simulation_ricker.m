%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Simulation experiments for Ricker model %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% True DGP:  logr = log(5), sigma_e = 0.03, phi = 5, T = 300
% Select the distribution of the innovation: dist = 'Student' or 'Gaussian'
addpath(genpath(pwd))
clear
clc
rng(1, 'twister' );
Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_ISMMD2 = zeros(3,max_lag,Sim);
param_PSMMD1 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);
param_SL = zeros(3,Sim);

lb = [log(1),0.01,1]; ub = [log(10),0.15,15];
Rsim = 1000;
opts = optimoptions('patternsearch','Display','iter', ...
    'MaxFunctionEvaluations',50000,'MaxIterations',10000, ...
    'UseCompletePoll',true,'PollMethod','GSSPositiveBasis2N');

% [log(r), sigma, phi]
logr = log(5); sigma_e = 0.03; phi = 5; param_true = [logr,sigma_e,phi];
%dist = 'Gaussian';
dist = 'Student';

for oo= 1:Sim
    T = 300;
    Y_obs = simulate_ricker(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_ricker_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_ricker_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_ricker_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_ricker_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_ricker_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_ricker_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_ricker_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_ricker_psmmd(Y_p,2000,p,sigma);
    end
    
    startcoeff = [log(4+(9-4)*rand(1));log(0.01+(0.1-0.01)*rand(1));log(4+(9-4)*rand(1))];
    [theta_SL, fval] = patternsearch(@(x)neglog_synthlik_ricker(x,Y_obs,Rsim), startcoeff, [], [], [], [], lb, ub, [], opts);
    param_SL(:,oo) = theta_SL';
end
%% True DGP:  logr = log(5), sigma_e = 0.03, phi = 5, T = 1000
% Select the distribution of the innovation: dist = 'Student' or 'Gaussian'
addpath(genpath(pwd))
clear
clc
rng(2, 'twister' );
Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_ISMMD2 = zeros(3,max_lag,Sim);
param_PSMMD1 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);
param_SL = zeros(3,Sim);

lb = [log(1),0.01,1]; ub = [log(10),0.15,15];
Rsim = 1000;
opts = optimoptions('patternsearch','Display','iter', ...
    'MaxFunctionEvaluations',50000,'MaxIterations',10000, ...
    'UseCompletePoll',true,'PollMethod','GSSPositiveBasis2N');

% [log(r), sigma, phi]
logr = log(5); sigma_e = 0.03; phi = 5; param_true = [logr,sigma_e,phi];
%dist = 'Gaussian';
dist = 'Student';

for oo= 1:Sim
    T = 1000;
    Y_obs = simulate_ricker(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_ricker_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_ricker_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_ricker_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_ricker_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_ricker_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_ricker_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_ricker_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_ricker_psmmd(Y_p,2000,p,sigma);
    end
    
    startcoeff = [log(4+(9-4)*rand(1));log(0.01+(0.1-0.01)*rand(1));log(4+(9-4)*rand(1))];
    [theta_SL, fval] = patternsearch(@(x)neglog_synthlik_ricker(x,Y_obs,Rsim), startcoeff, [], [], [], [], lb, ub, [], opts);
    param_SL(:,oo) = theta_SL';
end
%% True DGP:  logr = log(7), sigma_e = 0.05, phi = 7, T = 300
% Select the distribution of the innovation: dist = 'Student' or 'Gaussian'
addpath(genpath(pwd))
clear
clc
rng(3, 'twister' );
Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_ISMMD2 = zeros(3,max_lag,Sim);
param_PSMMD1 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);
param_SL = zeros(3,Sim);

lb = [log(1),0.01,1]; ub = [log(10),0.15,15];
Rsim = 1000;
opts = optimoptions('patternsearch','Display','iter', ...
    'MaxFunctionEvaluations',50000,'MaxIterations',10000, ...
    'UseCompletePoll',true,'PollMethod','GSSPositiveBasis2N');

% [log(r), sigma, phi]
logr = log(7); sigma_e = 0.05; phi = 7; param_true = [logr,sigma_e,phi];
%dist = 'Gaussian';
dist = 'Student';

for oo= 1:Sim
    T = 300;
    Y_obs = simulate_ricker(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_ricker_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_ricker_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_ricker_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_ricker_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_ricker_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_ricker_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_ricker_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_ricker_psmmd(Y_p,2000,p,sigma);
    end
    
    startcoeff = [log(4+(9-4)*rand(1));log(0.01+(0.1-0.01)*rand(1));log(4+(9-4)*rand(1))];
    [theta_SL, fval] = patternsearch(@(x)neglog_synthlik_ricker(x,Y_obs,Rsim), startcoeff, [], [], [], [], lb, ub, [], opts);
    param_SL(:,oo) = theta_SL';
end
%% True DGP:  logr = log(7), sigma_e = 0.05, phi = 7, T = 300
% Select the distribution of the innovation: dist = 'Student' or 'Gaussian'
addpath(genpath(pwd))
clear
clc
rng(4, 'twister' );
Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_ISMMD2 = zeros(3,max_lag,Sim);
param_PSMMD1 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);
param_SL = zeros(3,Sim);

lb = [log(1),0.01,1]; ub = [log(10),0.15,15];
Rsim = 1000;
opts = optimoptions('patternsearch','Display','iter', ...
    'MaxFunctionEvaluations',50000,'MaxIterations',10000, ...
    'UseCompletePoll',true,'PollMethod','GSSPositiveBasis2N');

% [log(r), sigma, phi]
logr = log(7); sigma_e = 0.05; phi = 7; param_true = [logr,sigma_e,phi];
%dist = 'Gaussian';
dist = 'Student';

for oo= 1:Sim
    T = 1000;
    Y_obs = simulate_ricker(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_ricker_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_ricker_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_ricker_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_ricker_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_ricker_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_ricker_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_ricker_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_ricker_psmmd(Y_p,2000,p,sigma);
    end
    
    startcoeff = [log(4+(9-4)*rand(1));log(0.01+(0.1-0.01)*rand(1));log(4+(9-4)*rand(1))];
    [theta_SL, fval] = patternsearch(@(x)neglog_synthlik_ricker(x,Y_obs,Rsim), startcoeff, [], [], [], [], lb, ub, [], opts);
    param_SL(:,oo) = theta_SL';
end

%% \|\|_2-error, apply mean for averaging over the 100 batches
dist_NT = zeros(Sim,max_lag); dist_T = zeros(Sim,max_lag);
dist_NT2 = zeros(Sim,max_lag); dist_T2 = zeros(Sim,max_lag);
param_true = [logr,sigma_e,phi]';
dist_SL = zeros(Sim,1);
for oo=1:Sim
    for k = 1:max_lag
        dist_NT(oo,k) = norm(param_ISMMD1(:,k,oo)-param_true);
        dist_T(oo,k) = norm(param_PSMMD1(:,k,oo)-param_true);
        dist_NT2(oo,k) = norm(param_ISMMD2(:,k,oo)-param_true);
        dist_T2(oo,k) = norm(param_PSMMD2(:,k,oo)-param_true);
    end
    dist_SL(oo) = norm(param_SL(:,oo)-param_true);
end