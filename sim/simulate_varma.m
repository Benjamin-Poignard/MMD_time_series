function y = simulate_varma(T,p,q,Phi,Theta,Sigma,dist)

% ARMA model DGP

% inputs: - T: Number of time points
%         - p: Order of the AR component
%         - q: Order of the MA component
%         - Phi: AR coefficient matrices (cell array of size p)
%         - Theta: MA coefficient matrices (cell array of size q)
%         - Sigma: Covariance matrix of the Gaussian white noise
%         - dist: type of distribution: 'Gaussian' or 'Student'

% ouput: ARMA(1,1) based generated data

T = T+100;
% Initialize the time series with zeros or some initial values
y = zeros(T,1); e = zeros(T,1);

switch dist
    case 'Gaussian'
        for t = 1:T
            e(t) = sqrt(Sigma)*randn(1);
        end
    case 'Student'
        gamma = 3;
        for t = 1:T
            e(t) = sqrt(Sigma)*(sqrt(gamma-2)*trnd(gamma)/sqrt(gamma));
        end
        
end
% Simulate the VARMA process
for t = max(p,q) + 1:T
    % AR component
    ar_term = zeros(1,1);
    for i = 1:p
        ar_term = ar_term + y(t-i, :) * Phi{i}';
    end
    
    % MA component
    ma_term = zeros(1,1);
    for j = 1:q
        ma_term = ma_term + e(t-j, :) * Theta{j}';
    end
    
    % Update the time series
    y(t, :) = ar_term + ma_term + e(t);
end
y = y(101:end);
end