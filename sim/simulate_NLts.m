function y = simulate_NLts(param,T,dist)

% Non-linear MA model DGP

% inputs: - param: model parameter 
%         - T: number of time points
%         - dist: type of distribution: 'Gaussian' or 'Student'

% ouput: non-linear MA based generated data

T = T+100;
y = zeros(T,1); e = zeros(T,1); gamma = 3;
switch dist
    case 'Gaussian'
        e(1) = mvnrnd(0,1);
    case 'Student'
        e(1) = (sqrt(gamma-2)*trnd(gamma)/sqrt(gamma));
end
for t = 2:T
    switch dist
        case 'Gaussian'
            e(t) = mvnrnd(0,1);
            y(t) =  e(t)+param*e(t-1)^2;
        case 'Student'
            e(t) = (sqrt(gamma-2)*trnd(gamma)/sqrt(gamma));
            y(t) =  e(t)+param*e(t-1)^2;
    end
end
y = y(101:end);