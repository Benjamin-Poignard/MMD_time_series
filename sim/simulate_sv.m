function y = simulate_sv(param,T,dist)

% SV model DGP

% inputs: - param: parameter of SV model
%         - T: number of time points
%         - dist: type of distribution: 'Gaussian' or 'Student'

% ouput: SV based generated data

b = param(1); sigma_e = param(2); sigma_y = param(3);
T = T+100;
y = zeros(T,1); vol = zeros(T,1);
gamma = 3;
for t = 2:T
    switch dist
        case 'Gaussian'
            vol(t) = b*vol(t-1) + sigma_e*randn(1);
        case 'Student'
            vol(t) = b*vol(t-1) + sigma_e*(sqrt(gamma-2)*trnd(gamma)/sqrt(gamma));
    end
    y(t) = sigma_y*exp(vol(t)/2)*randn(1);
end
y = y(101:end);