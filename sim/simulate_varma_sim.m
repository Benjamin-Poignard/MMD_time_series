function Y = simulate_varma_sim(E, T, p, q, Phi, Theta, Sigma)

% inputs: - T: Number of time points
%         - p: Order of the AR component
%         - q: Order of the MA component
%         - Phi: AR coefficient matrices (cell array of size p)
%         - Theta: MA coefficient matrices (cell array of size q)
%         - Sigma: variance of the white noise

% output: - Y: simulated path

    % Initialize the time series with zeros or some initial values
    Y = zeros(T,1);
    for t = 1:T
       E(t) = sqrt(Sigma)*E(t);
    end
    % Simulate the VARMA process
    for t = max(p, q) + 1:T
        % AR component
        ar_term = zeros(1,1);
        for i = 1:p
            ar_term = ar_term + Y(t-i) * Phi;
        end

        % MA component
        ma_term = zeros(1,1);
        for j = 1:q
            ma_term = ma_term + E(t-j) * Theta;
        end

        % Update the time series
        Y(t, :) = ar_term + ma_term + E(t);
    end
end