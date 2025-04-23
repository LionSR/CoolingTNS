function rhoA = TrXMulti(rho, N)
    % TrXMulti - Trace out N bath spins from a combined system+bath density matrix
    %
    % Inputs:
    %   rho - Full density matrix for system+bath (2^(2N) x 2^(2N))
    %   N - Number of spins in the system (and bath)
    %
    % Output:
    %   rhoA - Reduced density matrix for system only (2^N x 2^N)
    
    % Initialize the reduced density matrix
    dimA = 2^N;
    dimB = 2^N;
    rhoA = sparse(dimA, dimA);
    
    % More efficient approach for tracing out multiple spins
    % Loop through all possible bath states in computational basis
    for b = 0:dimB-1
        % Convert to binary representation for the bath state
        bBin = bitget(b, N:-1:1);
        
        % Create a projector onto this bath state
        projB = sparse(dimB, 1);
        projB(b+1) = 1;  % +1 because MATLAB is 1-indexed
        
        % Create full projector
        projBdm = kron(speye(dimA), projB);
        
        % Apply projector and accumulate reduced density matrix
        temp = projBdm' * rho * projBdm;
        rhoA = rhoA + temp(1:dimA, 1:dimA);
    end
    
    % Verify trace is approximately 1
    tr = full(trace(rhoA));
    if abs(tr - 1) > 1e-5
        fprintf('\nWarning: trace of reduced density matrix = %.8f', tr);
    end
end 