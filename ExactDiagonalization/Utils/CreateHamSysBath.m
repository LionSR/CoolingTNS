function Ham = CreateHamSysBath(HamS, N, delta, g, coupling_type)
    % CreateHamSysBath - Creates a Hamiltonian for N system spins coupled to N bath spins
    % with the same connectivity pattern as the Julia version
    %
    % The Hamiltonian structure is:
    % H = H_sys ⊗ I_bath + I_sys ⊗ H_bath + H_coupling
    %
    % Where:
    % - H_sys: System Hamiltonian (e.g., Ising model)
    % - H_bath = Σᵢ -Δ/2 * Z_i (Sum over all bath spins)
    % - H_coupling = Σᵢ g * O_sys_i ⊗ O_bath_i (Local coupling between each system spin and its bath)
    %
    % For H_bath = -Δ/2 * Z:
    % - If Δ < 0: ground state is |1⟩ (down)
    % - If Δ > 0: ground state is |0⟩ (up)
    %
    % Indexing scheme for the full 2N-spin space:
    % - System spins: indices 1 to N
    % - Bath spins: indices N+1 to 2N
    %
    % Inputs:
    %   HamS - System Hamiltonian (2^N x 2^N sparse matrix)
    %   N - Number of spins in the system
    %   delta - Detuning parameter for the bath
    %   g - Coupling strength between system and bath
    %   coupling_type - String specifying the coupling type (e.g., "XX", "YY", "XY")
    %
    % Output:
    %   Ham - Full Hamiltonian for system+bath (2^(2N) x 2^(2N) sparse matrix)
    
    % Process coupling type
    coupling_str = parseCouplingType(coupling_type);
    
    % Map operator character to index (1=X, 2=Y, 3=Z as expected by MultiSingleSpin)
    sys_op_idx = getOpIndex(coupling_str(1));
    bath_op_idx = getOpIndex(coupling_str(2));
    
    % Initialize the full Hamiltonian
    Ham = sparse(2^(2*N), 2^(2*N));
    
    % Add system Hamiltonian: H_sys ⊗ I_bath
    Ham = Ham + kron(HamS, speye(2^N));
    
    % Add bath Hamiltonian and coupling terms
    for i = 1:N
        % Bath spin index
        bath_idx = N + i;
        
        % Add bath Hamiltonian term: -delta/2 * Z_i
        Ham = Ham + (-delta/2) * MultiSingleSpin(2*N, bath_idx, 3); % 3 = Z
        
        % Add coupling term
        Ham = Ham + g * MultiSingleSpin(2*N, i, sys_op_idx) * ...
                     MultiSingleSpin(2*N, bath_idx, bath_op_idx);
    end
    
    % Helper function to get operator index from character
    function idx = getOpIndex(c)
        switch c
            case {'X', 'x'}
                idx = 1; % σˣ
            case {'Y', 'y'}
                idx = 2; % σʸ
            case {'Z', 'z'}
                idx = 3; % σᶻ
            otherwise
                error('Invalid Pauli operator: must be X, Y, or Z');
        end
    end
    
    % Helper function to parse coupling type from various input formats
    function coupling_str = parseCouplingType(ct)
        if iscell(ct)
            % Cell array like {"XX"}
            coupling_str = char(ct{1});
        elseif isstring(ct)
            % MATLAB string object
            coupling_str = char(ct);
        else
            % Already a char array
            coupling_str = ct;
        end
        
        if length(coupling_str) < 2
            error('Coupling type must be at least two characters (e.g., "XX", "YY")');
        end
        
        fprintf('Parsed coupling operators: %c and %c\n', coupling_str(1), coupling_str(2));
    end
end 