function [sZ,E,GSO,purity,popul] = EvolveMultiBath(rho, Vs, steps, M, coupling_type, coupling_param, N, HamS, psi_gs, thetaind)
    % EvolveMultiBath - Evolve density matrix with N system spins + N bath spins
    %
    % Inputs:
    %   rho - Initial density matrix of system (2^N x 2^N)
    %   Vs - Eigenvectors of system Hamiltonian
    %   steps - Number of cooling steps
    %   M - Number of eigenenergies to track
    %   coupling_type - String with coupling type (e.g., "XX")
    %   coupling_param - Structure with delta, g, and t (evolution time)
    %   N - Number of spins in the system
    %   HamS - System Hamiltonian
    %   psi_gs - Ground state of system
    %   thetaind - Initial state indicator (for printing)
    
    % Convert coupling_type to string if needed
    if iscell(coupling_type)
        coupling_type = coupling_type{1};
    end
    
    % Extract parameters from the structure
    delta = coupling_param.delta;
    g = coupling_param.g;
    
    % Get evolution time (te) from parameters
    if isfield(coupling_param, 't')
        te = coupling_param.t;  % Still accept 't' field for now
    else
        te = abs(delta/g);
        fprintf('\nNote: te not provided, using default te = |delta/g| = %.2f', te);
    end
    
    % Initialize result arrays - steps rows
    E = zeros(steps, 1);
    GSO = zeros(steps, 1);
    purity = zeros(steps, 1);
    popul = zeros(steps, M);
    sZ = zeros(steps, 1);
    
    % Initialize bath spins in their ground state based on delta sign
    % For H_bath = -Δ/2 * Z, the ground state depends on sign of Δ
    % If Δ < 0: ground state is |1⟩ (down)
    % If Δ > 0: ground state is |0⟩ (up)
    if delta < 0
        % Ground state is |1⟩ (down) for negative delta
        fprintf('\nInitializing bath in |1⟩ (down) state (GS for delta < 0)');
        vzminus = [0;1];
        rhobath = vzminus*vzminus';
    else
        % Ground state is |0⟩ (up) state for positive delta
        fprintf('\nInitializing bath in |0⟩ (up) state (GS for delta > 0)');
        vzplus = [1;0];
        rhobath = vzplus*vzplus';
    end
    
    % Create full bath state
    bath_state = rhobath;
    for i = 2:N
        bath_state = kron(bath_state, rhobath);
    end
    
    % Bath Z operator for each bath spin
    sZ_bath = cell(N,1);
    for i = 1:N
        sZ_i = sparse([1 0; 0 -1]);
        for j = 1:N
            if j ~= i
                sZ_i = kron(sZ_i, speye(2));
            end
        end
        sZ_bath{i} = kron(speye(2^N), sZ_i);
    end
    
    % Initial energy and GS overlap
    E0 = real(trace(HamS*rho));
    GSO0 = real(psi_gs'*rho*psi_gs);
    fprintf('\nInitial energy: E/N = %.5f, GSO = %.5f', E0/N, GSO0);
    
    % Create time evolution operator with provided te
    fprintf('\nCreating Hamiltonian with N=%d, delta=%.6f, g=%.6f, coupling=%s...', ...
        N, delta, g, coupling_type);
    HamSB = CreateHamSysBath(HamS, N, delta, g, coupling_type);
    
    fprintf('\nCreating time evolution operator with te=%.2f...', te);
    Ut = fastExpm(-1i*te*HamSB);
    
    tic;
    for i = 1:steps
        % Append bath in ground state to system density matrix
        rho_full = kron(rho, bath_state);
        
        % Evolve
        rho_full = Ut * rho_full * Ut';
        
        % Measure average Z for each bath spin
        sZ_avg = 0;
        for j = 1:N
            sZ_avg = sZ_avg + real(trace(rho_full * sZ_bath{j}))/N;
        end
        sZ(i) = sZ_avg;
        
        % Trace out bath spins
        rho_traced = TrXMulti(rho_full, N);
        
        % Renormalize the density matrix to ensure trace=1
        tr_rho = trace(rho_traced);
        % if abs(tr_rho - 1.0) > 1e-5
        %     fprintf('\nWarning: trace of reduced density matrix = %g, renormalizing', tr_rho);
        %     rho_traced = rho_traced / tr_rho;
        % end
        rho = rho_traced / tr_rho;
        
        % Calculate observables
        E(i) = real(trace(HamS*rho));
        GSO(i) = real(psi_gs'*rho*psi_gs);
        purity(i) = real(trace(rho^2));
        
        for j = 1:M
            popul(i,j) = real(Vs(:,j)'*rho*Vs(:,j));
        end
        
        if mod(i, 5) == 0 || i == 1
            fprintf('\nIteration %d: E/N = %.5f, GSO = %.5f, bath Z = %.5f', i, E(i)/N, GSO(i), sZ(i));
        end
    end
    
    % Print final summary
    time_elapsed = toc;
    fprintf("\nState #%d: Final E/N=%.5f, Purity=%.3f, GS Overlap=%.5f (completed in %.2f sec)", ...
            thetaind, E(end)/N, purity(end), GSO(end), time_elapsed);
end 