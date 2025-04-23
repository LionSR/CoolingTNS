%% CoolingMultiBath - Cooling simulation with N system spins coupled to N bath spins
% This unified implementation focuses on exact resonant cooling (Delta = -gap)
% with sufficient iterations to study long-term cooling dynamics

addpath('../Utils/','../Utils/expmv');

%% Parameters
if ~exist('N','var')
    N = 6;
end

if ~exist('hx','var')
    hx = -1.05;
end
if ~exist('hz','var')
    hz = 0.5;
end
if ~exist('J','var')
    J = 1.0;
end

% Number of eigenenergies to plot - we need at least 2 to calculate the gap
if ~exist('M','var')
    M = 2;
end

% Coupling parameters
if ~exist('coupling_types','var')
    coupling_types = "XX";
end
if ~exist('Niter','var')
    Niter = 300;  % Default to 300 iterations
end

% Time evolution parameter t
if ~exist('t','var')
    % Default to Delta/g ratio calculated below if not provided
    t_provided = false;
else
    t_provided = true;
end

%% Initialize system Hamiltonian and compute gap dynamically
HamS = CreateHamZZXZ(N,J,hx,hz);
[V,D] = eigs(HamS,M,'sa');
[~,ind] = sort(diag(D));
Ds = diag(D(ind,ind));
Vs = V(:,ind);
gap = Ds(2)-Ds(1);  % Dynamically compute the gap
fprintf("E_{GS}/L=%.3f, gap=%.3f\n",Ds(1)/N, gap);
EGS = Ds(1);
psi_gs = V(:,1);

% Set Delta to exact resonance using the dynamically computed gap
if ~exist('Delta','var')
    Delta = -gap;  % Exact resonance (-gap) by default
end
if exist('factor','var')
    g = Delta/factor;
end
if ~exist('g','var')
    g = Delta/5.0;  % Standard coupling strength
end

% Set t if not provided
if ~t_provided
    t = abs(Delta/g);
    fprintf("t not provided, using default t = |Delta/g| = %.2f\n", t);
else
    fprintf("Using provided t = %.2f\n", t);
end

% Set up output directory and filenames
titlename = sprintf('CI_MB_N%dJ%.1fhx%.2fhz%.1f', N, J, hx, hz);
DirName = sprintf('N%dJ%.1fhx%.2fhz%.1f/', N, J, hx, hz);
if not(isfolder(DirName))
    mkdir(DirName);
end

% Create simple scheme name without multiple schemes complexity
if iscell(coupling_types)
    ct = coupling_types{1};
else
    ct = coupling_types;
end
SchemeName = sprintf('%s/SchemeMultiBath_%s%dDelta%.3fg%.3ft%.2f', ...
    DirName, ct, Niter, Delta, g, t);
titlename = sprintf('%s_%s%dDelta%.3fg%.3ft%.2f', ...
    titlename, ct, Niter, Delta, g, t);
disp(titlename);

%% Initial states
thetaList = -0.5:0.25:0.5;
thetaSelected = [thetaList(1), thetaList(3), thetaList(5)];
NStates = length(thetaSelected);

%% Run cooling for each of the selected states
% Initialize arrays with size Niter+1 to include index 0 for initial state
EAll = zeros(Niter+1, NStates);
GSOAll = zeros(Niter+1, NStates);
purityAll = zeros(Niter+1, NStates);
populAll = zeros(Niter+1, M, NStates);
sZaAll = zeros(Niter+1, NStates);
vstateAll = zeros(2^N, NStates);

% Display run information
fprintf("\nRunning resonant cooling simulation with %d iterations: \n", Niter);
fprintf("Using dynamically computed gap = %.4f and Delta = %.4f\n", gap, Delta);
fprintf("Using g = %.4f (g/Delta = %.4f) with %s coupling\n", g, g/Delta, coupling_types);
fprintf("Using t = %.4f for time evolution\n", t);

for i = 1:NStates
    theta = thetaSelected(i);
    v0 = [cos(theta); sin(theta)];
    vstate = 1;
    for k = 1:N
        vstate = kron(vstate, v0);
    end
    Nst = vstate'*vstate;
    vstate = vstate/sqrt(Nst); % init normalized
    vstateAll(:, i) = vstate;
    
    % Calculate initial values and store at index 0
    rho = vstate*vstate';
    for j = 1:M
        populAll(1, j, i) = (abs(vstate'*Vs(:,j)))^2;
    end
    EAll(1, i) = real(vstate'*HamS*vstate);
    GSOAll(1, i) = (abs(vstate'*psi_gs))^2;
    
    % All initial states are pure states with purity = 1.0
    purityAll(1, i) = 1.0;
    
    % Initial bath Z is always 1.0 (bathless state)
    sZaAll(1, i) = 1.0;
end

% Run cooling for each initial state
for i = 1:NStates
    fprintf("\ntheta=%.2f pi, E_init/N=%.2f", thetaSelected(i), EAll(1, i)/N);
    rho = vstateAll(:, i) * vstateAll(:, i)';
    
    % Create coupling parameters structure with t included
    coupling_param = struct('Delta', Delta, 'g', g, 't', t);
    
    % Run evolution with parameters
    [sZ, E, GSO, purity, popul] = EvolveMultiBath(rho, Vs, Niter, M, coupling_types, coupling_param, N, HamS, psi_gs, i);
    
    % Copy results to indices 2:Niter+1 (since 1 is the initial state)
    sZaAll(2:Niter+1, i) = sZ;
    EAll(2:Niter+1, i) = E;
    GSOAll(2:Niter+1, i) = GSO;
    purityAll(2:Niter+1, i) = purity;
    populAll(2:Niter+1,:, i) = popul;
end

fprintf("\n");
% Save results to the specific parameter directory, not the main MatlabRho folder
save(strcat(DirName, '/', titlename, '.mat'), 'EAll', 'GSOAll', 'purityAll', 'populAll', 'sZaAll', ...
     'thetaSelected', 'J', 'hx', 'hz', 'Delta', 'g', 't', 'Niter', 'EGS', 'Ds', 'gap', 'coupling_types', 'SchemeName');

% Save coupling_params for compatibility with plotting code
coupling_params = struct('Delta', Delta, 'g', g, 't', t);
save(strcat(DirName, '/', titlename, '.mat'), 'coupling_params', '-append');

% Call the plotting function to visualize results
plotCoolingMultiBath;
