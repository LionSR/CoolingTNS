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
if ~exist('coupling','var')
    coupling = "XX";
end
if ~exist('steps','var')
    steps = 300;  % Default to 300 steps
end

% Time evolution parameter te
if ~exist('te','var')
    % Default to delta/g ratio calculated below if not provided
    te_provided = false;
else
    te_provided = true;
end

%% Initialize system Hamiltonian and compute gap dynamically
HamS = CreateHamZZXZ(N,J,hx,hz);
[V,D] = eigs(HamS,M,'sa');
[~,ind] = sort(diag(D));
Ds = diag(D(ind,ind));
Vs = V(:,ind);
gap = Ds(2)-Ds(1);  % Dynamically compute the gap
fprintf("E_{GS}/N=%.3f, gap=%.3f\n",Ds(1)/N, gap);
EGS = Ds(1);
psi_gs = V(:,1);

% Set delta to exact resonance using the dynamically computed gap
if ~exist('delta','var')
    delta = -gap;  % Exact resonance (-gap) by default
end
if exist('factor','var')
    g = delta/factor;
end
if ~exist('g','var')
    g = delta/5.0;  % Standard coupling strength
end

% Set te if not provided
if ~te_provided
    te = abs(delta/g);
    fprintf("te not provided, using default te = |delta/g| = %.2f\n", te);
else
    fprintf("Using provided te = %.2f\n", te);
end

% Set up output directory and filenames
titlename = sprintf('CI_MB_N%dJ%.1fhx%.2fhz%.1f', N, J, hx, hz);
DirName = sprintf('N%dJ%.1fhx%.2fhz%.1f/', N, J, hx, hz);
if not(isfolder(DirName))
    mkdir(DirName);
end

% Create simple scheme name without multiple schemes complexity
if iscell(coupling)
    ct = coupling{1};
else
    ct = coupling;
end
SchemeName = sprintf('%s/SchemeMultiBath_%s%ddelta%.3fg%.3fte%.2f', ...
    DirName, ct, steps, delta, g, te);
titlename = sprintf('%s_%s%ddelta%.3fg%.3fte%.2f', ...
    titlename, ct, steps, delta, g, te);
disp(titlename);

%% Initial states
thetaList = -0.5:0.25:0.5;
thetaSelected = [thetaList(1), thetaList(3), thetaList(5)];
NStates = length(thetaSelected);

%% Run cooling for each of the selected states
% Initialize arrays with size steps+1 to include index 0 for initial state
EAll = zeros(steps+1, NStates);
GSOAll = zeros(steps+1, NStates);
purityAll = zeros(steps+1, NStates);
populAll = zeros(steps+1, M, NStates);
sZaAll = zeros(steps+1, NStates);
vstateAll = zeros(2^N, NStates);

% Display run information
fprintf("\nRunning resonant cooling simulation with %d steps: \n", steps);
fprintf("Using dynamically computed gap = %.4f and delta = %.4f\n", gap, delta);
fprintf("Using g = %.4f (g/delta = %.4f) with %s coupling\n", g, g/delta, coupling);
fprintf("Using te = %.4f for time evolution\n", te);

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
    
    % Create coupling parameters structure with te
    coupling_param = struct('delta', delta, 'g', g, 't', te);
    
    % Run evolution with parameters
    [sZ, E, GSO, purity, popul] = EvolveMultiBath(rho, Vs, steps, M, coupling, coupling_param, N, HamS, psi_gs, i);
    
    % Copy results to indices 2:steps+1 (since 1 is the initial state)
    sZaAll(2:steps+1, i) = sZ;
    EAll(2:steps+1, i) = E;
    GSOAll(2:steps+1, i) = GSO;
    purityAll(2:steps+1, i) = purity;
    populAll(2:steps+1,:, i) = popul;
end

fprintf("\n");
% Save results to the specific parameter directory
save(strcat(DirName, '/', titlename, '.mat'), 'EAll', 'GSOAll', 'purityAll', 'populAll', 'sZaAll', ...
     'thetaSelected', 'J', 'hx', 'hz', 'delta', 'g', 'te', 'steps', 'EGS', 'Ds', 'gap', 'coupling', 'SchemeName');

% Save coupling_params for compatibility with plotting code
coupling_params = struct('delta', delta, 'g', g, 't', te);
save(strcat(DirName, '/', titlename, '.mat'), 'coupling_params', '-append');

% Call the plotting function to visualize results
plotCoolingMultiBath;
