% Test script for resonant cooling with exact resonance condition (Delta = -gap)
clear;

%% Set parameters for cooling simulation
% Simulation size and physics parameters
N = 5;                  % System size (small for quick testing)
J = 1.0;                % Interaction strength
hx = -1.05;             % x-field
hz = 0.5;               % z-field

% J = cos(pi/3);        % Interaction strength
% hx = sin(pi/3);       % x-field
% hz = 0;               % z-field

% Coupling parameters
coupling_types = "XX"; % Use XX interaction
Niter = 1000;            

% Note: Delta will be automatically set to -gap in CoolingMultiBath.m
% where gap is dynamically computed for the given system parameters.
% g will be automatically set to Delta/5.0 unless specified here.

g = 0.1;

% Time evolution parameter
t = 5.0;  % Uncomment to set a specific t value
% Otherwise t = |Delta/g| will be used by default


%% Display test configuration
disp('=== Testing Resonant Cooling Implementation ===');
disp(['System size: N = ', num2str(N)]);
disp(['Physics parameters: J = ', num2str(J), ', hx = ', num2str(hx), ', hz = ', num2str(hz)]);
disp(['Coupling type: ', coupling_types]);
disp(['Iterations: ', num2str(Niter)]);
disp(['Resonance condition: Delta = -gap (exact resonance, calculated dynamically)']);
disp(['Coupling strength: g = Delta/5.0 (default)']);
if exist('t', 'var')
    disp(['Evolution time: t = ', num2str(t)]);
else
    disp('Evolution time: t = |Delta/g| (default)');
end
disp('-----------------------------------------------------');

%% Run the cooling simulation
try
    disp('Starting cooling simulation...');
    CoolingMultiBath;
    disp('Cooling simulation completed successfully!');

    % Add final results summary
    fprintf('\nFinal Results Summary:\n');
    for i = 1:length(thetaSelected)
        fprintf('Initial State θ=%.2fπ: Energy %.5f → %.5f (Energy reduction: %.5f)\n', ...
            thetaSelected(i), EAll(1,i)/N, EAll(end,i)/N, EAll(1,i)/N - EAll(end,i)/N);
        fprintf('                      GS Overlap %.5f → %.5f\n', ...
            GSOAll(1,i), GSOAll(end,i));
    end
    fprintf('Ground State Energy: %.5f\n', EGS/N);
    fprintf('System Gap: %.5f\n', gap);
    fprintf('Used Delta: %.5f (ratio to gap: %.2f)\n', Delta, Delta/gap);
    fprintf('Used g: %.5f (ratio to Delta: %.2f)\n', g, g/Delta);
    fprintf('Used t: %.5f\n', t);

catch ME
    disp('Error occurred during testing:');
    disp(ME.message);
    for i = 1:length(ME.stack)
        disp(['Function: ', ME.stack(i).name, ', Line: ', num2str(ME.stack(i).line)]);
    end
end