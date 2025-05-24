% Test script for resonant cooling with exact resonance condition (delta = -gap)
% Updated to use Julia-compatible naming conventions

clear;

%% Set parameters for cooling simulation
% Simulation size and physics parameters
N = 5;                  % System size (small for quick testing)
J = 1.0;                % Interaction strength
hx = -1.05;             % x-field
hz = 0.5;               % z-field

% Alternative parameters (commented out)
% J = cos(pi/3);        % Interaction strength
% hx = sin(pi/3);       % x-field
% hz = 0;               % z-field

% Coupling parameters (using Julia naming)
coupling = "XX";        % System-bath coupling operator
steps = 1000;           % Number of cooling iterations

% Note: delta will be automatically set to -gap in CoolingMultiBathV2.m
% where gap is dynamically computed for the given system parameters.
% g will be automatically set to delta/5.0 unless specified here.

g = 0.1;

% Time evolution parameter
te = 5.0;               % Total evolution time per step
% Otherwise te = |delta/g| will be used by default


%% Display test configuration
disp('=== Testing Resonant Cooling Implementation ===');
fprintf('System size: N = %d\n', N);
fprintf('Physics parameters: J = %.1f, hx = %.2f, hz = %.1f\n', J, hx, hz);
fprintf('Coupling type: %s\n', coupling);
fprintf('Steps: %d\n', steps);
fprintf('Resonance condition: delta = -gap (exact resonance, calculated dynamically)\n');
fprintf('Coupling strength: g = %.3f\n', g);
fprintf('Evolution time: te = %.1f\n', te);
disp('-----------------------------------------------------');

%% Run the cooling simulation
try
    disp('Starting cooling simulation...');
    CoolingMultiBath;
    disp('Cooling simulation completed successfully!');

    % Add final results summary (using Julia naming)
    fprintf('\nFinal Results Summary:\n');
    for i = 1:length(thetaSelected)
        fprintf('Initial State θ=%.2fπ: Energy %.5f → %.5f (Energy reduction: %.5f)\n', ...
            thetaSelected(i)/pi, EAll(1,i)/N, EAll(end,i)/N, EAll(1,i)/N - EAll(end,i)/N);
        fprintf('                      GS Overlap %.5f → %.5f\n', ...
            GSOAll(1,i), GSOAll(end,i));
    end
    fprintf('Ground State Energy: %.5f\n', EGS/N);
    fprintf('System Gap: %.5f\n', gap);
    fprintf('Used delta: %.5f (ratio to gap: %.2f)\n', delta, delta/gap);
    fprintf('Used g: %.5f (ratio to delta: %.2f)\n', g, g/delta);
    fprintf('Used te: %.5f\n', te);
    
    % Use plotting function
    fprintf('\nGenerating plots...\n');
    plotCoolingMultiBath;

catch ME
    disp('Error occurred during testing:');
    disp(ME.message);
    for i = 1:length(ME.stack)
        disp(['Function: ', ME.stack(i).name, ', Line: ', num2str(ME.stack(i).line)]);
    end
end