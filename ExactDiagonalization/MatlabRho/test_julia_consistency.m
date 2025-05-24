% test_julia_consistency.m - Test MATLAB-Julia consistency with unified naming

clear;
fprintf('=== Testing MATLAB-Julia Consistency ===\n\n');

%% Test with Julia-style parameter names
fprintf('MATLAB now uses Julia-style parameter names:\n');

% System parameters
N = 4;
J = 1.0;
hx = -1.05;
hz = 0.5;

% Julia-style names (now standard in MATLAB)
coupling = 'XX';    % was 'coupling_types'
steps = 10;         % was 'Niter'
te = 2.0;           % was 't'
delta = -1.0;       % was 'Delta'
g = 0.1;

fprintf('Parameters:\n');
fprintf('  N = %d\n', N);
fprintf('  coupling = %s\n', coupling);
fprintf('  steps = %d\n', steps);
fprintf('  te = %.1f\n', te);
fprintf('  delta = %.3f\n', delta);
fprintf('  g = %.1f\n', g);
fprintf('\nRunning CoolingMultiBath...\n');

% Run with standard MATLAB script (now uses Julia names)
CoolingMultiBath;

fprintf('\n=== Test Complete ===\n');
fprintf('\nTo run equivalent Julia simulation:\n');
fprintf('julia Cooling.jl --N %d --method MPO --init_state identity --steps %d --te %.1f --g %.1f\n', ...
        N, steps, te, g);

%% Show parameter naming convention
fprintf('\n=== Parameter Naming Convention ===\n');
fprintf('MATLAB and Julia now use the same names:\n');
fprintf('  steps       (was Niter in MATLAB)\n');
fprintf('  te          (was t in MATLAB)\n');
fprintf('  delta       (was Delta in MATLAB)\n');
fprintf('  coupling    (was coupling_types in MATLAB)\n');
fprintf('\nAll other parameters (N, J, hx, hz, g) remain the same.\n');