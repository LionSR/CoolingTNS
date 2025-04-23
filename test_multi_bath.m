% Test script for the multi-bath implementation
clear;

% Set parameters for a quick test
N = 4;  % Small system size for faster testing
hx = -1.05;
hz = 0.5;
J = 1.0;

% Use a single coupling scheme with standard parameters
coupling_types = {"XX"};
Niter = [30];  % Fewer iterations for quick test
GapList = [0 0 1.1480 0.8484 0.8006 0.6323 0.6327 0.5185 0.5327 0.4483 0.4660 0.4009 0.4182 0.3667 0.3824 0.3410 0.3546 0.3209 0.3325];
Delta = -GapList(N)/2;  % Same as Julia version
g = Delta/2.5;  % Standard g value

disp('=== Testing Multi-Bath Implementation ===');
disp(['System size: N = ', num2str(N)]);
disp(['Delta: ', num2str(Delta)]);
disp(['g: ', num2str(g)]);
disp(['Coupling type: ', coupling_types{1}]);
disp(['Iterations: ', num2str(Niter)]);

% Run the cooling simulation
try
    CoolingMultiBath;
    disp('Test completed successfully!');
catch ME
    disp('Error occurred during testing:');
    disp(ME.message);
    disp(getReport(ME));
end 