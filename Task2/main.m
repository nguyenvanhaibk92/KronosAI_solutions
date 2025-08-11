%% Buried Waveguide Mode Analysis Script
% This script analyzes vectorial mode profiles and effective indices
% for TE0 and TM0 modes in rib waveguides at different wavelengths

% Initialize workspace
clear all; 
close all; 
clc;

% Add functions directory to path
addpath(fullfile(pwd, 'functions'));

%% ========================================================================
%% TASK 1: Visualize vectorial mode profiles for TE0 and TM0 modes
%% ========================================================================

fprintf('=== TASK 1: Mode Profile Visualization ===\n');

% --- Analysis at 1550 nm wavelength ---
fprintf('Analyzing modes at 1550 nm...\n');
lam0 = 1550;        % Wavelength [nm]
rib_w = 45;         % Rib width [nm]

% Solve for waveguide modes
[Ex, Ey, Ez, Hx, Hy, Hz, NEFF, Gamma_Si] = rib_waveguide_solver(lam0, rib_w);

% Plot domain structure
plot_domain(1);

% Plot TE0 mode (Ex component)
plot_single_field(Ex, 'TE_0 (E_x)', 1, NEFF, ['TE0_lam0_' num2str(lam0) '.png']);

% Plot TM0 mode (Ey component)  
plot_single_field(Ey, 'TM_0 (E_y)', 2, NEFF, ['TM0_lam0_' num2str(lam0) '.png']);

% --- Analysis at 1310 nm wavelength ---
fprintf('Analyzing modes at 1310 nm...\n');
lam0 = 1310;        % Wavelength [nm]
rib_w = 450;        % Rib width [nm]

% Solve for waveguide modes
[Ex, Ey, Ez, Hx, Hy, Hz, NEFF, Gamma_Si] = rib_waveguide_solver(lam0, rib_w);

% Plot TE0 mode (Ex component)
plot_single_field(Ex, 'TE_0 (E_x)', 1, NEFF, ['TE0_lam0_' num2str(lam0) '.png']);

% Plot TM0 mode (Ey component)
plot_single_field(Ey, 'TM_0 (E_y)', 2, NEFF, ['TM0_lam0_' num2str(lam0) '.png']);

%% ========================================================================
%% TASK 2: Effective index vs wavelength for first 5 modes
%% ========================================================================

fprintf('\n=== TASK 2: Effective Index vs Wavelength ===\n');

% Define wavelength range and waveguide parameters
lam0s = 1300:100:1600;      % Wavelength range [nm]
rib_w = 450;                % Rib width [nm]
num_modes = 5;              % Number of modes to analyze

% Initialize storage for effective indices
NEFFs = zeros(length(lam0s), num_modes);

% Calculate effective indices for each wavelength
for i = 1:length(lam0s)
    fprintf('Calculating modes for wavelength: %d nm\n', lam0s(i));
    
    % Solve waveguide for current wavelength
    [Ex, Ey, Ez, Hx, Hy, Hz, NEFF, Gamma_Si] = rib_waveguide_solver(lam0s(i), rib_w);
    
    % Store effective indices for first 5 modes
    NEFFs(i, :) = NEFF;
end

% Plot effective index vs wavelength
plot_neff_vs_wavelength(lam0s, NEFFs, 'neff_vs_wavelength.png');

%% ========================================================================
%% TASK 3: Investigation of 450 nm width as industry standard
%% ========================================================================

fprintf('\n=== TASK 3: Why 450 nm Width is Industry Standard ===\n');

% Define parameter ranges for comprehensive analysis
lam0s = 1300:50:1600;       % Wavelength range [nm] (finer resolution)
rib_ws = 200:25:800;        % Rib width range [nm]

% Initialize storage arrays
cf_Hs = zeros(length(rib_ws), length(lam0s));           % Confinement factors
NEFFs = zeros(length(rib_ws), length(lam0s), 5);       % Effective indices

% Nested loop for parameter sweep
fprintf('Starting parameter sweep analysis...\n');
fprintf('Total calculations: %d\n', length(rib_ws) * length(lam0s));

for i = 1:length(rib_ws)
    fprintf('\nAnalyzing rib width: %d nm\n', rib_ws(i));
    
    for j = 1:length(lam0s)
        fprintf('  Wavelength: %d nm (Progress: %.1f%%)\n', ...
                lam0s(j), ((i-1)*length(lam0s) + j)/(length(rib_ws)*length(lam0s))*100);
        
        % Solve waveguide for current parameter combination
        [Ex, Ey, Ez, Hx, Hy, Hz, NEFF, cf_H] = rib_waveguide_solver(lam0s(j), rib_ws(i));
        
        % Store results
        cf_Hs(i, j) = cf_H;                    % Confinement factor
        NEFFs(i, j, :) = NEFF;                 % Effective indices for all modes
    end
end

% Plot analysis results
fprintf('\nGenerating analysis plot...\n');
plot_neff_vs_cf(rib_ws, lam0s, cf_Hs, 'neff_vs_cf_Hs.png');

fprintf('\n=== Analysis Complete ===\n');
fprintf('All plots have been saved to the current directory.\n');