clear all; close all; clc;

addpath(fullfile(pwd, 'functions'));

% Task 1
% visualize the vectorial mode profile of the TE0 and TM0 modes for 1310 and 1550 nm 

lam0 = 1550;
rib_w = 450;

[Ex, Ey, Ez, Hx, Hy, Hz, NEFF, Gamma_Si] = rib_waveguide_solver(lam0, rib_w);
% plot_Exyz_neff(Ex, Ey, Ez, NEFF)

plot_domain(1)

plot_single_field(Ex, 'TE_0 (E_x)', 1, NEFF,['TE0_lam0_' num2str(lam0) '.png'])
plot_single_field(Ey, 'TM_0 (E_y)', 2, NEFF,['TM0_lam0_' num2str(lam0) '.png'])

lam0 = 1310;
rib_w = 450;

[Ex, Ey, Ez, Hx, Hy, Hz, NEFF, Gamma_Si] = rib_waveguide_solver(lam0, rib_w);
% plot_Exyz_neff(Ex, Ey, Ez, NEFF)

plot_single_field(Ex, 'TE_0 (E_x)', 1, NEFF,['TE0_lam0_' num2str(lam0) '.png'])
plot_single_field(Ey, 'TM_0 (E_y)', 2, NEFF,['TM0_lam0_' num2str(lam0) '.png'])

% Task 2
% plot the effective index of the first 5 modes between 1300 and 1600 nm.

lam0s = 1300:25:1600;
rib_w = 450;
NEFFs = zeros(length(lam0s), 5);

for i = 1:length(lam0s)
    [Ex, Ey, Ez, Hx, Hy, Hz, NEFF, Gamma_Si] = rib_waveguide_solver(lam0s(i), rib_w);
    NEFFs(i, :) = NEFF;
end

plot_neff_vs_wavelength(lam0s, NEFFs, 'neff_vs_wavelength.png');

% Task 3
% Why has 450 nm width become the “industry standard”?
lam0s = 1300:50:1600; % nm
rib_ws = 200:25:800; % nm
cf_Hs = zeros(length(rib_ws), length(lam0s));
NEFFs = zeros(length(rib_ws), length(lam0s), 5);

for i = 1:length(rib_ws)
    for j = 1:length(lam0s)
        fprintf('Calculating for rib width %d nm at wavelength %d nm...\n', rib_ws(i), lam0s(j));
        % Call the solver for each combination of rib width and wavelength
        [Ex, Ey, Ez, Hx, Hy, Hz, NEFF, cf_H] = rib_waveguide_solver(lam0s(j), rib_ws(i));
        cf_Hs(i, j) = cf_H; % Store the confinement factor
        NEFFs(i, j, :) = NEFF; % Store the effective index of the first mode
    end
end

save('confinement.mat', 'cf_H')
save('NEFFs_all.mat', 'NEFFs')

plot_neff_vs_cf(rib_ws, lam0s, cf_Hs, 'neff_vs_cf_Hs.png');
