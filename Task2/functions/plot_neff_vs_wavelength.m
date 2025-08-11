function plot_neff_vs_wavelength(lam0s, NEFFs, save_name)

global xa ya ER2 xa2 ya2 NMODES Nx Ny rib_n1 rib_n2

colors = {'#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'};
mode_names = {'TE_0', 'TM_0', 'TE_1', 'TM_1', 'TE_2'};

figure('Position', [100, 100, 600, 250]);
hold on;
for m = 1:5
    plot(lam0s, NEFFs(:, m), 'o-', 'Color', colors{m}, 'LineWidth', 2, ...
         'MarkerSize', 6, 'DisplayName', mode_names{m});
end

xlabel('Wavelength (nm)');
ylabel('Effective Index (n_{eff})');
title('Effective Index vs Wavelength');
grid on;
legend('Location', 'best', 'NumColumns', 2);
xlim([min(lam0s)-20 max(lam0s)+20]);

if nargin >= 3 && ~isempty(save_name)
    if ~exist('imgs', 'dir')
        mkdir('imgs');
    end
    set(gcf, 'Color', 'white');
    set(gca, 'LooseInset', get(gca,'TightInset'));
    exportgraphics(gcf, fullfile('imgs', save_name), 'Resolution', 1200);
    fprintf('Plot saved as %s\n', fullfile('imgs', save_name));
end

end