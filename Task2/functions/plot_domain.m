function plot_domain(save_png)

% DECLARE GLOBAL VARIABLES
global xa ya ER2 xa2 ya2 NMODES Nx Ny rib_n1 rib_n2

if nargin < 1
    save_png = false;
end

micrometers = 1;

figure('Position', [100, 100, 600, 500]);
imagesc(xa2/micrometers, ya2/micrometers, sqrt(ER2).');
axis equal tight;
set(gca, 'DataAspectRatio', [1 1 1]);
colorbar;
colormap('parula');
xlabel('x (μm)');
ylabel('y (μm)');
title('Permittivity Distribution');

% Tight layout
set(gca, 'Position', [0.15 0.15 0.7 0.7]);

if save_png
    set(gcf, 'Color', 'white');
    set(gca, 'LooseInset', get(gca,'TightInset'));
    exportgraphics(gcf, 'imgs/computation_domain.png', 'Resolution', 600);
    fprintf('Plot saved as ER2_domain.png\n');
end

end

