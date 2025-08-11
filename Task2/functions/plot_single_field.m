function plot_single_field(field_data, field_name, mode_idx, NEFF, save_name)

global xa ya NMODES Nx Ny

micrometers = 1;

field_2d = reshape(field_data(:,mode_idx),Nx,Ny);
field_norm = field_2d / max(abs(field_2d(:)));

figure('Position', [100, 100, 600, 500]);
imagesc(xa/micrometers, ya/micrometers, abs(field_norm).');
axis equal tight;
set(gca, 'DataAspectRatio', [1 1 1]);
colorbar;
colormap('jet');
caxis([0 1]);
xlabel('x (μm)');
ylabel('y (μm)');
title(sprintf('%s, n_{eff} = %.2f', field_name, NEFF(mode_idx)));

% Tight layout
set(gca, 'Position', [0.15 0.15 0.7 0.7]);

if nargin >= 5 && ~isempty(save_name)
    set(gcf, 'Color', 'white');
    set(gca, 'LooseInset', get(gca,'TightInset'));
    exportgraphics(gcf, ['imgs' '/' save_name], 'Resolution', 600);
    fprintf('Plot saved as %s\n', save_name);
end

end

% plot_single_field(Ex, 'Ex', 1, NEFF)