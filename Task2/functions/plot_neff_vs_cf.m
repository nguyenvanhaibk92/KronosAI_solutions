function plot_neff_vs_cf(rib_ws, lam0s, cf_Hs, save_name)
% rib_ws : 1×NW widths (nm)
% lam0s  : 1×NL wavelengths (nm)
% cf_Hs  : NW×NL matrix (rows=widths, cols=wavelengths)

% If cf_Hs is transposed accidentally, fix it:
if size(cf_Hs,1) ~= numel(rib_ws) && size(cf_Hs,2) == numel(rib_ws)
    cf_Hs = cf_Hs.'; % make rows = widths
end

figure('Position',[120,120,700,300]); hold on; grid on;
colors = lines(numel(lam0s));

for j = 1:numel(lam0s)
    plot(rib_ws, cf_Hs(:,j), '-o', ...
        'LineWidth', 1.8, 'MarkerSize', 5, 'Color', colors(j,:), ...
        'DisplayName', sprintf('\\lambda = %d nm', lam0s(j)));
end

% % Optional thresholds (like the paper’s 20–35%)
% yline(0.20,'--','cf=0.20','Color',[0.5 0.5 0.5],'LabelHorizontalAlignment','left');
% yline(0.35,'--','cf=0.35','Color',[0.3 0.3 0.3],'LabelHorizontalAlignment','left');

xlabel('Waveguide width (nm)');
ylabel('Confinement factor (cf)');
title('Ex-based confinement vs. width at multiple wavelengths');
legend('Location','bestoutside');
xlim([min(rib_ws)-10, max(rib_ws)+10]);
ylim([0, .35]);

if nargin >= 4 && ~isempty(save_name)
    if ~exist('imgs','dir'); mkdir('imgs'); end
    set(gcf,'Color','w');
    exportgraphics(gcf, fullfile('imgs', save_name), 'Resolution', 1200);
    fprintf('Plot saved as %s\n', fullfile('imgs', save_name));
end

end
