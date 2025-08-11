function plot_Exyz_neff(Ex, Ey, Ez, NEFF, save_png)

% DECLARE GLOBAL VARIABLES
global xa ya ER2 xa2 ya2 NMODES Nx Ny rib_n1 rib_n2

if nargin < 5
    save_png = false;
end

micrometers = 1;

figure;
for m = 1 : NMODES
    ex = reshape(Ex(:,m),Nx,Ny);
    ey = reshape(Ey(:,m),Nx,Ny);
    ez = reshape(Ez(:,m),Nx,Ny);
    
    E_total = sqrt(abs(ex).^2 + abs(ey).^2);
    E_total_norm = E_total / max(E_total(:));
    
    fmax = max(abs([ ex(:) ; ey(:) ; ez(:) ]));
    ex = ex/fmax;
    ey = ey/fmax;
    ez = ez/fmax;
    
    subplot(NMODES,4,(m - 1)*4 + 1);
    imagesc(xa/micrometers,ya/micrometers,abs(ex).');
    axis equal tight off;
    colorbar;
    colormap('jet');
    caxis([0 1]);
    title(['Ex, n_{eff} = ' num2str(NEFF(m))]);
    
    subplot(NMODES,4,(m - 1)*4 + 2);
    imagesc(xa/micrometers,ya/micrometers,abs(ey).');
    axis equal tight off;
    colorbar;
    colormap('jet');
    caxis([0 1]);
    title('Ey');
    
    subplot(NMODES,4,(m - 1)*4 + 3);
    imagesc(xa/micrometers,ya/micrometers,abs(ez).');
    axis equal tight off;
    colorbar;
    colormap('jet');
    caxis([0 1]);
    title('Ez');
    
    subplot(NMODES,4,(m - 1)*4 + 4);
    imagesc(xa/micrometers,ya/micrometers,E_total_norm.');
    axis equal tight off;
    colorbar;
    colormap('jet');
    caxis([0 1]);
    title('|E| normalized');
end

if save_png
    saveas(gcf, 'Exyz_neff_modes.png');
    fprintf('Modes plot saved as Exyz_neff_modes.png\n');
end