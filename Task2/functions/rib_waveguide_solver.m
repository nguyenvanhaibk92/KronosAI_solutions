function [Ex, Ey, Ez, Hx, Hy, Hz, NEFF, cf_H] = rib_waveguide_solver(lam0, rib_w)

% DECLARE GLOBAL VARIABLES
global xa ya ER2 xa2 ya2 NMODES Nx Ny rib_n1 rib_n2

% UNITS
micrometers = 1;
nanometers = 1e-3 * micrometers;

lam0 = lam0 * nanometers;  % Convert wavelength to nanometers

% FIXED PARAMETERS
rib_h = 220 * nanometers;   % Height of the rib
rib_w = rib_w * nanometers;  % Width of the rib
rib_n1 = 1.444;               % Substrate refractive index (SiO2)
rib_n2 = 3.476;               % Core refractive index (silicon)
NMODES = 5;                 % Number of modes to calculate
NRES = 50;                  % Grid resolution parameter

% GRID PARAMETERS
nmax = max([rib_n1 rib_n2]);  % Maximum refractive index for resolution calc
SPACER = lam0*[0.8 0.8 0.6 0.6];  % Spacer regions [left, right, top, bottom]

% COMPUTE OPTIMIZED GRID
dx = lam0/nmax/NRES;  % Initial x-direction grid spacing
dy = lam0/nmax/NRES;  % Initial y-direction grid spacing

% X-direction optimization (rib width is critical dimension)
nx = ceil(rib_w/dx);    % Number of cells for rib width
dx = rib_w/nx;          % Adjusted resolution to exactly fit rib width

% Y-direction optimization (silicon thickness is critical dimension)  
ny = ceil(rib_h/dy);    % Number of cells for silicon thickness
dy = rib_h/ny;          % Adjusted resolution to exactly fit thickness

% CALCULATE TOTAL GRID SIZE
Sx = SPACER(1) + rib_w + SPACER(2);  % Total physical width
Nx = ceil(Sx/dx);                    % Number of cells in x
Sx = Nx*dx;                          % Recalculate physical size

Sy = SPACER(3) + rib_h + SPACER(4);  % Total physical height
Ny = ceil(Sy/dy);                    % Number of cells in y
Sy = Ny*dy;                          % Recalculate physical size

% 2X OVERSAMPLED GRID (for Yee grid material assignment)
Nx2 = 2*Nx;         dx2 = dx/2;  % Double resolution in x
Ny2 = 2*Ny;         dy2 = dy/2;  % Double resolution in y

% GRID AXES FOR VISUALIZATION
xa = [1:Nx]*dx;     xa = xa - mean(xa);    % X-axis centered at origin
ya = [1:Ny]*dy;     ya = ya - mean(ya);    % Y-axis centered at origin
xa2 = [1:Nx2]*dx2;  xa2 = xa2 - mean(xa2); % 2x grid x-axis
ya2 = [1:Ny2]*dy2;  ya2 = ya2 - mean(ya2); % 2x grid y-axis

% BUILD DEVICE ON 2X GRID
ER2 = rib_n1^2*ones(Nx2,Ny2);  % Permittivity on 2x grid (start with substrate)
UR2 = ones(Nx2,Ny2);           % Permeability on 2x grid (non-magnetic)

% CALCULATE POSITION INDICES ON 2X GRID
nx1 = 1 + round(SPACER(1)/dx2);        % Left edge of rib
nx2 = nx1 + round(rib_w/dx2) - 1;      % Right edge of rib (-1 for proper indexing)

ny1 = 1 + round(SPACER(3)/dy2);        % Bottom of air region (top of rib)
ny2 = ny1 + round(rib_h/dy2) - 1;      % Top of silicon layer

% BUILD RIB WAVEGUIDE STRUCTURE
ER2(nx1:nx2,ny1:ny2) = rib_n2^2;      % Rib (silicon)

% EXTRACT YEED GRID MATERIAL TENSORS
ERxx = ER2(2:2:Nx2,1:2:Ny2);  % εxx at Ex locations
ERyy = ER2(1:2:Nx2,2:2:Ny2);  % εyy at Ey locations  
ERzz = ER2(1:2:Nx2,1:2:Ny2);  % εzz at Ez locations

URxx = UR2(1:2:Nx2,2:2:Ny2);  % μxx at Hx locations
URyy = UR2(2:2:Nx2,1:2:Ny2);  % μyy at Hy locations
URzz = UR2(2:2:Nx2,2:2:Ny2);  % μzz at Hz locations

% CONVERT MATERIAL ARRAYS TO DIAGONAL SPARSE MATRICES
ERxx = diag(sparse(ERxx(:)));  % Convert εxx to diagonal matrix
ERyy = diag(sparse(ERyy(:)));  % Convert εyy to diagonal matrix
ERzz = diag(sparse(ERzz(:)));  % Convert εzz to diagonal matrix
URxx = diag(sparse(URxx(:)));  % Convert μxx to diagonal matrix
URyy = diag(sparse(URyy(:)));  % Convert μyy to diagonal matrix
URzz = diag(sparse(URzz(:)));  % Convert μzz to diagonal matrix

% BUILD FINITE-DIFFERENCE DERIVATIVE MATRICES
k0 = 2*pi/lam0;           % Free space wavenumber
NS = [Nx Ny];             % Grid size array
RES = [dx dy];            % Grid resolution array
BC = [0 0];               % Boundary conditions (0 = Dirichlet)

[DEX,DEY,DHX,DHY] = yeeder2d(NS,k0*RES,BC);

% BUILD P AND Q MATRICES FOR EIGENVALUE PROBLEM
P = [ DEX/ERzz*DHY , -(DEX/ERzz*DHX + URyy) ; ...
      DEY/ERzz*DHY + URxx , -DEY/ERzz*DHX ];

Q = [ DHX/URzz*DEY , -(DHX/URzz*DEX + ERyy) ; ...
      DHY/URzz*DEY + ERxx , -DHY/URzz*DEX ];

% SOLVE EIGENVALUE PROBLEM TO FIND GUIDED MODES
ev = -rib_n2^2;  % Eigenvalue estimate: -neff² ≈ -n_core²

[Exy,D2] = eigs(P*Q,NMODES,ev);

% Calculate propagation constants from eigenvalues
D = sqrt(D2);           % γ = sqrt(eigenvalue)
NEFF = -1i*diag(D);     % Effective index: neff = -i*γ

% CALCULATE OTHER FIELD COMPONENTS
M = Nx*Ny;                    % Total number of grid points

% Extract transverse electric field components
Ex = Exy(1:M,:);               % Ex component (first M rows)
Ey = Exy(M+1:2*M,:);          % Ey component (second M rows)

% Calculate transverse magnetic field components
Hxy = (Q*Exy)/D;               % H-field from Maxwell's equations
Hx = Hxy(1:M,:);             % Hx component  
Hy = Hxy(M+1:2*M,:);         % Hy component

% Calculate longitudinal field components
Ez = ERzz\(DHX*Hy - DHY*Hx);   % Ez from ∇×H = -iωεE
Hz = URzz\(DEX*Ey - DEY*Ex);   % Hz from ∇×E = iωμH


m = 1;

% Reshape fields to 2D (Yee grids)
Ex2 = reshape(Ex(:,m), Nx, Ny);   % Ex lives on Ex/Hx grid

% Build coordinate grids (to define the slot mask)
[X,Y] = meshgrid(xa, ya);  % Ny-by-Nx
X = X.'; Y = Y.';          % -> Nx-by-Ny to match Ex2

% ====== EDIT THESE SLOT PARAMETERS AS NEEDED ======
% Slot center (micrometers, since xa/ya are in micrometers)
slot_cx = 0.0;                       % x-center of slot
slot_cy = 0.0;                       % y-center of slot
% Slot size
slot_w_nm = 100;                     % slot width in nm (example)
slot_h_nm = 220;                     % slot height in nm (often ~ rib_h)
slot_w = slot_w_nm * 1e-3;           % to micrometers
slot_h = slot_h_nm * 1e-3;           % to micrometers
% ================================================

% Slot mask: rectangle centered at (slot_cx, slot_cy)
slot_mask = (abs(X - slot_cx) <= slot_w/2) & (abs(Y - slot_cy) <= slot_h/2);

% Total region: whole computational window (you can restrict if desired)
total_mask = true(size(Ex2));

% Compute confinement using |Ex|^2
num = sum(sum( abs(Ex2).^2 .* slot_mask ));
den = sum(sum( abs(Ex2).^2 .* total_mask ));

% Use the existing output variable name for compatibility
cf_H = real(num / den);

end
