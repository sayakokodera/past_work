% MUSE batch interpolation via scatteredInterpolant

% Load
S = load('npy_data/batch_itp.mat');
% Unpack
dx = S.dx;
dz = S.dz;
N_batch = S.N_batch;
M = S.M;
A_true = S.A_true;
V_smp = S.V_smp.';
x_smp = cast(S.x_smp, 'double').';
y_smp = cast(S.y_smp, 'double').';
z_smp = cast(S.z_smp, 'double').';
x_quer = cast(S.x_quer, 'double').';
y_quer = cast(S.y_quer, 'double').';
z_quer = cast(S.z_quer, 'double').';

%% Plot
figure;
title('A_true')
imagesc(squeeze(A_true(200, :, :)))

%% Interpolation
% Adjust the ratio of dx and dz
z_smp_rel = dz/dx.* z_smp;
z_quer_rel = dz/dx.* z_quer;

% Interpolant
% Scattererd interpolant
F = scatteredInterpolant(z_smp_rel, x_smp, y_smp, V_smp);
F.Method = 'natural';
% Gridded interpolant -> does not work, as the sampled positions to be on
% grids.
%F = griddedInterpolant(x_smp, y_smp, z_smp, V_smp);
%F.Method = 'spline';

% Interpolate
V_quer = F(z_quer_rel, x_quer, y_quer);
% Reshape
A_itp = reshape(V_quer, M, N_batch, N_batch);
%% Insert the already known measurements 
N_scan = 15;
p_scan = zeros(N_scan, 2);

for idx = 1:N_scan;
    % Take every M elements of x_smp and y_smp 
    x = S.x_smp((idx-1)*M + 1);
    y = S.y_smp((idx-1)*M + 1);
    % Assign
    A_itp(:, x+1, y+1) = A_true(:, x+1, y+1);
    % Save scan positions (just in case)
    p_scan(idx, 1) = x + 1;
    p_scan(idx, 2) = y + 1;
    
end
%% Plot
figure;
title('A_itp')
imagesc(squeeze(A_itp(200, :, :)))

%% Plot: C-scan
C_true = cscan(A_true);
C_itp = cscan(A_itp);

% Plots: Transpose -> flip up side down (to match the CAD image)
figure;
imagesc(flipud(C_true.'))
title('C_true')

figure;
imagesc(flipud(C_itp.'))
title('C_itp')

%% Surf plots
[X, Y] = meshgrid(1:N_batch, 1:N_batch);

figure;
surf(X, Y, squeeze(A_true(200, :, :)))
title('A_true')

figure;
surf(X, Y, squeeze(A_itp(200, :, :)))
title('A_itp')


%% Save
fname = 'npy_data/batch_itp_A_natural.mat'
save(fname, 'A_itp')

%% Function : should be at the end of the script!
% function out = cscan(data_3d)
%     out = squeeze(max(data_3d));
% end




