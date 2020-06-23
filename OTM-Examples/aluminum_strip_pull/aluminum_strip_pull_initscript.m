% Script to make the input files for elastic pull LAMMPS OTM tests
[x,y] = meshgrid(-10:10);
nd = [x(:),y(:), zeros(size(x(:)))]; % Nodal Coordinates

DT = delaunayTriangulation(nd(:,1:2));
mp = incenter(DT); % Material Point Coordinates

N = size(nd,1) + size(mp,1);

velnd = zeros(size(nd)); % Initial velocities
velmp = zeros(size(mp));

% Temporarily combine both atom types (eliminate once actually using
% different mps and nds
nd = [nd;mp];
vel = [velnd;velmp];

% Write to file
fileID = fopen('initfile.txt','w');
fprintf('LAMMPS Initialization File for Aluminum Strip Pull\n\n');

% Number and types of atoms
fprintf('%i atoms\n\n', N);
fprintf('1 atom types\n\n');

% simulation box
fprintf('-10 10 xlo xhi\n');
fprintf('-10 10 ylo yhi\n\n');

% Locations of atoms
fprintf('Atoms\n\n');
% order for SMD: ID atom-type moleculeID volume mass kernelradius contactradius x y z
for ii = 1:N
    fprintf('%i 1 0 %3.17e %3.17e %3.17e %3.17e %3.17e %3.17e %3.17e %3.17e', ...
        ii, );
end
 
% Okay, smd requires a lot of input. I think I will need to modify
% read_data at some point.


fclose(fileID);
