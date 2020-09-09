%% Write Input file for aluminum_strip_pull_2.lmp
% Lucas Caparini 53547155
% August 14 2020
clear, clc, close all;
%% Set up Geometry
bound = [-10.1 10.1; ... %xlo xhi
         -10.1 10.1; ... %ylo yhi
         -.5 .5];    %zlo zhi

rho = 1; % initial mass density

% Nodal positions
l0 = 1;

x = -10:l0:10;
[x,y] = meshgrid(x);
x = [x(:),y(:)];
N_nd = size(x,1);
nd.x = [x,zeros(N_nd,1)];


nd.vol = repmat(l0^2,N_nd,1);
nd.mass = rho*nd.vol;

nd.type = ones(N_nd,1); % type specifier

nd.v = zeros(N_nd,1); % nd velocity

% MP positions
% Set up temp nodes for denser mp positions
l_temp = l0/2;
x_temp = -10:l_temp:10;
[x_temp,y_temp] = meshgrid(x_temp);
x_temp = [x_temp(:) y_temp(:)];

DT = delaunayTriangulation(x_temp);
mp.x = incenter(DT);
N_mp = size(mp.x,1);
mp.x = [mp.x, zeros(N_mp,1)];

a_x = x_temp(DT.ConnectivityList(:,1),1)-x_temp(DT.ConnectivityList(:,2),1); % Volume of the mp
a_y = x_temp(DT.ConnectivityList(:,1),2)-x_temp(DT.ConnectivityList(:,2),2);
b_x = x_temp(DT.ConnectivityList(:,1),1)-x_temp(DT.ConnectivityList(:,3),1);
b_y = x_temp(DT.ConnectivityList(:,1),2)-x_temp(DT.ConnectivityList(:,3),2);
mp.vol = 1/2*abs(a_x.*b_y-a_y.*b_x); % assume unit thickness
mp.mass = rho*mp.vol; % mass of each mp

mp.type = repmat(2,N_mp,1); 

mp.v = zeros(N_mp,1); % mp velocity


% Concatenate stats

num_atoms = N_nd + N_mp;
type = [nd.type;mp.type];
num_types = 2;
ID = [1:num_atoms]'; % atom ID
M_ID = ones(num_atoms,1);%ID; % Molecule ID

x0 = [nd.x;mp.x];
x = [nd.x;mp.x];

volume = [nd.vol;mp.vol];
mass = [nd.mass;mp.mass];

v = zeros(num_atoms,3);

h = 2.01*l0;
k_rad = repmat(h,num_atoms,1); % kernel radius - generally 2 to 3 times particle spacing
c_rad = repmat(h/2,num_atoms,1); % contact radius - generally half particle spacing

%% Write data to files
filename1 = 'aluminum_strip_pull_moreMPs.data';
fileId1 = fopen(filename1,'w');

% Header
fprintf(fileId1,['First iteration of the aluminum strip pull input file. Still using SMD input format.\n',...
    '%d atoms\n', ...
    '%d atom types\n', ...
    '%f %f xlo xhi\n', ...
    '%f %f ylo yhi\n', ...
    '%f %f zlo zhi\n\n'], num_atoms,num_types,bound(1,1),bound(1,2),bound(2,1),bound(2,2),bound(3,1),bound(3,2));

% Atom section (SMD style)
fprintf(fileId1,'Atoms # otm\n\n');
for ii = 1:num_atoms
    fprintf(fileId1,"%d %d %d %.5f %.5e %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n", ...
        ID(ii),type(ii),M_ID(ii),volume(ii),mass(ii),k_rad(ii),c_rad(ii), ...
        x0(ii,1),x0(ii,2),x0(ii,3),x(ii,1),x(ii,2),x(ii,3));
end

% Velocity section
fprintf(fileId1,'\nVelocities\n\n');
for ii = 1:num_atoms
    fprintf(fileId1,'%d %f %f %f\n',ID(ii),v(ii,:));
end

fclose(fileId1);

%% Plot points to make sure it worked
plot(nd.x(:,1),nd.x(:,2),'ro');
hold on;
plot(mp.x(:,1),mp.x(:,2),'bx');

for ii = 1:num_atoms % I have no clue why this isn't working. Not even any duplicates. Fuck this.
    for jj = ii+1:num_atoms
        if ( x(ii,1)==x(jj,1) && x(ii,2)==x(jj,2) )
            disp("Error: mp duplicate");
            ii,jj
        end
    end
end
