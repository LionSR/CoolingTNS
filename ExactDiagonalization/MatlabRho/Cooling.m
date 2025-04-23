%% Variables
addpath('../Utils/','../Utils/expmv');

if ~exist('N','var')
    N = 8;
end

if ~exist('hx','var')
    hx = -1.05;
end
if ~exist('hz','var')
    hz = 0.5;
end
if ~exist('J','var')
    J = 1;
end

GapList = [0 0 1.1480 0.8484 0.8006 0.6323 0.6327 0.5185 0.5327 0.4483 0.4660 0.4009 0.4182 0.3667 0.3824 0.3410 0.3546 0.3209 0.3325];
M = 5; % number of eigenenergies to plot

if ~exist('Scheme','var')
    Scheme = ["XX"];
end
if ~exist('Niter','var')
    Niter = [100*N];
end
if ~exist('Delta','var')
    Delta = -GapList(N)/2;
end
if exist('factor','var')
    g = Delta/factor;
end
if ~exist('g','var')
    g = Delta/2.5;
end
numelSchemes = numel(Niter);

titlename = sprintf('CI_N%dJ%.1fhx%.2fhz%.1f',N,J,hx,hz);
DirName = sprintf('N%dJ%.1fhx%.2fhz%.1f/',N,J,hx,hz);
if not(isfolder(DirName))
    mkdir(DirName);
end


SchemeName = strcat(DirName,'Scheme');
for i=1:numelSchemes
    SchemeName = strcat(SchemeName,'_',Scheme(i),num2str(Niter(i)),'Delta',num2str(Delta(i),'%.3f'),'g',num2str(g(i),'%.3f'));
    titlename = strcat(titlename,'_',Scheme(i),num2str(Niter(i)),'Delta',num2str(Delta(i),'%.3f'),'g',num2str(g(i),'%.3f'));
end
disp(titlename);

%% Init Hamiltonians
sZ = sparse([1 0; 0 -1]);
sZa = kron(speye(2^N),sZ);

HamS = CreateHamZZXZ(N,J,hx,hz);
[V,D] = eigs(HamS,M,'sa');
[~,ind] = sort(diag(D));
Ds = diag(D(ind,ind));
Vs = V(:,ind);
gap = Ds(2)-Ds(1);
fprintf("E_{GS}/L=%.3f, gap=%.3f\n",Ds(1)/N, gap);
EGS = Ds(1);
psi_gs = V(:,1);

%% Initial state

thetaSelected=[1/6,1/4,1/3];
vzplus = [1;0];
vzminus = [0;1];
rhozplus = vzplus*vzplus';

%% Cooling parameters


lenScheme = strlength(Scheme);
Ncouples = N-lenScheme+2;
UtAll = zeros([numelSchemes,max(lenScheme),2^(N+1),2^(N+1)]);
fprintf("Initializing time evolution op: ")

for i=1:numelSchemes
    % T0 = abs(Delta(i)/g(i)^2);
    T0 = abs(Delta(i)/g(i));
    fprintf("\nScheme%s T0=%.3f: ",Scheme(i),T0)
    CreateHamCouple=str2func(strcat('CreateHamCouple',Scheme(i)));
    tic;
    for ind=1:Ncouples(i)
        HamCouple = CreateHamCouple(HamS,N,Delta(i),g(i),ind);
        UtAll(i,ind,:,:)=fastExpm(-1i*T0*HamCouple);
        fprintf("%d",ind)
    end
    fprintf("\ntime = %0.2f sec",toc);
end

Ut = zeros([2^(N+1),2^(N+1)]);

%% Run many steps of Cooling for three different initial states

EAll = zeros(sum(Niter),3);
GSOAll = zeros(sum(Niter),3);
purityAll = zeros(sum(Niter),3);
populAll = zeros(sum(Niter),M,3);
sZaAll = zeros(sum(Niter),3);
vstateAll = zeros(2^N,3);

fprintf("\nRunning cooling algorithms: \n")
for thetaind=1:3
    theta = thetaSelected(thetaind)*pi;
    v0=[cos(theta);sin(theta)];
    vstate=1;
    for k=1:N
        vstate=kron(vstate,v0);
    end
    Nst=vstate'*vstate;
    vstate=vstate/sqrt(Nst); % init normalized
    vstateAll(:,thetaind) = vstate;
    for j=1:M
        populAll(1,j,thetaind) = (abs(vstate'*Vs(:,j)))^2;
    end
    EAll(1,thetaind) = real(vstate'*HamS*vstate);
    GSOAll(1,thetaind) = (abs(vstate'*vstate))^2;
    purityAll(1,thetaind) = 1.0;
    sZaAll(1,thetaind) = 1;
    
end

for thetaind=1:3
    fprintf("\ntheta=%.2f pi, E_init/N=%.2f",thetaSelected(thetaind),EAll(1,thetaind)/N);
    psi_in = vstateAll(:,thetaind);
    rho = psi_in*psi_in';
    [sZ,E,GSO,purity,popul] = Evolve(rho,UtAll,Vs,Niter,M,numelSchemes,rhozplus,Ncouples,sZa,N,HamS,psi_gs,Scheme,thetaind);
    sZaAll(:,thetaind) = sZ;
    EAll(:,thetaind) = E;
    GSOAll(:,thetaind) = GSO;
    purityAll(:,thetaind) = purity;
    populAll(:,:,thetaind) = popul;
end

fprintf("\n");
save(strcat(DirName,'/',titlename,'.mat'),'EAll','GSOAll','purityAll','populAll','sZaAll','thetaSelected','J','hx','hz','Delta','g','T0','Niter','EGS','Ds','gap','Scheme','SchemeName');
