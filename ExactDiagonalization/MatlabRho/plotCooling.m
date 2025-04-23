%% Plot utilities
set(0,'DefaultLineMarkerSize',3);
set(0,'Defaultlinelinewidth',1);
set(0,'DefaultAxesFontSize',10);
set(0,'DefaultAxesXMinorTick','on');
set(0,'DefaultAxesYMinorTick','on');

colorDefault = lines;
colorDefault = colorDefault(1:7,:);
Marker = {'o','s','d','x','p','^','v','*','+','<','>','h'};

colorScienceEdge = [
    161,49,47 % red
    54,91,149; % blue
    0,161,59; % green
    152,78,163; % purple
    217,83,25; % orange
    85,88,86; % grey
    0,0,0; % black
]/256;
colorScienceFace = [
    195,137,133; % red
    164,190,216; % blue
    161,199,136; % green
    184,135,195;% purple
    252,141,98; % orange
    210,210,205; % grey
    75,77,75; % black
]/256;

%% Variables

if ~exist('N','var')
    N = 5;
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
numelSchemes = numel(Niter);

titlename = sprintf('CI_N%dJ%.1fhx%.2fhz%.1f',N,J,hx,hz);
DirName = sprintf('N%dJ%.1fhx%.2fhz%.1f/',N,J,hx,hz);
SchemeName = strcat(DirName,'Scheme');
for i=1:numelSchemes
    SchemeName = strcat(SchemeName,'_',Scheme(i),num2str(Niter(i)),'Delta',num2str(Delta(i),'%.3f'),'g',num2str(g(i),'%.3f'));
    titlename = strcat(titlename,'_',Scheme(i),num2str(Niter(i)),'Delta',num2str(Delta(i),'%.3f'),'g',num2str(g(i),'%.3f'));
end

load(DirName+titlename+".mat");

thetaLbl={'$\theta=\pi/6$','$\theta=\pi/4$','$\theta=\pi/3$'};
thetaLbls={'(\theta=\pi{/6})','(\theta=\pi/4)','(\theta=\pi/3)'};

div = floor(sum(Niter) / 20);
StepsAll = 0:div:sum(Niter);
StepsAll(1)=1;
yinds = StepsAll;

SumNiter = zeros([1 numelSchemes]);
for i=1:numelSchemes
    SumNiter(i) = sum(Niter(1:i));
end



%% side by side: Energy and ground state overlap vs steps

figure;set(gcf, 'Position',[1,1,510,510/3*3/4*2]);

subplot(2,3,1);grid on;box on;hold on;
for pind=1:3
    plot(StepsAll,sZaAll(yinds,pind),'LineStyle','-','Marker',Marker{pind},'MarkerFaceColor',colorScienceFace(pind,:),'color',colorScienceEdge(pind,:));
end
%leg = legend(thetaLbl,'Location','southeast','Interpreter','Latex','fontsize',7);
%leg.ItemTokenSize = [10,20];
ylabel('$\langle \sigma_z^a(t=\Delta/g^2)\rangle$','Interpreter','Latex','fontsize',9);
for i=1:numelSchemes
    xline(SumNiter(i));
end

subplot(2,3,2);grid on;box on;hold on;
for pind=1:3
    plot(StepsAll,EAll(yinds,pind)/N,'LineStyle','-','Marker',Marker{pind},'MarkerFaceColor',colorScienceFace(pind,:),'color',colorScienceEdge(pind,:));
end
yline(EGS/N);
% ylim([EGS/N-0.1,-0.4]);
%leg = legend({'$\theta=\pi/6$','$\theta=\pi/4$','$\theta=\pi/3$','GS'},'Location','northeast','Interpreter','Latex','fontsize',7);
%leg.ItemTokenSize = [10,20];
xlabel('Steps');
ylabel('Energy density','fontsize',9);
for i=1:numelSchemes
    xline(SumNiter(i));
end
titlestr = {strcat('$N=',num2str(N),'J=',sprintf("%.2f",J),'h_x=',sprintf("%.2f",hx),'h_z=',sprintf("%.2f",hz),'$;gap=$',sprintf("%.3f",gap),'$')};
for i=1:numelSchemes
    titlestr{i+1}=strcat('$S_',num2str(i),':\Delta=',sprintf("%.2f",Delta(i)),',g=',sprintf("%.2f",g(i)),',g/\Delta=',sprintf("%.2f",g(i)/Delta(i)),',H_{SB}=',Scheme(i),',',num2str(Niter(i)),'$',' steps');
end
title(titlestr,'Interpreter','Latex','fontsize',7); 


subplot(2,3,3);grid on;box on;hold on;
for pind=1:3
    plot(StepsAll,purityAll(yinds,pind),'LineStyle','-','Marker',Marker{pind},'MarkerFaceColor',colorScienceFace(pind,:),'color',colorScienceEdge(pind,:));
end
leg = legend(thetaLbl,'Location','best','Interpreter','Latex','fontsize',7);
leg.ItemTokenSize = [10,20];
ylim([0,1.05]);
ylabel('Purity','fontsize',9);
for i=1:numelSchemes
    xline(SumNiter(i),'HandleVisibility','off');
end


for pind=1:3
    subplot(2,3,3+pind);grid on;box on;hold on;
    for j=1:M
        plot(StepsAll, populAll(yinds,j,pind));
    end
    ylim([0 1.05]);
    for i=1:numelSchemes
        xline(SumNiter(i));
    end
    if pind==3
        legendCell = strcat(string(num2cell(0:M-1)),'E');legendCell(1)='GS';
        leg = legend(legendCell,'fontsize',7,'Location','best');
        leg.ItemTokenSize = [10,20];
    end
    if pind==2
        xlabel('Steps');
    end
    ylabel(strcat('Population',thetaLbls{pind}),'fontsize',9);
end

tightfig;
saveas(gcf,strcat(DirName,'/',titlename,'.pdf'));%saveas(gcf,strcat(SchemeName,'/Figs/',titlename,'.fig'));
close;
