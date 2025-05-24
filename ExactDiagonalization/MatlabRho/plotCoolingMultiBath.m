%% plotCoolingMultiBath - Unified plotting script for multi-bath cooling simulations
% This script visualizes results from CoolingMultiBath simulations

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
    161,49,47; % red
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

%% Load variables if not already in workspace
if ~exist('N','var') || ~exist('EAll','var')
    % Try to find the most recent result file
    DirName = pwd;
    files = dir(fullfile(DirName, 'CI_MB_*.mat'));
    if isempty(files)
        error('No multi-bath result files found in current directory');
    end
    [~, mostRecentIdx] = max([files.datenum]);
    mostRecentFile = fullfile(DirName, files(mostRecentIdx).name);
    fprintf('Loading most recent results file: %s\n', mostRecentFile);
    load(mostRecentFile);
end

% Create labels for initial states
thetaLbl={'$\theta=\pi/6$','$\theta=\pi/4$','$\theta=\pi/3$'};
thetaLbls={'(\theta=\pi{/6})','(\theta=\pi/4)','(\theta=\pi/3)'};

%% Standard plotting style: detailed visualization with multiple subplots
% Determine points to plot (downsample for clarity if there are many iterations)
if size(EAll, 1) > 20
    % Get evenly spaced indices, ensuring 0 and final point are included
    div = floor(size(EAll, 1) / 20);
    StepsAll = 0:div:size(EAll, 1)-1; % Steps start from 0 now
    if StepsAll(end) ~= size(EAll, 1)-1
        StepsAll = [StepsAll, size(EAll, 1)-1];
    end
    yinds = StepsAll + 1; % Convert to 1-based indexing for MATLAB array access
else
    StepsAll = 0:size(EAll, 1)-1; % Steps now start from 0 to steps
    yinds = StepsAll + 1; % Convert to 1-based indexing for MATLAB array access
end

figure;
set(gcf, 'Position',[1,1,510,510/3*3/4*2]);

% Plot average bath spin Z
subplot(2,3,1);grid on;box on;hold on;
for pind=1:3
    plot(StepsAll,sZaAll(yinds,pind),'LineStyle','-','Marker',Marker{pind},'MarkerFaceColor',colorScienceFace(pind,:),'color',colorScienceEdge(pind,:));
end
ylabel('$\langle \sigma_z^a \rangle$ (avg)','Interpreter','Latex','fontsize',9);

% Plot energy density
subplot(2,3,2);grid on;box on;hold on;
for pind=1:3
    plot(StepsAll,EAll(yinds,pind)/N,'LineStyle','-','Marker',Marker{pind},'MarkerFaceColor',colorScienceFace(pind,:),'color',colorScienceEdge(pind,:));
end
yline(EGS/N);
xlabel('Steps');
ylabel('Energy density','fontsize',9);

% Get coupling type
if iscell(coupling)
    ct = coupling{1};
else
    ct = coupling;
end

% Add title with simulation parameters including t
titlestr = {sprintf('$N=%d, J=%.2f, h_x=%.2f, h_z=%.2f$; gap=$%.3f$', N, J, hx, hz, gap)};
titlestr{2}=sprintf('$\\delta=%.3f, g=%.3f, g/\\delta=%.3f, t_e=%.3f, H_{SB}=%s, %d$ steps', ...
    delta, g, g/delta, te, ct, steps);
title(titlestr,'Interpreter','Latex','fontsize',7); 

% Plot purity
subplot(2,3,3);grid on;box on;hold on;
for pind=1:3
    plot(StepsAll,purityAll(yinds,pind),'LineStyle','-','Marker',Marker{pind},'MarkerFaceColor',colorScienceFace(pind,:),'color',colorScienceEdge(pind,:));
end
leg = legend(thetaLbl,'Location','best','Interpreter','Latex','fontsize',7);
leg.ItemTokenSize = [10,20];
ylim([0,1.05]);
ylabel('Purity','fontsize',9);

% Plot eigenstate populations for each initial state
for pind=1:3
    subplot(2,3,3+pind);grid on;box on;hold on;
    % Check size of populAll to accommodate different values of M
    M = size(populAll, 2);
    for j=1:M
        plot(StepsAll, populAll(yinds,j,pind));
    end
    ylim([0 1.05]);
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

% Make the figure compact
if exist('tightfig', 'file')
    tightfig;
end

% Add te to the filename if it exists
if exist('te', 'var')
    te_suffix = sprintf('te%.2f', te);
else
    te_suffix = '';
end

% Save figure with full filename for consistency
savename = sprintf('CI_MB_N%dJ%.1fhx%.2fhz%.1f', N, J, hx, hz);
savename = [savename sprintf('_%s%d_delta%.3f_g%.3f%s', ct, steps, delta, g, te_suffix)];
if exist('DirName', 'var')
    if ~exist(DirName, 'dir')
        mkdir(DirName);
    end
    savename = strcat(DirName, '/', savename, '.pdf');
end
% Use saveas with individual arguments to avoid path issues
saveas(gcf, savename, 'pdf');
fprintf('Figure saved as: %s.pdf\n', savename); 