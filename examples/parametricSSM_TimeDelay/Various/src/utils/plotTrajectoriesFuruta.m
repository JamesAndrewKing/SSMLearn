function plotTrajectoriesFuruta(yData, varargin)
% plotTrajectories(yData)
% plotTrajectories(yData, yRec)
% plotTrajectories(yData, 'Color', 'r', 'PlotCoordinate', n)
% plotTrajectories(yData, yRec, 'm', 'DisplayName', {'Test', 'Prediction'})
% Plot the coordinate with index plotCoord in the trajectories yData over 
% time. Optionally overlay a dashed plot of the same trajectories in yRec.
%
% INPUT
% yData             cell (nTraj x 2)   full trajectories
% yRec              cell (nTraj x 2)   optional, reconstructed trajectories
% Color             char               color for yData plot, e.g. 'r--'
% PlotCoordinate    int                index in dim to plot

p = inputParser;
addOptional(p, 'yRec', {});
addOptional(p, 'Color', 'b', @(x)ischar(x));
addOptional(p, 'PlotCoordinate', 1);
addOptional(p, 'DisplayName', {'Full','Reconstructed'});
addOptional(p, 'Titles', {});
parse(p, varargin{:});
yRec = p.Results.yRec;
plotCoord = p.Results.PlotCoordinate;
    
nTraj = size(yData,1);

if nTraj > 1
    nCol = ceil(sqrt(nTraj));
    nRow = nCol - (nCol * nCol - nTraj > nCol - 1);
    fig = customFigure('subPlot',[nRow nCol]);
    set(fig, 'Color', 'w');
    tiledlayout(nRow,nCol, 'TileSpacing', 'compact')
else
    fig = customFigure();
    set(fig, 'Color', 'w');
end
allData = horzcat(yData{:,2});
maxAmp = max(abs(allData(plotCoord,:)));
minTime = min(min(horzcat(yData{:,1})));
maxTime = max(max(horzcat(yData{:,1})));

for iTraj = 1:nTraj
    if nTraj > 1; nexttile; end
    hold on
    plot(yData{iTraj,1}, yData{iTraj,2}(plotCoord,:), p.Results.Color, 'Linewidth', 0.3, 'DisplayName', p.Results.DisplayName{1})
    if ~isempty(yRec)
        %hh = plot(yRec{iTraj,1}, rad2deg(yRec{iTraj,2}(plotCoord,:)), 'k:', 'Linewidth', 2, 'DisplayName', p.Results.DisplayName{2});
        hh = plot( yRec{iTraj,1}, ...
           yRec{iTraj,2}(plotCoord,:), ...
           'LineStyle', ':',    ...  % still dashed
           'LineWidth', 2, ...
           'Color',     [0 0 0], ...
           'DisplayName', p.Results.DisplayName{2} );
    end
    xlim([-minTime,maxTime]);
    ylim([-max(maxAmp),max(maxAmp)]);
    if nTraj > 9
        set(gca,'YTick',[])
        set(gca,'XTick', [])
    elseif length(p.Results.DisplayName) == 2 - isempty(yRec)
        if iTraj == nTraj; legend; end
    end
    if length(p.Results.Titles) == nTraj
        title(p.Results.Titles{iTraj}, 'Interpreter', 'latex')
    end
end

if nTraj == 1
    ylabel(['$\theta_{', num2str(plotCoord), '}[deg]$'], 'Interpreter', 'latex');
    xlabel('$t[s]$', 'Interpreter', 'latex');
else
    hnd = axes(fig,'visible','off');
    hnd.Title.Visible = 'on';
    hnd.XLabel.Visible = 'on';
    hnd.YLabel.Visible = 'on';
    ylabel(hnd, ['$\theta_{', num2str(plotCoord), '}[deg]$'], 'Interpreter', 'latex');
    xlabel(hnd, '$t$ [s]', 'Interpreter', 'latex');
    xlabel('time')
end