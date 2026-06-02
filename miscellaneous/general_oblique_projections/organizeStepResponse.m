function [yData, fData, fDataMeasured, info] = organizeStepResponse(T,embedDim, delaySteps,smooth_window)

%% Load data and split
Tstep = T;
xDataAllStep = {Tstep.Time.', Tstep.PositionFiltered.'};
fDataAllStep = {Tstep.Time.', Tstep.CommandedVoltage.'};
fDataMeasuredAllStep = {Tstep.Time.', Tstep.MeasuredInputVoltage.'};
dt = Tstep.Time(2)-Tstep.Time(1);
info.dt = dt;

index_level = find(diff(fDataAllStep{1,2}) == 0);
index_level = index_level(fDataAllStep{1,2}(index_level)~=0);
index_level_end = [index_level(diff(index_level)~=1) index_level(end-5000)];
fl_index_level = flip(index_level);
index_level_start = fl_index_level(diff(fl_index_level)~=-1);
index_level_start = [index_level(1) flip(index_level_start)];

% index_level_start = index_level_start(2:end);
% index_level_end = index_level_end(2:end);


figure; hold on; grid on;
plot(fDataAllStep{1,1}, fDataAllStep{1,2},'r.','LineWidth',2)
plot(fDataAllStep{1,1}(index_level), fDataAllStep{1,2}(index_level),'go')
plot(fDataAllStep{1,1}(index_level_end), fDataAllStep{1,2}(index_level_end),'b*','MarkerSize',15,'LineWidth',2)
plot(fDataAllStep{1,1}(index_level_start), fDataAllStep{1,2}(index_level_start),'r*','MarkerSize',15,'LineWidth',2)


figure; hold on; grid on;
plot(xDataAllStep{1,1}, xDataAllStep{1,2},'r.','LineWidth',2)
% plot(xDataAllStep{1,1}(index_level), xDataAllStep{1,2}(index_level),'go')
plot(xDataAllStep{1,1}(index_level_end), xDataAllStep{1,2}(index_level_end),'b*','MarkerSize',15,'LineWidth',2)
plot(xDataAllStep{1,1}(index_level_start), xDataAllStep{1,2}(index_level_start),'r*','MarkerSize',15,'LineWidth',2)



% [pks, locs] = findpeaks(abs(diff(fDataAllStep{1,2})),MinPeakDistance=5000);
% locs = locs(pks>70)+1;
% locs = [1,locs];
% 
% diffF = abs(diff(fDataAllStep{1,2}));

% figure; hold on;
% plot(diffF)
% plot(locs,diffF(locs),'ro')
% 
% figure; hold on; grid on;
% plot(xDataAllStep{1,2},'k')
% plot(locs,xDataAllStep{1,2}(locs),'co')
% 



% xDataStep = makeTrajectories(xDataAllStep, locs(1:end-1), locs(2:end)-100);
% fDataStep = makeTrajectories(fDataAllStep, locs(1:end-1), locs(2:end)-100);


xDataStep = makeTrajectories(xDataAllStep, index_level_start, index_level_end);
fDataStep = makeTrajectories(fDataAllStep, index_level_start, index_level_end);
fDataMeasuredStep = makeTrajectories(fDataMeasuredAllStep, index_level_start, index_level_end);

% interpolate the data to regularize the sampling 
t_end_vect = zeros(size(xDataStep,1),1);
for vv = 1:size(xDataStep,1)
    t_end_vect(vv) = xDataStep{vv,1}(end);
end
t_end_min = min(t_end_vect);
t_interp = 0:dt:t_end_min;
for vv = 1:size(xDataStep,1)
    % smoothen the signal
    signal = interp1(xDataStep{vv,1}, xDataStep{vv,2}, t_interp);
    xDataStep{vv,2} = transpose(smooth(signal,smooth_window));
    xDataStep{vv,1} = t_interp;

%     figure; hold on; grid on;
%     plot(t_interp,signal,'k')
%     plot(t_interp,xDataStep{vv,2},'r')
end


xData = [xDataStep]; fData = [fDataStep]; fDataMeasured = [fDataMeasuredStep];
yData = embedCoordinates(xData, embedDim, delaySteps);
fData = embedCoordinates(fData, embedDim, delaySteps);
fDataMeasured = embedCoordinates(fDataMeasured, embedDim, delaySteps);
nTraj = size(xData, 1);
for iTraj = 1:nTraj
    fData{iTraj,2} = fData{iTraj,2}(1,:);
    info.Voltages(iTraj) = round(mean(fData{iTraj,2}), 3);
end
end

function xData = makeTrajectories(xDataAll, locsstart, locsend)
for iTraj = 1:length(locsstart)
    trajInds = locsstart(iTraj):(locsend(iTraj)-1);
    t = xDataAll{1,1}(trajInds)-xDataAll{1,1}(trajInds(1));
    x = xDataAll{1,2}(:,trajInds);
    xData(iTraj,:) = {t, x};
end
end